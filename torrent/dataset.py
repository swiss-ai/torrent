from __future__ import annotations

import threading
from tqdm import tqdm
from time import time
from typing import Dict, Any, List, Optional
from datasets import Dataset as HuggingFaceDataset

from torrent.types import Batch


class Dataset:
    def __init__(self, dataset: HuggingFaceDataset, batch_size: int, timeout: int = 120) -> None:
        self.map: Dict[str, List[Batch]] = {}
        self.len_input_dataset = len(dataset)
        self.num_batches = self.len_input_dataset // batch_size
        self.current_index = 0
        self.timeout = timeout
        self.input_dataset = dataset
        self.output_dataset = []
        self.map_lock = threading.Lock()
        self.input_dataset_lock = threading.Lock()
        self.output_dataset_lock = threading.Lock()

        def map_fn(x: Dict[str, Any], index: int) -> Dict[str, Any]:
            x["batch_index"] = index // batch_size
            return x

        self.input_dataset = self.input_dataset.map(map_fn, with_indices=True)
        self.input_dataset_iterator = self.input_dataset.iter(batch_size=batch_size)

        self.progress_bar = tqdm(total=self.num_batches, desc="Processing dataset")

    def is_done(self) -> bool:
        return self.output_dataset == self.num_batches

    def pull(self, worker_id: str) -> Optional[Dict[str, Any]]:
        with self.input_dataset_lock:
            input_batch = None
            if self.current_index < self.num_batches:
                input_batch = next(self.input_dataset_iterator)
            else:
                input_batch = self.get_one_timed_out_batch()

            if input_batch is None:
                return None

            self.current_index += 1
        with self.map_lock:
            batches = self.map.get(worker_id, []) + [
                Batch(index=input_batch["batch_index"][0], start_time=time(), data=input_batch)
            ]
            self.map[worker_id] = batches
        return input_batch

    def push(self, worker_id: str, batch: Dict[str, Any]) -> None:
        with self.map_lock:
            if worker_id not in self.map:
                return

            worker_batches = self.map[worker_id]
            if batch["batch_index"][0] not in [
                worker_batch.index for worker_batch in worker_batches
            ]:
                return

            self.map[worker_id] = [
                batch
                for worker_batch in worker_batches
                if worker_batch.index != batch["batch_index"][0]
            ]
            with self.output_dataset_lock:
                self.output_dataset.append(batch)
                self.progress_bar.update(1)

    def get_one_timed_out_batch(self) -> Optional[Dict[str, Any]]:
        with self.map_lock:
            for worker_id, batches in self.map.items():
                for batch in batches:
                    if time() - batch.start_time > self.timeout:
                        self.map[worker_id] = [
                            worker_batch
                            for worker_batch in batches
                            if worker_batch.index != batch.index
                        ]
                        return batch.data
        return None


from datasets import load_dataset

dataset = Dataset(
    load_dataset("openai/gsm8k", name="main", split="train"), batch_size=32
)

batch = dataset.pull("worker_id")
dataset.push("worker_id", batch)
print(dataset.output_dataset)
print(dataset.map)
