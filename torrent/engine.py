from __future__ import annotations

import logging
import time
import threading
import aiohttp
import asyncio
import random
from dacite import from_dict
from omegaconf import OmegaConf
from datasets.features import Value
from importlib.resources import files
from transformers import AutoTokenizer
from typing import Dict, Any, Optional, List
from datasets import Dataset as HuggingFaceDataset

from torrent.db import ServingDB
from torrent.utils import nanoid
from torrent.launcher import Launcher
from torrent.dataset import Dataset as TorrentDataset
from torrent.types import RunMetadata, ModelConfig, ServingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Engine:
    def __init__(self) -> None:
        self.launcher = Launcher()
        self.serving_db = ServingDB()

        # Tracking variables for throughput control
        self._worker_health: Dict[str, bool] = {}
        self._target_concurrent_requests = 2000
        self._request_lock = threading.Lock()

        # Start health check thread
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self._health_check_thread.start()

    def _health_check_loop(self) -> None:
        """Periodically checks the health of all workers."""
        asyncio.run(self._async_health_check_loop())

    async def _async_health_check_loop(self) -> None:
        while True:
            await self._check_all_workers_health()
            await asyncio.sleep(10)

    async def _check_all_workers_health(self) -> None:
        """Checks the health of all workers across all models."""
        tasks = []
        try:
            async with aiohttp.ClientSession() as session:
                for model_path in self.serving_db.keys():
                    model_instances = self.serving_db.get(model_path)
                    if not model_instances:
                        continue

                    for instance in model_instances.model_instances:
                        if instance.state == "running" and instance.workers_head_ids:
                            for worker_id in instance.workers_head_ids:
                                task = self._check_worker_health(
                                    session, worker_id, instance.port
                                )
                                tasks.append(task)
                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"An error occurred during health check: {e}")

    async def _check_worker_health(
        self, session: aiohttp.ClientSession, worker_id: str, port: int
    ) -> None:
        """Check the health of a single worker and update its status."""
        is_healthy = False
        try:
            url = f"http://{worker_id}:{port}/health"
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    is_healthy = True
        except Exception as e:
            logger.warning(
                f"Health check failed for worker {worker_id} on port {port}: {e}"
            )

        with self._request_lock:
            if self._worker_health.get(worker_id) != is_healthy:
                if is_healthy:
                    logger.info(f"Worker {worker_id} is now healthy")
                else:
                    logger.warning(f"Worker {worker_id} is now unhealthy")
            self._worker_health[worker_id] = is_healthy

    def run(
        self,
        dataset: HuggingFaceDataset,
        model_path: str,
        batch_size: int = 32,
    ) -> None:
        model_config = self.get_model_config(model_path)
        if model_config is None:
            raise ValueError(
                f"Model config not found for {model_path}. Before using this model, you need to add a config file here: https://github.com/swiss-ai/torrent/tree/main/torrent/multi_node/configs"
            )

        self.validate_dataset(dataset)

        run_id = nanoid()
        metadata = self.get_metadata(dataset, model_path)
        torrent_dataset = TorrentDataset(dataset, batch_size)

        model_instances = self.serving_db.get(model_path)
        serving_config = self.get_serving_config(model_config, metadata)

        # Launch instances if needed
        has_to_wait = False
        if (
            model_instances is None
            or model_instances.get_num_workers() < serving_config.num_workers
        ):
            if model_instances is None:
                needed_workers = serving_config.num_workers
            else:
                needed_workers = (
                    serving_config.num_workers - model_instances.get_num_workers()
                )

            serving_config.num_workers = needed_workers
            new_instance = self.launcher.launch(model_config, serving_config)
            self.serving_db.add_instance(model_path, new_instance)
            has_to_wait = True

        # Add run to queue and update metadata
        self.serving_db.add_to_queue(model_path, run_id)

        # Update num_waiting_requests with dataset size
        model_instances = self.serving_db.get(model_path)
        model_instances.num_waiting_requests += metadata.num_rows
        self.serving_db.set(model_path, model_instances)

        # Wait for new instance to be ready and start monitoring
        if has_to_wait:
            new_instance = self.launcher.wait_for_model_instance(new_instance)
            self.serving_db.update_instance(model_path, new_instance)

        # Get updated model instances and start background monitoring
        model_instances = self.serving_db.get(model_path)

        # Wait in the queue
        while True:
            model_instances = self.serving_db.get(model_path)
            if model_instances.queue[0] == run_id:
                break
            logger.info("Waiting in the queue...")
            time.sleep(1)

        # Main processing loop
        asyncio.run(
            self._process_dataset(
                torrent_dataset, model_path, batch_size, serving_config.port, run_id
            )
        )

        # Remove from queue and update waiting requests
        self.serving_db.remove_from_queue(model_path, run_id)
        model_instances = self.serving_db.get(model_path)
        model_instances.num_waiting_requests = max(
            0, model_instances.num_waiting_requests - metadata.num_rows
        )
        self.serving_db.set(model_path, model_instances)

    async def _process_dataset(
        self,
        dataset: TorrentDataset,
        model_path: str,
        batch_size: int,
        port: int,
        run_id: str,
    ) -> None:
        logger.info(f"Starting dataset processing for run {run_id}")

        while not self._get_healthy_workers(model_path):
            logger.info("Waiting for healthy workers...")
            await asyncio.sleep(5)

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._request_worker(session, port, dataset, model_path)
                for _ in range(self._target_concurrent_requests)
            ]
            await asyncio.gather(*tasks)

        logger.info(f"Completed processing for run {run_id}")

    async def _request_worker(
        self,
        session: aiohttp.ClientSession,
        port: int,
        dataset: TorrentDataset,
        model_path: str,
    ) -> None:
        """A worker task that continuously processes batches from the dataset."""
        while not dataset.is_done():
            healthy_workers = self._get_healthy_workers(model_path)
            if not healthy_workers:
                await asyncio.sleep(1)
                continue

            worker_id = random.choice(healthy_workers)
            batch = dataset.pull(worker_id)

            if batch is None:
                await asyncio.sleep(0.1)
                continue

            try:
                request_data = {
                    "text": batch["input"],
                    "sampling_params": batch.get("sampling_params", {}),
                }
                results = await self._send_batch_request(
                    worker_id, request_data, port, session
                )
                output_batch = batch.copy()
                output_batch["output"] = results
                dataset.push(worker_id, output_batch)
                logger.debug(
                    f"Worker {worker_id} processed batch {batch['batch_index'][0]}"
                )
            except Exception as e:
                logger.error(f"Failed to process batch on worker {worker_id}: {e}")
                await asyncio.sleep(1)

    def validate_dataset(self, dataset: HuggingFaceDataset) -> None:
        features = dataset.features

        if "input" not in features:
            raise ValueError(
                f"Dataset must have an 'input' column, got {features.keys()}"
            )

        input_feature = features["input"]
        if not isinstance(input_feature, Value) or input_feature.dtype != "string":
            raise ValueError(f"Input feature must be a string, got {input_feature}")

        if "sampling_params" in features:
            sampling_params_feature = features["sampling_params"]
            if not isinstance(sampling_params_feature, dict):
                raise ValueError(
                    f"Sampling params feature must be a dictionary, got {sampling_params_feature}"
                )

    def get_model_config(self, model_path: str) -> Optional[ModelConfig]:
        mode_path = self.launcher.get_mode_path()
        config_files = files(f"torrent.{mode_path}")
        config_path = config_files / "configs" / f"{model_path.replace('/', '_')}.yaml"

        if not config_path.exists():
            return None

        return from_dict(ModelConfig, OmegaConf.create(config_path.read_text()))

    def get_metadata(
        self, dataset: HuggingFaceDataset, model_path: str
    ) -> RunMetadata:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def map_fn(x: Dict[str, Any]) -> Dict[str, Any]:
            input_ids = tokenizer(x["input"])["input_ids"]

            x["token_counts"] = [len(id_list) for id_list in input_ids]

            return x

        dataset_with_token_counts = dataset.map(
            map_fn,
            batched=True,
            num_proc=32,
        )

        return RunMetadata(
            num_rows=len(dataset_with_token_counts),
            num_input_tokens=sum(dataset_with_token_counts["token_counts"]),
        )

    def get_serving_config(
        self, model_config: ModelConfig, run_metadata: RunMetadata
    ) -> ServingConfig:
        num_nodes_per_worker = (
            max(model_config.dp_size, model_config.tp_size, model_config.ep_size) // 4
        )
        num_workers = 2  # TODO: add a way to get the number of workers

        return ServingConfig(
            num_workers=num_workers,
            num_nodes_per_worker=num_nodes_per_worker,
        )

    def _get_healthy_workers(self, model_path: str) -> List[str]:
        """Get list of healthy worker IDs for a model."""
        model_instances = self.serving_db.get(model_path)
        if not model_instances:
            return []

        all_workers = model_instances.workers_head_ids
        with self._request_lock:
            return [
                worker_id
                for worker_id in all_workers
                if self._worker_health.get(worker_id, False)
            ]

    async def _send_batch_request(
        self,
        worker_id: str,
        request_data: Dict[str, Any],
        port: int,
        session: aiohttp.ClientSession,
    ) -> List[Any]:
        """Send a single batch request to a worker and return the results."""
        try:
            url = f"http://{worker_id}:{port}/generate"
            async with session.post(
                url, json=request_data, timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                logger.debug(f"Request completed on worker {worker_id}")
                return result
        except Exception as e:
            logger.error(f"Failed to send request to worker {worker_id}: {e}")
            raise

    # def __del__(self) -> None:
    #     # Clean up model instances
    #     for model_path in self.serving_db.keys():
    #         model_instances = self.serving_db.get(model_path)
    #         if model_instances.queue == []:
    #             for model_instance in model_instances.model_instances:
    #                 self.launcher.cancel_instance(model_instance)
    #             self.serving_db.delete(model_path)
