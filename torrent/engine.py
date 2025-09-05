from __future__ import annotations

import logging
import time
import threading
import aiohttp
import asyncio
from atexit import register
from dacite import from_dict
from omegaconf import OmegaConf
from importlib.resources import files
from transformers import AutoTokenizer
from datasets.features import Value
from datasets import Dataset, IterableDataset
from typing import Union, Dict, Any, Optional, List

from torrent.db import ServingDB
from torrent.utils import nanoid
from torrent.monitor import Monitor
from torrent.launcher import Launcher
from torrent.types import RunMetadata, ModelConfig, ServingConfig, WorkerMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Engine:
    def __init__(self) -> None:
        self.launcher = Launcher()
        self.serving_db = ServingDB()
        self.monitor = Monitor()
        
        # Tracking variables for throughput control
        self._worker_health: Dict[str, bool] = {}
        self._worker_metrics: Dict[str, WorkerMetrics] = {}
        self._target_waiting_requests = 2000
        self._request_lock = threading.Lock()
        
        # Set up monitoring callbacks
        self.monitor.add_health_callback(self._on_worker_health_change)
        self.monitor.add_metrics_callback(self._on_worker_metrics_update)

    def run(
        self,
        dataset: Dataset,
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

        dataset = dataset.to_iterable_dataset()

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
                needed_workers = serving_config.num_workers - model_instances.get_num_workers()

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
        if model_instances.workers_head_ids:
            self.monitor.start_background_monitoring(
                model_instances.workers_head_ids, 
                serving_config.port,
                interval=2  # Check every 2 seconds for responsiveness
            )

        # Wait in the queue
        while True:
            model_instances = self.serving_db.get(model_path)
            if model_instances.queue[0] == run_id:
                break
            logger.info("Waiting in the queue...")
            time.sleep(1)

        # Main processing loop
        self._process_dataset(dataset, model_path, batch_size, serving_config.port, run_id)

        # Remove from queue and update waiting requests
        self.serving_db.remove_from_queue(model_path, run_id)
        model_instances = self.serving_db.get(model_path)
        model_instances.num_waiting_requests = max(0, model_instances.num_waiting_requests - metadata.num_rows)
        self.serving_db.set(model_path, model_instances)

    def _process_dataset(
        self, 
        dataset: Union[Dataset, IterableDataset], 
        model_path: str, 
        batch_size: int, 
        port: int,
        run_id: str
    ) -> None:
        """Main processing loop that distributes dataset while maintaining throughput."""
        logger.info(f"Starting dataset processing for run {run_id}")
        
        dataset_iterator = dataset.iter(batch_size=batch_size)
        total_batches_sent = 0
        
        for batch in dataset_iterator:
            # Wait until we have healthy workers
            while True:
                healthy_workers = self._get_healthy_workers(model_path)
                if healthy_workers:
                    break
                logger.info("Waiting for healthy workers...")
                time.sleep(5)
            
            # Throttle based on current waiting requests to maintain target throughput
            while True:
                current_waiting = self._get_current_waiting_requests()
                if current_waiting < self._target_waiting_requests:
                    break
                logger.debug(f"Throttling: current waiting requests ({current_waiting}) >= target ({self._target_waiting_requests})")
                time.sleep(1)
            
            # Prepare batch requests
            batch_requests = []
            for input, sampling_params in zip(batch["input"], batch["sampling_params"]):
                request_data = {
                    "text": input,
                    "sampling_params": sampling_params
                }
                batch_requests.append(request_data)
            
            # Send batch to workers
            try:
                asyncio.run(self._send_batch_requests(batch_requests, healthy_workers, port))
                total_batches_sent += 1
                logger.debug(f"Sent batch {total_batches_sent} ({len(batch_requests)} requests)")
                
            except Exception as e:
                logger.error(f"Failed to send batch: {e}")
                time.sleep(1)  # Brief pause before retrying
        
        logger.info(f"Completed processing {total_batches_sent} batches for run {run_id}")

    def validate_dataset(self, dataset: Union[Dataset, IterableDataset]) -> None:
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
        self, dataset: Union[Dataset, IterableDataset], model_path: str
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
        num_nodes_per_worker = max(
            model_config.dp_size, model_config.tp_size, model_config.ep_size
        ) // 4
        num_workers = 2  # TODO: add a way to get the number of workers

        return ServingConfig(
            num_workers=num_workers,
            num_nodes_per_worker=num_nodes_per_worker,
        )

    def _on_worker_health_change(self, worker_id: str, is_healthy: bool) -> None:
        """Callback for when worker health status changes."""
        with self._request_lock:
            self._worker_health[worker_id] = is_healthy
            if is_healthy:
                logger.info(f"Worker {worker_id} is now healthy")
            else:
                logger.warning(f"Worker {worker_id} is now unhealthy")

    def _on_worker_metrics_update(self, worker_id: str, metrics: WorkerMetrics) -> None:
        """Callback for when worker metrics are updated."""
        with self._request_lock:
            self._worker_metrics[worker_id] = metrics
            logger.debug(f"Worker {worker_id}: queue_reqs={metrics.num_queue_reqs}, running_reqs={metrics.num_running_reqs}")

    def _get_current_waiting_requests(self) -> int:
        """Get total waiting requests across all healthy workers."""
        with self._request_lock:
            return sum(
                int(metrics.num_queue_reqs) 
                for worker_id, metrics in self._worker_metrics.items() 
                if self._worker_health.get(worker_id, False)
            )

    def _get_healthy_workers(self, model_path: str) -> List[str]:
        """Get list of healthy worker IDs for a model."""
        model_instances = self.serving_db.get(model_path)
        if not model_instances:
            return []
        
        all_workers = model_instances.workers_head_ids
        with self._request_lock:
            return [worker_id for worker_id in all_workers if self._worker_health.get(worker_id, False)]

    async def _send_batch_requests(self, batch: List[Dict[str, Any]], workers: List[str], port: int) -> None:
        """Send batch requests to available workers."""
        if not workers:
            logger.warning("No healthy workers available to send requests")
            return

        async def send_request(session: aiohttp.ClientSession, worker_id: str, request_data: Dict[str, Any]) -> None:
            try:
                url = f"http://{worker_id}:{port}/generate"
                async with session.post(url, json=request_data, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    response.raise_for_status()
                    result = await response.json()
                    logger.debug(f"Request completed on worker {worker_id}")
            except Exception as e:
                logger.error(f"Failed to send request to worker {worker_id}: {e}")

        # Distribute requests across workers using round-robin
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, request_data in enumerate(batch):
                worker_id = workers[i % len(workers)]
                task = send_request(session, worker_id, request_data)
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)

    # def __del__(self) -> None:
    #     # Stop monitoring
    #     self.monitor.stop_background_monitoring()
        
    #     # Clean up model instances
    #     for model_path in self.serving_db.keys():
    #         model_instances = self.serving_db.get(model_path)
    #         if model_instances.queue == []:
    #             for model_instance in model_instances.model_instances:
    #                 self.launcher.cancel_instance(model_instance)
    #             self.serving_db.delete(model_path)
