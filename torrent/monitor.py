import logging
import re
import threading
import time
from typing import List, Dict, Callable, Optional

import aiohttp
import asyncio
from torrent.types import WorkerMetrics

logger = logging.getLogger(__name__)


class Monitor:
    def __init__(self) -> None:
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._health_callbacks: List[Callable[[str, bool], None]] = []
        self._metrics_callbacks: List[Callable[[str, WorkerMetrics], None]] = []
        self._worker_health_status: Dict[str, bool] = {}

    def add_health_callback(self, callback: Callable[[str, bool], None]) -> None:
        """Add a callback that will be called when worker health status changes."""
        self._health_callbacks.append(callback)

    def add_metrics_callback(self, callback: Callable[[str, WorkerMetrics], None]) -> None:
        """Add a callback that will be called when worker metrics are updated."""
        self._metrics_callbacks.append(callback)

    def start_background_monitoring(self, workers_head_ids: List[str], port: int, interval: int = 5) -> None:
        """Start background monitoring thread for health checks and metrics."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Background monitoring is already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(workers_head_ids, port, interval),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info(f"Started background monitoring for {len(workers_head_ids)} workers")

    def stop_background_monitoring(self) -> None:
        """Stop background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=10)
            logger.info("Stopped background monitoring")

    def _monitoring_loop(self, workers_head_ids: List[str], port: int, interval: int) -> None:
        """Main monitoring loop that runs in background thread."""
        while not self._stop_monitoring.is_set():
            try:
                # Check health status
                health_results = self.check_workers_health(workers_head_ids, port)
                
                # Update health status and trigger callbacks
                for worker_id, is_healthy in health_results.items():
                    previous_status = self._worker_health_status.get(worker_id, None)
                    if previous_status != is_healthy:
                        self._worker_health_status[worker_id] = is_healthy
                        for callback in self._health_callbacks:
                            try:
                                callback(worker_id, is_healthy)
                            except Exception as e:
                                logger.error(f"Error in health callback: {e}")
                
                # Get metrics from healthy workers
                healthy_workers = [worker_id for worker_id, is_healthy in health_results.items() if is_healthy]
                if healthy_workers:
                    metrics_list = self.get_metrics(healthy_workers, port)
                    for worker_id, metrics in zip(healthy_workers, metrics_list):
                        if metrics:
                            for callback in self._metrics_callbacks:
                                try:
                                    callback(worker_id, metrics)
                                except Exception as e:
                                    logger.error(f"Error in metrics callback: {e}")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next iteration
            self._stop_monitoring.wait(interval)

    def check_workers_health(self, workers_head_ids: List[str], port: int) -> Dict[str, bool]:
        """Check health status of all workers."""
        async def check_worker_health(session: aiohttp.ClientSession, url: str, worker_id: str) -> dict:
            try:
                async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    is_healthy = response.status == 200
                    return {"worker_id": worker_id, "is_healthy": is_healthy}
            except Exception as e:
                logger.debug(f"Health check failed for {url}: {e}")
                return {"worker_id": worker_id, "is_healthy": False}

        async def check_all():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    check_worker_health(session, f"http://{worker_id}:{port}", worker_id)
                    for worker_id in workers_head_ids
                ]
                results = await asyncio.gather(*tasks)
                return {result["worker_id"]: result["is_healthy"] for result in results}

        return asyncio.run(check_all())

    def get_metrics(
        self, workers_head_ids: List[str], port: int
    ) -> List[Optional[WorkerMetrics]]:
        async def fetch_worker_metrics(
            session: aiohttp.ClientSession, url: str, index: int, worker_id: str
        ) -> dict:
            try:
                async with session.get(f"{url}/metrics") as response:
                    response.raise_for_status()
                    metrics_text = await response.text()
                    return {"index": index, "worker_id": worker_id, "metrics_text": metrics_text}
            except Exception as e:
                logger.error(f"Failed to get worker metrics from {url}: {e}")
                return {"index": index, "worker_id": worker_id, "metrics_text": None}

        async def fetch_all():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    fetch_worker_metrics(session, f"http://{worker_id}:{port}", index, worker_id)
                    for index, worker_id in enumerate(workers_head_ids)
                ]
                results = sorted(await asyncio.gather(*tasks), key=lambda x: x["index"])
                return [
                    self.parse_metrics(result["metrics_text"], result["worker_id"])
                    if result["metrics_text"] is not None
                    else None
                    for result in results
                ]

        return asyncio.run(fetch_all())

    def parse_metrics(self, metrics: str, worker_id: str) -> WorkerMetrics:
        worker_metrics = WorkerMetrics(worker_id=worker_id)

        for line in metrics.strip().splitlines():
            line = line.strip()

            match = re.match(r"sglang:(\w+)(?:\{([^}]*)\})?\s+([\d.]+)", line)
            if not match:
                continue

            metric_name, _, value_str = match.groups()
            value = float(value_str)

            if hasattr(worker_metrics, metric_name):
                setattr(worker_metrics, metric_name, value)

        return worker_metrics
