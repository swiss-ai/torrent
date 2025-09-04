import logging
import re
from typing import List

import aiohttp
import asyncio
from torrent.types import WorkerMetrics

logger = logging.getLogger(__name__)


class Monitor:
    def __init__(self) -> None:
        pass

    def get_metrics(
        self, workers_head_ids: List[str], port: int
    ) -> List[WorkerMetrics]:
        async def fetch_worker_metrics(
            session: aiohttp.ClientSession, url: str, index: int
        ) -> dict:
            try:
                async with session.get(f"{url}/metrics") as response:
                    response.raise_for_status()
                    return {"index": index, "worker_metrics": await response.json()}
            except Exception as e:
                logger.error(f"Failed to get worker metrics from {url}: {e}")
                return {"index": index, "worker_metrics": None}

        async def fetch_all():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    fetch_worker_metrics(session, f"http://{id}:{port}", index)
                    for index, id in enumerate(workers_head_ids)
                ]
                return [
                    x["worker_metrics"]
                    for x in sorted(
                        await asyncio.gather(*tasks), key=lambda x: x["index"]
                    )
                    if x["worker_metrics"] is not None
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
