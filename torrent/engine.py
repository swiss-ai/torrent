from __future__ import annotations

import logging
import aiohttp
import asyncio
from typing import Optional, Union, List
from datasets.features import Features, Value
from datasets import Dataset, IterableDataset

from torrent.db import ServingDB
from torrent.launcher import Launcher
from torrent.types import WorkerStatistics

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self) -> None:
        self.launcher = Launcher()
        self.serving_db = ServingDB()

    def run(
        self,
        dataset: Union[Dataset, IterableDataset],
        model_path: str,
    ) -> None:
        self.validate_dataset(dataset)

        if isinstance(dataset, Dataset):
            dataset = dataset.to_iterable_dataset()

        dataset_iterator = dataset.iter(batch_size=8)

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
            if not isinstance(sampling_params_feature, Features):
                raise ValueError(
                    f"Sampling params feature must be a dictionary, got {sampling_params_feature}"
                )

    def get_statistics(self, urls: List[str]) -> List[WorkerStatistics]:
        async def fetch_worker_statistics(
            session: aiohttp.ClientSession, url: str, index: int
        ) -> dict:
            try:
                async with session.get(f"{url}/metrics") as response:
                    response.raise_for_status()
                    return {"index": index, "worker_statistics": await response.json()}
            except Exception as e:
                logger.error(f"Failed to get worker statistics from {url}: {e}")
                return {"index": index, "worker_statistics": None}

        async def fetch_all():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    fetch_worker_statistics(session, url, index)
                    for index, url in enumerate(urls)
                ]
                return [
                    x["worker_statistics"]
                    for x in sorted(
                        await asyncio.gather(*tasks), key=lambda x: x["index"]
                    )
                    if x["worker_statistics"] is not None
                ]

        return asyncio.run(fetch_all())
