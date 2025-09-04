from __future__ import annotations

import logging
from atexit import register
from dacite import from_dict
from omegaconf import OmegaConf
from importlib.resources import files
from transformers import AutoTokenizer
from typing import Union, Dict, Any, Optional
from datasets.features import Features, Value
from datasets import Dataset, IterableDataset

from torrent.db import ServingDB
from torrent.utils import nanoid
from torrent.monitor import Monitor
from torrent.launcher import Launcher
from torrent.types import RunMetadata, ModelConfig, ServingConfig

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self) -> None:
        self.launcher = Launcher()
        self.serving_db = ServingDB()
        self.monitor = Monitor()

    def run(
        self,
        dataset: Union[Dataset, IterableDataset],
        model_path: str,
        batch_size: int = 32,
    ) -> None:
        model_config = self.get_model_config(model_path)
        if model_config is None:
            raise ValueError(
                f"Model config not found for {model_path}. Before using this model, you need to add a config file here: https://github.com/swiss-ai/torrent/tree/main/torrent/multi_node/configs"
            )

        self.validate_dataset(dataset)

        if isinstance(dataset, Dataset):
            dataset = dataset.to_iterable_dataset()

        run_id = nanoid()
        metadata = self.get_metadata(dataset, model_path)

        model_instances = self.serving_db.get(model_path)
        serving_config = self.get_serving_config(model_config, metadata)

        has_to_wait = False
        if (
            model_instances is None
            or model_instances.get_num_workers("launched") < serving_config.num_workers
        ):
            serving_config.num_workers = (
                serving_config.num_workers - model_instances.get_num_workers("launched")
            )

            new_instance = self.launcher.launch(model_config, serving_config)
            self.serving_db.add_instance(model_path, new_instance)
            has_to_wait = True

        self.serving_db.add_to_queue(model_path, run_id)

        if has_to_wait:
            new_instance = self.launcher.wait_for_model_instance(new_instance)
            self.serving_db.update_instance(model_path, new_instance)

        dataset_iterator = dataset.iter(batch_size=batch_size)

        self.serving_db.remove_from_queue(model_path, run_id)

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

    def get_model_config(self, model_path: str) -> Optional[ModelConfig]:
        mode_path = self.launcher.get_mode_path()
        config_files = files(f"torrent.{mode_path}")
        config_path = config_files / "configs" / f"{model_path}.yaml"

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
        )
        num_workers = 8  # TODO: add a way to get the number of workers

        return ServingConfig(
            num_workers=num_workers,
            num_nodes_per_worker=num_nodes_per_worker,
        )

    @register
    def cleanup(self) -> None:
        for model_path in self.serving_db.keys():
            model_instances = self.serving_db.get(model_path)
            if model_instances.queue == []:
                for model_instance in model_instances.model_instances:
                    self.launcher.cancel_instance(model_instance)
                self.serving_db.delete(model_path)
