from redislite import Redis
from dacite import from_dict
from json import loads, dumps
from dataclasses import asdict

from torrent.utils import TMP_PATH
from torrent.types import ModelInstances, ModelInstance


class ServingDB:
    def __init__(self) -> None:
        self.db = Redis(TMP_PATH)

    def get(self, model_path: str) -> ModelInstances:
        return from_dict(ModelInstances, loads(self.db.get(model_path)))

    def set(self, model_instances: ModelInstances) -> None:
        self.db.set(
            model_instances.model_config.model_path, dumps(asdict(model_instances))
        )

    def delete(self, model_path: str) -> None:
        self.db.delete(model_path)

    def add_instance(self, model_path: str, instance: ModelInstance) -> None:
        model_instances = self.get(model_path)
        model_instances.model_instances.append(instance)
        self.set(model_instances)
