from typing import List
from redislite import Redis
from dacite import from_dict
from json import loads, dumps
from dataclasses import asdict

from torrent.utils import TMP_PATH
from torrent.types import ModelInstance


class ServingDB:
    def __init__(self) -> None:
        self.db = Redis(TMP_PATH)

    def get(self, model_path: str) -> ModelInstance:
        return from_dict(ModelInstance, loads(self.db.get(model_path)))

    def set(self, model_instance: ModelInstance) -> None:
        self.db.set(
            model_instance.model_config.model_path, dumps(asdict(model_instance))
        )

    def delete(self, model_path: str) -> None:
        self.db.delete(model_path)

    def is_available(self, model_path: str) -> bool:
        return self.db.get(model_path) is not None
