import os
from redislite import Redis
from dacite import from_dict
from json import loads, dumps
from dataclasses import asdict
from typing import List, Optional

from torrent.utils import TMP_PATH
from torrent.types import ModelInstances, ModelInstance


class ServingDB:
    def __init__(self) -> None:
        os.makedirs(TMP_PATH, mode=0o777, exist_ok=True)
        self.db = Redis(f"{TMP_PATH}/serving.db")

    def keys(self) -> List[str]:
        return self.db.keys()

    def get(self, model_path: str) -> Optional[ModelInstances]:
        value = self.db.get(model_path)
        if value is None:
            return None
        return from_dict(ModelInstances, loads(value))

    def set(self, model_path: str, model_instances: ModelInstances) -> None:
        self.db.set(model_path, dumps(asdict(model_instances)))

    def delete(self, model_path: str) -> None:
        self.db.delete(model_path)

    def add_instance(self, model_path: str, model_instance: ModelInstance) -> None:
        model_instances = self.get(model_path) or ModelInstances(
            model_instances=[], queue=[]
        )
        model_instances.model_instances.append(model_instance)
        self.set(model_path, model_instances)

    def add_to_queue(self, model_path: str, run_id: str) -> None:
        model_instances = self.get(model_path)
        model_instances.queue.append(run_id)
        self.set(model_path, model_instances)

    def remove_from_queue(self, model_path: str, run_id: str) -> None:
        model_instances = self.get(model_path)
        model_instances.queue = [
            queue_run_id
            for queue_run_id in model_instances.queue
            if queue_run_id != run_id
        ]
        self.set(model_path, model_instances)

    def update_instance(self, model_path: str, model_instance: ModelInstance) -> None:
        model_instances = self.get(model_path)
        model_instances.model_instances = [
            existing_instance
            for existing_instance in model_instances.model_instances
            if existing_instance.job_id != model_instance.job_id
        ]
        model_instances.model_instances.append(model_instance)
        self.set(model_path, model_instances)
