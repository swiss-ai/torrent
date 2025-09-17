import os
import warnings
from typing import List
from json import dumps, loads
from dataclasses import asdict
from dacite import from_dict, Config
from filelock import FileLock

from torrent.types import RunMetadata, WorkerInfos, WorkerStatus, RunStatus, Usage

warnings.filterwarnings("ignore")


class TorrentDB:
    def __init__(self, path: str) -> None:
        self.db_path = os.path.join(os.path.abspath(path), "db")
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)

        self.dacite_config = Config(cast=[RunStatus, WorkerStatus])

    def _key_to_path(self, key: str) -> str:
        return os.path.join(self.db_path, *key.split(":"))

    def add_run(self, run_metadata: RunMetadata) -> None:
        metadata_path = self._key_to_path(f"{run_metadata.id}:metadata")
        if os.path.exists(metadata_path):
            raise ValueError(f"Run {run_metadata.id} already exists")

        index_path = self._key_to_path(f"{run_metadata.id}:index")

        lock_path = metadata_path + ".lock"
        with FileLock(lock_path):
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, "w") as f:
                f.write(dumps(asdict(run_metadata)))

        lock_path_index = index_path + ".lock"
        with FileLock(lock_path_index):
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            with open(index_path, "w") as f:
                f.write("0")

    def get_run(self, id: str) -> RunMetadata:
        path = self._key_to_path(f"{id}:metadata")
        lock_path = path + ".lock"
        with FileLock(lock_path):
            with open(path, "r") as f:
                content = f.read()
        return from_dict(RunMetadata, loads(content), config=self.dacite_config)

    def get_run_index(self, id: str) -> int:
        path = self._key_to_path(f"{id}:index")
        lock_path = path + ".lock"
        with FileLock(lock_path):
            with open(path, "r") as f:
                content = f.read()
        return int(content)

    def incr_run_index(self, id: str, incr: int = 1) -> int:
        path = self._key_to_path(f"{id}:index")
        lock_path = path + ".lock"
        with FileLock(lock_path):
            current_value = 0
            if os.path.exists(path):
                with open(path, "r") as f:
                    content = f.read()
                    if content:
                        current_value = int(content)

            new_value = current_value + incr

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(str(new_value))

            return new_value

    def add_worker(self, id: str, worker_infos: WorkerInfos) -> None:
        key = f"{id}:workers:{worker_infos.worker_head_node_id}"
        path = self._key_to_path(key)
        lock_path = path + ".lock"
        with FileLock(lock_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(dumps(asdict(worker_infos)))

    def get_worker(self, id: str, worker_head_node_id: str) -> WorkerInfos:
        key = f"{id}:workers:{worker_head_node_id}"
        path = self._key_to_path(key)
        lock_path = path + ".lock"
        with FileLock(lock_path):
            with open(path, "r") as f:
                content = f.read()
        return from_dict(
            WorkerInfos,
            loads(content),
            config=self.dacite_config,
        )

    def update_worker_status(
        self, id: str, worker_head_node_id: str, status: WorkerStatus
    ) -> None:
        key = f"{id}:workers:{worker_head_node_id}"
        path = self._key_to_path(key)
        lock_path = path + ".lock"
        with FileLock(lock_path):
            with open(path, "r") as f:
                content = f.read()
            worker_infos = from_dict(
                WorkerInfos, loads(content), config=self.dacite_config
            )
            worker_infos.status = status
            with open(path, "w") as f:
                f.write(dumps(asdict(worker_infos)))

    def update_worker_usage(
        self, id: str, worker_head_node_id: str, usage: Usage
    ) -> Usage:
        key = f"{id}:workers:{worker_head_node_id}"
        path = self._key_to_path(key)
        lock_path = path + ".lock"
        with FileLock(lock_path):
            with open(path, "r") as f:
                content = f.read()
            worker_infos = from_dict(
                WorkerInfos, loads(content), config=self.dacite_config
            )
            worker_infos.add_usage(usage)
            with open(path, "w") as f:
                f.write(dumps(asdict(worker_infos)))
            return worker_infos.usage

    def update_run_status(self, id: str, status: RunStatus) -> None:
        path = self._key_to_path(f"{id}:metadata")
        lock_path = path + ".lock"
        with FileLock(lock_path):
            with open(path, "r") as f:
                content = f.read()
            run_metadata = from_dict(
                RunMetadata, loads(content), config=self.dacite_config
            )
            run_metadata.status = status
            with open(path, "w") as f:
                f.write(dumps(asdict(run_metadata)))

    def list_runs(self) -> List[RunMetadata]:
        runs = []
        if not os.path.exists(self.db_path):
            return runs
        for run_id in os.listdir(self.db_path):
            metadata_path = os.path.join(self.db_path, run_id, "metadata")
            if os.path.isfile(metadata_path):
                runs.append(self.get_run(run_id))
        return runs

    def list_workers(self, run_id: str) -> List[WorkerInfos]:
        workers = []
        workers_dir = os.path.join(self.db_path, run_id, "workers")
        if not os.path.isdir(workers_dir):
            return workers

        for worker_id in os.listdir(workers_dir):
            worker_path = os.path.join(workers_dir, worker_id)
            if os.path.isfile(worker_path) and not worker_id.endswith(".lock"):
                workers.append(self.get_worker(run_id, worker_id))
        return workers

    def get_full_path(self) -> str:
        return self.db_path

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
