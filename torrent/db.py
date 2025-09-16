import os
import warnings

warnings.filterwarnings("ignore")

from typing import List
from redislite import Redis
from dacite import from_dict, Config
from json import dumps, loads
from dataclasses import asdict

from torrent.types import RunMetadata, WorkerInfos, WorkerStatus, RunStatus, Usage


class TorrentDB:
    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        self.db = Redis(f"{path}/torrent.db")

        self.dacite_config = Config(cast=[RunStatus, WorkerStatus])

    def add_run(self, run_metadata: RunMetadata) -> None:
        if self.db.get(run_metadata.id) is not None:
            raise ValueError(f"Run {run_metadata.id} already exists")

        self.db.set(f"{run_metadata.id}:index", 0)
        self.db.set(f"{run_metadata.id}:metadata", dumps(asdict(run_metadata)))

    def get_run(self, id: str) -> RunMetadata:
        return from_dict(
            RunMetadata, loads(self.db.get(f"{id}:metadata")), config=self.dacite_config
        )

    def get_run_index(self, id: str) -> int:
        return self.db.get(f"{id}:index")

    def incr_run_index(self, id: str, incr: int = 1) -> int:
        return self.db.incrby(f"{id}:index", incr)

    def add_worker(self, id: str, worker_infos: WorkerInfos) -> None:
        out = self.db.set(
            f"{id}:workers:{worker_infos.worker_head_node_id}",
            dumps(asdict(worker_infos)),
        )
        print(worker_infos, out)

        retrieved = self.get_worker(id, worker_infos.worker_head_node_id)
        print(worker_infos, retrieved)

    def get_worker(self, id: str, worker_head_node_id: str) -> WorkerInfos:
        return from_dict(
            WorkerInfos,
            loads(self.db.get(f"{id}:workers:{worker_head_node_id}")),
            config=self.dacite_config,
        )

    def update_worker_status(
        self, id: str, worker_head_node_id: str, status: WorkerStatus
    ) -> None:
        worker_infos = self.get_worker(id, worker_head_node_id)
        worker_infos.status = status
        self.db.set(
            f"{id}:workers:{worker_head_node_id}",
            dumps(asdict(worker_infos)),
        )

    def update_worker_usage(
        self, id: str, worker_head_node_id: str, usage: Usage
    ) -> Usage:
        worker_infos = self.get_worker(id, worker_head_node_id)
        worker_infos.add_usage(usage)
        self.db.set(
            f"{id}:workers:{worker_head_node_id}",
            dumps(asdict(worker_infos)),
        )
        return worker_infos.usage

    def list_runs(self) -> List[RunMetadata]:
        keys = self.db.keys(pattern="*:metadata")
        return [self.get_run(key.decode("utf-8").split(":")[0]) for key in keys]

    def list_workers(self, id: str) -> List[WorkerInfos]:
        keys = self.db.keys(pattern=f"{id}:workers:*")
        return [self.get_worker(id, key.decode("utf-8").split(":")[2]) for key in keys]

    def get_full_path(self) -> str:
        return "/".join(self.db.db.split("/")[:-1])
