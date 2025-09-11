from typing import List
from redislite import Redis
from dacite import from_dict
from json import dumps, loads
from dataclasses import asdict

from torrent.utils import TORRENT_PATH
from torrent.types import RunMetadata, WorkerInfos, WorkerStatus, Usage


class TorrentDB:
    def __init__(self, path: str = TORRENT_PATH) -> None:
        self.db = Redis(f"{path}/torrent.db")

    def add_run(self, run_metadata: RunMetadata) -> None:
        if self.db.get(run_metadata.id) is not None:
            raise ValueError(f"Run {run_metadata.id} already exists")

        self.db.set(f"{run_metadata.id}:index", 0)
        self.db.set(f"{run_metadata.id}:metadata", dumps(asdict(run_metadata)))

    def get_run(self, id: str) -> RunMetadata:
        return from_dict(RunMetadata, loads(self.db.get(f"{id}:metadata")))

    def get_run_index(self, id: str) -> int:
        return self.db.get(f"{id}:index")

    def incr_run_index(self, id: str, incr: int = 1) -> int:
        return self.db.incrby(f"{id}:index", incr)

    def add_worker(self, id: str, worker_infos: WorkerInfos) -> None:
        self.db.set(
            f"{id}:workers:{worker_infos.worker_head_node_id}",
            dumps(asdict(worker_infos)),
        )

    def get_worker(self, id: str, worker_head_node_id: str) -> WorkerInfos:
        return from_dict(
            WorkerInfos, loads(self.db.get(f"{id}:workers:{worker_head_node_id}"))
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
    ) -> None:
        worker_infos = self.get_worker(id, worker_head_node_id)
        worker_infos.usage = usage
        self.db.set(
            f"{id}:workers:{worker_head_node_id}",
            dumps(asdict(worker_infos)),
        )

    def list_runs(self) -> List[RunMetadata]:
        keys = self.db.keys(pattern="*:metadata")
        return [self.get_run(key.split(":")[0]) for key in keys]

    def list_workers(self, id: str) -> List[WorkerInfos]:
        keys = self.db.keys(pattern=f"{id}:workers:*")
        return [self.get_worker(id, key.split(":")[2]) for key in keys]
