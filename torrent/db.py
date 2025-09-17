import os
import sqlite3
import warnings

warnings.filterwarnings("ignore")

from json import dumps, loads
from dataclasses import asdict
from typing import List, Optional
from dacite import from_dict, Config

from torrent.types import RunMetadata, WorkerInfos, WorkerStatus, RunStatus, Usage


class TorrentDB:
    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        self.db_path = f"{path}/torrent.db"
        self.db = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=10
        )
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute("PRAGMA synchronous=NORMAL;")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.db.commit()

        self.dacite_config = Config(cast=[RunStatus, WorkerStatus])

    def _get(self, key: str) -> Optional[str]:
        cursor = self.db.cursor()
        cursor.execute("SELECT value FROM kv WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def _set(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value)
            )

    def _keys(self, pattern: str) -> List[str]:
        cursor = self.db.cursor()
        sql_pattern = pattern.replace("*", "%")
        cursor.execute("SELECT key FROM kv WHERE key LIKE ?", (sql_pattern,))
        return [row[0] for row in cursor.fetchall()]

    def add_run(self, run_metadata: RunMetadata) -> None:
        if self._get(f"{run_metadata.id}:metadata") is not None:
            raise ValueError(f"Run {run_metadata.id} already exists")

        self._set(f"{run_metadata.id}:index", "0")
        self._set(f"{run_metadata.id}:metadata", dumps(asdict(run_metadata)))

    def get_run(self, id: str) -> RunMetadata:
        value = self._get(f"{id}:metadata")
        if value is None:
            raise ValueError(f"Run {id} not found")
        return from_dict(RunMetadata, loads(value), config=self.dacite_config)

    def get_run_index(self, id: str) -> int:
        value = self._get(f"{id}:index")
        return int(value) if value is not None else 0

    def incr_run_index(self, id: str, incr: int = 1) -> int:
        key = f"{id}:index"
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cursor.fetchone()
            current_value = int(row[0]) if row and row[0] is not None else 0
            new_value = current_value + incr
            cursor.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (key, str(new_value)),
            )
        return new_value

    def add_worker(self, id: str, worker_infos: WorkerInfos) -> None:
        key = f"{id}:workers:{worker_infos.worker_head_node_id}"
        self._set(key, dumps(asdict(worker_infos)))

    def get_worker(self, id: str, worker_head_node_id: str) -> WorkerInfos:
        value = self._get(f"{id}:workers:{worker_head_node_id}")
        if value is None:
            raise ValueError(f"Worker {worker_head_node_id} for run {id} not found")
        return from_dict(
            WorkerInfos,
            loads(value),
            config=self.dacite_config,
        )

    def update_worker_status(
        self, id: str, worker_head_node_id: str, status: WorkerStatus
    ) -> None:
        worker_infos = self.get_worker(id, worker_head_node_id)
        worker_infos.status = status
        self._set(
            f"{id}:workers:{worker_head_node_id}",
            dumps(asdict(worker_infos)),
        )

    def update_worker_usage(
        self, id: str, worker_head_node_id: str, usage: Usage
    ) -> Usage:
        worker_infos = self.get_worker(id, worker_head_node_id)
        worker_infos.add_usage(usage)
        self._set(
            f"{id}:workers:{worker_head_node_id}",
            dumps(asdict(worker_infos)),
        )
        return worker_infos.usage

    def update_run_status(self, id: str, status: RunStatus) -> None:
        run_metadata = self.get_run(id)
        run_metadata.status = status
        self._set(f"{id}:metadata", dumps(asdict(run_metadata)))

    def list_runs(self) -> List[RunMetadata]:
        keys = self._keys("*:metadata")
        return [self.get_run(key.split(":")[0]) for key in keys]

    def list_workers(self, run_id: str) -> List[WorkerInfos]:
        keys = self._keys(f"{run_id}:workers:*")
        return [
            self.get_worker(run_id, key.split(":")[2]) for key in keys
        ]

    def get_full_path(self) -> str:
        return os.path.dirname(self.db_path)

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
