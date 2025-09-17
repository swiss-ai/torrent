import os
import sqlite3
import warnings
from filelock import FileLock

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
        lock_path = f"{self.db_path}.lock"
        self.db = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)

        with FileLock(lock_path):
            cursor = self.db.cursor()

            cursor.execute("PRAGMA synchronous")
            if cursor.fetchone()[0] != 1:  # NORMAL is 1
                cursor.execute("PRAGMA synchronous=NORMAL;")

            cursor.execute("PRAGMA journal_mode")
            if cursor.fetchone()[0] != "wal":
                self.db.execute("PRAGMA journal_mode=WAL;")

            self.db.execute(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, metadata TEXT)"
            )
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS workers (
                    run_id TEXT,
                    worker_head_node_id TEXT,
                    infos TEXT,
                    PRIMARY KEY (run_id, worker_head_node_id)
                )"""
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS indices (run_id TEXT PRIMARY KEY, value INTEGER)"
            )
            self.db.commit()

        self.dacite_config = Config(cast=[RunStatus, WorkerStatus])

    def add_run(self, run_metadata: RunMetadata) -> None:
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("SELECT id FROM runs WHERE id = ?", (run_metadata.id,))
            if cursor.fetchone() is not None:
                raise ValueError(f"Run {run_metadata.id} already exists")

            cursor.execute(
                "INSERT INTO runs (id, metadata) VALUES (?, ?)",
                (run_metadata.id, dumps(asdict(run_metadata))),
            )
            cursor.execute(
                "INSERT INTO indices (run_id, value) VALUES (?, ?)", (run_metadata.id, 0)
            )

    def get_run(self, id: str) -> RunMetadata:
        cursor = self.db.cursor()
        cursor.execute("SELECT metadata FROM runs WHERE id = ?", (id,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Run {id} not found")
        return from_dict(RunMetadata, loads(row[0]), config=self.dacite_config)

    def get_run_index(self, id: str) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT value FROM indices WHERE run_id = ?", (id,))
        row = cursor.fetchone()
        return row[0] if row else 0

    def incr_run_index(self, id: str, incr: int = 1) -> int:
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("SELECT value FROM indices WHERE run_id = ?", (id,))
            row = cursor.fetchone()
            current_value = row[0] if row else 0
            new_value = current_value + incr
            cursor.execute(
                "INSERT OR REPLACE INTO indices (run_id, value) VALUES (?, ?)",
                (id, new_value),
            )
        return new_value

    def add_worker(self, id: str, worker_infos: WorkerInfos) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO workers (run_id, worker_head_node_id, infos) VALUES (?, ?, ?)",
                (id, worker_infos.worker_head_node_id, dumps(asdict(worker_infos))),
            )

    def get_worker(self, id: str, worker_head_node_id: str) -> WorkerInfos:
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT infos FROM workers WHERE run_id = ? AND worker_head_node_id = ?",
            (id, worker_head_node_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Worker {worker_head_node_id} for run {id} not found")
        return from_dict(
            WorkerInfos,
            loads(row[0]),
            config=self.dacite_config,
        )

    def update_worker_status(
        self, id: str, worker_head_node_id: str, status: WorkerStatus
    ) -> None:
        worker_infos = self.get_worker(id, worker_head_node_id)
        worker_infos.status = status
        with self.db:
            self.db.execute(
                "UPDATE workers SET infos = ? WHERE run_id = ? AND worker_head_node_id = ?",
                (dumps(asdict(worker_infos)), id, worker_head_node_id),
            )

    def update_worker_usage(
        self, id: str, worker_head_node_id: str, usage: Usage
    ) -> Usage:
        worker_infos = self.get_worker(id, worker_head_node_id)
        worker_infos.add_usage(usage)
        with self.db:
            self.db.execute(
                "UPDATE workers SET infos = ? WHERE run_id = ? AND worker_head_node_id = ?",
                (dumps(asdict(worker_infos)), id, worker_head_node_id),
            )
        return worker_infos.usage

    def update_run_status(self, id: str, status: RunStatus) -> None:
        run_metadata = self.get_run(id)
        run_metadata.status = status
        with self.db:
            self.db.execute(
                "UPDATE runs SET metadata = ? WHERE id = ?",
                (dumps(asdict(run_metadata)), id),
            )

    def list_runs(self) -> List[RunMetadata]:
        cursor = self.db.cursor()
        cursor.execute("SELECT metadata FROM runs")
        return [
            from_dict(RunMetadata, loads(row[0]), config=self.dacite_config)
            for row in cursor.fetchall()
        ]

    def list_workers(self, run_id: str) -> List[WorkerInfos]:
        cursor = self.db.cursor()
        cursor.execute("SELECT infos FROM workers WHERE run_id = ?", (run_id,))
        return [
            from_dict(WorkerInfos, loads(row[0]), config=self.dacite_config)
            for row in cursor.fetchall()
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
