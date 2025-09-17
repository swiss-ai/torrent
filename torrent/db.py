import os
import time
import random
import sqlite3
import warnings
from json import dumps, loads
from dataclasses import asdict
from typing import List, Generator
from dacite import from_dict, Config
from contextlib import contextmanager

from torrent.types import RunMetadata, WorkerInfos, WorkerStatus, RunStatus, Usage

warnings.filterwarnings("ignore")


class TorrentDB:
    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        self.db_path = f"{path}/torrent.db"
        self.dacite_config = Config(cast=[RunStatus, WorkerStatus])

        self._initialize_db()

    def _initialize_db(self) -> None:
        db_exists = os.path.exists(self.db_path)
        if not db_exists:
            with self._get_connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA busy_timeout=30000;")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, metadata TEXT)"
                )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS workers (
                        run_id TEXT,
                        worker_head_node_id TEXT,
                        infos TEXT,
                        PRIMARY KEY (run_id, worker_head_node_id)
                    )"""
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS indices (run_id TEXT PRIMARY KEY, value INTEGER)"
                )
                conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        try:
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            yield conn
        finally:
            conn.close()

    def _retry_operation(
        self, operation, max_retries: int = 5, base_delay: float = 0.1
    ):
        for attempt in range(max_retries):
            try:
                return operation()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                print(f"[DB] {e}")
                if (
                    "locking protocol" in str(e).lower()
                    or "database is locked" in str(e).lower()
                ):
                    if attempt == max_retries - 1:
                        raise

                    delay = base_delay * (2**attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                else:
                    raise

    def add_run(self, run_metadata: RunMetadata) -> None:
        def _add_run_operation():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM runs WHERE id = ?", (run_metadata.id,))
                if cursor.fetchone() is not None:
                    raise ValueError(f"Run {run_metadata.id} already exists")

                cursor.execute(
                    "INSERT INTO runs (id, metadata) VALUES (?, ?)",
                    (run_metadata.id, dumps(asdict(run_metadata))),
                )
                cursor.execute(
                    "INSERT INTO indices (run_id, value) VALUES (?, ?)",
                    (run_metadata.id, 0),
                )
                conn.commit()

        self._retry_operation(_add_run_operation)

    def get_run(self, id: str) -> RunMetadata:
        def _get_run_operation():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT metadata FROM runs WHERE id = ?", (id,))
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"Run {id} not found")
                return from_dict(RunMetadata, loads(row[0]), config=self.dacite_config)

        return self._retry_operation(_get_run_operation)

    def get_run_index(self, id: str) -> int:
        def _get_run_index_operation():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM indices WHERE run_id = ?", (id,))
                row = cursor.fetchone()
                return row[0] if row else 0

        return self._retry_operation(_get_run_index_operation)

    def incr_run_index(self, id: str, incr: int = 1) -> int:
        def _incr_run_index_operation():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM indices WHERE run_id = ?", (id,))
                row = cursor.fetchone()
                current_value = row[0] if row else 0
                new_value = current_value + incr
                cursor.execute(
                    "INSERT OR REPLACE INTO indices (run_id, value) VALUES (?, ?)",
                    (id, new_value),
                )
                conn.commit()
                return new_value

        return self._retry_operation(_incr_run_index_operation)

    def add_worker(self, id: str, worker_infos: WorkerInfos) -> None:
        def _add_worker_operation():
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO workers (run_id, worker_head_node_id, infos) VALUES (?, ?, ?)",
                    (id, worker_infos.worker_head_node_id, dumps(asdict(worker_infos))),
                )
                conn.commit()

        self._retry_operation(_add_worker_operation)

    def get_worker(self, id: str, worker_head_node_id: str) -> WorkerInfos:
        def _get_worker_operation():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT infos FROM workers WHERE run_id = ? AND worker_head_node_id = ?",
                    (id, worker_head_node_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(
                        f"Worker {worker_head_node_id} for run {id} not found"
                    )
                return from_dict(
                    WorkerInfos,
                    loads(row[0]),
                    config=self.dacite_config,
                )

        return self._retry_operation(_get_worker_operation)

    def update_worker_status(
        self, id: str, worker_head_node_id: str, status: WorkerStatus
    ) -> None:
        def _update_worker_status_operation():
            worker_infos = self.get_worker(id, worker_head_node_id)
            worker_infos.status = status
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE workers SET infos = ? WHERE run_id = ? AND worker_head_node_id = ?",
                    (dumps(asdict(worker_infos)), id, worker_head_node_id),
                )
                conn.commit()

        self._retry_operation(_update_worker_status_operation)

    def update_worker_usage(
        self, id: str, worker_head_node_id: str, usage: Usage
    ) -> Usage:
        def _update_worker_usage_operation():
            worker_infos = self.get_worker(id, worker_head_node_id)
            worker_infos.add_usage(usage)
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE workers SET infos = ? WHERE run_id = ? AND worker_head_node_id = ?",
                    (dumps(asdict(worker_infos)), id, worker_head_node_id),
                )
                conn.commit()
            return worker_infos.usage

        return self._retry_operation(_update_worker_usage_operation)

    def update_run_status(self, id: str, status: RunStatus) -> None:
        def _update_run_status_operation():
            run_metadata = self.get_run(id)
            run_metadata.status = status
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE runs SET metadata = ? WHERE id = ?",
                    (dumps(asdict(run_metadata)), id),
                )
                conn.commit()

        self._retry_operation(_update_run_status_operation)

    def list_runs(self) -> List[RunMetadata]:
        def _list_runs_operation():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT metadata FROM runs")
                return [
                    from_dict(RunMetadata, loads(row[0]), config=self.dacite_config)
                    for row in cursor.fetchall()
                ]

        return self._retry_operation(_list_runs_operation)

    def list_workers(self, run_id: str) -> List[WorkerInfos]:
        def _list_workers_operation():
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT infos FROM workers WHERE run_id = ?", (run_id,))
                return [
                    from_dict(WorkerInfos, loads(row[0]), config=self.dacite_config)
                    for row in cursor.fetchall()
                ]

        return self._retry_operation(_list_workers_operation)

    def get_full_path(self) -> str:
        return os.path.dirname(self.db_path)
