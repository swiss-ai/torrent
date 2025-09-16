import os
import sqlite3
import threading
import warnings
from typing import List
from json import dumps, loads
from dataclasses import asdict
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
        self._local = threading.local()

        with self._get_connection() as conn:
            self._configure_connection(conn)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    metadata TEXT NOT NULL,
                    index_count INTEGER NOT NULL DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS workers (
                    run_id TEXT NOT NULL,
                    worker_head_node_id TEXT NOT NULL,
                    worker_data TEXT NOT NULL,
                    PRIMARY KEY (run_id, worker_head_node_id),
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)

            conn.commit()
    
    def _configure_connection(self, conn):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            try:
                conn.execute("PRAGMA journal_mode=DELETE")
            except sqlite3.OperationalError:
                pass
        
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            try:
                conn.execute("PRAGMA synchronous=OFF")
            except sqlite3.OperationalError:
                pass
        
        try:
            conn.execute("PRAGMA busy_timeout=30000")
        except sqlite3.OperationalError:
            pass

    @contextmanager
    def _get_connection(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=30.0
            )
            self._configure_connection(self._local.conn)

        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def add_run(self, run_metadata: RunMetadata) -> None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_metadata.id,))
            if cursor.fetchone() is not None:
                raise ValueError(f"Run {run_metadata.id} already exists")

            conn.execute(
                "INSERT INTO runs (id, metadata, index_count) VALUES (?, ?, ?)",
                (run_metadata.id, dumps(asdict(run_metadata)), 0),
            )
            conn.commit()

    def get_run(self, id: str) -> RunMetadata:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT metadata FROM runs WHERE id = ?", (id,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"Run {id} not found")
            return from_dict(RunMetadata, loads(row[0]), config=self.dacite_config)

    def get_run_index(self, id: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT index_count FROM runs WHERE id = ?", (id,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"Run {id} not found")
            return row[0]

    def incr_run_index(self, id: str, incr: int = 1) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE runs SET index_count = index_count + ? WHERE id = ? RETURNING index_count",
                (incr, id),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"Run {id} not found")
            conn.commit()
            return row[0]

    def add_worker(self, id: str, worker_infos: WorkerInfos) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO workers (run_id, worker_head_node_id, worker_data) VALUES (?, ?, ?)",
                (id, worker_infos.worker_head_node_id, dumps(asdict(worker_infos))),
            )
            conn.commit()

    def get_worker(self, id: str, worker_head_node_id: str) -> WorkerInfos:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT worker_data FROM workers WHERE run_id = ? AND worker_head_node_id = ?",
                (id, worker_head_node_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"Worker {worker_head_node_id} not found for run {id}")
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
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE workers SET worker_data = ? WHERE run_id = ? AND worker_head_node_id = ?",
                (dumps(asdict(worker_infos)), id, worker_head_node_id),
            )
            conn.commit()

    def update_worker_usage(
        self, id: str, worker_head_node_id: str, usage: Usage
    ) -> Usage:
        worker_infos = self.get_worker(id, worker_head_node_id)
        worker_infos.add_usage(usage)
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE workers SET worker_data = ? WHERE run_id = ? AND worker_head_node_id = ?",
                (dumps(asdict(worker_infos)), id, worker_head_node_id),
            )
            conn.commit()
        return worker_infos.usage

    def update_run_status(self, id: str, status: RunStatus) -> None:
        run_metadata = self.get_run(id)
        run_metadata.status = status
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE runs SET metadata = ? WHERE id = ?",
                (dumps(asdict(run_metadata)), id),
            )
            conn.commit()

    def list_runs(self) -> List[RunMetadata]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT metadata FROM runs")
            runs = []
            for (metadata_json,) in cursor.fetchall():
                runs.append(
                    from_dict(
                        RunMetadata, loads(metadata_json), config=self.dacite_config
                    )
                )
            return runs

    def list_workers(self, run_id: str) -> List[WorkerInfos]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT worker_data FROM workers WHERE run_id = ?", (run_id,)
            )
            workers = []
            for (worker_data_json,) in cursor.fetchall():
                workers.append(
                    from_dict(
                        WorkerInfos, loads(worker_data_json), config=self.dacite_config
                    )
                )
            return workers

    def get_full_path(self) -> str:
        return os.path.dirname(self.db_path)

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
