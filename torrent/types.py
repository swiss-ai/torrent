from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkerStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int

    @classmethod
    def zero(cls) -> Usage:
        return cls(prompt_tokens=0, completion_tokens=0, cached_tokens=0)

    def add(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass
class RunMetadata:
    id: str
    status: RunStatus
    start_time: int
    model_path: str
    input_dataset_path: str
    input_dataset_split: str
    output_dataset_path: str
    batch_size: Optional[int] = None
    total_rows: Optional[int] = None
    end_time: Optional[int] = None


@dataclass
class WorkerInfos:
    job_id: str
    worker_head_node_id: str
    status: WorkerStatus
    usage: Usage


@dataclass
class WorkerArgs:
    run_id: str
    job_id: str
    worker_head_node_id: str
    input_dataset_path: str
    input_dataset_split: str
    output_dataset_path: str


@dataclass
class ServerArgs:
    dp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1
    cuda_graph_max_bs: Optional[int] = 512
    grammar_backend: Literal["llguidance", "xgrammar"] = "llguidance"
    batch_size: int = 256
    max_concurrent_requests: int = 16
    token_usage_threshold: float = 0.8
