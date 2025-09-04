from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List, Optional


@dataclass
class ModelConfig:
    model_path: str
    dp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1
    moe_a2a_backend: Literal["none", "deepep"] = "none"


@dataclass
class ServingConfig:
    num_workers: int
    num_nodes_per_worker: int
    partition: str = "normal"
    environment: str = "sglang_gb200"
    time: str = "04:00:00"
    account: str = "a-infra01"
    port: int = 30000
    cuda_graph_max_bs: Optional[int] = None


@dataclass
class ModelInstance:
    model_config: ModelConfig
    job_id: str
    job_dir: str
    start_time: int
    duration: int
    queue: List[str]
    nodes_ids: List[str]
    port: int
    workers_head_ids: List[str]
    num_waiting_requests: int = 0
    avg_time_per_request: float = 0.0


@dataclass
class WorkerMetrics:
    worker_id: str
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    num_requests_total: float = 0.0
    cached_tokens_total: float = 0.0
    num_running_reqs: float = 0.0
    num_used_tokens: float = 0.0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: float = 0.0
    num_grammar_queue_reqs: float = 0.0
    cache_hit_rate: float = 0.0
    spec_accept_length: float = 0.0
    total_retracted_reqs: float = 0.0
