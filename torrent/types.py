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
    extra_args: Optional[str] = None


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
    port: int
    nodes_ids: List[str]
    workers_head_ids: List[str]


@dataclass
class ModelInstances:
    model_instances: List[ModelInstance]
    queue: List[str]
    num_waiting_requests: int = 0
    avg_time_per_request: float = 0.0

    @property
    def num_workers(self) -> int:
        return sum(
            model_instance.num_workers for model_instance in self.model_instances
        )

    @property
    def nodes_ids(self) -> List[str]:
        nodes_ids = []
        for model_instance in self.model_instances:
            nodes_ids.extend(model_instance.nodes_ids)
        return nodes_ids

    @property
    def workers_head_ids(self) -> List[str]:
        workers_head_ids = []
        for model_instance in self.model_instances:
            workers_head_ids.extend(model_instance.workers_head_ids)
        return workers_head_ids


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


@dataclass
class RunMetadata:
    num_rows: int
    num_input_tokens: int
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    num_output_tokens: Optional[int] = None
