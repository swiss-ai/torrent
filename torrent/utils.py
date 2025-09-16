import os
import time
import random
import string
from tqdm import tqdm
from typing import Optional
from dacite import from_dict
from datetime import datetime
from omegaconf import OmegaConf

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from torrent.db import TorrentDB
from torrent.types import ServerArgs
from prettytable import PrettyTable

BATCH_SIZE = 512
NUM_GPU_PER_NODE = 4
TORRENT_PATH = os.path.expanduser("~/.torrent")


def nanoid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_letters, k=length))


def get_default_db() -> TorrentDB:
    return TorrentDB(TORRENT_PATH)


def get_server_args(model_path: str) -> Optional[ServerArgs]:
    config_filename = f"{model_path.replace('/', '_')}.yaml"
    try:
        config_content = (files("torrent") / "models" / config_filename).read_text()
        return from_dict(ServerArgs, OmegaConf.create(config_content))
    except FileNotFoundError:
        return None


def format_int(value: Optional[int]) -> str:
    return f"{value:,}" if value is not None else "n/a"


def print_runs(db: TorrentDB) -> None:
    runs = db.list_runs()

    if not runs:
        print("No runs found. Run `torrent launch` to start a new run.")
        return

    table = PrettyTable()
    table.field_names = [
        "ID",
        "Status",
        "Start Time",
        "Model",
        "Input Dataset",
        "Output Dataset",
        "Total Rows",
        "Batch Size",
        "Total Prompt Tokens",
        "Total Completion Tokens",
        "Total Cached Tokens",
        "End Time",
    ]
    table.align = "l"

    for run in runs:
        workers = db.list_workers(run.id)

        total_prompt_tokens = sum(worker.usage.prompt_tokens for worker in workers)
        total_completion_tokens = sum(
            worker.usage.completion_tokens for worker in workers
        )
        total_cached_tokens = sum(worker.usage.cached_tokens for worker in workers)

        start_time = datetime.fromtimestamp(run.start_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        end_time = (
            datetime.fromtimestamp(run.end_time).strftime("%Y-%m-%d %H:%M:%S")
            if run.end_time
            else "n/a"
        )

        model_name = (
            run.model_path.split("/")[-1] if "/" in run.model_path else run.model_path
        )
        input_dataset_name = (
            run.input_dataset_path.split("/")[-1]
            if "/" in run.input_dataset_path
            else run.input_dataset_path
        )
        output_dataset_name = (
            run.output_dataset_path.split("/")[-1]
            if "/" in run.output_dataset_path
            else run.output_dataset_path
        )

        table.add_row(
            [
                run.id,
                run.status.value,
                start_time,
                model_name,
                input_dataset_name,
                output_dataset_name,
                format_int(run.total_rows),
                format_int(run.batch_size),
                format_int(total_prompt_tokens),
                format_int(total_completion_tokens),
                format_int(total_cached_tokens),
                end_time,
            ]
        )

    print(table)


def attach_run(db: TorrentDB, run_id: str) -> None:
    run = db.get_run(run_id)

    with tqdm(total=run.total_rows, desc=run_id) as pbar:
        while (index := db.get_run_index(run_id)) < run.total_rows:
            pbar.n = index
            pbar.refresh()
            time.sleep(0.5)
