import os
import time
import random
import string
import subprocess
from tqdm import tqdm
from dacite import from_dict
from datetime import datetime
from omegaconf import OmegaConf
from typing import Optional

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from torrent.db import TorrentDB
from prettytable import PrettyTable
from torrent.types import ServerArgs, RunStatus

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


def clean_runs(
    db: TorrentDB,
) -> None:  # TODO: this function is wrong, it should be fixed
    for run in [run for run in db.list_runs() if run.status == RunStatus.RUNNING]:
        try:
            result = subprocess.run(
                ["sacct", "-j", run.id, "--format=JobID,State,ExitCode", "--parsable2"],
                capture_output=True,
                text=True,
                check=True,
            )

            lines = result.stdout.strip().splitlines()
            if len(lines) < 2:
                continue

            for line in lines[1:]:
                parts = line.split("|")
                if len(parts) >= 2:
                    job_id = parts[0].strip()
                    job_state = parts[1].strip()

                    if job_id == run.id and "." not in job_id:
                        if job_state == "COMPLETED":
                            run.status = RunStatus.COMPLETED
                            if run.end_time is None:
                                run.end_time = int(time.time())
                            db.update_run_status(run.id, run.status)
                            break
                        elif job_state in [
                            "FAILED",
                            "CANCELLED",
                            "TIMEOUT",
                            "NODE_FAIL",
                            "OUT_OF_MEMORY",
                        ]:
                            run.status = RunStatus.FAILED
                            if run.end_time is None:
                                run.end_time = int(time.time())
                            db.update_run_status(run.id, run.status)
                            break

        except subprocess.CalledProcessError as e:
            print(f"Error checking run {run.id}: {e}")
        except Exception as e:
            print(f"Unexpected error checking run {run.id}: {e}")


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

    for run in sorted(runs, key=lambda x: x.start_time, reverse=True):
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
        while pbar.n < run.total_rows:
            pbar.n = db.get_run_index(run_id)
            pbar.refresh()
            time.sleep(0.5)
