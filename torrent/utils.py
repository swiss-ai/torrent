import random
import string
from datetime import datetime
from torrent.db import TorrentDB
from prettytable import PrettyTable

BATCH_SIZE = 512
NUM_GPU_PER_NODE = 4
TORRENT_PATH = "$HOME/.torrent"


def nanoid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_letters, k=length))


def print_runs(db: TorrentDB) -> None:
    runs = db.list_runs()

    if not runs:
        print("No runs found.")
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
            else "N/A"
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
                run.total_rows if run.total_rows else "N/A",
                run.batch_size if run.batch_size else "N/A",
                total_prompt_tokens,
                total_completion_tokens,
                total_cached_tokens,
                end_time,
            ]
        )

    print(table)
