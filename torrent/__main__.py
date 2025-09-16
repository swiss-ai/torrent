import time
from argparse import ArgumentParser
from datasets import load_from_disk

from torrent.job_manager import JobManager
from torrent.types import RunMetadata, RunStatus
from torrent.utils import (
    nanoid,
    get_default_db,
    print_runs,
    get_server_args,
    attach_run,
)


def launch_run(
    run_id: str,
    model_path: str,
    input_dataset_path: str,
    split: str,
    output_dataset_path: str,
    workers: int,
    partition: str,
    environment: str,
    account: str,
    port: int,
    time_str: str,
):
    db = get_default_db()

    dataset = load_from_disk(input_dataset_path)[split]

    if "input" not in dataset.column_names:
        raise ValueError("Dataset must have an 'input' column")

    server_args = get_server_args(model_path)
    if server_args is None:
        raise ValueError(
            f"Model currently not supported: {run_metadata.model_path}. You need to add a config file here: https://github.com/swiss-ai/torrent/tree/main/torrent/models"
        )

    batch_size = min(server_args.batch_size, dataset.num_rows)
    run_metadata = RunMetadata(
        id=run_id,
        model_path=model_path,
        input_dataset_path=input_dataset_path,
        input_dataset_split=split,
        output_dataset_path=output_dataset_path,
        total_rows=dataset.num_rows,
        batch_size=batch_size,
        status=RunStatus.RUNNING,
        start_time=int(time.time()),
        end_time=None,
    )
    db.add_run(run_metadata)

    job_manager = JobManager()
    job_manager.launch(
        run_metadata,
        server_args,
        workers,
        time_str,
        partition,
        environment,
        account,
        port,
        db.get_full_path(),
    )


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List all runs")

    attach_parser = subparsers.add_parser("attach", help="Attach to a run")
    attach_parser.add_argument(
        "run_id", type=str, help="The ID of the run to attach to"
    )

    launch_parser = subparsers.add_parser("launch", help="Launch a new run")
    launch_parser.add_argument(
        "model_path", type=str, help="The path to the model to launch"
    )
    launch_parser.add_argument(
        "input_dataset_path", type=str, help="The path to the input dataset"
    )
    launch_parser.add_argument(
        "--split", type=str, default="train", help="The split of the input dataset"
    )
    launch_parser.add_argument(
        "output_dataset_path", type=str, help="The path to the output dataset"
    )
    launch_parser.add_argument(
        "--workers", type=int, default=1, help="The number of workers to use"
    )
    launch_parser.add_argument(
        "--partition", type=str, default="normal", help="The partition to use"
    )
    launch_parser.add_argument(
        "--environment", type=str, default="sglang_gb200", help="The environment to use"
    )
    launch_parser.add_argument(
        "--account", type=str, default="a-infra01", help="The account to use"
    )
    launch_parser.add_argument(
        "--port", type=int, default=30000, help="The port to use"
    )
    launch_parser.add_argument(
        "--time", type=str, default="04:00:00", help="The time to use"
    )

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a run")
    cancel_parser.add_argument("run_id", type=str, help="The ID of the run to cancel")

    args = parser.parse_args()

    if args.command == "list":
        db = get_default_db()
        print_runs(db)
    elif args.command == "attach":
        db = get_default_db()
        attach_run(db, args.run_id)
    elif args.command == "launch":
        launch_run(
            nanoid(),
            args.model_path,
            args.input_dataset_path,
            args.split,
            args.output_dataset_path,
            args.workers,
            args.partition,
            args.environment,
            args.account,
            args.port,
            args.time,
        )
    elif args.command == "cancel":
        run_id = args.run_id
        db = get_default_db()
        job_manager = JobManager()

        for worker in db.list_workers(run_id):
            job_manager.cancel_job(worker.job_id)


if __name__ == "__main__":
    main()
