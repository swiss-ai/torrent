from argparse import ArgumentParser

from torrent.db import TorrentDB
from torrent.job_manager import JobManager
from torrent.utils import print_runs
from torrent.types import RunMetadata, RunStatus


def launch_run(
    run_id: str,
    model_path: str,
    input_dataset_path: str,
    split: str,
    output_dataset_path: str,
    workers: int,
    time: str,
    partition: str,
    environment: str,
    account: str,
    port: int,
):
    db = TorrentDB()

    run_metadata = RunMetadata(
        id=run_id,
        model_path=model_path,
        input_dataset_path=input_dataset_path,
        input_dataset_split=split,
        output_dataset_path=output_dataset_path,
        status=RunStatus.RUNNING,
        start_time=int(time.time()),
        end_time=None,
    )
    db.add_run(run_metadata)

    job_manager = JobManager()
    job_manager.launch(
        run_metadata, workers, time, partition, environment, account, port
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
        "--time", type=str, default="04:00:00", help="The time to use"
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

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a run")
    cancel_parser.add_argument("run_id", type=str, help="The ID of the run to cancel")

    args = parser.parse_args()

    if args.command == "list":
        db = TorrentDB()
        print_runs(db)
    elif args.command == "attach":
        pass
    elif args.command == "launch":
        launch_run(
            args.model_path,
            args.input_dataset_path,
            args.split,
            args.output_dataset_path,
            args.workers,
            args.time,
            args.partition,
            args.environment,
            args.account,
            args.port,
        )
    elif args.command == "cancel":
        run_id = args.run_id
        db = TorrentDB()
        job_manager = JobManager()

        for worker in db.list_workers(run_id):
            job_manager.cancel_job(worker.job_id)


if __name__ == "__main__":
    main()
