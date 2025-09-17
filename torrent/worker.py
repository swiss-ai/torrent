import os
import zmq
import sys
import asyncio
from dacite import from_dict
from datasets import Dataset, load_from_disk
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.srt.entrypoints.engine import Engine
from typing import Dict, Any, List, Tuple, Optional
from sglang.srt.server_args import prepare_server_args
from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput

from torrent.db import TorrentDB
from torrent.types import WorkerArgs, WorkerStatus, Usage, WorkerInfos


def parse_worker_args(argv: str) -> WorkerArgs:
    args = argv.split("<SEP>")
    return from_dict(
        WorkerArgs,
        {
            "run_id": args[0],
            "job_id": args[1],
            "db_full_path": args[2],
            "worker_head_node_id": args[3],
            "input_dataset_path": args[4],
            "input_dataset_split": args[5],
            "output_dataset_path": args[6],
            "batch_size": int(args[7]),
            "max_concurrent_requests": int(args[8]),
            "token_usage_threshold": float(args[9]),
        },
    )


def get_token_usage(engine: Engine) -> Optional[float]:
    req = RpcReqInput(method="_get_token_info")
    
    engine.send_to_rpc.send_pyobj(req)
    recv_req = engine.send_to_rpc.recv_pyobj(zmq.BLOCKY)
    assert isinstance(recv_req, RpcReqOutput)

    if recv_req.success:
        _, token_usage, _, _ = recv_req.message
        return token_usage

    return None


def load_dataset(dataset_path: str, split: str) -> Dataset:
    return load_from_disk(dataset_path)[split]


async def async_processing_loop(
    id: str,
    batch_size: int,
    engine: Engine,
    db: TorrentDB,
    input_dataset: Dataset,
    worker_infos: WorkerInfos,
    max_concurrent_requests: int,
    token_usage_threshold: float,
) -> Dataset:
    pending_requests = []
    completed_outputs = []

    async def process_batch(
        start_index: int, batch_data: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Usage]:
        outputs = await engine.async_generate(
            batch_data["input"], sampling_params=batch_data.get("sampling_params", {})
        )

        flattened_results = []
        for i, output in enumerate(outputs):
            flattened_results.append(
                {"index": start_index + i, "output": output["text"]}
            )

        meta_infos = [output["meta_info"] for output in outputs]
        usage = Usage(
            prompt_tokens=sum(meta_info["prompt_tokens"] for meta_info in meta_infos),
            completion_tokens=sum(
                meta_info["completion_tokens"] for meta_info in meta_infos
            ),
            cached_tokens=sum(meta_info["cached_tokens"] for meta_info in meta_infos),
        )

        return flattened_results, usage

    while True:
        while len(pending_requests) < max_concurrent_requests:
            try:
                token_usage = get_token_usage(engine.tokenizer_manager)
                if token_usage >= token_usage_threshold:
                    break
            except Exception:
                pass

            index = db.incr_run_index(id, batch_size)
            if index >= len(input_dataset):
                break

            task = asyncio.create_task(
                process_batch(index, input_dataset[index : index + batch_size])
            )
            pending_requests.append(task)

        if not pending_requests:
            if db.get_run_index(id) >= len(input_dataset):
                break
            await asyncio.sleep(0.01)
            continue

        done, pending_requests = await asyncio.wait(
            pending_requests, return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            try:
                outputs, usage = await task
                completed_outputs.extend(outputs)
                worker_infos.usage = db.update_worker_usage(
                    worker_args.run_id, worker_args.worker_head_node_id, usage
                )
            except Exception as e:
                print(f"error: {e}")

        pending_requests = list(pending_requests)

    if pending_requests:
        results = await asyncio.gather(*pending_requests)
        for outputs, usage in results:
            completed_outputs.extend(outputs)
            worker_infos.usage = db.update_worker_usage(
                worker_args.run_id, worker_args.worker_head_node_id, usage
            )

    return Dataset.from_list(completed_outputs)


async def main(worker_args: WorkerArgs, server_args: ServerArgs):
    db = TorrentDB(worker_args.db_full_path)

    worker_infos = WorkerInfos(
        job_id=worker_args.job_id,
        worker_head_node_id=worker_args.worker_head_node_id,
        status=WorkerStatus.STARTING,
        usage=Usage.zero(),
    )

    db.add_worker(worker_args.run_id, worker_infos)
    dataset = load_dataset(
        worker_args.input_dataset_path, worker_args.input_dataset_split
    )

    with Engine(server_args=server_args) as engine:
        db.update_worker_status(
            worker_args.run_id, worker_args.worker_head_node_id, WorkerStatus.RUNNING
        )

        output_dataset = await async_processing_loop(
            id=worker_args.run_id,
            batch_size=worker_args.batch_size,
            engine=engine,
            db=db,
            input_dataset=dataset,
            worker_infos=worker_infos,
            max_concurrent_requests=worker_args.max_concurrent_requests,
            token_usage_threshold=worker_args.token_usage_threshold,
        )

        output_dataset.save_to_disk(
            f"{worker_args.output_dataset_path}/worker_{worker_args.worker_head_node_id}"
        )

    db.update_worker_status(
        worker_args.run_id, worker_args.worker_head_node_id, WorkerStatus.STOPPED
    )


if __name__ == "__main__":
    worker_args = parse_worker_args(sys.argv[1])
    server_args = prepare_server_args(sys.argv[2:])
    print(worker_args)

    try:
        asyncio.run(main(worker_args, server_args))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
