import os
import asyncio
import aiohttp
import concurrent.futures
from typing import Any, Dict, List
from urllib.parse import urlparse, urlunparse

import requests
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_from_disk, Dataset, concatenate_datasets

BATCH_SIZE = 2048
DATASET_PATH = "/capstor/store/cscs/swissai/infra01/posttrain_data/06_sft_mixtures_newformat_linearised/apertus-sft-mixture-8e"


def format_messages(x: Dict[str, Any], template: Template) -> Dict[str, Any]:
    x["formatted_messages"] = template.render(messages=x["messages"]).rsplit(
        "<|assistant_end|>", 1
    )[0]
    return x


def filter_batch_by_length(batch: Dict[str, List[Any]], tokenizer: PreTrainedTokenizer):
    return [
        len(ids) < 32768 for ids in tokenizer(batch["formatted_messages"]).input_ids
    ]


async def process_batch(
    worker_url: str,
    formatted_messages: List[str],
    conversation_ids: List[str],
):
    payload = {
        "text": formatted_messages,
    }

    try:
        response = requests.post(f"{worker_url}/encode", json=payload, timeout=30)
        if response.status_code == 200:
            outputs = response.json()
        else:
            print(f"HTTP {response.status_code} from {worker_url}: {response.text}.")
            return []
    except Exception as e:
        print(f"Error with {worker_url}: {e}")
        return []

    flattened_samples = []
    for id, output in zip(conversation_ids, outputs):
        embedding = output["embedding"]
        flattened_samples.append(
            {
                "conversation_id": id,
                "embedding": embedding,
            }
        )

    return flattened_samples


async def worker(
    worker_url: str,
    template: Template,
    pbar: tqdm,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    num_shards: int,
    index: int,
) -> Dataset:
    partial_dataset = dataset.shard(num_shards=num_shards, index=index)

    iterable_dataset = partial_dataset.to_iterable_dataset()

    iterable_dataset = iterable_dataset.map(
        format_messages,
        fn_kwargs={"template": template},
    )

    iterable_dataset = iterable_dataset.filter(
        filter_batch_by_length,
        batched=True,
        batch_size=1024,
        fn_kwargs={"tokenizer": tokenizer},
    )

    iterable_dataset = iterable_dataset.batch(BATCH_SIZE).batch(2)

    partial_outputs = []
    for batches in iterable_dataset:
        tasks = [
            process_batch(worker_url, formatted_messages, conversation_ids)
            for formatted_messages, conversation_ids in zip(
                batches["formatted_messages"], batches["conversation_id"]
            )
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result:
                partial_outputs.extend(result)

        pbar.update(sum([result != [] for result in results]))

    return Dataset.from_list(partial_outputs)


async def check_health(url: str):
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        port = parsed_url.port

        if not hostname:
            raise ValueError(f"Could not parse hostname from URL: {url}")

        loop = asyncio.get_running_loop()
        addr_info = await loop.getaddrinfo(hostname, port)
        ip = addr_info[0][4][0]

        ip_url = urlunparse(parsed_url._replace(netloc=f"{ip}:{port}"))

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health", timeout=10) as response:
                if response.status == 200:
                    return ip_url, True
                print(
                    f"Worker {url} ({ip_url}) is unhealthy with status code: {response.status}"
                )
                return ip_url, False
    except Exception as e:
        print(f"Worker {url} is unhealthy with exception: {e}")
        return url, False


async def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "template.jinja")

    with open(template_path, "r") as f:
        template = Template(f.read())

    worker_urls = []

    print("Checking health of workers...")
    healthy_urls = []
    results = await asyncio.gather(*[check_health(url) for url in worker_urls])
    for url, is_healthy in results:
        if is_healthy:
            healthy_urls.append(url)

    if not healthy_urls:
        print("No healthy workers found. Exiting.")
        return

    print(f"{len(healthy_urls)} out of {len(worker_urls)} workers are healthy.")
    print(healthy_urls)
    worker_urls = healthy_urls

    dataset = load_from_disk(DATASET_PATH)["train"]
    print(f"processing dataset: {dataset}")

    max_num_batches = len(dataset) // BATCH_SIZE

    pbar = tqdm(total=max_num_batches, desc="processing batches")

    futures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(worker_urls)
    ) as executor:
        for i, url in enumerate(worker_urls):
            future = executor.submit(
                asyncio.run,
                worker(url, template, pbar, dataset, tokenizer, len(worker_urls), i),
            )
            futures.append(future)

    all_outputs = []
    for future in futures:
        all_outputs.append(future.result())

    pbar.close()

    output_dataset = concatenate_datasets(all_outputs)

    print(output_dataset)

    output_dataset.save_to_disk("/iopsstor/scratch/cscs/nathanrchn/sft_dataset")
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
