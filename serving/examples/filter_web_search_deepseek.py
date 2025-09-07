import os
import json
import asyncio
from tqdm import tqdm
from typing import Any, Dict, List

from datasets import load_from_disk, Dataset
from mini_lb import create_load_balancer, LoadBalancer
from transformers import AutoTokenizer, PreTrainedTokenizer

BATCH_SIZE = 128
DATASETS_PATH = "datasets/"
CONCURRENT_REQUEST_LIMIT = 8
SOURCE_DATASET_PATH = "datasets/qa_dataset"


async def generate_response(
    lb: LoadBalancer,
    semaphore: asyncio.Semaphore,
    tokenizer: PreTrainedTokenizer,
    sample: Dict[str, List[Any]],
):
    conversations = [
        [
            {
                "role": "system",
                "content": """### System
You are DATA-FILTER-DEEPSEEK, a highly precise quality-gate for question-answering datasets.

...

(Continue processing the next sample.)""",
            },
            {
                "role": "user",
                "content": f"""INPUT:
{{"query":"{question}",
"gold":["{answer}"]}}
OUTPUT:""",
            },
        ]
        for question, answer in zip(sample["question"], sample["answer"])
    ]

    texts = [
        tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for conversation in conversations
    ]

    async with semaphore:
        try:
            outputs = await lb.generate(
                {
                    "text": texts,
                    "sampling_params": {
                        "max_new_tokens": 512,
                        "json_schema": json.dumps(
                            {
                                "type": "object",
                                "properties": {
                                    "keep": {"type": "boolean"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["keep", "reason"],
                                "additionalProperties": False,
                            }
                        ),
                    },
                }
            )
        except Exception as e:
            print(e)
            return None

    parsed_outputs = [json.loads(output["text"]) for output in outputs]

    sample["keep"] = [parsed_output["keep"] for parsed_output in parsed_outputs]
    sample["reason"] = [parsed_output["reason"] for parsed_output in parsed_outputs]

    flattened_samples = []
    for question, answer, keep, reason in zip(
        sample["question"], sample["answer"], sample["keep"], sample["reason"]
    ):
        flattened_samples.append(
            {"question": question, "answer": answer, "keep": keep, "reason": reason}
        )

    return flattened_samples


async def main():
    model_id = "deepseek-ai/DeepSeek-V3.1"
    output_folder = os.path.join(DATASETS_PATH, f"qa_dataset_filtered_deepseek_v31")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_from_disk(SOURCE_DATASET_PATH).batch(BATCH_SIZE)

    lb = await create_load_balancer(
        worker_urls=[
            "http://172.28.39.148:5000",
            "http://172.28.39.176:5000",
            "http://172.28.48.169:5000",
            "http://172.28.39.200:5000",
            "http://172.28.39.220:5000",
            "http://172.28.39.236:5000",
            "http://172.28.39.244:5000",
            "http://172.28.40.4:5000",
        ],
        max_requests_per_worker=CONCURRENT_REQUEST_LIMIT,
    )
    print("load balancer is ready")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT * (len(lb.workers) - 1))
    tasks = [generate_response(lb, semaphore, tokenizer, sample) for sample in dataset]

    outputs = []

    try:
        iterator = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="filtering...")
        for task in iterator:
            outputs += await task

    finally:
        output_dataset = Dataset.from_list(outputs)

        print(f"dataset length: {len(output_dataset)}")

        output_dataset = output_dataset.filter(lambda x: x["keep"])

        print(f"filtered dataset length: {len(output_dataset)}")

        print("saving to", output_folder)
        output_dataset.save_to_disk(output_folder)

        await lb.close()
        print("done")


if __name__ == "__main__":
    asyncio.run(main())
