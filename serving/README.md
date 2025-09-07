# Single and Multi Node Serving

## Arguments

### Single node serving
- `--job-name`: the name of the job (default: random 4 characters)
- `--workers`: the number of workers (default: 1)
- `--partition`: the partition to run the job on (default: "normal")
- `--environment`: the environment to run the job on (default: "sglang_gb200")
- `--model-path`: the path to the model (required)
- `--dp-size`: the data parallel size (default: 1)
- `--tp-size`: the tensor parallel size (default: 1)
- `--ep-size`: the expert parallel size (default: 1)
- `--cuda-graph-max-bs`: the maximum batch size for CUDA graphs (default: 256)
- `--grammar-backend`: the grammar backend to use (default: "llguidance")


### Multi node serving
- `--job-name`: the name of the job (default: random 4 characters)
- `--workers`: the number of workers (default: 1)
- `--nodes-per-worker`: the number of nodes per worker (required)
- `--partition`: the partition to run the job on (default: "normal")
- `--environment`: the environment to run the job on (default: "sglang_gb200")
- `--model-path`: the path to the model (required)
- `--dp-size`: the data parallel size (default: 1)
- `--tp-size`: the tensor parallel size (default: 1)
- `--ep-size`: the expert parallel size (default: 1)
- `--cuda-graph-max-bs`: the maximum batch size for CUDA graphs (default: 256)
- `--grammar-backend`: the grammar backend to use (default: "llguidance")


## Usage

Realistically, you should only change the number of workers, the model path and the TP/DP/EP sizes. To launch a set of workers:

### Single node serving

```bash
python serving/single-node/submit_job.py --workers 4 --model-path Qwen/Qwen3-Embedding-4B --dp-size 4
```

### Multi node serving

```bash
python serving/multi-node/submit_job.py --workers 4 --nodes-per-worker 4 --model-path deepseek-ai/DeepSeek-V3.1 --tp-size 16
```

Then you need to wait a bit for the job to start. Then you can open the logs and copy the list of workers urls.

These urls are compatible with the OpenAI API. But wait at least 5min (and 10min if you are using a large model) for the server to be ready.

If you want the model to use a very specific output format, please use [structured outputs](https://platform.openai.com/docs/guides/structured-outputs). The format will be used by the model 100% of the time. The easiest way is to use the `json_schema` parameter.

## Examples
- [Filter Web Search DeepSeek](examples/filter_web_search_deepseek.py): this example uses the mini_lb module to balance the work between the workers. But this vibe coded LB is not very good and I am planning to replace it with a more robust LB.
- [Visualize SFT Dataset](examples/visualize_sft_dataset.py): this example simply shard the dataset between the workers and process the dataset in parallel. It's simpler but not very robust if the dataset is not homogenous.

## Settings per model

Note that all the models are served in the original precision.

| Model | Precision | TP size | Nodes per worker |
|-------|---------|---------|----|
| Qwen/Qwen3-235B-A22B-Instruct-2507 | bf16 | 8 | 2 |
| Qwen/Qwen3-Coder-480B-A35B-Instruct | bf16 | 16 | 4 |
| deepseek-ai/DeepSeek-V3.1 | fp8 | 16 | 4 |
| meta-llama/Llama-3.1-405B-Instruct | bf16 | 16 | 4 |
| swiss-ai/Apertus-8B-Instruct-2509 | bf16 | 1 | 1 |

## Models weights

I have downloaded the weights of the models above. If you want to avoid downloading the weights, set the `HF_HOME` to:

```bash
export HF_HOME=/iopsstor/scratch/cscs/nathanrchn/.cache/huggingface
```
