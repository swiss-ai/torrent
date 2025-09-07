# Single and Multi Node Serving

## Single node serving

Arguments:
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

Usage:
Realistically, you should only change the number of workers, the model path and the TP/DP/EP sizes. To launch a set of workers:

```bash
python serving/single-node/submit_job.py --workers 4 --model-path Qwen/Qwen3-Embedding-4B --dp-size 4
```

## Multi node serving

Arguments:
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

Usage:

```bash
python serving/multi-node/submit_job.py --workers 4 --nodes-per-worker 4 --model-path deepseek-ai/DeepSeek-V3.1 --tp-size 16
```

## Examples
- [Filter Web Search DeepSeek](examples/filter_web_search_deepseek.py): this example uses the mini_lb module to balance the work between the workers. But this vibe coded LB is not very good and I am planning to replace it with a more robust LB.
- [Visualize SFT Dataset](examples/visualize_sft_dataset.py): this example simply shard the dataset between the workers and process the dataset in parallel. It's simpler but not very robust if the dataset is not homogenous.
