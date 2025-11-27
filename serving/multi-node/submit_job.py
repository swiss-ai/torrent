import os
import stat
import random
import string
import logging
import argparse
import tempfile
import subprocess

from jinja2 import Template


def nanoid(length: int = 4) -> str:
    return "".join(random.choices(string.ascii_letters, k=length))


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level)


def generate_job_script(template_path, output_path, **kwargs):
    with open(template_path, "r") as f:
        template = Template(f.read())

    rendered_script = template.render(**kwargs)
    with open(output_path, "w") as f:
        f.write(rendered_script)


def submit_job(job_script_path):
    try:
        result = subprocess.run(
            ["sbatch", job_script_path], capture_output=True, text=True, check=True
        )
        output_lines = result.stdout.strip().split("\n")

        job_id = output_lines[-1].split()[-1]
        logging.info(f"Job submitted successfully with ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting job: {e}")
        logging.error(f"stderr: {e.stderr}")
        raise
    except (IndexError, ValueError):
        logging.error(f"Error parsing job ID from sbatch output: {result.stdout}")
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str, default=nanoid())
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--nodes-per-worker", type=int, required=True)
    parser.add_argument("--partition", type=str, default="normal")
    parser.add_argument("--environment", type=str, default="sglang_gb200")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=256)
    parser.add_argument("--grammar-backend", type=str, default="llguidance")
    parser.add_argument("--use-router", action="store_true", help="Start a router when workers > 1")
    parser.add_argument("--router-policy", type=str, default="cache_aware", help="Routing policy (random, round_robin, cache_aware, power_of_two)")
    parser.add_argument("--router-environment", type=str, default="sglang_router", help="Environment for the router")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "template.jinja")

    start_server_with_env_path = os.path.join(script_dir, "start_server_with_env.sh")

    if os.path.exists(start_server_with_env_path):
        current_permissions = os.stat(start_server_with_env_path).st_mode
        os.chmod(start_server_with_env_path, current_permissions | stat.S_IEXEC)

    template_args = {
        "job_name": args.job_name,
        "nodes": args.nodes_per_worker * args.workers,
        "nodes_per_worker": args.nodes_per_worker,
        "workers": args.workers,
        "partition": args.partition,
        "environment": args.environment,
        "model_path": args.model_path,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "ep_size": args.ep_size,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "grammar_backend": args.grammar_backend,
        "start_server_with_env_path": start_server_with_env_path,
        "use_router": args.use_router,
        "router_policy": args.router_policy,
        "router_environment": args.router_environment,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as temp_file:
        generate_job_script(template_path, temp_file.name, **template_args)
        job_id = submit_job(temp_file.name)
        logging.info(f"worker urls will be available in: logs/{job_id}/log.out")


if __name__ == "__main__":
    main()
