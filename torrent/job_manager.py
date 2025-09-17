import os
import stat
import logging
import subprocess
from jinja2 import Template
from dataclasses import asdict

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from torrent.types import RunMetadata, ServerArgs
from torrent.utils import TORRENT_PATH, NUM_GPU_PER_NODE


WORKER_COMMAND_CONTENT = """\
#!/bin/bash

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

pip install git+https://github.com/swiss-ai/torrent.git

python -m torrent.worker "$@"
"""


class JobManager:
    def __init__(self) -> None:
        self.template = self.get_template()

    def launch(
        self,
        run_metadata: RunMetadata,
        server_args: ServerArgs,
        workers: int,
        time: str,
        partition: str,
        environment: str,
        account: str,
        port: int,
        db_full_path: str,
    ) -> None:
        h, m, s = map(int, time.split(":"))
        duration = (
            h * 3600 + m * 60 + s
        )  # TODO: attach the duration to each worker so that it can restart if needed

        run_dir = self.create_run_dir(run_metadata.id)

        num_nodes_per_worker = server_args.tp_size // NUM_GPU_PER_NODE
        self.create_script(run_dir, run_metadata.model_path)

        self.create_sbtach(
            run_dir,
            model_path=run_metadata.model_path,
            workers=workers,
            num_nodes_per_worker=num_nodes_per_worker,
            partition=partition,
            environment=environment,
            time=time,
            account=account,
            port=port,
            log_dir=run_dir,
            worker_cmd=f"{run_dir}/worker_command.sh",
            job_name=run_metadata.id,
            wa_run_id=run_metadata.id,
            wa_db_full_path=db_full_path,
            wa_input_dataset_path=run_metadata.input_dataset_path,
            wa_input_dataset_split=run_metadata.input_dataset_split,
            wa_output_dataset_path=run_metadata.output_dataset_path,
            wa_batch_size=server_args.batch_size,
            wa_max_concurrent_requests=server_args.max_concurrent_requests,
            wa_token_usage_threshold=server_args.token_usage_threshold,
            **asdict(server_args),
        )

        for _ in range(workers):
            self.submit_job(run_dir)

    def get_template(self) -> Template:
        template_content = (files("torrent") / "template.jinja").read_text()
        return Template(template_content)

    def create_run_dir(self, run_id: str) -> str:
        run_dir = f"{TORRENT_PATH}/{run_id}"
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def get_worker_command_content(self, model_path: str) -> str:
        extra_env = (
            f'export SGL_ENABLE_JIT_DEEPGEMM="false"'
            if model_path == "deepseek-ai/DeepSeek-V3.1"
            else ""
        )
        return f"""\
#!/bin/bash

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

local_rank=$1

{extra_env}

if [ "$local_rank" -eq 0 ]; then
    shift 1
    pip install git+https://github.com/swiss-ai/torrent.git

    python -m torrent.worker "$@"
else
    shift 2
    python -m sglang.launch_server "$@"
fi
        """

    def create_script(self, job_dir: str, model_path: str) -> None:
        script_path = f"{job_dir}/worker_command.sh"
        with open(script_path, "w") as f:
            f.write(self.get_worker_command_content(model_path))

        current_permissions = os.stat(script_path).st_mode
        os.chmod(script_path, current_permissions | stat.S_IEXEC)

    def create_sbtach(self, run_dir: str, **kwargs) -> None:
        sbatch_content = self.template.render(**kwargs)

        sbatch_path = f"{run_dir}/sbatch.sh"
        with open(sbatch_path, "w") as f:
            f.write(sbatch_content)

        current_permissions = os.stat(sbatch_path).st_mode
        os.chmod(sbatch_path, current_permissions | stat.S_IEXEC)

    def submit_job(self, job_dir: str) -> int:
        try:
            result = subprocess.run(
                ["sbatch", f"{job_dir}/sbatch.sh"],
                capture_output=True,
                text=True,
                check=True,
            )
            output_lines = result.stdout.strip().split("\n")

            job_id = output_lines[-1].split()[-1]
            logging.info(f"Job submitted successfully with ID: {job_id}")
            return int(job_id)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error submitting job: {e}")
            logging.error(f"stderr: {e.stderr}")
            raise
        except (IndexError, ValueError):
            logging.error(f"Error parsing job ID from sbatch output: {result.stdout}")
            raise

    def cancel_job(self, job_id: int) -> None:
        try:
            subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                check=True,
            )
            logging.info(f"Job {job_id} cancelled successfully")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error cancelling job {job_id}: {e}")
            logging.error(f"stderr: {e.stderr}")
            raise
