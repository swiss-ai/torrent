import os
import stat
import logging
import subprocess
from jinja2 import Template
from typing import Optional
from dacite import from_dict
from dataclasses import asdict
from omegaconf import OmegaConf
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from torrent.types import RunMetadata, ServerArgs
from torrent.utils import TORRENT_PATH, NUM_GPU_PER_NODE


class JobManager:
    def __init__(self) -> None:
        self.template = self.get_template()

    def launch(
        self,
        run_metadata: RunMetadata,
        workers: int,
        time: str,
        partition: str,
        environment: str,
        account: str,
        port: int,
    ) -> None:
        h, m, s = map(int, time.split(":"))
        duration = (
            h * 3600 + m * 60 + s
        )  # TODO: attach the duration to each worker so that it can restart if needed

        server_args = self.get_server_args(run_metadata.model_path)
        if server_args is None:
            raise ValueError(
                f"Model currently not supported: {run_metadata.model_path}. You need to add a config file here: https://github.com/swiss-ai/torrent/tree/main/torrent/models"
            )

        run_dir = self.create_run_dir(run_metadata.id)

        num_nodes_per_worker = server_args.tp_size // NUM_GPU_PER_NODE
        self.create_sbtach(
            run_dir,
            model_path=run_metadata.model_path,
            nodes=num_nodes_per_worker * workers,
            workers=workers,
            num_nodes_per_worker=num_nodes_per_worker,
            partition=partition,
            environment=environment,
            time=time,
            account=account,
            port=port,
            log_dir=run_dir,
            worker_cmd="python -m torrent.worker",
            job_name=run_metadata.id,
            **asdict(server_args),
        )

        for _ in range(workers):
            self.submit_job(run_dir)

    def get_template(self) -> Template:
        template_content = (files("torrent") / "template.jinja").read_text()
        return Template(template_content)

    def get_server_args(self, model_path: str) -> Optional[ServerArgs]:
        config_filename = f"{model_path.replace('/', '_')}.yaml"
        try:
            config_content = (files("torrent") / "models" / config_filename).read_text()
            return from_dict(ServerArgs, OmegaConf.create(config_content))
        except FileNotFoundError:
            return None

    def create_run_dir(self, run_id: str) -> str:
        run_dir = f"{TORRENT_PATH}/{run_id}"
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

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
