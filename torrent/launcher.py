from __future__ import annotations

import os
import stat
import time
import aiohttp
import asyncio
import logging
import subprocess
from jinja2 import Template
from datetime import datetime
from dataclasses import asdict
from importlib.resources import files
from typing import Tuple, List, Literal

from torrent.utils import TMP_PATH, nanoid
from torrent.types import ModelConfig, ServingConfig, ModelInstance

ENV_SCRIPT_CONTENT = """\
#!/bin/bash

export no_proxy="0.0.0.0,$no_proxy"
export NO_PROXY="0.0.0.0,$NO_PROXY"

python -m sglang.launch_server "$@"
"""


class Launcher:
    def __init__(
        self, mode: Literal["multi-node", "pd-dissaggregation"] = "multi-node"
    ) -> None:
        self.mode = mode
        self.template = self.get_template()

    def get_mode_path(self) -> str:
        return self.mode.replace("-", "_")

    def launch(
        self, model_config: ModelConfig, serving_config: ServingConfig
    ) -> ModelInstance:
        start_time = time.time()
        duration = datetime.strptime(serving_config.time, "%H:%M:%S").total_seconds()

        job_dir = self.create_job_dir()
        self.create_script(job_dir)

        additional_kwargs = {
            "job_name": f"torrent-{job_dir.split('/')[-1]}",
        }

        self.create_sbtach(
            job_dir,
            **asdict(serving_config),
            **asdict(model_config),
            **additional_kwargs,
        )
        job_id = self.submit_job(job_dir)

        return ModelInstance(
            model_config=model_config,
            job_id=job_id,
            job_dir=job_dir,
            start_time=start_time,
            duration=duration,
            port=serving_config.port,
        )

    def wait_for_model_instance(self, model_instance: ModelInstance) -> ModelInstance:
        self.wait_for_job_logs(model_instance.job_dir)
        nodes_ids, workers_head_ids = self.parse_job_logs(model_instance.job_dir)

        model_instance.nodes_ids = nodes_ids
        model_instance.workers_head_ids = workers_head_ids
        model_instance.state = "running"

        return model_instance

    def get_template(self) -> Template:
        mode_path = self.get_mode_path()
        template_files = files(f"torrent.{mode_path}")
        template_content = (template_files / "template.jinja").read_text()

        return Template(template_content)

    def create_job_dir(self) -> str:
        job_dir = f"{TMP_PATH}/{nanoid()}"
        os.makedirs(job_dir, exist_ok=True)
        return job_dir

    def create_script(self, job_dir: str) -> None:
        script_path = f"{job_dir}/launch_with_env.sh"
        with open(script_path, "w") as f:
            f.write(ENV_SCRIPT_CONTENT)

        current_permissions = os.stat(script_path).st_mode
        os.chmod(script_path, current_permissions | stat.S_IEXEC)

    def create_sbtach(self, job_dir: str, **kwargs) -> None:
        sbatch_content = self.template.render(**kwargs)

        sbatch_path = f"{job_dir}/launch.sbatch"
        with open(sbatch_path, "w") as f:
            f.write(sbatch_content)

        current_permissions = os.stat(sbatch_path).st_mode
        os.chmod(sbatch_path, current_permissions | stat.S_IEXEC)

    def submit_job(self, job_dir: str) -> int:
        try:
            result = subprocess.run(
                ["sbatch", f"{job_dir}/launch.sbatch"],
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

    def cancel_instance(self, model_instance: ModelInstance) -> None:
        try:
            subprocess.run(
                ["scancel", model_instance.job_id],
                capture_output=True,
                text=True,
                check=True,
            )
            logging.info(f"Job {model_instance.job_id} cancelled successfully")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error cancelling job {model_instance.job_id}: {e}")
            logging.error(f"stderr: {e.stderr}")
            raise

    def wait_for_job_logs(
        self, job_dir: str, timeout: int = 60, interval: int = 1
    ) -> None:
        log_path = f"{job_dir}/log.out"
        while True:
            if os.path.exists(log_path) and os.path.isfile(log_path):
                try:
                    with open(log_path, "r") as f:
                        log_content = f.read()
                    if "[DONE]" in log_content:
                        break
                except Exception:
                    pass
            time.sleep(interval)
            timeout -= interval
            if timeout <= 0:
                raise TimeoutError(f"Job timed out after {timeout} seconds")

    def parse_job_logs(self, job_dir: str) -> Tuple[List[str], List[str]]:
        with open(f"{job_dir}/log.out", "r") as f:
            log_content = f.read()

        nodes_ids = []
        workers_head_ids = []
        for line in log_content.strip().splitlines():
            if "[NODES_IDS]" in line:
                nodes_ids = line.split(":")[1].strip().split()
            elif "[WORKERS_HEAD_IDS]" in line:
                workers_head_ids = line.split(":")[1].strip().split()

        return nodes_ids, workers_head_ids
