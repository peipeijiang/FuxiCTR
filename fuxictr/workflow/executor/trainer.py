import asyncio
import subprocess
import os
from typing import Optional
from fuxictr.workflow.config import Config
from fuxictr.workflow.db import DatabaseManager
from fuxictr.workflow.utils.logger import WorkflowLogger
from fuxictr.workflow.models import StepStatus

class Trainer:
    def __init__(self, config: Config, db_manager: DatabaseManager,
                 logger: WorkflowLogger, task_id: int):
        self.config = config
        self.db = db_manager
        self.logger = logger
        self.task_id = task_id

    async def execute(self, experiment_id: str, data_path: str) -> bool:
        """Execute training stage"""
        step_name = "train"

        self.db.update_step_status(self.task_id, step_name, StepStatus.RUNNING)
        self.logger.log(step_name, f"Starting training with experiment_id: {experiment_id}")

        try:
            run_script = os.path.join(
                os.path.dirname(__file__),
                "../../model_zoo/multitask/run_expid.py"
            )

            cmd = [
                "python", run_script,
                "--expid", experiment_id,
                "--dataset_path", data_path
            ]

            self.logger.log(step_name, f"Executing: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                log_line = line.decode().strip()
                self.logger.log(step_name, log_line)

            returncode = await process.wait()

            if returncode == 0:
                self.db.update_step_status(self.task_id, step_name, StepStatus.COMPLETED)
                self.logger.complete(step_name)
                return True
            else:
                error_output = await process.stderr.read()
                raise Exception(f"Training failed: {error_output.decode()}")

        except Exception as e:
            self.db.update_step_status(
                self.task_id, step_name, StepStatus.FAILED,
                error_message=str(e)
            )
            self.logger.error(step_name, str(e))
            return False
