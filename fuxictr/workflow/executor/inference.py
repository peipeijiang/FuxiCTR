# =========================================================================
# Copyright (C) 2026. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
Inference, Transport, and Monitor Executors.

Stage executors for inference (3), transport (4), and monitoring (5).
"""

import asyncio
import subprocess
import os
import re
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from fuxictr.workflow.executor.base import BaseExecutor
from fuxictr.workflow.utils.ssh_transfer import SSHTransferManager
from fuxictr.workflow.models import InferenceCheckpoint


logger = logging.getLogger(__name__)


class InferenceExecutor(BaseExecutor):
    """
    Executor for Stage 3: Inference.

    Workflow:
    1. Load trained model
    2. Load inference data
    3. Run distributed inference
    4. Save results to parquet
    """

    def __init__(
        self,
        db_manager,
        config: Dict[str, Any],
        logger
    ):
        """Initialize inference executor."""
        super().__init__(db_manager, config, logger)
        self.storage = config.get("storage", {})
        self.fuxictr_paths = config.get("fuxictr_paths", {})

    async def execute(
        self,
        task_id: int,
        task: Dict,
        step: Dict,
        checkpoint: Optional[Dict[str, Any]],
        cancel_event: Optional[asyncio.Event]
    ) -> Dict[str, Any]:
        """Execute inference stage."""
        step_name = "infer"
        result = {
            "success": False,
            "error": None,
            "rows_processed": 0,
            "throughput": 0.0
        }

        try:
            self.logger.log(step_name, f"Starting inference for task {task_id}")

            # Get parameters
            experiment_id = task.get("experiment_id", "")
            model = task.get("model", "")

            # Get checkpoints from previous stages
            data_fetch_checkpoint = self.db.get_checkpoint(task_id, "data_fetch")
            train_checkpoint = self.db.get_checkpoint(task_id, "train")

            if not data_fetch_checkpoint or not data_fetch_checkpoint.get("processed_data_paths"):
                raise ValueError(
                    "Data fetch step not completed. Please ensure the data_fetch step has completed successfully."
                )

            if not train_checkpoint:
                raise ValueError(
                    "Training step not completed. Please ensure the training step has completed successfully."
                )

            # Get paths from checkpoints
            processed_paths = data_fetch_checkpoint.get("processed_data_paths", {})
            dataset_id = processed_paths.get("dataset_id", f"{experiment_id}.unknown")
            inference_output_path = data_fetch_checkpoint.get("inference_output_path", "")
            model_path = train_checkpoint.get("model_path", "")
            raw_infer_data = processed_paths.get("infer_data", "")

            self.logger.log(
                step_name,
                f"Dataset ID: {dataset_id}"
            )
            self.logger.log(
                step_name,
                f"Model path: {model_path}"
            )
            self.logger.log(
                step_name,
                f"Inference output: {inference_output_path}"
            )

            # Run inference
            inference_result = await self._run_inference(
                experiment_id=experiment_id,
                dataset_id=dataset_id,
                model_path=model_path,
                infer_data=raw_infer_data,
                output_path=inference_output_path,
                step_name=step_name,
                checkpoint=checkpoint,
                cancel_event=cancel_event
            )

            result.update(inference_result)

            if result.get("success"):
                # Save checkpoint
                checkpoint_data = {
                    "dataset_id": dataset_id,
                    "total_rows": result.get("rows_processed", 0),
                    "processed_rows": result.get("rows_processed", 0),
                    "output_path": result.get("output_path"),
                    "completed_parts": result.get("completed_parts", []),
                    "timestamp": datetime.now().isoformat()
                }
                self._save_checkpoint(task_id, step_name, checkpoint_data)

            return result

        except Exception as e:
            logger.exception(f"Inference failed for task {task_id}")
            result["error"] = str(e)
            return result

    async def _run_inference(
        self,
        experiment_id: str,
        dataset_id: str,
        model_path: str,
        infer_data: str,
        output_path: str,
        step_name: str,
        checkpoint: Optional[Dict[str, Any]],
        cancel_event: Optional[asyncio.Event]
    ) -> Dict[str, Any]:
        """
        Run the inference process.

        Args:
            experiment_id: Experiment ID
            dataset_id: Dataset ID (exp_id.timestamp)
            model_path: Path to trained model
            infer_data: Path to raw inference data
            output_path: Path to store inference results
            step_name: Step name for logging
            checkpoint: Optional checkpoint for resume
            cancel_event: Optional cancellation event

        Returns:
            Inference result dictionary
        """
        result = {
            "success": False,
            "rows_processed": 0,
            "throughput": 0.0,
            "output_path": output_path,
            "completed_parts": []
        }

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Find run_expid.py script
        script_dir = os.path.dirname(os.path.dirname(__file__))
        run_script = os.path.join(script_dir, "../../model_zoo/multitask/run_expid.py")

        if not os.path.exists(run_script):
            run_script = os.path.join(
                os.path.dirname(__file__),
                "../../../model_zoo/multitask/run_expid.py"
            )

        # Build command for inference
        # Note: The actual inference implementation depends on your FuxiCTR setup
        # This is a placeholder that should be adapted to your inference script
        cmd = [
            "python",
            run_script,
            "--expid", experiment_id,
            "--dataset_id", dataset_id,
            "--mode", "infer",
            "--model_path", model_path,
            "--infer_data", infer_data,
            "--output_path", output_path
        ]

        self.logger.log(step_name, f"Executing inference: {' '.join(cmd[:6])}...")

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Start inference process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        )

        # Parse output
        rows_processed = 0
        start_time = datetime.now()

        while True:
            if cancel_event and cancel_event.is_set():
                process.terminate()
                result["error"] = "Cancelled by user"
                return result

            line = await process.stdout.readline()
            if not line:
                break

            log_line = line.decode().strip()
            if log_line:
                # Parse metrics
                rows_match = re.search(r'(\d+)\s+rows?', log_line, re.IGNORECASE)
                if rows_match:
                    rows_processed = int(rows_match.group(1))

                self.logger.log(step_name, log_line)

        returncode = await process.wait()
        duration = (datetime.now() - start_time).total_seconds()

        if returncode == 0:
            result["success"] = True
            result["rows_processed"] = rows_processed
            result["throughput"] = rows_processed / duration if duration > 0 else 0
            return result
        else:
            stderr = await process.stderr.read()
            result["error"] = f"Inference failed (code {returncode}): {stderr.decode()}"
            return result


class TransportExecutor(BaseExecutor):
    """Executor for Stage 4: Transport results to Server 21."""

    def __init__(
        self,
        db_manager,
        transfer_manager: SSHTransferManager,
        config: Dict[str, Any],
        logger
    ):
        """Initialize transport executor."""
        super().__init__(db_manager, config, logger)
        self.transfer_manager = transfer_manager
        self.servers = config.get("servers", {})
        self.storage = config.get("storage", {})
        self.fuxictr_paths = config.get("fuxictr_paths", {})

    async def execute(
        self,
        task_id: int,
        task: Dict,
        step: Dict,
        checkpoint: Optional[Dict[str, Any]],
        cancel_event: Optional[asyncio.Event]
    ) -> Dict[str, Any]:
        """Execute transport stage."""
        step_name = "transport"
        result = {
            "success": False,
            "error": None,
            "bytes_transferred": 0,
            "files_transferred": 0
        }

        try:
            self.logger.log(step_name, f"Starting transport for task {task_id}")

            # Get parameters
            experiment_id = task.get("experiment_id", "")
            hive_table = task.get("hive_table", "")

            # Get checkpoints from previous stages
            data_fetch_checkpoint = self.db.get_checkpoint(task_id, "data_fetch")
            infer_checkpoint = self.db.get_checkpoint(task_id, "infer")

            if not data_fetch_checkpoint or not data_fetch_checkpoint.get("inference_output_path"):
                raise ValueError(
                    "Data fetch step not completed or inference_output_path not found."
                )

            if not infer_checkpoint or not infer_checkpoint.get("output_path"):
                raise ValueError(
                    "Inference step not completed or output path not found."
                )

            # Get paths from checkpoints
            processed_paths = data_fetch_checkpoint.get("processed_data_paths", {})
            dataset_id = processed_paths.get("dataset_id", f"{experiment_id}.unknown")
            inference_output = infer_checkpoint.get("output_path", "")

            self.logger.log(
                step_name,
                f"Dataset ID: {dataset_id}"
            )
            self.logger.log(
                step_name,
                f"Transferring from: {inference_output}"
            )

            # Transfer to Server 21
            server_21 = self.servers.get("server_21", {})
            if not server_21:
                raise ValueError("server_21 not configured")

            transfer_result = await self.transfer_manager.transfer_directory(
                source_host="localhost",
                source_path=inference_output,
                dest_host=server_21.get("host"),
                dest_path=f"/tmp/staging/{dataset_id}/",
                ssh_key=server_21.get("key_path"),
                ssh_user=server_21.get("username"),
                ssh_port=server_21.get("port", 22),
                task_id=task_id,
                step_name=step_name
            )

            if not transfer_result.success:
                raise Exception(f"Transfer failed: {transfer_result.error_message}")

            # Load to Hive
            await self._load_to_hive(
                server_21, dataset_id, experiment_id,
                hive_table, step_name
            )

            result["success"] = True
            result["bytes_transferred"] = transfer_result.bytes_transferred
            result["files_transferred"] = transfer_result.files_transferred

            return result

        except Exception as e:
            logger.exception(f"Transport failed for task {task_id}")
            result["error"] = str(e)
            return result

    async def _load_to_hive(
        self,
        server: Dict[str, Any],
        dataset_id: str,
        experiment_id: str,
        hive_table: str,
        step_name: str
    ):
        """Load inference results to Hive table."""
        self.logger.log(step_name, f"Loading data to Hive table: {hive_table}")

        host = server.get("host")
        port = server.get("port", 22)
        username = server.get("username")
        key_path = server.get("key_path")

        # Build Hive load command
        # The path on Server 21 where inference results were transferred
        staging_path = f"/tmp/staging/{dataset_id}/"

        hive_cmd = (
            f"ssh -i {key_path} -p {port} {username}@{host} "
            f"'" + f'hive -e "LOAD DATA INPATH \'{staging_path}\' '
            f'INTO TABLE {hive_table}' + "'"
        )

        proc = await asyncio.create_subprocess_shell(
            hive_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        await proc.wait()

        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            self.logger.log(
                step_name,
                f"Warning: Hive load may have failed: {stderr.decode()}"
            )
        else:
            self.logger.log(
                step_name,
                f"Successfully loaded data to Hive table: {hive_table}"
            )


class MonitorExecutor(BaseExecutor):
    """Executor for Stage 5: Monitor and cleanup."""

    def __init__(
        self,
        db_manager,
        config: Dict[str, Any],
        logger
    ):
        """Initialize monitor executor."""
        super().__init__(db_manager, config, logger)

    async def execute(
        self,
        task_id: int,
        task: Dict,
        step: Dict,
        checkpoint: Optional[Dict[str, Any]],
        cancel_event: Optional[asyncio.Event]
    ) -> Dict[str, Any]:
        """Execute monitor stage."""
        step_name = "monitor"
        result = {
            "success": False,
            "error": None
        }

        try:
            self.logger.log(step_name, f"Generating final report for task {task_id}")

            # Get all metrics
            metrics = self.db.get_metrics(task_id)

            self.logger.log(
                step_name,
                f"Workflow completed. Collected metrics for {len(metrics)} stages."
            )

            for m in metrics:
                m_step = m.get("step_name")
                duration = m.get("duration_seconds", 0)
                self.logger.log(
                    step_name,
                    f"  {m_step}: {duration:.1f}s"
                )

            result["success"] = True
            return result

        except Exception as e:
            logger.exception(f"Monitor failed for task {task_id}")
            result["error"] = str(e)
            return result
