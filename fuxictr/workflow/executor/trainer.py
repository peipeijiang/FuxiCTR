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
Training Executor - Stage 2

Runs model training on training server with checkpoint/resume support.

Features:
- Multi-GPU DDP training
- Checkpoint after each epoch
- Resume from last checkpoint
- TensorBoard logging
"""

import asyncio
import subprocess
import os
import re
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from fuxictr.workflow.executor.base import BaseExecutor
from fuxictr.workflow.models import TrainingCheckpoint
from fuxictr.workflow.utils.config_merge import prepare_training_config


logger = logging.getLogger(__name__)


class TrainingExecutor(BaseExecutor):
    """
    Executor for Stage 2: Training.

    Workflow:
    1. Load training data from staging
    2. Initialize model with config
    3. Run training with DDP
    4. Save checkpoints after each epoch
    5. Monitor metrics
    """

    def __init__(
        self,
        db_manager,
        config: Dict[str, Any],
        logger
    ):
        """
        Initialize training executor.

        Args:
            db_manager: Database manager
            config: Configuration dictionary
            logger: Workflow logger
        """
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
        """
        Execute training stage.

        Args:
            task_id: Task ID
            task: Task dictionary
            step: Step dictionary
            checkpoint: Optional checkpoint for resume
            cancel_event: Optional cancellation event

        Returns:
            Result dictionary
        """
        step_name = "train"
        result = {
            "success": False,
            "error": None,
            "epochs_completed": 0,
            "best_auc": 0.0,
            "best_loss": 0.0,
            "final_loss": 0.0
        }

        try:
            self.logger.log(
                step_name,
                f"Starting training for task {task_id}"
            )

            # Parse checkpoint if resuming
            training_checkpoint = TrainingCheckpoint()
            if checkpoint:
                training_checkpoint = TrainingCheckpoint.from_dict(checkpoint)
                self.logger.log(
                    step_name,
                    f"Resuming from epoch {training_checkpoint.epoch}"
                )

            # Get training parameters
            experiment_id = task.get("experiment_id", "")
            model = task.get("model", "")
            user = task.get("user", "")

            # Get data_fetch checkpoint to retrieve processed data paths and features
            data_fetch_checkpoint = self.db.get_checkpoint(task_id, "data_fetch")

            if not data_fetch_checkpoint or not data_fetch_checkpoint.get("processed_data_paths"):
                raise ValueError(
                    "Data fetch step not completed or processed data not found. "
                    "Please ensure the data_fetch step has completed successfully."
                )

            processed_paths = data_fetch_checkpoint.get("processed_data_paths", {})
            dataset_id = processed_paths.get("dataset_id", f"{experiment_id}.unknown")
            dataset_dir = processed_paths.get("dataset_dir", "")

            # Use the processed directory as data_root for training
            # New structure: datasets_root/{exp_id.dataset_id}/processed/
            data_root = dataset_dir
            feature_cols = processed_paths.get("feature_cols", [])
            label_col = processed_paths.get("label_col", ["label"])

            # Model root: FuxiCTR will append dataset_id to this path
            # Final structure: model_zoo/{model}/checkpoints/{dataset_id}/
            # FuxiCTR rank_model.py does: model_dir = os.path.join(model_root, feature_map.dataset_id)
            model_zoo_root = self.fuxictr_paths.get("model_root", "../../../model_zoo")
            model_root = f"{model_zoo_root}/{model}/checkpoints/"

            # Create model directory
            os.makedirs(model_root, exist_ok=True)

            self.logger.log(
                step_name,
                f"Dataset ID: {dataset_id}"
            )
            self.logger.log(
                step_name,
                f"Data root: {data_root}"
            )
            self.logger.log(
                step_name,
                f"Model root: {model_root}"
            )

            # Prepare merged training config
            # This loads original experiment config and only replaces data paths and features
            try:
                config_path, merged_config, original_config_path = prepare_training_config(
                    user=user,
                    model=model,
                    experiment_id=experiment_id,
                    data_root=data_root,  # Use processed directory
                    train_data=processed_paths.get("train_data", ""),
                    valid_data=processed_paths.get("valid_data", ""),
                    test_data=processed_paths.get("test_data", ""),
                    feature_cols=feature_cols,
                    label_col=label_col,  # User-configured
                    task_id=task_id,
                    dataset_id=dataset_id,  # Pass for FuxiCTR subdirectory creation
                    model_root=model_root,  # Pass model_root for FuxiCTR
                    processed_root=data_root  # Pass processed_root (same as data_root for workflow)
                )

                self.logger.log(
                    step_name,
                    f"Using merged config (original: {original_config_path})"
                )
                self.logger.log(
                    step_name,
                    f"FuxiCTR will use: dataset_id={dataset_id}, model_root={model_root}"
                )

            except FileNotFoundError as e:
                raise ValueError(
                    f"Failed to find original experiment config for {user}/{model}/{experiment_id}. "
                    f"Please ensure the experiment exists in the dashboard."
                ) from e

            # Run training
            training_result = await self._run_training(
                experiment_id=experiment_id,
                dataset_id=dataset_id,
                data_root=data_root,
                model_root=model_root,
                step_name=step_name,
                checkpoint=training_checkpoint,
                cancel_event=cancel_event,
                task=task,
                config_path=config_path  # Pass merged config path
            )

            result.update(training_result)

            if result.get("success"):
                # Save final checkpoint
                checkpoint_data = {
                    "epoch": result.get("epochs_completed", 0),
                    "total_epochs": result.get("total_epochs", 0),
                    "best_metric": result.get("best_auc", 0.0),
                    "best_epoch": result.get("best_epoch", 0),
                    "dataset_id": dataset_id,
                    "model_path": f"{model_root}/{experiment_id}.model",
                    "final_loss": result.get("final_loss", 0.0),
                    "timestamp": datetime.now().isoformat()
                }
                self._save_checkpoint(task_id, step_name, checkpoint_data)

            return result

        except Exception as e:
            logger.exception(f"Training failed for task {task_id}")
            result["error"] = str(e)
            return result

    async def _run_training(
        self,
        experiment_id: str,
        dataset_id: str,
        data_root: str,
        model_root: str,
        step_name: str,
        checkpoint: TrainingCheckpoint,
        cancel_event: Optional[asyncio.Event],
        task: Dict[str, Any],
        config_path: str
    ) -> Dict[str, Any]:
        """
        Run the training process.

        Args:
            experiment_id: Experiment ID
            dataset_id: Dataset ID
            data_root: Data root directory (should match data_fetch step)
            model_root: Model root directory
            step_name: Step name for logging
            checkpoint: Training checkpoint
            cancel_event: Optional cancellation event
            task: Task dictionary containing model and config info
            config_path: Path to merged training config

        Returns:
            Training result dictionary
        """
        result = {
            "success": False,
            "epochs_completed": 0,
            "best_auc": 0.0,
            "best_loss": 0.0,
            "final_loss": 0.0
        }

        # Find run_expid.py script
        script_dir = os.path.dirname(os.path.dirname(__file__))
        run_script = os.path.join(script_dir, "../../model_zoo/multitask/run_expid.py")

        if not os.path.exists(run_script):
            # Try alternative paths
            run_script = os.path.join(
                os.path.dirname(__file__),
                "../../../model_zoo/multitask/run_expid.py"
            )

        if not os.path.exists(run_script):
            raise FileNotFoundError(f"run_expid.py not found at {run_script}")

        # Use the merged config file directly
        # config_path is the full path to the merged config file
        config_dir = os.path.dirname(config_path)
        config_filename = os.path.basename(config_path)
        # Extract expid from filename (format: {experiment_id}_task{task_id}.yaml)
        merged_expid = config_filename.replace(".yaml", "")

        self.logger.log(
            step_name,
            f"Using merged config: {config_path}"
        )

        # Build command
        cmd = [
            "python",
            run_script,
            "--config", config_dir,
            "--expid", merged_expid
        ]

        # Add resume checkpoint if exists
        if checkpoint and checkpoint.epoch > 0:
            cmd.extend([
                "--resume_epoch", str(checkpoint.epoch),
                "--resume_model", checkpoint.model_path or ""
            ])

        self.logger.log(
            step_name,
            f"Executing: {' '.join(cmd[:5])}..."
        )

        # Create model directory (model_root already contains dataset_id)
        os.makedirs(model_root, exist_ok=True)

        # Set up environment with data_root for run_expid.py
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "FUXICTR_DATA_ROOT": data_root
        }

        # Start training process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        # Parse output for metrics
        best_auc = 0.0
        best_loss = float('inf')
        current_epoch = 0
        final_loss = 0.0
        total_epochs = 10  # Default, will be updated if detected
        last_logged_epoch = 0

        # Monitor process
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
                # Parse metrics from log
                # Look for patterns like "Epoch 1/10 - loss: 0.5 - auc: 0.8"
                epoch_match = re.search(r'[Ee]poch\s+(\d+)(?:/(\d+))?', log_line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    if epoch_match.group(2):
                        total_epochs = int(epoch_match.group(2))

                    # Log structured progress when epoch changes
                    if current_epoch != last_logged_epoch:
                        self.logger.progress(
                            step_name,
                            current_epoch,
                            total_epochs,
                            f"Training epoch {current_epoch}/{total_epochs}"
                        )
                        last_logged_epoch = current_epoch

                loss_match = re.search(r'[Ll]oss[:\s]+([\d.]+)', log_line)
                if loss_match:
                    final_loss = float(loss_match.group(1))
                    if final_loss < best_loss:
                        best_loss = final_loss

                    # Log metric update
                    self.logger.metric(step_name, "loss", final_loss)

                auc_match = re.search(r'[Aa][Uu][Cc][:\s]+([\d.]+)', log_line)
                if auc_match:
                    best_auc = float(auc_match.group(1))

                    # Log metric update
                    self.logger.metric(step_name, "auc", best_auc)

                # Log to workflow
                self.logger.log(step_name, log_line)

        # Monitor stderr separately for error messages
        async def monitor_stderr():
            """Monitor stderr for error messages."""
            while True:
                if cancel_event and cancel_event.is_set():
                    break
                line = await process.stderr.readline()
                if not line:
                    break
                error_line = line.decode().strip()
                if error_line:
                    self.logger.log(step_name, f"[STDERR] {error_line}", level="WARNING")

        # Start stderr monitoring task
        stderr_task = asyncio.create_task(monitor_stderr())

        # Wait for stdout to complete
        await process.wait()
        returncode = process.returncode

        # Cancel stderr task
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass

        if returncode == 0:
            result["success"] = True
            result["epochs_completed"] = current_epoch
            result["best_auc"] = best_auc
            result["best_loss"] = best_loss
            result["final_loss"] = final_loss

            # Log completion
            self.logger.complete(step_name, {
                "epochs_completed": current_epoch,
                "best_auc": best_auc,
                "best_loss": best_loss,
                "final_loss": final_loss
            })
            return result
        else:
            # Read remaining stderr
            remaining_stderr = await process.stderr.read()
            error_msg = remaining_stderr.decode()
            result["error"] = f"Training failed (code {returncode}): {error_msg}"

            # Log error
            self.logger.error(step_name, result["error"])
            return result


# Backward compatibility alias
Trainer = TrainingExecutor
