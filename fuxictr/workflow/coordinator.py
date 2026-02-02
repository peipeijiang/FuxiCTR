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
Workflow Coordinator - Main orchestration engine for multi-server workflow.

Coordinates:
- Data fetch from Server 21
- Training on training server
- Inference on inference server
- Result transport back to Server 21
- Monitoring and metrics collection
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from pathlib import Path

from fuxictr.workflow.db import DatabaseManager
from fuxictr.workflow.models import (
    TaskStatus, StepStatus, StepName, Task, WorkflowMetrics
)
from fuxictr.workflow.utils.logger import WorkflowLogger
from fuxictr.workflow.utils.ssh_transfer import SSHTransferManager
from fuxictr.workflow.executor.data_fetch import DataFetchExecutor
from fuxictr.workflow.executor.trainer import TrainingExecutor
from fuxictr.workflow.executor.inference import InferenceExecutor
from fuxictr.workflow.executor.transport import TransportExecutor
from fuxictr.workflow.executor.monitor import MonitorExecutor


logger = logging.getLogger(__name__)


class WorkflowCoordinator:
    """
    Main coordinator for workflow execution.

    Features:
    - Automatic retry from failed stages
    - Checkpoint-based resume
    - Server-to-server SSH transfers
    - Real-time progress reporting via WebSocket
    - Graceful shutdown and cancellation
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        workflow_logger: WorkflowLogger,
        config: Dict[str, Any]
    ):
        """
        Initialize workflow coordinator.

        Args:
            db_manager: Database manager for state persistence
            workflow_logger: Logger for progress reporting
            config: Configuration dictionary containing:
                - servers: Server configurations
                - storage: Storage paths
                - transfer: Transfer settings
                - workflow: Workflow settings
        """
        self.db = db_manager
        self.logger = workflow_logger
        self.config = config

        # Extract server configurations
        self.servers = config.get("servers", {})
        self.storage = config.get("storage", {})
        self.transfer_config = config.get("transfer", {})
        self.workflow_config = config.get("workflow", {})

        # Initialize SSH transfer manager
        self.transfer_manager = SSHTransferManager(
            db_manager=db_manager,
            workflow_logger=workflow_logger,
            chunk_size=self.transfer_config.get("chunk_size", 100 * 1024 * 1024),
            max_retries=self.transfer_config.get("max_retries", 10),
            compression=self.transfer_config.get("compression", True),
            bandwidth_limit=self.transfer_config.get("bandwidth_limit"),
            verify_checksum=self.transfer_config.get("verify_checksum", True)
        )

        # Initialize executors (lazy initialization)
        self._executors: Dict[str, Any] = {}
        self._running_tasks: Dict[int, bool] = {}

    async def execute_workflow(
        self,
        task_id: int,
        cancel_event: Optional[asyncio.Event] = None
    ) -> TaskStatus:
        """
        Execute complete workflow for a task.

        Args:
            task_id: Task ID to execute
            cancel_event: Optional event for graceful cancellation

        Returns:
            Final task status
        """
        # Mark task as running
        self.db.update_task_status(task_id, TaskStatus.RUNNING.value, current_step=0)
        self._running_tasks[task_id] = True

        try:
            # Load task details
            task = self.db.get_task(task_id)
            if not task:
                self.logger.error("workflow", f"Task {task_id} not found")
                return TaskStatus.FAILED

            self.logger.log(
                "workflow",
                f"Starting workflow: {task['name']} (ID: {task_id})"
            )

            # Get task steps
            steps = self.db.get_task_steps(task_id)

            # Execute stages in order
            stage_executors = [
                ("data_fetch", self._execute_data_fetch),
                ("train", self._execute_training),
                ("infer", self._execute_inference),
                ("transport", self._execute_transport),
                ("monitor", self._execute_monitor),
            ]

            final_status = TaskStatus.COMPLETED

            for step_name, executor_func in stage_executors:
                # Check if cancelled
                if cancel_event and cancel_event.is_set():
                    self.logger.log("workflow", "Workflow cancelled by user")
                    final_status = TaskStatus.CANCELLED
                    break

                # Check if we should stop running
                if not self._running_tasks.get(task_id, False):
                    final_status = TaskStatus.CANCELLED
                    break

                # Find the step record
                step = next((s for s in steps if s["step_name"] == step_name), None)
                if not step:
                    self.logger.error("workflow", f"Step {step_name} not found")
                    final_status = TaskStatus.FAILED
                    break

                # Skip if already completed
                if step["status"] == StepStatus.COMPLETED.value:
                    self.logger.log(
                        "workflow",
                        f"Step {step_name} already completed, skipping"
                    )
                    # Update current_step
                    current_step = list(StepName).index(StepName[step_name.upper()])
                    self.db.update_task_status(
                        task_id, TaskStatus.RUNNING.value, current_step=current_step
                    )
                    continue

                # Execute stage
                try:
                    success = await executor_func(task_id, task, step, cancel_event)
                    if not success:
                        final_status = TaskStatus.FAILED
                        break

                    # Update current_step
                    current_step = list(StepName).index(StepName[step_name.upper()])
                    self.db.update_task_status(
                        task_id, TaskStatus.RUNNING.value, current_step=current_step + 1
                    )

                except Exception as e:
                    self.logger.error(step_name, f"Stage failed: {e}")
                    self.db.update_step_status(
                        task_id, step_name, StepStatus.FAILED, error_message=str(e)
                    )
                    final_status = TaskStatus.FAILED
                    break

            # Update final task status
            self.db.update_task_status(task_id, final_status.value)
            self._running_tasks.pop(task_id, None)

            if final_status == TaskStatus.COMPLETED:
                self.logger.log(
                    "workflow",
                    f"Workflow completed successfully: {task['name']}"
                )
            else:
                self.logger.log(
                    "workflow",
                    f"Workflow ended with status: {final_status.value}"
                )

            return final_status

        except Exception as e:
            logger.exception(f"Workflow execution failed for task {task_id}")
            self.db.update_task_status(task_id, TaskStatus.FAILED.value)
            self._running_tasks.pop(task_id, None)
            return TaskStatus.FAILED

    async def resume_workflow(
        self,
        task_id: int,
        cancel_event: Optional[asyncio.Event] = None
    ) -> TaskStatus:
        """
        Resume a workflow from its last checkpoint.

        Args:
            task_id: Task ID to resume
            cancel_event: Optional event for graceful cancellation

        Returns:
            Final task status
        """
        self.logger.log("workflow", f"Resuming workflow for task {task_id}")

        # Check which steps are completed
        steps = self.db.get_task_steps(task_id)
        completed_steps = [s["step_name"] for s in steps
                          if s["status"] == StepStatus.COMPLETED.value]

        self.logger.log(
            "workflow",
            f"Completed steps: {', '.join(completed_steps) if completed_steps else 'None'}"
        )

        # Execute from the first incomplete step
        return await self.execute_workflow(task_id, cancel_event)

    def cancel_workflow(self, task_id: int) -> bool:
        """
        Request cancellation of a running workflow.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if cancellation requested
        """
        if task_id in self._running_tasks:
            self._running_tasks[task_id] = False
            self.db.update_task_status(task_id, TaskStatus.CANCELLED.value)
            self.logger.log("workflow", f"Cancellation requested for task {task_id}")
            return True
        return False

    def pause_workflow(self, task_id: int) -> bool:
        """
        Pause a running workflow (same as cancel for now).

        Args:
            task_id: Task ID to pause

        Returns:
            True if pause requested
        """
        return self.cancel_workflow(task_id)

    # ========================================================================
    # Stage Executors
    # ========================================================================

    async def _execute_data_fetch(
        self,
        task_id: int,
        task: Dict,
        step: Dict,
        cancel_event: Optional[asyncio.Event]
    ) -> bool:
        """Execute Stage 1: Data Fetch from Server 21."""
        step_name = "data_fetch"

        # Check if we should resume from checkpoint
        checkpoint = self.db.get_checkpoint(task_id, step_name)
        if checkpoint:
            self.logger.log(
                step_name,
                f"Resuming from checkpoint: {checkpoint}"
            )

        # Update step status
        self.db.update_step_status(task_id, step_name, StepStatus.RUNNING)

        try:
            # Get or create executor
            executor = self._get_executor("data_fetch")
            start_time = datetime.now()

            # Execute data fetch
            result = await executor.execute(
                task_id=task_id,
                task=task,
                step=step,
                checkpoint=checkpoint,
                cancel_event=cancel_event
            )

            # Calculate metrics
            duration = (datetime.now() - start_time).total_seconds()
            metrics = {
                "duration_seconds": duration,
                "bytes_transferred": result.get("bytes_transferred", 0),
                "files_transferred": result.get("files_transferred", 0)
            }
            self.db.save_metrics(task_id, step_name, metrics)

            if result.get("success"):
                self.db.update_step_status(task_id, step_name, StepStatus.COMPLETED)
                self.logger.complete(step_name)
                return True
            else:
                error_msg = result.get("error", "Unknown error")
                self.db.update_step_status(
                    task_id, step_name, StepStatus.FAILED, error_message=error_msg
                )
                return False

        except Exception as e:
            logger.exception(f"Data fetch failed for task {task_id}")
            self.db.update_step_status(
                task_id, step_name, StepStatus.FAILED, error_message=str(e)
            )
            return False

    async def _execute_training(
        self,
        task_id: int,
        task: Dict,
        step: Dict,
        cancel_event: Optional[asyncio.Event]
    ) -> bool:
        """Execute Stage 2: Training."""
        step_name = "train"

        # Check if we should resume from checkpoint
        checkpoint = self.db.get_checkpoint(task_id, step_name)
        if checkpoint:
            self.logger.log(
                step_name,
                f"Resuming training from epoch {checkpoint.get('epoch', 0)}"
            )

        self.db.update_step_status(task_id, step_name, StepStatus.RUNNING)

        try:
            executor = self._get_executor("train")
            start_time = datetime.now()

            result = await executor.execute(
                task_id=task_id,
                task=task,
                step=step,
                checkpoint=checkpoint,
                cancel_event=cancel_event
            )

            duration = (datetime.now() - start_time).total_seconds()
            metrics = {
                "duration_seconds": duration,
                "epochs_completed": result.get("epochs_completed", 0),
                "best_auc": result.get("best_auc", 0.0),
                "best_loss": result.get("best_loss", 0.0),
                "final_loss": result.get("final_loss", 0.0)
            }
            self.db.save_metrics(task_id, step_name, metrics)

            if result.get("success"):
                self.db.update_step_status(task_id, step_name, StepStatus.COMPLETED)
                self.logger.complete(step_name)
                return True
            else:
                error_msg = result.get("error", "Unknown error")
                self.db.update_step_status(
                    task_id, step_name, StepStatus.FAILED, error_message=error_msg
                )
                return False

        except Exception as e:
            logger.exception(f"Training failed for task {task_id}")
            self.db.update_step_status(
                task_id, step_name, StepStatus.FAILED, error_message=str(e)
            )
            return False

    async def _execute_inference(
        self,
        task_id: int,
        task: Dict,
        step: Dict,
        cancel_event: Optional[asyncio.Event]
    ) -> bool:
        """Execute Stage 3: Inference."""
        step_name = "infer"

        checkpoint = self.db.get_checkpoint(task_id, step_name)
        if checkpoint:
            self.logger.log(
                step_name,
                f"Resuming inference from part {checkpoint.get('current_part', 0)}"
            )

        self.db.update_step_status(task_id, step_name, StepStatus.RUNNING)

        try:
            executor = self._get_executor("infer")
            start_time = datetime.now()

            result = await executor.execute(
                task_id=task_id,
                task=task,
                step=step,
                checkpoint=checkpoint,
                cancel_event=cancel_event
            )

            duration = (datetime.now() - start_time).total_seconds()
            metrics = {
                "duration_seconds": duration,
                "rows_processed": result.get("rows_processed", 0),
                "inference_throughput": result.get("throughput", 0.0)
            }
            self.db.save_metrics(task_id, step_name, metrics)

            if result.get("success"):
                self.db.update_step_status(task_id, step_name, StepStatus.COMPLETED)
                self.logger.complete(step_name)
                return True
            else:
                error_msg = result.get("error", "Unknown error")
                self.db.update_step_status(
                    task_id, step_name, StepStatus.FAILED, error_message=error_msg
                )
                return False

        except Exception as e:
            logger.exception(f"Inference failed for task {task_id}")
            self.db.update_step_status(
                task_id, step_name, StepStatus.FAILED, error_message=str(e)
            )
            return False

    async def _execute_transport(
        self,
        task_id: int,
        task: Dict,
        step: Dict,
        cancel_event: Optional[asyncio.Event]
    ) -> bool:
        """Execute Stage 4: Transport results to Server 21."""
        step_name = "transport"

        checkpoint = self.db.get_checkpoint(task_id, step_name)
        if checkpoint:
            self.logger.log(
                step_name,
                f"Resuming transport: {checkpoint.get('transferred_bytes', 0)} bytes transferred"
            )

        self.db.update_step_status(task_id, step_name, StepStatus.RUNNING)

        try:
            executor = self._get_executor("transport")
            start_time = datetime.now()

            result = await executor.execute(
                task_id=task_id,
                task=task,
                step=step,
                checkpoint=checkpoint,
                cancel_event=cancel_event
            )

            duration = (datetime.now() - start_time).total_seconds()
            metrics = {
                "duration_seconds": duration,
                "bytes_transferred": result.get("bytes_transferred", 0),
                "files_transferred": result.get("files_transferred", 0)
            }
            self.db.save_metrics(task_id, step_name, metrics)

            if result.get("success"):
                self.db.update_step_status(task_id, step_name, StepStatus.COMPLETED)
                self.logger.complete(step_name)
                return True
            else:
                error_msg = result.get("error", "Unknown error")
                self.db.update_step_status(
                    task_id, step_name, StepStatus.FAILED, error_message=error_msg
                )
                return False

        except Exception as e:
            logger.exception(f"Transport failed for task {task_id}")
            self.db.update_step_status(
                task_id, step_name, StepStatus.FAILED, error_message=str(e)
            )
            return False

    async def _execute_monitor(
        self,
        task_id: int,
        task: Dict,
        step: Dict,
        cancel_event: Optional[asyncio.Event]
    ) -> bool:
        """Execute Stage 5: Monitor and cleanup."""
        step_name = "monitor"

        self.db.update_step_status(task_id, step_name, StepStatus.RUNNING)

        try:
            executor = self._get_executor("monitor")

            result = await executor.execute(
                task_id=task_id,
                task=task,
                step=step,
                checkpoint=None,
                cancel_event=cancel_event
            )

            if result.get("success"):
                self.db.update_step_status(task_id, step_name, StepStatus.COMPLETED)

                # Generate final report
                await self._generate_final_report(task_id)

                self.logger.complete(step_name)
                return True
            else:
                error_msg = result.get("error", "Unknown error")
                self.db.update_step_status(
                    task_id, step_name, StepStatus.FAILED, error_message=error_msg
                )
                return False

        except Exception as e:
            logger.exception(f"Monitor failed for task {task_id}")
            self.db.update_step_status(
                task_id, step_name, StepStatus.FAILED, error_message=str(e)
            )
            return False

    async def _generate_final_report(self, task_id: int):
        """Generate final workflow report."""
        task = self.db.get_task(task_id)
        metrics = self.db.get_aggregate_metrics(task_id)

        report = {
            "task_id": task_id,
            "task_name": task.get("name"),
            "experiment_id": task.get("experiment_id"),
            "model": task.get("model"),
            "status": task.get("status"),
            "created_at": task.get("created_at"),
            "completed_at": task.get("completed_at"),
            "steps": self.db.get_task_steps(task_id),
            "metrics": metrics
        }

        # Save report as additional metrics
        self.db.save_metrics(task_id, "workflow_report", report)

        self.logger.log(
            "workflow",
            f"Final report generated for task {task_id}"
        )

    def _get_executor(self, step_name: str):
        """Get or create executor for a step."""
        if step_name not in self._executors:
            if step_name == "data_fetch":
                self._executors[step_name] = DataFetchExecutor(
                    db_manager=self.db,
                    transfer_manager=self.transfer_manager,
                    config=self.config,
                    logger=self.logger
                )
            elif step_name == "train":
                self._executors[step_name] = TrainingExecutor(
                    db_manager=self.db,
                    config=self.config,
                    logger=self.logger
                )
            elif step_name == "infer":
                self._executors[step_name] = InferenceExecutor(
                    db_manager=self.db,
                    config=self.config,
                    logger=self.logger
                )
            elif step_name == "transport":
                self._executors[step_name] = TransportExecutor(
                    db_manager=self.db,
                    transfer_manager=self.transfer_manager,
                    config=self.config,
                    logger=self.logger
                )
            elif step_name == "monitor":
                self._executors[step_name] = MonitorExecutor(
                    db_manager=self.db,
                    config=self.config,
                    logger=self.logger
                )

        return self._executors[step_name]

    async def get_task_progress(self, task_id: int) -> Dict[str, Any]:
        """
        Get current progress of a task.

        Args:
            task_id: Task ID

        Returns:
            Progress dictionary with step statuses and metrics
        """
        task = self.db.get_task(task_id)
        if not task:
            return {"error": "Task not found"}

        steps = self.db.get_task_steps(task_id)
        metrics = self.db.get_aggregate_metrics(task_id)

        return {
            "task_id": task_id,
            "task_name": task.get("name"),
            "status": task.get("status"),
            "current_step": task.get("current_step"),
            "total_steps": task.get("total_steps", 5),
            "steps": [
                {
                    "name": s["step_name"],
                    "status": s["status"],
                    "started_at": s.get("started_at"),
                    "completed_at": s.get("completed_at"),
                    "error_message": s.get("error_message"),
                    "retry_count": s.get("retry_count", 0)
                }
                for s in steps
            ],
            "metrics": metrics,
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "completed_at": task.get("completed_at")
        }


class WorkflowOrchestrator:
    """
    Higher-level orchestrator that manages multiple workflows.

    Features:
    - Queue management
    - Concurrent workflow execution
    - Resource allocation
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        config: Dict[str, Any]
    ):
        """
        Initialize workflow orchestrator.

        Args:
            db_manager: Database manager
            config: Configuration dictionary
        """
        self.db = db_manager
        self.config = config
        self.coordinators: Dict[int, WorkflowCoordinator] = {}
        self.active_workflows: Dict[int, asyncio.Task] = {}
        self.cancel_events: Dict[int, asyncio.Event] = {}

    async def submit_workflow(
        self,
        task_id: int,
        workflow_logger: WorkflowLogger
    ) -> bool:
        """
        Submit a workflow for execution.

        Args:
            task_id: Task ID to execute
            workflow_logger: Logger for the workflow

        Returns:
            True if workflow submitted successfully
        """
        if task_id in self.active_workflows:
            return False  # Already running

        # Create cancel event
        cancel_event = asyncio.Event()
        self.cancel_events[task_id] = cancel_event

        # Create coordinator
        coordinator = WorkflowCoordinator(
            db_manager=self.db,
            workflow_logger=workflow_logger,
            config=self.config
        )
        self.coordinators[task_id] = coordinator

        # Start workflow in background
        task = asyncio.create_task(
            coordinator.execute_workflow(task_id, cancel_event)
        )
        self.active_workflows[task_id] = task

        return True

    async def cancel_workflow(self, task_id: int) -> bool:
        """
        Cancel a running workflow.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if cancellation successful
        """
        if task_id not in self.coordinators:
            return False  # Not running

        # Set cancel event
        if task_id in self.cancel_events:
            self.cancel_events[task_id].set()

        # Cancel coordinator
        coordinator = self.coordinators[task_id]
        coordinator.cancel_workflow(task_id)

        # Cancel task
        if task_id in self.active_workflows:
            self.active_workflows[task_id].cancel()

        # Clean up
        self.active_workflows.pop(task_id, None)
        self.coordinators.pop(task_id, None)
        self.cancel_events.pop(task_id, None)

        return True

    async def get_workflow_status(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a workflow."""
        if task_id not in self.coordinators:
            # Not actively running, get from database
            task = self.db.get_task(task_id)
            if task:
                return {
                    "task_id": task_id,
                    "status": task.get("status"),
                    "active": False
                }
            return None

        coordinator = self.coordinators[task_id]
        progress = await coordinator.get_task_progress(task_id)
        progress["active"] = True
        return progress

    async def cleanup_completed_workflows(self):
        """Clean up completed workflows."""
        completed = []
        for task_id, task in self.active_workflows.items():
            if task.done():
                completed.append(task_id)

        for task_id in completed:
            self.active_workflows.pop(task_id, None)
            self.coordinators.pop(task_id, None)
            self.cancel_events.pop(task_id, None)

            logger.info(f"Cleaned up completed workflow: {task_id}")
