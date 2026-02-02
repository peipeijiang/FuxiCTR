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
Workflow executors for each stage.

Executors:
- BaseExecutor: Base class for all executors
- DataFetchExecutor: Fetch data from Server 21 via SSH
- TrainingExecutor: Run training on training server
- InferenceExecutor: Run inference on inference server
- TransportExecutor: Transport results to Server 21
- MonitorExecutor: Monitor and generate reports
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio


class BaseExecutor(ABC):
    """Base class for all workflow executors."""

    def __init__(
        self,
        db_manager,
        config: Dict[str, Any],
        logger
    ):
        """
        Initialize base executor.

        Args:
            db_manager: Database manager
            config: Configuration dictionary
            logger: Workflow logger
        """
        self.db = db_manager
        self.config = config
        self.logger = logger

    @abstractmethod
    async def execute(
        self,
        task_id: int,
        task: Dict,
        step: Dict,
        checkpoint: Optional[Dict[str, Any]],
        cancel_event: Optional[asyncio.Event]
    ) -> Dict[str, Any]:
        """
        Execute the stage.

        Args:
            task_id: Task ID
            task: Task dictionary
            step: Step dictionary
            checkpoint: Optional checkpoint data for resume
            cancel_event: Optional cancellation event

        Returns:
            Result dictionary with keys:
                - success: bool
                - error: Optional error message
                - ... (stage-specific results)
        """
        pass

    def _save_checkpoint(
        self,
        task_id: int,
        step_name: str,
        checkpoint_data: Dict[str, Any]
    ):
        """Save checkpoint data to database."""
        self.db.save_checkpoint(task_id, step_name, checkpoint_data)

    def _load_checkpoint(
        self,
        task_id: int,
        step_name: str
    ) -> Dict[str, Any]:
        """Load checkpoint data from database."""
        return self.db.get_checkpoint(task_id, step_name)


# Import executors
from fuxictr.workflow.executor.base import BaseExecutor
from fuxictr.workflow.executor.data_fetcher import DataFetchExecutor
from fuxictr.workflow.executor.trainer import TrainingExecutor
from fuxictr.workflow.executor.inference import (
    InferenceExecutor,
    TransportExecutor,
    MonitorExecutor
)

__all__ = [
    "BaseExecutor",
    "DataFetchExecutor",
    "TrainingExecutor",
    "InferenceExecutor",
    "TransportExecutor",
    "MonitorExecutor"
]
