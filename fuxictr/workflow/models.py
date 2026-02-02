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
Data models for workflow orchestration.

Supports:
- Multi-server workflow without shared directories
- Checkpoint-based resume capability
- Chunk-based transfer tracking
- Server assignment and coordination
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json


class TaskStatus(str, Enum):
    """Status of a workflow task."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepName(str, Enum):
    """Named workflow steps in execution order."""
    DATA_FETCH = "data_fetch"
    TRAIN = "train"
    INFER = "infer"
    TRANSPORT = "transport"
    MONITOR = "monitor"


class ChunkStatus(str, Enum):
    """Status of a transfer chunk."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ServerRole(str, Enum):
    """Role of a server in the workflow."""
    DATA_SOURCE = "data_source"      # Server 21: HDFS/Hive access
    TRAINING = "training"             # Training server
    INFERENCE = "inference"           # Inference server
    ORCHESTRATOR = "orchestrator"     # Workflow coordinator


@dataclass
class Task:
    """Workflow task model."""
    id: Optional[int]
    name: str
    user: str
    model: str
    experiment_id: str
    sample_sql: str
    infer_sql: str
    hdfs_path: str
    hive_table: str

    # Task state
    status: TaskStatus = TaskStatus.PENDING
    current_step: int = 0
    total_steps: int = 5  # Default number of steps

    # Server assignment
    training_server: Optional[str] = None
    inference_server: Optional[str] = None

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Additional config (JSON)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling enums and datetime."""
        return {
            "id": self.id,
            "name": self.name,
            "user": self.user,
            "model": self.model,
            "experiment_id": self.experiment_id,
            "sample_sql": self.sample_sql,
            "infer_sql": self.infer_sql,
            "hdfs_path": self.hdfs_path,
            "hive_table": self.hive_table,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "training_server": self.training_server,
            "inference_server": self.inference_server,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config": self.config
        }


@dataclass
class TaskStep:
    """Workflow step model with checkpoint support."""
    id: Optional[int]
    task_id: int
    step_name: str  # StepName enum value as string
    step_order: int

    # Step state
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    # Checkpoint data (JSON string)
    checkpoint_data: Optional[str] = None

    def get_checkpoint(self) -> Dict[str, Any]:
        """Parse and return checkpoint data as dictionary."""
        if self.checkpoint_data:
            try:
                return json.loads(self.checkpoint_data)
            except json.JSONDecodeError:
                return {}
        return {}

    def set_checkpoint(self, data: Dict[str, Any]):
        """Serialize checkpoint data to JSON."""
        self.checkpoint_data = json.dumps(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "step_name": self.step_name,
            "step_order": self.step_order,
            "status": self.status.value if isinstance(self.status, StepStatus) else self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "checkpoint_data": self.checkpoint_data
        }


@dataclass
class TransferChunk:
    """Transfer chunk model for resume capability."""
    id: Optional[int]
    task_id: int
    step_name: str
    chunk_id: str

    # File info
    source_path: Optional[str] = None
    dest_path: Optional[str] = None
    offset: int = 0
    size: int = 0
    checksum: Optional[str] = None

    # Transfer state
    status: ChunkStatus = ChunkStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "step_name": self.step_name,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "dest_path": self.dest_path,
            "offset": self.offset,
            "size": self.size,
            "checksum": self.checksum,
            "status": self.status.value if isinstance(self.status, ChunkStatus) else self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "error_message": self.error_message
        }


@dataclass
class ServerConfig:
    """Server configuration for workflow."""
    name: str
    host: str
    port: int = 22
    username: str = ""
    key_path: Optional[str] = None
    role: ServerRole = ServerRole.TRAINING
    gpus: List[int] = field(default_factory=list)

    # Storage paths
    data_root: str = "./data/"
    model_root: str = "./checkpoints/"
    staging_dir: str = "/data/staging"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "key_path": self.key_path,
            "role": self.role.value if isinstance(self.role, ServerRole) else self.role,
            "gpus": self.gpus,
            "data_root": self.data_root,
            "model_root": self.model_root,
            "staging_dir": self.staging_dir
        }


@dataclass
class TrainingCheckpoint:
    """Training checkpoint data for resume."""
    epoch: int = 0
    total_epochs: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    model_path: Optional[str] = None
    optimizer_state_path: Optional[str] = None

    # Training state
    loss: float = 0.0
    learning_rate: float = 0.0
    global_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "model_path": self.model_path,
            "optimizer_state_path": self.optimizer_state_path,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "global_step": self.global_step
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingCheckpoint':
        """Create from dictionary."""
        return cls(
            epoch=data.get("epoch", 0),
            total_epochs=data.get("total_epochs", 0),
            best_metric=data.get("best_metric", 0.0),
            best_epoch=data.get("best_epoch", 0),
            model_path=data.get("model_path"),
            optimizer_state_path=data.get("optimizer_state_path"),
            loss=data.get("loss", 0.0),
            learning_rate=data.get("learning_rate", 0.0),
            global_step=data.get("global_step", 0)
        )


@dataclass
class InferenceCheckpoint:
    """Inference checkpoint data for resume."""
    total_rows: int = 0
    processed_rows: int = 0
    completed_parts: List[str] = field(default_factory=list)
    output_path: Optional[str] = None

    # Progress tracking
    current_file: Optional[str] = None
    current_offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_rows": self.total_rows,
            "processed_rows": self.processed_rows,
            "completed_parts": self.completed_parts,
            "output_path": self.output_path,
            "current_file": self.current_file,
            "current_offset": self.current_offset
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceCheckpoint':
        """Create from dictionary."""
        return cls(
            total_rows=data.get("total_rows", 0),
            processed_rows=data.get("processed_rows", 0),
            completed_parts=data.get("completed_parts", []),
            output_path=data.get("output_path"),
            current_file=data.get("current_file"),
            current_offset=data.get("current_offset", 0)
        )


@dataclass
class TransferCheckpoint:
    """Transfer checkpoint data for resume."""
    total_bytes: int = 0
    transferred_bytes: int = 0
    completed_chunks: List[str] = field(default_factory=list)
    current_chunk: Optional[str] = None

    # Source and destination info
    source_host: Optional[str] = None
    source_path: Optional[str] = None
    dest_host: Optional[str] = None
    dest_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_bytes": self.total_bytes,
            "transferred_bytes": self.transferred_bytes,
            "completed_chunks": self.completed_chunks,
            "current_chunk": self.current_chunk,
            "source_host": self.source_host,
            "source_path": self.source_path,
            "dest_host": self.dest_host,
            "dest_path": self.dest_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransferCheckpoint':
        """Create from dictionary."""
        return cls(
            total_bytes=data.get("total_bytes", 0),
            transferred_bytes=data.get("transferred_bytes", 0),
            completed_chunks=data.get("completed_chunks", []),
            current_chunk=data.get("current_chunk"),
            source_host=data.get("source_host"),
            source_path=data.get("source_path"),
            dest_host=data.get("dest_host"),
            dest_path=data.get("dest_path")
        )


@dataclass
class WorkflowMetrics:
    """Metrics collected during workflow execution."""
    task_id: int
    step_name: str

    # Timing metrics
    duration_seconds: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Transfer metrics
    bytes_transferred: int = 0
    files_transferred: int = 0
    transfer_rate_mbps: float = 0.0

    # Training metrics
    epochs_completed: int = 0
    best_auc: float = 0.0
    best_loss: float = 0.0
    final_loss: float = 0.0

    # Inference metrics
    rows_processed: int = 0
    inference_throughput: float = 0.0  # rows per second

    # Additional metrics
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "task_id": self.task_id,
            "step_name": self.step_name,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "bytes_transferred": self.bytes_transferred,
            "files_transferred": self.files_transferred,
            "transfer_rate_mbps": self.transfer_rate_mbps,
            "epochs_completed": self.epochs_completed,
            "best_auc": self.best_auc,
            "best_loss": self.best_loss,
            "final_loss": self.final_loss,
            "rows_processed": self.rows_processed,
            "inference_throughput": self.inference_throughput,
            "additional_metrics": self.additional_metrics
        }

        # Filter out None values
        return {k: v for k, v in result.items() if v is not None}
