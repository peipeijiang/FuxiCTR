from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class StepName(str, Enum):
    DATA_FETCH = "data_fetch"
    TRAIN = "train"
    INFER = "infer"
    MONITOR = "monitor"
    TRANSPORT = "transport"
    UPLOAD = "upload"

class ChunkStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: Optional[int]
    name: str
    user: str
    model: str
    experiment_id: str
    sample_sql: str
    infer_sql: str
    hdfs_path: str
    hive_table: str
    status: TaskStatus
    current_step: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

@dataclass
class TaskStep:
    id: Optional[int]
    task_id: int
    step_name: StepName
    status: StepStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int

@dataclass
class TransferChunk:
    id: Optional[int]
    task_id: int
    step_name: str
    chunk_id: str
    file_path: str
    offset: int
    size: int
    status: ChunkStatus
    retry_count: int
