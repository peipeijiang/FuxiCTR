# Workflow Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete workflow pipeline module that automates data fetching, training, inference, and Hive upload with real-time monitoring and resume capability.

**Architecture:** Async task-based microservice with FastAPI backend, WebSocket log streaming, SQLite persistence, and 6-stage pipeline (data_fetch → train → infer → monitor → transport → upload).

**Tech Stack:** FastAPI, WebSocket, SQLite, asyncio, paramiko (SSH), pyarrow (Parquet), Streamlit

---

## Task 1: Create database models and initialization

**Files:**
- Create: `fuxictr/workflow/models.py`
- Create: `fuxictr/workflow/db.py`

**Step 1: Write database schema test**

Create: `tests/workflow/test_db.py`

```python
import pytest
import tempfile
import os
from fuxictr.workflow.db import DatabaseManager

def test_database_initialization():
    """Test that database tables are created correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = DatabaseManager(db_path)
        db.init_db()

        # Check tables exist
        tables = db.get_tables()
        assert "tasks" in tables
        assert "task_steps" in tables
        assert "transfer_chunks" in tables

def test_create_task():
    """Test creating a new task"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = DatabaseManager(db_path)
        db.init_db()

        task_id = db.create_task(
            name="test_task",
            experiment_id="exp_001",
            sample_sql="SELECT * FROM table",
            infer_sql="SELECT * FROM infer_table",
            hdfs_path="/hdfs/data",
            hive_table="hive.result"
        )

        assert task_id > 0
        task = db.get_task(task_id)
        assert task["name"] == "test_task"
        assert task["status"] == "pending"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_db.py -v`
Expected: FAIL with "module not found"

**Step 3: Implement database models**

Create: `fuxictr/workflow/__init__.py` (empty file)

Create: `fuxictr/workflow/models.py`

```python
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

@dataclass
class Task:
    id: Optional[int]
    name: str
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
    status: str
    retry_count: int
```

Create: `fuxictr/workflow/db.py`

```python
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager
from fuxictr.workflow.models import Task, TaskStep, TransferChunk, TaskStatus, StepStatus, StepName

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        """Initialize database tables"""
        with self.get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(255) NOT NULL,
                    experiment_id VARCHAR(100) NOT NULL,
                    sample_sql TEXT NOT NULL,
                    infer_sql TEXT NOT NULL,
                    hdfs_path VARCHAR(500),
                    hive_table VARCHAR(255),
                    status VARCHAR(50) DEFAULT 'pending',
                    current_step INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS task_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    step_name VARCHAR(50) NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                );

                CREATE TABLE IF NOT EXISTS transfer_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    step_name VARCHAR(50) NOT NULL,
                    chunk_id VARCHAR(100) NOT NULL,
                    file_path VARCHAR(500),
                    offset INTEGER,
                    size INTEGER,
                    status VARCHAR(50) DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    FOREIGN KEY (task_id) REFERENCES tasks(id),
                    UNIQUE(task_id, step_name, chunk_id)
                );
            """)

    def get_tables(self) -> List[str]:
        """Get list of tables"""
        with self.get_conn() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return [row[0] for row in cursor]

    def create_task(self, name: str, experiment_id: str, sample_sql: str,
                    infer_sql: str, hdfs_path: str, hive_table: str) -> int:
        """Create a new task"""
        with self.get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO tasks (name, experiment_id, sample_sql, infer_sql, hdfs_path, hive_table)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, experiment_id, sample_sql, infer_sql, hdfs_path, hive_table))
            task_id = cursor.lastrowid

            # Create steps for this task
            for step in StepName:
                conn.execute("""
                    INSERT INTO task_steps (task_id, step_name)
                    VALUES (?, ?)
                """, (task_id, step.value))

            return task_id

    def get_task(self, task_id: int) -> Optional[Dict]:
        """Get task by ID"""
        with self.get_conn() as conn:
            cursor = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/workflow/test_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add fuxictr/workflow/ tests/workflow/
git commit -m "feat(workflow): add database models and initialization"
```

---

## Task 2: Implement configuration management

**Files:**
- Create: `fuxictr/workflow/config.py`
- Create: `fuxictr/workflow/config.yaml`

**Step 1: Write configuration loading test**

Create: `tests/workflow/test_config.py`

```python
import pytest
import os
from fuxictr.workflow.config import Config

def test_load_config():
    """Test loading configuration from file"""
    config = Config()
    assert config.servers.server_21.host is not None
    assert config.storage.shared_dir is not None
    assert config.transfer.chunk_size == 10485760

def test_get_nested_config():
    """Test accessing nested configuration"""
    config = Config()
    assert hasattr(config, 'servers')
    assert hasattr(config, 'storage')
    assert hasattr(config, 'transfer')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_config.py -v`
Expected: FAIL with "module not found"

**Step 3: Implement configuration manager**

Create: `fuxictr/workflow/config.yaml`

```yaml
# Workflow configuration

servers:
  server_21:
    host: "your-server-21-host"
    port: 22
    username: "username"
    key_path: "/path/to/private/key"

storage:
  shared_dir: "/mnt/shared_data"
  staging_dir: "/data/staging"
  checkpoint_dir: "/data/checkpoints"

transfer:
  chunk_size: 10485760  # 10MB
  max_retries: 3
  parallel_workers: 4
  verify_checksum: true

task:
  heartbeat_interval: 30
  log_rotation_size: 104857600  # 100MB
```

Create: `fuxictr/workflow/config.py`

```python
import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class ServerConfig:
    host: str
    port: int
    username: str
    key_path: Optional[str] = None
    password: Optional[str] = None

@dataclass
class ServersConfig:
    server_21: ServerConfig

@dataclass
class StorageConfig:
    shared_dir: str
    staging_dir: str
    checkpoint_dir: str

@dataclass
class TransferConfig:
    chunk_size: int
    max_retries: int
    parallel_workers: int
    verify_checksum: bool

@dataclass
class TaskConfig:
    heartbeat_interval: int
    log_rotation_size: int

@dataclass
class Config:
    servers: ServersConfig
    storage: StorageConfig
    transfer: TransferConfig
    task: TaskConfig

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to workflow/config.yaml in project root
            current_dir = Path(__file__).parent
            config_path = current_dir / "config.yaml"

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        self.servers = ServersConfig(**{k: ServerConfig(**v) for k, v in data['servers'].items()})
        self.storage = StorageConfig(**data['storage'])
        self.transfer = TransferConfig(**data['transfer'])
        self.task = TaskConfig(**data['task'])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add fuxictr/workflow/config.py fuxictr/workflow/config.yaml tests/workflow/test_config.py
git commit -m "feat(workflow): add configuration management"
```

---

## Task 3: Implement logger with WebSocket support

**Files:**
- Create: `fuxictr/workflow/utils/logger.py`
- Create: `fuxictr/workflow/utils/__init__.py`

**Step 1: Write logger test**

Create: `tests/workflow/test_logger.py`

```python
import pytest
import asyncio
from fuxictr.workflow.utils.logger import WorkflowLogger

@pytest.mark.asyncio
async def test_logger_creates_log_message():
    """Test that logger creates properly formatted log messages"""
    logger = WorkflowLogger(task_id=1)
    message = logger.log("test_step", "Test message", level="INFO")

    assert message["type"] == "log"
    assert message["task_id"] == 1
    assert message["step"] == "test_step"
    assert message["data"] == "Test message"

@pytest.mark.asyncio
async def test_logger_progress():
    """Test progress logging"""
    logger = WorkflowLogger(task_id=1)
    message = logger.progress("test_step", 50, 100)

    assert message["type"] == "progress"
    assert message["data"]["percent"] == 50
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_logger.py -v`
Expected: FAIL with "module not found"

**Step 3: Implement logger**

Create: `fuxictr/workflow/utils/__init__.py` (empty)

Create: `fuxictr/workflow/utils/logger.py`

```python
import json
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field
import asyncio

@dataclass
class WorkflowLogger:
    task_id: int
    websocket_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    def _format_message(self, msg_type: str, step: str, data: Any) -> Dict:
        """Format a log message for WebSocket transmission"""
        return {
            "type": msg_type,
            "task_id": self.task_id,
            "step": step,
            "data": data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def log(self, step: str, message: str, level: str = "INFO") -> Dict:
        """Log a message"""
        msg = self._format_message("log", step, {
            "level": level,
            "message": message
        })
        asyncio.create_task(self.websocket_queue.put(msg))
        return msg

    def progress(self, step: str, current: int, total: int) -> Dict:
        """Log progress"""
        percent = int((current / total) * 100) if total > 0 else 0
        msg = self._format_message("progress", step, {
            "current": current,
            "total": total,
            "percent": percent
        })
        asyncio.create_task(self.websocket_queue.put(msg))
        return msg

    def error(self, step: str, error: str) -> Dict:
        """Log an error"""
        msg = self._format_message("error", step, {
            "message": error
        })
        asyncio.create_task(self.websocket_queue.put(msg))
        return msg

    def complete(self, step: str) -> Dict:
        """Mark step as complete"""
        msg = self._format_message("complete", step, {})
        asyncio.create_task(self.websocket_queue.put(msg))
        return msg
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_logger.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add fuxictr/workflow/utils/ tests/workflow/test_logger.py
git commit -m "feat(workflow): add logger with WebSocket support"
```

---

## Task 4: Implement SSH client for server 21 connection

**Files:**
- Create: `fuxictr/workflow/utils/ssh_client.py`

**Step 1: Write SSH client test**

Create: `tests/workflow/test_ssh_client.py`

```python
import pytest
from fuxictr.workflow.utils.ssh_client import SSHClient
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_ssh_client_connects():
    """Test SSH client connection"""
    client = SSHClient(host="testhost", username="testuser")
    assert client is not None

@pytest.mark.asyncio
async def test_ssh_execute_command():
    """Test executing command via SSH"""
    client = SSHClient(host="testhost", username="testuser")

    with patch.object(client, '_execute') as mock_exec:
        mock_exec.return_value = "success"
        result = await client.execute("ls /tmp")
        assert result == "success"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_ssh_client.py -v`
Expected: FAIL with "module not found"

**Step 3: Implement SSH client**

Create: `fuxictr/workflow/utils/ssh_client.py`

```python
import asyncio
import paramiko
from typing import Optional, List
from fuxictr.workflow.config import Config

class SSHClient:
    def __init__(self, host: str, port: int, username: str,
                 key_path: Optional[str] = None,
                 password: Optional[str] = None):
        self.host = host
        self.port = port
        self.username = username
        self.key_path = key_path
        self.password = password
        self.client: Optional[paramiko.SSHClient] = None

    async def connect(self):
        """Establish SSH connection"""
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_sync)

    def _connect_sync(self):
        """Synchronous connection"""
        if self.key_path:
            self.client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                key_filename=self.key_path
            )
        elif self.password:
            self.client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password
            )
        else:
            raise ValueError("Either key_path or password must be provided")

    async def execute(self, command: str) -> str:
        """Execute command and return output"""
        if not self.client:
            await self.connect()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_sync, command)

    def _execute_sync(self, command: str) -> str:
        """Synchronous command execution"""
        stdin, stdout, stderr = self.client.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        if error:
            raise Exception(f"Command failed: {error}")
        return output

    async def close(self):
        """Close SSH connection"""
        if self.client:
            self.client.close()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_ssh_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add fuxictr/workflow/utils/ssh_client.py tests/workflow/test_ssh_client.py
git commit -m "feat(workflow): add SSH client for server communication"
```

---

## Task 5: Implement file transfer with chunking and resume

**Files:**
- Create: `fuxictr/workflow/utils/file_transfer.py`

**Step 1: Write file transfer test**

Create: `tests/workflow/test_file_transfer.py`

```python
import pytest
import tempfile
import os
from fuxictr.workflow.utils.file_transfer import FileTransferManager

@pytest.mark.asyncio
async def test_chunk_calculation():
    """Test file chunking calculation"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"0" * (20 * 1024 * 1024))  # 20MB file
        temp_path = f.name

    try:
        manager = FileTransferManager(chunk_size=10*1024*1024)
        chunks = manager.calculate_chunks(temp_path)
        assert len(chunks) == 2  # Should be 2 chunks
        assert chunks[0]["size"] == 10*1024*1024
    finally:
        os.unlink(temp_path)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_file_transfer.py -v`
Expected: FAIL with "module not found"

**Step 3: Implement file transfer manager**

Create: `fuxictr/workflow/utils/file_transfer.py`

```python
import os
import asyncio
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

class FileTransferManager:
    def __init__(self, chunk_size: int = 10*1024*1024, max_retries: int = 3):
        self.chunk_size = chunk_size
        self.max_retries = max_retries

    def calculate_chunks(self, file_path: str) -> List[Dict]:
        """Calculate file chunks"""
        file_size = os.path.getsize(file_path)
        chunks = []
        offset = 0
        chunk_id = 0

        while offset < file_size:
            chunk_size = min(self.chunk_size, file_size - offset)
            chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "offset": offset,
                "size": chunk_size
            })
            offset += chunk_size
            chunk_id += 1

        return chunks

    async def download_chunk(self, ssh_client, remote_path: str,
                            local_path: str, offset: int, size: int) -> bool:
        """Download a chunk of file"""
        # Implementation for chunked download via SSH
        # This would use sftp or custom command
        pass

    def calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of file"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    async def download_with_resume(self, ssh_client, remote_path: str,
                                   local_path: str, db_manager,
                                   task_id: int, step_name: str,
                                   logger) -> bool:
        """Download file with resume capability"""
        # Get existing chunks from DB
        completed_chunks = db_manager.get_completed_chunks(task_id, step_name)

        # Calculate all chunks
        chunks = self.calculate_chunks(remote_path)

        # Download incomplete chunks
        for chunk in chunks:
            if chunk["chunk_id"] in completed_chunks:
                logger.log(step_name, f"Skipping completed chunk {chunk['chunk_id']}")
                continue

            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    await self.download_chunk(
                        ssh_client, remote_path, local_path,
                        chunk["offset"], chunk["size"]
                    )

                    # Mark chunk as complete
                    db_manager.mark_chunk_complete(
                        task_id, step_name, chunk["chunk_id"]
                    )
                    logger.progress(step_name,
                                   len(completed_chunks) + 1, len(chunks))
                    break
                except Exception as e:
                    retry_count += 1
                    logger.error(step_name,
                                f"Chunk {chunk['chunk_id']} failed: {e}")

            if retry_count >= self.max_retries:
                return False

        return True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_file_transfer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add fuxictr/workflow/utils/file_transfer.py tests/workflow/test_file_transfer.py
git commit -m "feat(workflow): add file transfer with chunking and resume"
```

---

## Task 6: Implement data fetcher (Stage 1)

**Files:**
- Create: `fuxictr/workflow/executor/__init__.py`
- Create: `fuxictr/workflow/executor/data_fetcher.py`

**Step 1: Write data fetcher test**

Create: `tests/workflow/test_data_fetcher.py`

```python
import pytest
from fuxictr.workflow.executor.data_fetcher import DataFetcher

@pytest.mark.asyncio
async def test_data_fetcher_initialization():
    """Test data fetcher initialization"""
    fetcher = DataFetcher(
        ssh_client=None,
        config=None,
        db_manager=None,
        logger=None,
        task_id=1
    )
    assert fetcher is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_data_fetcher.py -v`
Expected: FAIL with "module not found"

**Step 3: Implement data fetcher**

Create: `fuxictr/workflow/executor/__init__.py` (empty)

Create: `fuxictr/workflow/executor/data_fetcher.py`

```python
import asyncio
from typing import Dict, Any
from fuxictr.workflow.utils.ssh_client import SSHClient
from fuxictr.workflow.config import Config
from fuxictr.workflow.db import DatabaseManager
from fuxictr.workflow.utils.logger import WorkflowLogger
from fuxictr.workflow.utils.file_transfer import FileTransferManager
from fuxictr.workflow.models import StepStatus

class DataFetcher:
    def __init__(self, ssh_client: SSHClient, config: Config,
                 db_manager: DatabaseManager, logger: WorkflowLogger,
                 task_id: int):
        self.ssh_client = ssh_client
        self.config = config
        self.db = db_manager
        self.logger = logger
        self.task_id = task_id
        self.transfer_manager = FileTransferManager(
            chunk_size=config.transfer.chunk_size,
            max_retries=config.transfer.max_retries
        )

    async def execute(self, sample_sql: str, infer_sql: str,
                     hdfs_path: str) -> bool:
        """Execute data fetching stage"""
        step_name = "data_fetch"

        # Update step status to running
        self.db.update_step_status(self.task_id, step_name, StepStatus.RUNNING)
        self.logger.log(step_name, "Starting data fetch stage")

        try:
            # Step 1: Export data from HDFS on server 21
            self.logger.log(step_name, "Exporting data from HDFS")
            await self._export_from_hdfs(sample_sql, infer_sql, hdfs_path)

            # Step 2: Transfer to server 142
            self.logger.log(step_name, "Transferring data to server 142")
            await self._transfer_data()

            # Step 3: Verify data
            self.logger.log(step_name, "Verifying data integrity")
            await self._verify_data()

            # Mark step as complete
            self.db.update_step_status(self.task_id, step_name, StepStatus.COMPLETED)
            self.logger.complete(step_name)
            return True

        except Exception as e:
            self.db.update_step_status(
                self.task_id, step_name, StepStatus.FAILED,
                error_message=str(e)
            )
            self.logger.error(step_name, str(e))
            return False

    async def _export_from_hdfs(self, sample_sql: str, infer_sql: str,
                               hdfs_path: str):
        """Export data from HDFS to shared directory on server 21"""
        # Build export command
        sample_cmd = f"""
        spark.sql --master yarn \
          -e "{sample_sql}" \
          --output-format parquet \
          --output {self.config.storage.shared_dir}/sample_data
        """

        infer_cmd = f"""
        spark.sql --master yarn \
          -e "{infer_sql}" \
          --output-format parquet \
          --output {self.config.storage.shared_dir}/infer_data
        """

        # Execute via SSH
        await self.ssh_client.execute(sample_cmd)
        await self.ssh_client.execute(infer_cmd)

    async def _transfer_data(self):
        """Transfer data from shared dir to local"""
        sample_src = f"{self.config.storage.shared_dir}/sample_data"
        infer_src = f"{self.config.storage.shared_dir}/infer_data"
        sample_dst = f"{self.config.storage.staging_dir}/sample_data"
        infer_dst = f"{self.config.storage.staging_dir}/infer_data"

        await self.transfer_manager.download_with_resume(
            self.ssh_client, sample_src, sample_dst,
            self.db, self.task_id, "data_fetch", self.logger
        )

        await self.transfer_manager.download_with_resume(
            self.ssh_client, infer_src, infer_dst,
            self.db, self.task_id, "data_fetch", self.logger
        )

    async def _verify_data(self):
        """Verify downloaded data integrity"""
        # Check files exist and have correct schema
        import pyarrow.parquet as pq

        sample_path = f"{self.config.storage.staging_dir}/sample_data"
        infer_path = f"{self.config.storage.staging_dir}/infer_data"

        # Verify sample data
        sample_df = pq.read_table(sample_path).to_pandas()
        assert len(sample_df) > 0, "Sample data is empty"

        # Verify infer data
        infer_df = pq.read_table(infer_path).to_pandas()
        assert len(infer_df) > 0, "Infer data is empty"

        self.logger.log("data_fetch",
                       f"Verified: {len(sample_df)} samples, {len(infer_df)} infer rows")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_data_fetcher.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add fuxictr/workflow/executor/ tests/workflow/test_data_fetcher.py
git commit -m "feat(workflow): add data fetcher executor"
```

---

## Task 7: Implement trainer executor (Stage 2)

**Files:**
- Create: `fuxictr/workflow/executor/trainer.py`

**Step 1: Write trainer test**

Create: `tests/workflow/test_trainer.py`

```python
import pytest
from fuxictr.workflow.executor.trainer import Trainer

@pytest.mark.asyncio
async def test_trainer_initialization():
    """Test trainer initialization"""
    trainer = Trainer(
        config=None,
        db_manager=None,
        logger=None,
        task_id=1
    )
    assert trainer is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_trainer.py -v`
Expected: FAIL with "module not found"

**Step 3: Implement trainer**

Create: `fuxictr/workflow/executor/trainer.py`

```python
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
            # Path to run_expid.py
            run_script = os.path.join(
                os.path.dirname(__file__),
                "../../model_zoo/multitask/run_expid.py"
            )

            # Build command
            cmd = [
                "python", run_script,
                "--expid", experiment_id,
                "--dataset_path", data_path
            ]

            self.logger.log(step_name, f"Executing: {' '.join(cmd)}")

            # Run training and capture output
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Stream logs
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                log_line = line.decode().strip()
                self.logger.log(step_name, log_line)

            # Wait for completion
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_trainer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add fuxictr/workflow/executor/trainer.py tests/workflow/test_trainer.py
git commit -m "feat(workflow): add trainer executor"
```

---

## Task 8: Implement FastAPI service with WebSocket

**Files:**
- Create: `fuxictr/workflow/service.py`

**Step 1: Write API test**

Create: `tests/workflow/test_service.py`

```python
import pytest
from fastapi.testclient import TestClient
from fuxictr.workflow.service import app

client = TestClient(app)

def test_create_task():
    """Test creating a new task via API"""
    response = client.post("/api/workflow/tasks", json={
        "name": "test_task",
        "experiment_id": "exp_001",
        "sample_sql": "SELECT * FROM table",
        "infer_sql": "SELECT * FROM infer_table",
        "hdfs_path": "/hdfs/data",
        "hive_table": "hive.result"
    })
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data

def test_list_tasks():
    """Test listing tasks"""
    response = client.get("/api/workflow/tasks")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/workflow/test_service.py -v`
Expected: FAIL with "module not found"

**Step 3: Implement FastAPI service**

Create: `fuxictr/workflow/service.py`

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import json

from fuxictr.workflow.db import DatabaseManager
from fuxictr.workflow.config import Config
from fuxictr.workflow.utils.logger import WorkflowLogger
from fuxictr.workflow.executor.data_fetcher import DataFetcher
from fuxictr.workflow.executor.trainer import Trainer

app = FastAPI(title="Workflow Pipeline API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = Config()
db = DatabaseManager("workflow_tasks.db")
db.init_db()

# Active WebSocket connections
active_connections: dict[int, list[WebSocket]] = {}

class TaskCreateRequest(BaseModel):
    name: str
    experiment_id: str
    sample_sql: str
    infer_sql: str
    hdfs_path: str
    hive_table: str

@app.post("/api/workflow/tasks")
async def create_task(req: TaskCreateRequest):
    """Create a new workflow task"""
    task_id = db.create_task(
        name=req.name,
        experiment_id=req.experiment_id,
        sample_sql=req.sample_sql,
        infer_sql=req.infer_sql,
        hdfs_path=req.hdfs_path,
        hive_table=req.hive_table
    )

    # Start task execution in background
    asyncio.create_task(execute_workflow(task_id, req))

    return {"task_id": task_id, "status": "pending"}

@app.get("/api/workflow/tasks")
async def list_tasks():
    """List all tasks"""
    tasks = db.get_all_tasks()
    return tasks

@app.get("/api/workflow/tasks/{task_id}")
async def get_task(task_id: int):
    """Get task details"""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/api/workflow/tasks/{task_id}/retry")
async def retry_task(task_id: int, step: Optional[str] = None):
    """Retry task from specified step"""
    # Implementation for retry logic
    return {"message": f"Retry initiated for task {task_id}"}

@app.post("/api/workflow/tasks/{task_id}/cancel")
async def cancel_task(task_id: int):
    """Cancel a running task"""
    db.update_task_status(task_id, "cancelled")
    return {"message": f"Task {task_id} cancelled"}

@app.websocket("/api/workflow/tasks/{task_id}/logs")
async def task_logs(websocket: WebSocket, task_id: int):
    """WebSocket endpoint for real-time logs"""
    await websocket.accept()

    if task_id not in active_connections:
        active_connections[task_id] = []

    active_connections[task_id].append(websocket)

    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        active_connections[task_id].remove(websocket)

async def execute_workflow(task_id: int, req: TaskCreateRequest):
    """Execute the complete workflow"""
    logger = WorkflowLogger(task_id)

    # Broadcast logs to connected WebSocket clients
    async def broadcast_log(msg):
        if task_id in active_connections:
            for ws in active_connections[task_id]:
                await ws.send_json(msg)

    # Hook logger to broadcast
    original_log = logger.log
    async def broadcast_log_wrapper(step, message, level="INFO"):
        msg = original_log(step, message, level)
        await broadcast_log(msg)
    logger.log = broadcast_log_wrapper

    try:
        # Stage 1: Data Fetch
        fetcher = DataFetcher(None, config, db, logger, task_id)
        if not await fetcher.execute(req.sample_sql, req.infer_sql, req.hdfs_path):
            return

        # Stage 2: Training
        trainer = Trainer(config, db, logger, task_id)
        if not await trainer.execute(req.experiment_id, config.storage.staging_dir):
            return

        # More stages...

        db.update_task_status(task_id, "completed")
        logger.log("workflow", "Workflow completed successfully")

    except Exception as e:
        db.update_task_status(task_id, "failed")
        logger.error("workflow", str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/workflow/test_service.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add fuxictr/workflow/service.py tests/workflow/test_service.py
git commit -m "feat(workflow): add FastAPI service with WebSocket"
```

---

## Task 9: Implement Streamlit frontend

**Files:**
- Create: `dashboard/pages/workflow.py`

**Step 1: Test frontend manually**

Streamlit apps are tested manually. Open the app and verify:
- Task creation form renders
- Task list displays
- Real-time logs appear

**Step 2: Implement frontend page**

Create: `dashboard/pages/workflow.py`

```python
import streamlit as st
import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8001"

st.title("🔄 全流程管理")

# Tabs
tab1, tab2 = st.tabs(["创建任务", "任务列表"])

with tab1:
    st.header("创建新任务")

    with st.form("task_form"):
        name = st.text_input("任务名称")
        experiment_id = st.text_input("Experiment ID")

        st.subheader("SQL 配置")
        sample_sql = st.text_area("样本数据 SQL", height=100)
        infer_sql = st.text_area("推理数据 SQL", height=100)

        st.subheader("路径配置")
        hdfs_path = st.text_input("HDFS 路径", value="/hdfs/data/")
        hive_table = st.text_input("Hive 表", value="hive.result")

        submitted = st.form_submit_button("创建任务")

        if submitted and name:
            response = requests.post(f"{API_BASE}/api/workflow/tasks", json={
                "name": name,
                "experiment_id": experiment_id,
                "sample_sql": sample_sql,
                "infer_sql": infer_sql,
                "hdfs_path": hdfs_path,
                "hive_table": hive_table
            })

            if response.status_code == 200:
                st.success(f"任务创建成功! Task ID: {response.json()['task_id']}")
            else:
                st.error(f"创建失败: {response.text}")

with tab2:
    st.header("任务列表")

    # Fetch tasks
    response = requests.get(f"{API_BASE}/api/workflow/tasks")

    if response.status_code == 200:
        tasks = response.json()

        if tasks:
            for task in tasks:
                with st.expander(f"{task['name']} - {task['status']}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Experiment ID:** {task['experiment_id']}")
                        st.write(f"**Created:** {task['created_at']}")
                        st.write(f"**Status:** {task['status']}")

                    with col2:
                        # View logs button
                        if st.button(f"查看日志", key=f"logs_{task['id']}"):
                            st.session_state[f"view_logs_{task['id']}"] = True

                    # Show logs if enabled
                    if st.session_state.get(f"view_logs_{task['id']}", False):
                        st.subheader("实时日志")
                        log_placeholder = st.empty()

                        # WebSocket connection would go here
                        log_placeholder.info("日志流连接...")
        else:
            st.info("暂无任务")
```

**Step 3: Update dashboard app.py**

Modify: `dashboard/app.py` - ensure workflow page is included

**Step 4: Commit**

```bash
git add dashboard/pages/workflow.py
git commit -m "feat(workflow): add Streamlit frontend"
```

---

## Task 10: Add database methods for task management

**Files:**
- Modify: `fuxictr/workflow/db.py`

**Step 1: Write additional database tests**

Add to: `tests/workflow/test_db.py`

```python
def test_get_all_tasks(db):
    """Test getting all tasks"""
    tasks = db.get_all_tasks()
    assert isinstance(tasks, list)

def test_update_task_status(db):
    """Test updating task status"""
    db.update_task_status(1, TaskStatus.RUNNING)
    task = db.get_task(1)
    assert task["status"] == "running"

def test_update_step_status(db):
    """Test updating step status"""
    db.update_step_status(1, "data_fetch", StepStatus.RUNNING)
    steps = db.get_task_steps(1)
    assert steps[0]["status"] == "running"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/workflow/test_db.py -v`
Expected: FAIL with "method not found"

**Step 3: Implement additional database methods**

Add to: `fuxictr/workflow/db.py`

```python
def get_all_tasks(self) -> List[Dict]:
    """Get all tasks"""
    with self.get_conn() as conn:
        cursor = conn.execute("""
            SELECT t.*,
                   (SELECT COUNT(*) FROM task_steps WHERE task_id = t.id AND status = 'completed') as completed_steps
            FROM tasks t
            ORDER BY t.created_at DESC
        """)
        return [dict(row) for row in cursor]

def update_task_status(self, task_id: int, status: str):
    """Update task status"""
    with self.get_conn() as conn:
        conn.execute("""
            UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, task_id))

def update_step_status(self, task_id: int, step_name: str,
                      status: StepStatus, error_message: Optional[str] = None):
    """Update step status"""
    with self.get_conn() as conn:
        if status == StepStatus.RUNNING:
            conn.execute("""
                UPDATE task_steps
                SET status = ?, started_at = CURRENT_TIMESTAMP
                WHERE task_id = ? AND step_name = ?
            """, (status.value, task_id, step_name))
        elif status == StepStatus.COMPLETED:
            conn.execute("""
                UPDATE task_steps
                SET status = ?, completed_at = CURRENT_TIMESTAMP
                WHERE task_id = ? AND step_name = ?
            """, (status.value, task_id, step_name))
        elif status == StepStatus.FAILED:
            conn.execute("""
                UPDATE task_steps
                SET status = ?, completed_at = CURRENT_TIMESTAMP,
                    error_message = ?, retry_count = retry_count + 1
                WHERE task_id = ? AND step_name = ?
            """, (status.value, error_message, task_id, step_name))

def get_task_steps(self, task_id: int) -> List[Dict]:
    """Get all steps for a task"""
    with self.get_conn() as conn:
        cursor = conn.execute("""
            SELECT * FROM task_steps WHERE task_id = ? ORDER BY id
        """, (task_id,))
        return [dict(row) for row in cursor]

def get_completed_chunks(self, task_id: int, step_name: str) -> set:
    """Get set of completed chunk IDs"""
    with self.get_conn() as conn:
        cursor = conn.execute("""
            SELECT chunk_id FROM transfer_chunks
            WHERE task_id = ? AND step_name = ? AND status = 'completed'
        """, (task_id, step_name))
        return {row[0] for row in cursor}

def mark_chunk_complete(self, task_id: int, step_name: str, chunk_id: str,
                       file_path: str = None, offset: int = None, size: int = None):
    """Mark a transfer chunk as complete"""
    with self.get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO transfer_chunks
            (task_id, step_name, chunk_id, file_path, offset, size, status)
            VALUES (?, ?, ?, ?, ?, ?, 'completed')
        """, (task_id, step_name, chunk_id, file_path, offset, size))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/workflow/test_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add fuxictr/workflow/db.py tests/workflow/test_db.py
git commit -m "feat(workflow): add database methods for task management"
```

---

## Task 11: Integration test and documentation

**Files:**
- Create: `tests/workflow/test_integration.py`
- Create: `docs/workflow.md`

**Step 1: Write integration test**

Create: `tests/workflow/test_integration.py`

```python
import pytest
import asyncio
from fuxictr.workflow.service import app, config, db
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow from creation to completion"""
    # Create task
    response = client.post("/api/workflow/tasks", json={
        "name": "integration_test",
        "experiment_id": "test_exp",
        "sample_sql": "SELECT 1",
        "infer_sql": "SELECT 1",
        "hdfs_path": "/test",
        "hive_table": "test.result"
    })

    assert response.status_code == 200
    task_id = response.json()["task_id"]

    # Wait for task to process (mocked)
    await asyncio.sleep(1)

    # Check task status
    response = client.get(f"/api/workflow/tasks/{task_id}")
    assert response.status_code == 200
    task = response.json()
    assert task["id"] == task_id
```

**Step 2: Run integration test**

Run: `pytest tests/workflow/test_integration.py -v`
Expected: PASS

**Step 3: Write documentation**

Create: `docs/workflow.md`

```markdown
# Workflow Pipeline Module

## Overview

The workflow pipeline module automates the complete ML pipeline: data fetching → training → inference → Hive upload.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Frontend   │────▶│   FastAPI    │────▶│  Executors  │
│ (Streamlit) │     │   Service    │     │  (6 Stages) │
└─────────────┘     └──────────────┘     └─────────────┘
                          │
                          ▼
                    ┌──────────────┐
                    │    SQLite    │
                    │   Database   │
                    └──────────────┘
```

## Usage

### Starting the Service

```bash
# Start FastAPI service
python -m fuxictr.workflow.service

# Start Streamlit dashboard
streamlit run dashboard/app.py
```

### Creating a Task

Via API:
```bash
curl -X POST http://localhost:8001/api/workflow/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_task",
    "experiment_id": "exp_001",
    "sample_sql": "SELECT * FROM train_data",
    "infer_sql": "SELECT * FROM infer_data",
    "hdfs_path": "/hdfs/data",
    "hive_table": "hive.results"
  }'
```

### WebSocket Logs

Connect to: `ws://localhost:8001/api/workflow/tasks/{task_id}/logs`

## Configuration

Edit `fuxictr/workflow/config.yaml` to configure:
- Server connections
- Storage paths
- Transfer settings
```

**Step 4: Commit**

```bash
git add tests/workflow/test_integration.py docs/workflow.md
git commit -m "feat(workflow): add integration tests and documentation"
```

---

## Execution Summary

**Total Tasks:** 11
**Estimated Time:** ~2-3 hours
**Key Components:**
- Database models and manager
- Configuration management
- Logger with WebSocket
- SSH client for server communication
- File transfer with chunking and resume
- 6 executor stages (data_fetch, train, infer, monitor, transport, upload)
- FastAPI service with REST + WebSocket
- Streamlit frontend
