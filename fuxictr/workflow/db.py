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
Database Manager for workflow orchestration.

Supports:
- Task and step tracking
- Checkpoint data persistence
- Transfer chunk tracking for resume capability
- Server assignment management
"""

import sqlite3
import json
from typing import List, Dict, Optional, Iterator, Set, Any
from contextlib import contextmanager
from datetime import datetime

from fuxictr.workflow.models import (
    Task, TaskStep, TransferChunk, TaskStatus, StepStatus,
    StepName, ChunkStatus, TrainingCheckpoint, InferenceCheckpoint,
    TransferCheckpoint
)


class DatabaseManager:
    """Manages SQLite database for workflow state persistence."""

    def __init__(self, db_path: str):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

    @contextmanager
    def get_conn(self) -> Iterator[sqlite3.Connection]:
        """
        Get database connection with context management.

        Automatically commits on success, rolls back on error.
        """
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
        """Initialize database tables with updated schema."""
        with self.get_conn() as conn:
            conn.executescript("""
                -- Tasks table with server assignment
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(255) NOT NULL,
                    user VARCHAR(100) NOT NULL,
                    model VARCHAR(255) NOT NULL,
                    experiment_id VARCHAR(100) NOT NULL,
                    sample_sql TEXT NOT NULL,
                    infer_sql TEXT NOT NULL,
                    hdfs_path VARCHAR(500),
                    hive_table VARCHAR(255),

                    -- Task state
                    status VARCHAR(50) DEFAULT 'pending',
                    current_step INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 5,

                    -- Server assignment
                    training_server VARCHAR(100),
                    inference_server VARCHAR(100),

                    -- Additional config (JSON)
                    config TEXT,

                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                );

                -- Task steps with checkpoint support
                CREATE TABLE IF NOT EXISTS task_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    step_name VARCHAR(50) NOT NULL,
                    step_order INTEGER NOT NULL,

                    -- Step state
                    status VARCHAR(50) DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,

                    -- Checkpoint data (JSON)
                    checkpoint_data TEXT,

                    FOREIGN KEY (task_id) REFERENCES tasks(id),
                    UNIQUE(task_id, step_name)
                );

                -- Transfer chunks with resume support
                CREATE TABLE IF NOT EXISTS transfer_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    step_name VARCHAR(50) NOT NULL,
                    chunk_id VARCHAR(100) NOT NULL,

                    -- File info
                    source_path VARCHAR(500),
                    dest_path VARCHAR(500),
                    offset INTEGER DEFAULT 0,
                    size INTEGER DEFAULT 0,
                    checksum VARCHAR(64),

                    -- Transfer state
                    status VARCHAR(50) DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,

                    FOREIGN KEY (task_id) REFERENCES tasks(id),
                    UNIQUE(task_id, step_name, chunk_id)
                );

                -- Metrics table for tracking performance
                CREATE TABLE IF NOT EXISTS workflow_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    step_name VARCHAR(50) NOT NULL,

                    -- Timing metrics
                    duration_seconds REAL DEFAULT 0.0,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,

                    -- Transfer metrics
                    bytes_transferred INTEGER DEFAULT 0,
                    files_transferred INTEGER DEFAULT 0,
                    transfer_rate_mbps REAL DEFAULT 0.0,

                    -- Training metrics
                    epochs_completed INTEGER DEFAULT 0,
                    best_auc REAL DEFAULT 0.0,
                    best_loss REAL DEFAULT 0.0,
                    final_loss REAL DEFAULT 0.0,

                    -- Inference metrics
                    rows_processed INTEGER DEFAULT 0,
                    inference_throughput REAL DEFAULT 0.0,

                    -- Additional metrics (JSON)
                    additional_metrics TEXT,

                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_task_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_task_user ON tasks(user);
                CREATE INDEX IF NOT EXISTS idx_step_status ON task_steps(task_id, status);
                CREATE INDEX IF NOT EXISTS idx_transfer_status ON transfer_chunks(task_id, step_name, status);
                CREATE INDEX IF NOT EXISTS idx_metrics_task ON workflow_metrics(task_id);
            """)

    def get_tables(self) -> List[str]:
        """Get list of tables in database."""
        with self.get_conn() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return [row[0] for row in cursor]

    # ========================================================================
    # Task Management
    # ========================================================================

    def create_task(
        self,
        name: str,
        user: str,
        model: str,
        experiment_id: str,
        sample_sql: str,
        infer_sql: str,
        hdfs_path: str,
        hive_table: str,
        training_server: Optional[str] = None,
        inference_server: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Create a new workflow task.

        Args:
            name: Task name
            user: Username who created the task
            model: Model name (e.g., "MMoE", "PLE")
            experiment_id: Experiment configuration ID
            sample_sql: SQL for sample data extraction
            infer_sql: SQL for inference data extraction
            hdfs_path: HDFS path for data
            hive_table: Destination Hive table
            training_server: Optional training server assignment
            inference_server: Optional inference server assignment
            config: Additional configuration as dictionary

        Returns:
            Created task ID
        """
        with self.get_conn() as conn:
            config_json = json.dumps(config) if config else None
            cursor = conn.execute("""
                INSERT INTO tasks (
                    name, user, model, experiment_id, sample_sql, infer_sql,
                    hdfs_path, hive_table, training_server, inference_server, config
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, user, model, experiment_id, sample_sql, infer_sql,
                hdfs_path, hive_table, training_server, inference_server, config_json
            ))
            task_id = cursor.lastrowid

            # Create steps for this task in order
            for i, step in enumerate(StepName):
                conn.execute("""
                    INSERT INTO task_steps (task_id, step_name, step_order)
                    VALUES (?, ?, ?)
                """, (task_id, step.value, i))

            return task_id

    def get_task(self, task_id: int) -> Optional[Dict]:
        """Get task by ID."""
        with self.get_conn() as conn:
            cursor = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                task = dict(row)
                # Parse config JSON
                if task.get("config"):
                    try:
                        task["config"] = json.loads(task["config"])
                    except json.JSONDecodeError:
                        task["config"] = {}
                return task
            return None

    def get_all_tasks(
        self,
        user: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get all tasks with optional filtering.

        Args:
            user: Filter by username
            status: Filter by status
            limit: Maximum number of tasks to return

        Returns:
            List of task dictionaries
        """
        with self.get_conn() as conn:
            query = "SELECT * FROM tasks WHERE 1=1"
            params = []

            if user:
                query += " AND user = ?"
                params.append(user)
            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            tasks = []
            for row in cursor.fetchall():
                task = dict(row)
                # Parse config JSON
                if task.get("config"):
                    try:
                        task["config"] = json.loads(task["config"])
                    except json.JSONDecodeError:
                        task["config"] = {}
                tasks.append(task)
            return tasks

    def update_task_status(
        self,
        task_id: int,
        status: str,
        current_step: Optional[int] = None
    ) -> bool:
        """
        Update task status.

        Args:
            task_id: Task ID
            status: New status
            current_step: Optional current step number

        Returns:
            True if update successful
        """
        with self.get_conn() as conn:
            if status == TaskStatus.RUNNING.value:
                # Set started_at on first run
                cursor = conn.execute("""
                    UPDATE tasks
                    SET status = ?,
                        current_step = COALESCE(?, current_step),
                        started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, current_step, task_id))
            elif status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value]:
                cursor = conn.execute("""
                    UPDATE tasks
                    SET status = ?,
                        current_step = COALESCE(?, current_step),
                        completed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, current_step, task_id))
            else:
                cursor = conn.execute("""
                    UPDATE tasks
                    SET status = ?,
                        current_step = COALESCE(?, current_step),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, current_step, task_id))
            return cursor.rowcount > 0

    def update_task_servers(
        self,
        task_id: int,
        training_server: Optional[str] = None,
        inference_server: Optional[str] = None
    ) -> bool:
        """Update server assignments for task."""
        with self.get_conn() as conn:
            cursor = conn.execute("""
                UPDATE tasks
                SET training_server = COALESCE(?, training_server),
                    inference_server = COALESCE(?, inference_server),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (training_server, inference_server, task_id))
            return cursor.rowcount > 0

    def delete_task(self, task_id: int) -> bool:
        """
        Delete a task and all associated data.

        Args:
            task_id: Task ID to delete

        Returns:
            True if deletion successful
        """
        with self.get_conn() as conn:
            # Delete in order of dependencies
            conn.execute("DELETE FROM workflow_metrics WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM transfer_chunks WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM task_steps WHERE task_id = ?", (task_id,))
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            return cursor.rowcount > 0

    # ========================================================================
    # Step Management
    # ========================================================================

    def get_task_steps(self, task_id: int) -> List[Dict]:
        """Get all steps for a task, ordered by step_order."""
        with self.get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM task_steps
                WHERE task_id = ?
                ORDER BY step_order ASC
            """, (task_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_step(self, task_id: int, step_name: str) -> Optional[Dict]:
        """Get a specific step for a task."""
        with self.get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM task_steps
                WHERE task_id = ? AND step_name = ?
            """, (task_id, step_name))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_step_status(
        self,
        task_id: int,
        step_name: str,
        status: StepStatus,
        error_message: Optional[str] = None
    ):
        """
        Update step status.

        Args:
            task_id: Task ID
            step_name: Step name
            status: New status
            error_message: Optional error message on failure
        """
        with self.get_conn() as conn:
            if status == StepStatus.RUNNING:
                conn.execute("""
                    UPDATE task_steps
                    SET status = ?,
                        started_at = CURRENT_TIMESTAMP,
                        retry_count = retry_count
                    WHERE task_id = ? AND step_name = ?
                """, (status.value, task_id, step_name))
            elif status == StepStatus.COMPLETED:
                conn.execute("""
                    UPDATE task_steps
                    SET status = ?,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE task_id = ? AND step_name = ?
                """, (status.value, task_id, step_name))
            elif status == StepStatus.FAILED:
                conn.execute("""
                    UPDATE task_steps
                    SET status = ?,
                        completed_at = CURRENT_TIMESTAMP,
                        error_message = ?,
                        retry_count = retry_count + 1
                    WHERE task_id = ? AND step_name = ?
                """, (status.value, error_message, task_id, step_name))
            else:
                conn.execute("""
                    UPDATE task_steps
                    SET status = ?
                    WHERE task_id = ? AND step_name = ?
                """, (status.value, task_id, step_name))

    def save_checkpoint(
        self,
        task_id: int,
        step_name: str,
        checkpoint_data: Dict[str, Any]
    ) -> bool:
        """
        Save checkpoint data for a step.

        Args:
            task_id: Task ID
            step_name: Step name
            checkpoint_data: Checkpoint data as dictionary

        Returns:
            True if save successful
        """
        with self.get_conn() as conn:
            cursor = conn.execute("""
                UPDATE task_steps
                SET checkpoint_data = ?
                WHERE task_id = ? AND step_name = ?
            """, (json.dumps(checkpoint_data), task_id, step_name))
            return cursor.rowcount > 0

    def get_checkpoint(
        self,
        task_id: int,
        step_name: str
    ) -> Dict[str, Any]:
        """
        Get checkpoint data for a step.

        Args:
            task_id: Task ID
            step_name: Step name

        Returns:
            Checkpoint data as dictionary, empty if none exists
        """
        step = self.get_step(task_id, step_name)
        if step and step.get("checkpoint_data"):
            try:
                return json.loads(step["checkpoint_data"])
            except json.JSONDecodeError:
                return {}
        return {}

    # ========================================================================
    # Transfer Chunk Management
    # ========================================================================

    def create_transfer_chunks(
        self,
        task_id: int,
        step_name: str,
        chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Create transfer chunk records.

        Args:
            task_id: Task ID
            step_name: Step name
            chunks: List of chunk dictionaries with keys:
                    chunk_id, source_path, dest_path, offset, size, checksum

        Returns:
            Number of chunks created
        """
        with self.get_conn() as conn:
            count = 0
            for chunk in chunks:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO transfer_chunks
                        (task_id, step_name, chunk_id, source_path, dest_path,
                         offset, size, checksum, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                    """, (
                        task_id, step_name,
                        chunk["chunk_id"],
                        chunk.get("source_path"),
                        chunk.get("dest_path"),
                        chunk.get("offset", 0),
                        chunk.get("size", 0),
                        chunk.get("checksum")
                    ))
                    count += 1
                except sqlite3.IntegrityError:
                    # Chunk already exists, skip
                    pass
            return count

    def get_transfer_chunks(
        self,
        task_id: int,
        step_name: str,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        Get transfer chunks for a step.

        Args:
            task_id: Task ID
            step_name: Step name
            status: Optional status filter

        Returns:
            List of chunk dictionaries
        """
        with self.get_conn() as conn:
            if status:
                cursor = conn.execute("""
                    SELECT * FROM transfer_chunks
                    WHERE task_id = ? AND step_name = ? AND status = ?
                    ORDER BY chunk_id
                """, (task_id, step_name, status))
            else:
                cursor = conn.execute("""
                    SELECT * FROM transfer_chunks
                    WHERE task_id = ? AND step_name = ?
                    ORDER BY chunk_id
                """, (task_id, step_name))
            return [dict(row) for row in cursor.fetchall()]

    def get_completed_chunks(
        self,
        task_id: int,
        step_name: str
    ) -> Set[str]:
        """
        Get set of completed chunk IDs.

        Args:
            task_id: Task ID
            step_name: Step name

        Returns:
            Set of chunk_id strings that are completed
        """
        with self.get_conn() as conn:
            cursor = conn.execute("""
                SELECT chunk_id FROM transfer_chunks
                WHERE task_id = ? AND step_name = ? AND status = 'completed'
            """, (task_id, step_name))
            return {row[0] for row in cursor.fetchall()}

    def update_chunk_status(
        self,
        task_id: int,
        step_name: str,
        chunk_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update transfer chunk status.

        Args:
            task_id: Task ID
            step_name: Step name
            chunk_id: Chunk ID
            status: New status
            error_message: Optional error message

        Returns:
            True if update successful
        """
        with self.get_conn() as conn:
            if status == ChunkStatus.IN_PROGRESS.value:
                cursor = conn.execute("""
                    UPDATE transfer_chunks
                    SET status = ?,
                        started_at = CURRENT_TIMESTAMP
                    WHERE task_id = ? AND step_name = ? AND chunk_id = ?
                """, (status, task_id, step_name, chunk_id))
            elif status == ChunkStatus.COMPLETED.value:
                cursor = conn.execute("""
                    UPDATE transfer_chunks
                    SET status = ?,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE task_id = ? AND step_name = ? AND chunk_id = ?
                """, (status, task_id, step_name, chunk_id))
            elif status == ChunkStatus.FAILED.value:
                cursor = conn.execute("""
                    UPDATE transfer_chunks
                    SET status = ?,
                        error_message = ?,
                        retry_count = retry_count + 1
                    WHERE task_id = ? AND step_name = ? AND chunk_id = ?
                """, (status, error_message, task_id, step_name, chunk_id))
            else:
                cursor = conn.execute("""
                    UPDATE transfer_chunks
                    SET status = ?
                    WHERE task_id = ? AND step_name = ? AND chunk_id = ?
                """, (status, task_id, step_name, chunk_id))
            return cursor.rowcount > 0

    def mark_chunk_complete(
        self,
        task_id: int,
        step_name: str,
        chunk_id: str,
        source_path: Optional[str] = None,
        dest_path: Optional[str] = None,
        offset: Optional[int] = None,
        size: Optional[int] = None,
        checksum: Optional[str] = None
    ) -> bool:
        """
        Mark a transfer chunk as completed (upsert).

        Args:
            task_id: Task ID
            step_name: Step name
            chunk_id: Chunk ID
            source_path: Optional source path
            dest_path: Optional destination path
            offset: Optional offset
            size: Optional size
            checksum: Optional checksum

        Returns:
            True if successful
        """
        with self.get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO transfer_chunks
                (task_id, step_name, chunk_id, source_path, dest_path,
                 offset, size, checksum, status, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed', CURRENT_TIMESTAMP)
            """, (
                task_id, step_name, chunk_id,
                source_path, dest_path, offset, size, checksum
            ))
            return True

    def clear_transfer_chunks(
        self,
        task_id: int,
        step_name: Optional[str] = None
    ) -> int:
        """
        Clear transfer chunks for a task/step.

        Args:
            task_id: Task ID
            step_name: Optional step name, if not specified clears all steps

        Returns:
            Number of chunks deleted
        """
        with self.get_conn() as conn:
            if step_name:
                cursor = conn.execute("""
                    DELETE FROM transfer_chunks
                    WHERE task_id = ? AND step_name = ?
                """, (task_id, step_name))
            else:
                cursor = conn.execute("""
                    DELETE FROM transfer_chunks
                    WHERE task_id = ?
                """, (task_id,))
            return cursor.rowcount

    # ========================================================================
    # Metrics Management
    # ========================================================================

    def save_metrics(
        self,
        task_id: int,
        step_name: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Save metrics for a step.

        Args:
            task_id: Task ID
            step_name: Step name
            metrics: Metrics dictionary

        Returns:
            True if save successful
        """
        with self.get_conn() as conn:
            # Extract metrics
            duration = metrics.get("duration_seconds", 0.0)
            bytes_transferred = metrics.get("bytes_transferred", 0)
            files_transferred = metrics.get("files_transferred", 0)
            transfer_rate = metrics.get("transfer_rate_mbps", 0.0)
            epochs = metrics.get("epochs_completed", 0)
            best_auc = metrics.get("best_auc", 0.0)
            best_loss = metrics.get("best_loss", 0.0)
            final_loss = metrics.get("final_loss", 0.0)
            rows_processed = metrics.get("rows_processed", 0)
            throughput = metrics.get("inference_throughput", 0.0)

            # Additional metrics as JSON
            additional = {k: v for k, v in metrics.items()
                         if k not in ["duration_seconds", "bytes_transferred",
                                     "files_transferred", "transfer_rate_mbps",
                                     "epochs_completed", "best_auc", "best_loss",
                                     "final_loss", "rows_processed", "inference_throughput"]}
            additional_json = json.dumps(additional) if additional else None

            # Check if metrics already exist
            existing = conn.execute("""
                SELECT id FROM workflow_metrics
                WHERE task_id = ? AND step_name = ?
            """, (task_id, step_name)).fetchone()

            if existing:
                # Update existing
                cursor = conn.execute("""
                    UPDATE workflow_metrics
                    SET duration_seconds = ?,
                        bytes_transferred = ?,
                        files_transferred = ?,
                        transfer_rate_mbps = ?,
                        epochs_completed = ?,
                        best_auc = ?,
                        best_loss = ?,
                        final_loss = ?,
                        rows_processed = ?,
                        inference_throughput = ?,
                        additional_metrics = ?
                    WHERE task_id = ? AND step_name = ?
                """, (
                    duration, bytes_transferred, files_transferred, transfer_rate,
                    epochs, best_auc, best_loss, final_loss,
                    rows_processed, throughput, additional_json,
                    task_id, step_name
                ))
            else:
                # Insert new
                cursor = conn.execute("""
                    INSERT INTO workflow_metrics
                    (task_id, step_name, duration_seconds, bytes_transferred,
                     files_transferred, transfer_rate_mbps, epochs_completed,
                     best_auc, best_loss, final_loss, rows_processed,
                     inference_throughput, additional_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_id, step_name, duration, bytes_transferred,
                    files_transferred, transfer_rate, epochs,
                    best_auc, best_loss, final_loss, rows_processed,
                    throughput, additional_json
                ))
            return cursor.rowcount > 0

    def get_metrics(
        self,
        task_id: int,
        step_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Get metrics for a task.

        Args:
            task_id: Task ID
            step_name: Optional step name filter

        Returns:
            List of metrics dictionaries
        """
        with self.get_conn() as conn:
            if step_name:
                cursor = conn.execute("""
                    SELECT * FROM workflow_metrics
                    WHERE task_id = ? AND step_name = ?
                """, (task_id, step_name))
            else:
                cursor = conn.execute("""
                    SELECT * FROM workflow_metrics
                    WHERE task_id = ?
                    ORDER BY step_name
                """, (task_id,))

            results = []
            for row in cursor.fetchall():
                metrics = dict(row)
                # Parse additional metrics
                if metrics.get("additional_metrics"):
                    try:
                        additional = json.loads(metrics["additional_metrics"])
                        metrics.update(additional)
                        del metrics["additional_metrics"]
                    except json.JSONDecodeError:
                        pass
                results.append(metrics)
            return results

    def get_aggregate_metrics(self, task_id: int) -> Dict[str, Any]:
        """
        Get aggregate metrics for a task.

        Args:
            task_id: Task ID

        Returns:
            Dictionary with aggregate metrics
        """
        metrics_list = self.get_metrics(task_id)
        if not metrics_list:
            return {}

        # Aggregate by step
        aggregate = {}
        for m in metrics_list:
            step = m["step_name"]
            if step not in aggregate:
                aggregate[step] = {
                    "duration_seconds": 0,
                    "bytes_transferred": 0,
                    "rows_processed": 0
                }
            aggregate[step]["duration_seconds"] += m.get("duration_seconds", 0)
            aggregate[step]["bytes_transferred"] += m.get("bytes_transferred", 0)
            aggregate[step]["rows_processed"] += m.get("rows_processed", 0)

        return aggregate
