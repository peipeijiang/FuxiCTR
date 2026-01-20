import sqlite3
from typing import List, Dict, Optional, Iterator
from contextlib import contextmanager
from fuxictr.workflow.models import Task, TaskStep, TransferChunk, TaskStatus, StepStatus, StepName

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    @contextmanager
    def get_conn(self) -> Iterator[sqlite3.Connection]:
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
                    user VARCHAR(100) NOT NULL,
                    model VARCHAR(255) NOT NULL,
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

    def create_task(self, name: str, user: str, model: str, experiment_id: str,
                    sample_sql: str, infer_sql: str, hdfs_path: str, hive_table: str) -> int:
        """Create a new task"""
        with self.get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO tasks (name, user, model, experiment_id, sample_sql, infer_sql, hdfs_path, hive_table)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, user, model, experiment_id, sample_sql, infer_sql, hdfs_path, hive_table))
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

    def get_all_tasks(self) -> List[Dict]:
        """Get all tasks"""
        with self.get_conn() as conn:
            cursor = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def update_task_status(self, task_id: int, status: str) -> bool:
        """Update task status"""
        with self.get_conn() as conn:
            cursor = conn.execute(
                "UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, task_id)
            )
            return cursor.rowcount > 0

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
            else:
                conn.execute("""
                    UPDATE task_steps
                    SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
                    WHERE task_id = ? AND step_name = ?
                """, (status.value, error_message, task_id, step_name))

    def get_task_steps(self, task_id: int) -> List[Dict]:
        """Get all steps for a task"""
        with self.get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM task_steps
                WHERE task_id = ?
                ORDER BY id ASC
            """, (task_id,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_task(self, task_id: int) -> bool:
        """Delete a task and its associated steps"""
        with self.get_conn() as conn:
            # Delete associated steps first (foreign key)
            conn.execute("DELETE FROM task_steps WHERE task_id = ?", (task_id,))
            # Delete associated transfer chunks
            conn.execute("DELETE FROM transfer_chunks WHERE task_id = ?", (task_id,))
            # Delete the task
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            return cursor.rowcount > 0
