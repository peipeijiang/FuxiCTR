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
            user="yeshao",
            model="multitask/APG_SharedBottom",
            experiment_id="exp_001",
            sample_sql="SELECT * FROM table",
            infer_sql="SELECT * FROM infer_table",
            hdfs_path="/hdfs/data",
            hive_table="hive.result"
        )

        assert task_id > 0
        task = db.get_task(task_id)
        assert task["name"] == "test_task"
        assert task["user"] == "yeshao"
        assert task["model"] == "multitask/APG_SharedBottom"
        assert task["status"] == "pending"
