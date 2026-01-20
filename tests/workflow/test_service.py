import pytest
import tempfile
import os
from fastapi.testclient import TestClient
import fuxictr.workflow.service as service_module
from fuxictr.workflow.service import app
from fuxictr.workflow.db import DatabaseManager

@pytest.fixture
def client():
    """Create a test client with a fresh database for each test"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        # Temporarily override the module-level db variable
        original_db = service_module.db
        service_module.db = DatabaseManager(db_path)
        service_module.db.init_db()
        yield TestClient(app)
        # Restore original database
        service_module.db = original_db

def test_create_task(client):
    """Test creating a new task via API"""
    response = client.post("/api/workflow/tasks", json={
        "name": "test_task",
        "user": "yeshao",
        "model": "multitask/APG_SharedBottom",
        "experiment_id": "exp_001",
        "sample_sql": "SELECT * FROM table",
        "infer_sql": "SELECT * FROM infer_table",
        "hdfs_path": "/hdfs/data",
        "hive_table": "hive.result"
    })
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data

def test_list_tasks(client):
    """Test listing tasks"""
    response = client.get("/api/workflow/tasks")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
