"""
Integration tests for the workflow pipeline API.

Tests the full FastAPI service including:
- Task creation via POST endpoint
- Task listing via GET endpoint
- Task detail retrieval via GET endpoint
- Error handling for nonexistent tasks
"""
import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Import the service app
from fuxictr.workflow.service import app
from fuxictr.workflow.db import DatabaseManager


@pytest.fixture
def test_db():
    """Create a temporary test database"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name

    # Initialize test database
    db = DatabaseManager(db_path)
    db.init_db()

    # Replace the global db in service module
    import fuxictr.workflow.service as service_module
    original_db = service_module.db
    service_module.db = db

    yield db

    # Cleanup
    service_module.db = original_db
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def client(test_db):
    """Create a test client with the test database"""
    return TestClient(app)


def test_create_and_list_tasks(client):
    """Test creating a task and listing it"""
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
    data = response.json()
    assert "task_id" in data
    task_id = data["task_id"]
    assert task_id > 0
    assert data["status"] == "pending"

    # List tasks
    response = client.get("/api/workflow/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert isinstance(tasks, list)
    assert len(tasks) > 0

    # Verify our created task is in the list
    created_task = next((t for t in tasks if t["id"] == task_id), None)
    assert created_task is not None
    assert created_task["name"] == "integration_test"


def test_get_task_details(client):
    """Test getting task details"""
    # First create a task
    create_response = client.post("/api/workflow/tasks", json={
        "name": "detail_test",
        "experiment_id": "test_exp",
        "sample_sql": "SELECT 1",
        "infer_sql": "SELECT 1",
        "hdfs_path": "/test",
        "hive_table": "test.result"
    })
    assert create_response.status_code == 200
    task_id = create_response.json()["task_id"]

    # Get task details
    response = client.get(f"/api/workflow/tasks/{task_id}")
    assert response.status_code == 200
    task = response.json()
    assert task["id"] == task_id
    assert task["name"] == "detail_test"
    assert task["experiment_id"] == "test_exp"
    assert task["sample_sql"] == "SELECT 1"
    assert task["infer_sql"] == "SELECT 1"
    assert task["hdfs_path"] == "/test"
    assert task["hive_table"] == "test.result"
    assert "status" in task
    assert "created_at" in task


def test_get_nonexistent_task(client):
    """Test getting a task that doesn't exist"""
    response = client.get("/api/workflow/tasks/99999")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_create_multiple_tasks(client):
    """Test creating multiple tasks and verifying they're all listed"""
    task_names = ["task_1", "task_2", "task_3"]
    task_ids = []

    for name in task_names:
        response = client.post("/api/workflow/tasks", json={
            "name": name,
            "experiment_id": f"exp_{name}",
            "sample_sql": f"SELECT * FROM {name}",
            "infer_sql": f"SELECT * FROM {name}_infer",
            "hdfs_path": f"/hdfs/{name}",
            "hive_table": f"hive.{name}"
        })
        assert response.status_code == 200
        task_ids.append(response.json()["task_id"])

    # List all tasks
    response = client.get("/api/workflow/tasks")
    assert response.status_code == 200
    tasks = response.json()

    # Verify all our tasks are present
    listed_task_names = [t["name"] for t in tasks]
    for name in task_names:
        assert name in listed_task_names


def test_task_schema_validation(client):
    """Test that the API properly validates request schema"""
    # Missing required fields
    response = client.post("/api/workflow/tasks", json={
        "name": "invalid_task"
        # Missing other required fields
    })
    assert response.status_code == 422  # Validation error


def test_task_contains_all_fields(client):
    """Test that task responses contain all expected fields"""
    response = client.post("/api/workflow/tasks", json={
        "name": "field_test",
        "experiment_id": "exp_1",
        "sample_sql": "SELECT 1",
        "infer_sql": "SELECT 1",
        "hdfs_path": "/test",
        "hive_table": "test.result"
    })
    task_id = response.json()["task_id"]

    # Get task details
    response = client.get(f"/api/workflow/tasks/{task_id}")
    task = response.json()

    # Verify all expected fields are present
    expected_fields = [
        "id", "name", "experiment_id", "sample_sql", "infer_sql",
        "hdfs_path", "hive_table", "status", "current_step",
        "created_at", "updated_at"
    ]
    for field in expected_fields:
        assert field in task, f"Missing field: {field}"
