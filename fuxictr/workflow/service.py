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
FastAPI Service for Workflow Orchestration.

Provides REST API and WebSocket endpoints for workflow management.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml

from fuxictr.workflow.db import DatabaseManager
from fuxictr.workflow.models import TaskStatus, StepStatus
from fuxictr.workflow.utils.logger import WorkflowLogger, get_broadcaster
from fuxictr.workflow.coordinator import WorkflowCoordinator, WorkflowOrchestrator


logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FuxiCTR Workflow API",
    description="Multi-server workflow orchestration for ML pipelines",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG_PATH = os.environ.get(
    "WORKFLOW_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "config.yaml")
)

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return default config
        return {
            "servers": {
                "server_21": {
                    "host": os.environ.get("SERVER_21_HOST", "localhost"),
                    "port": int(os.environ.get("SERVER_21_PORT", "22")),
                    "username": os.environ.get("SERVER_21_USER", ""),
                    "key_path": os.environ.get("SERVER_21_KEY", "")
                }
            },
            "storage": {
                "staging_dir": os.environ.get("STAGING_DIR", "/data/staging"),
                "server_21_staging": os.environ.get("SERVER_21_STAGING", "/tmp/staging"),
                "checkpoint_dir": os.environ.get("CHECKPOINT_DIR", "/data/checkpoints")
            },
            "fuxictr_paths": {
                "data_root": os.environ.get("DATA_ROOT", "./data/"),
                "model_root": os.environ.get("MODEL_ROOT", "./checkpoints/")
            },
            "transfer": {
                "chunk_size": 100 * 1024 * 1024,
                "max_retries": 10,
                "compression": True,
                "verify_checksum": True
            },
            "workflow": {
                "heartbeat_interval": 30,
                "log_rotation_size": 100 * 1024 * 1024
            }
        }

config = load_config()

# Initialize database
DB_PATH = os.environ.get("WORKFLOW_DB_PATH", "workflow_tasks.db")
db = DatabaseManager(DB_PATH)
db.init_db()

# Initialize orchestrator
orchestrator = WorkflowOrchestrator(db_manager=db, config=config)

# Active WebSocket connections
active_connections: Dict[int, List[WebSocket]] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class TaskCreateRequest(BaseModel):
    """Request model for creating a new task."""
    name: str = Field(..., description="Task name")
    user: str = Field(..., description="Username who created the task")
    model: str = Field(..., description="Model name (e.g., MMoE, PLE)")
    experiment_id: str = Field(..., description="Experiment configuration ID")
    sample_sql: str = Field(..., description="SQL for sample data extraction")
    infer_sql: str = Field(..., description="SQL for inference data extraction")
    hdfs_path: str = Field(..., description="HDFS path for data")
    hive_table: str = Field(..., description="Destination Hive table")

    # Optional parameters
    training_server: Optional[str] = Field(None, description="Training server assignment")
    inference_server: Optional[str] = Field(None, description="Inference server assignment")
    gpu_count: Optional[int] = Field(None, description="Number of GPUs to use")
    batch_size: Optional[int] = Field(None, description="Batch size for training")
    epochs: Optional[int] = Field(None, description="Number of training epochs")


class TaskResponse(BaseModel):
    """Response model for task details."""
    task_id: int
    name: str
    user: str
    model: str
    experiment_id: str
    status: str
    current_step: int
    total_steps: int
    training_server: Optional[str]
    inference_server: Optional[str]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


class StepResponse(BaseModel):
    """Response model for step details."""
    id: int
    task_id: int
    step_name: str
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    retry_count: int


class ProgressResponse(BaseModel):
    """Response model for task progress."""
    task_id: int
    task_name: str
    status: str
    current_step: int
    total_steps: int
    steps: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "FuxiCTR Workflow API",
        "version": "2.0.0",
        "description": "Multi-server workflow orchestration for ML pipelines",
        "docs": "/docs",
        "endpoints": {
            "tasks": "/api/workflow/tasks",
            "create": "/api/workflow/tasks",
            "detail": "/api/workflow/tasks/{task_id}",
            "steps": "/api/workflow/tasks/{task_id}/steps",
            "progress": "/api/workflow/tasks/{task_id}/progress",
            "logs_ws": "/api/workflow/tasks/{task_id}/logs",
            "retry": "/api/workflow/tasks/{task_id}/retry",
            "cancel": "/api/workflow/tasks/{task_id}/cancel"
        }
    }


@app.post("/api/workflow/tasks", response_model=TaskResponse)
async def create_task(req: TaskCreateRequest, background_tasks: BackgroundTasks):
    """
    Create a new workflow task.

    The task will be executed in the background immediately after creation.
    """
    task_id = db.create_task(
        name=req.name,
        user=req.user,
        model=req.model,
        experiment_id=req.experiment_id,
        sample_sql=req.sample_sql,
        infer_sql=req.infer_sql,
        hdfs_path=req.hdfs_path,
        hive_table=req.hive_table,
        training_server=req.training_server,
        inference_server=req.inference_server,
        config={
            "gpu_count": req.gpu_count,
            "batch_size": req.batch_size,
            "epochs": req.epochs
        }
    )

    # Create workflow logger
    workflow_logger = WorkflowLogger(task_id)
    workflow_logger.broadcast_callback = broadcast_log

    # Register with broadcaster
    get_broadcaster().register_logger(task_id, workflow_logger)

    # Start workflow execution in background
    background_tasks.add_task(submit_and_execute_workflow(task_id, workflow_logger))

    # Get created task
    task = db.get_task(task_id)

    return TaskResponse(**_convert_task_for_response(task))


def _convert_task_for_response(task: Dict) -> Dict:
    """Convert database task record to TaskResponse format."""
    if not task:
        return task
    # Create a copy to avoid modifying the original
    result = dict(task)
    # Map database 'id' to response 'task_id'
    result['task_id'] = result.pop('id', None)
    return result


@app.get("/api/workflow/tasks", response_model=List[TaskResponse])
async def list_tasks(
    user: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """
    List all workflow tasks with optional filtering.

    Query parameters:
    - user: Filter by username
    - status: Filter by status (pending, running, completed, failed, cancelled)
    - limit: Maximum number of tasks to return (default: 100)
    """
    tasks = db.get_all_tasks(user=user, status=status, limit=limit)
    return [TaskResponse(**_convert_task_for_response(task)) for task in tasks]


@app.get("/api/workflow/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int):
    """Get details of a specific task."""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskResponse(**_convert_task_for_response(task))


@app.get("/api/workflow/tasks/{task_id}/steps", response_model=List[StepResponse])
async def get_task_steps(task_id: int):
    """Get all steps for a task with their status."""
    steps = db.get_task_steps(task_id)
    return [StepResponse(**step) for step in steps]


@app.get("/api/workflow/tasks/{task_id}/progress", response_model=ProgressResponse)
async def get_task_progress(task_id: int):
    """
    Get current progress of a task.

    Returns detailed progress including step statuses and metrics.
    """
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    steps = db.get_task_steps(task_id)
    metrics = db.get_aggregate_metrics(task_id)

    return ProgressResponse(
        task_id=task_id,
        task_name=task.get("name"),
        status=task.get("status"),
        current_step=task.get("current_step", 0),
        total_steps=task.get("total_steps", 5),
        steps=[
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
        metrics=metrics,
        created_at=task.get("created_at"),
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at")
    )


@app.get("/api/workflow/tasks/{task_id}/metrics")
async def get_task_metrics(task_id: int):
    """Get metrics for a task."""
    metrics = db.get_metrics(task_id)
    return {"task_id": task_id, "metrics": metrics}


@app.post("/api/workflow/tasks/{task_id}/retry")
async def retry_task(task_id: int, step: Optional[str] = None):
    """
    Retry a failed task from the specified step.

    If step is not specified, retry from the first failed step.
    """
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.get("status") not in [TaskStatus.FAILED.value, TaskStatus.CANCELLED.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed or cancelled tasks, current status: {task.get('status')}"
        )

    # Reset failed steps to pending
    steps = db.get_task_steps(task_id)
    for s in steps:
        if s["status"] == StepStatus.FAILED.value:
            if step is None or s["step_name"] == step:
                db.update_step_status(task_id, s["step_name"], StepStatus.PENDING)

    # Create logger and retry
    workflow_logger = WorkflowLogger(task_id)
    workflow_logger.broadcast_callback = broadcast_log

    # Register with broadcaster
    get_broadcaster().register_logger(task_id, workflow_logger)

    # Start workflow in background
    asyncio.create_task(execute_workflow_with_logger(task_id, workflow_logger))

    return {"message": f"Task {task_id} retry initiated"}


@app.post("/api/workflow/tasks/{task_id}/cancel")
async def cancel_task(task_id: int):
    """
    Cancel a running task.

    Attempts to gracefully cancel the task. If the task is not actively running,
    this will mark it as cancelled.
    """
    success = await orchestrator.cancel_workflow(task_id)

    if not success:
        # Task might not be running, just mark as cancelled
        db.update_task_status(task_id, TaskStatus.CANCELLED.value)

    return {"message": f"Task {task_id} cancellation requested"}


@app.delete("/api/workflow/tasks/{task_id}")
async def delete_task(task_id: int):
    """
    Delete a task and all its associated data.

    This will delete the task, its steps, transfer chunks, and metrics.
    """
    success = db.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")

    # Cancel if running
    await orchestrator.cancel_workflow(task_id)

    return {"message": "Task deleted successfully"}


@app.get("/api/workflow/servers")
async def list_servers():
    """List configured servers."""
    servers = []
    for name, server_config in config.get("servers", {}).items():
        servers.append({
            "name": name,
            "host": server_config.get("host"),
            "port": server_config.get("port"),
            "username": server_config.get("username")
        })
    return {"servers": servers}


@app.get("/api/workflow/config")
async def get_config():
    """Get current configuration (without sensitive data)."""
    safe_config = {
        "servers": {k: {"host": v.get("host"), "port": v.get("port")}
                   for k, v in config.get("servers", {}).items()},
        "storage": config.get("storage", {}),
        "fuxictr_paths": config.get("fuxictr_paths", {}),
        "transfer": config.get("transfer", {}),
        "workflow": config.get("workflow", {})
    }
    return safe_config


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/api/workflow/tasks/{task_id}/logs")
async def task_logs(websocket: WebSocket, task_id: int):
    """
    WebSocket endpoint for real-time log streaming.

    Connect to this endpoint to receive real-time logs from a running workflow.
    """
    await websocket.accept()

    if task_id not in active_connections:
        active_connections[task_id] = []
    active_connections[task_id].append(websocket)

    logger.info(f"WebSocket connected for task {task_id}")

    try:
        # Send initial status
        task = db.get_task(task_id)
        if task:
            await websocket.send_json({
                "type": "status",
                "task_id": task_id,
                "data": {
                    "status": task.get("status"),
                    "current_step": task.get("current_step")
                }
            })

        # Keep connection alive and send logs
        while True:
            # Check for new logs
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        if task_id in active_connections:
            active_connections[task_id].remove(websocket)
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
        if task_id in active_connections:
            active_connections[task_id].remove(websocket)


async def broadcast_log(task_id: int, message: Dict[str, Any]):
    """Broadcast a log message to all WebSocket connections for a task."""
    if task_id in active_connections:
        for ws in active_connections[task_id]:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")


# ============================================================================
# Workflow Execution Functions
# ============================================================================

async def submit_and_execute_workflow(task_id: int, workflow_logger: WorkflowLogger):
    """Submit task to orchestrator and execute."""
    try:
        success = await orchestrator.submit_workflow(task_id, workflow_logger)
        if not success:
            workflow_logger.error("workflow", "Failed to submit workflow")
    except Exception as e:
        logger.exception(f"Failed to submit workflow {task_id}")
        workflow_logger.error("workflow", f"Failed to submit: {e}")
        db.update_task_status(task_id, TaskStatus.FAILED.value)


async def execute_workflow_with_logger(task_id: int, workflow_logger: WorkflowLogger):
    """Execute workflow with logger."""

    # Hook logger to broadcast via WebSocket
    original_log = workflow_logger.log

    async def broadcast_log_wrapper(step, message, level="INFO"):
        msg = original_log(step, message, level)
        await broadcast_log(task_id, msg)
        return msg

    workflow_logger.log = broadcast_log_wrapper

    try:
        status = await orchestrator.resume_workflow(task_id)
        logger.info(f"Workflow {task_id} completed with status: {status.value}")
    except Exception as e:
        logger.exception(f"Workflow {task_id} failed")
        db.update_task_status(task_id, TaskStatus.FAILED.value)
        workflow_logger.error("workflow", str(e))


# ============================================================================
# Background Tasks
# ============================================================================

async def cleanup_completed_workflows():
    """Periodically clean up completed workflows."""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        await orchestrator.cleanup_completed_workflows()


@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup."""
    logger.info("Starting FuxiCTR Workflow API")

    # Start logger broadcaster
    broadcaster = get_broadcaster()
    broadcaster.start()
    logger.info("Logger broadcaster started")

    # Start cleanup task
    asyncio.create_task(cleanup_completed_workflows())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down FuxiCTR Workflow API")

    # Stop logger broadcaster
    broadcaster = get_broadcaster()
    await broadcaster.stop()
    logger.info("Logger broadcaster stopped")

    # Cancel all running workflows
    for task_id in list(orchestrator.active_workflows.keys()):
        await orchestrator.cancel_workflow(task_id)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting FuxiCTR Workflow API on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
