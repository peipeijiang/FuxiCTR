from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio

from fuxictr.workflow.db import DatabaseManager
from fuxictr.workflow.config import Config
from fuxictr.workflow.utils.logger import WorkflowLogger
from fuxictr.workflow.executor.data_fetcher import DataFetcher
from fuxictr.workflow.executor.trainer import Trainer

app = FastAPI(title="Workflow Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config()
db = DatabaseManager("workflow_tasks.db")
db.init_db()

active_connections: dict[int, list[WebSocket]] = {}

class TaskCreateRequest(BaseModel):
    name: str
    user: str
    model: str
    experiment_id: str
    sample_sql: str
    infer_sql: str
    hdfs_path: str
    hive_table: str

@app.post("/api/workflow/tasks")
async def create_task(req: TaskCreateRequest):
    task_id = db.create_task(
        name=req.name,
        user=req.user,
        model=req.model,
        experiment_id=req.experiment_id,
        sample_sql=req.sample_sql,
        infer_sql=req.infer_sql,
        hdfs_path=req.hdfs_path,
        hive_table=req.hive_table
    )
    asyncio.create_task(execute_workflow(task_id, req))
    return {"task_id": task_id, "status": "pending"}

@app.get("/api/workflow/tasks")
async def list_tasks():
    tasks = db.get_all_tasks()
    return tasks

@app.get("/api/workflow/tasks/{task_id}")
async def get_task(task_id: int):
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/api/workflow/tasks/{task_id}/steps")
async def get_task_steps(task_id: int):
    """Get all steps for a task"""
    steps = db.get_task_steps(task_id)
    return steps

@app.delete("/api/workflow/tasks/{task_id}")
async def delete_task(task_id: int):
    """Delete a task and its associated steps"""
    success = db.delete_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task deleted successfully"}

@app.websocket("/api/workflow/tasks/{task_id}/logs")
async def task_logs(websocket: WebSocket, task_id: int):
    await websocket.accept()
    if task_id not in active_connections:
        active_connections[task_id] = []
    active_connections[task_id].append(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        active_connections[task_id].remove(websocket)

async def execute_workflow(task_id: int, req: TaskCreateRequest):
    logger = WorkflowLogger(task_id)
    async def broadcast_log(msg):
        if task_id in active_connections:
            for ws in active_connections[task_id]:
                await ws.send_json(msg)
    original_log = logger.log
    def broadcast_log_wrapper(step, message, level="INFO"):
        msg = original_log(step, message, level)
        asyncio.create_task(broadcast_log(msg))
        return msg
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
        db.update_task_status(task_id, "completed")
        logger.log("workflow", "Workflow completed successfully")
    except Exception as e:
        db.update_task_status(task_id, "failed")
        logger.error("workflow", str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
