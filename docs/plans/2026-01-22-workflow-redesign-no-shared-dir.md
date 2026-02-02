# Workflow System Redesign - No Shared Directory Architecture

**Date:** 2026-01-22
**Author:** Shane (with Claude)
**Status:** Design Document

---

## 1. Overview

### 1.1 Problem Statement

The original workflow design (commit 5ef16574) relied on a **shared directory architecture** where:
- Server 21 (HDFS/Spark) exports data to a shared mount point
- Training server reads from shared directory
- Results are written back to shared directory

**Constraint Change:** After discussion with developers, **shared directories cannot be used**. All data must be transferred between servers via SSH.

### 1.2 Design Goals

1. **No Shared Dependencies**: All data transfer via SSH (rsync/scp/sftp)
2. **Fault Tolerance**: Support checkpoint/resume at every stage
3. **Minimal Interruption**: Workflow continues from last successful step on failure
4. **Compatibility**: Works with existing FuxiCTR framework
5. **Scalability**: Support multi-GPU training and inference

---

## 2. Current Path Architecture

### 2.1 Training Paths

```
{model_root}/
├── {dataset_id}/
│   ├── {model_id}.model           # Model checkpoint
│   ├── {model_id}.log             # Training log
│   └── {model_id}/                # TensorBoard logs
│       └── events.out.tfevents.*
```

**Default values:**
- `model_root`: `./checkpoints/`
- `dataset_id`: e.g., `tiny_parquet`
- `model_id`: e.g., `MMoE_test`

### 2.2 Inference Paths

```
{data_root}/
├── {dataset_id}/
│   ├── feature_map.json           # Feature schema
│   ├── feature_processor.pkl      # Feature encoder
│   ├── train/
│   │   ├── *.npz or *.parquet    # Training data
│   ├── valid/
│   └── {expid}_inference_result/  # Inference output
│       ├── part_0.parquet
│       ├── part_1.parquet
│       └── .inference_lock        # Lock file
```

**Default values:**
- `data_root`: `./data/`

### 2.3 Log Paths

- **Training logs**: `{model_root}/{dataset_id}/{model_id}.log`
- **TensorBoard logs**: `{model_root}/{dataset_id}/{model_id}/`
- **Workflow logs**: `workflow_tasks.db` + WebSocket stream

---

## 3. New Architecture Design

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Workflow Orchestrator                           │
│                    (FastAPI + WebSocket)                            │
└────────────────────────┬────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌────────────────┐ ┌──────────┐ ┌────────────────┐
│  Data Fetcher  │ │ Trainer  │ │   Inference    │
│  (Server 21)   │ │(Local)   │ │  (Local)       │
└────────┬───────┘ └────┬─────┘ └────────┬───────┘
         │               │                  │
    ┌────▼────┐      ┌───▼──────┐     ┌───▼──────┐
    │SSH Sync │      │ Local    │     │SSH Sync  │
    │ Client  │      │ Storage  │     │ Client   │
    └─────────┘      └──────────┘     └──────────┘
         │                                  │
    ┌────▼───────────────────────────────▼────┐
    │         Server-to-Server Transfer        │
    │           (rsync/scp/sftp)              │
    └─────────────────────────────────────────┘
```

### 3.2 Server Roles

| Server | Role | Responsibilities |
|--------|------|-------------------|
| **Server 21** | Data Source | - Execute Hive/Spark SQL<br>- Export data to local staging<br>- Transfer data via SSH |
| **Training Server** | Compute | - Receive training data<br>- Run multi-GPU training<br>- Save model checkpoints |
| **Inference Server** | Compute | - Load trained models<br>- Run inference<br>- Transfer results back to Server 21 |
| **Orchestrator** | Coordination | - Manage workflow state<br>- Coordinate transfers<br>- Handle failures |

---

## 4. Workflow Stages

### Stage 1: Data Fetch (Server 21 → Training Server)

```
┌─────────────────┐                    ┌─────────────────────────┐
│   Server 21     │                    │   Training Server       │
├─────────────────┤                    ├─────────────────────────┤
│ 1. Execute SQL  │                    │                         │
│    (Hive/Spark) │                    │                         │
│       │         │                    │                         │
│       ▼         │                    │                         │
│ 2. Export to    │                    │                         │
│    local staging│                    │                         │
│    /tmp/staging/│                    │                         │
│    sample_data/ │                    │                         │
│    infer_data/  │                    │                         │
│       │         │                    │                         │
│       ▼         │                    │                         │
│ 3. rsync via SSH│ ──────────────────▶│ 4. Receive to          │
│    - Port 22    │   SSH Transfer     │    {data_root}/         │
│    - Key auth   │                    │    {dataset_id}/        │
│    - Compress   │                    │    train/, valid/       │
│                 │                    │                         │
│ 5. Verify       │ ◀────────────────── │ 6. Verify checksum     │
│    checksum     │    Confirmation     │                         │
└─────────────────┘                    └─────────────────────────┘
```

**Key Features:**
- **Chunked transfer**: Use `rsync --partial` for resume capability
- **Compression**: `rsync -z` to reduce bandwidth
- **Progress tracking**: Store completed chunks in database
- **Checksum verification**: MD5/SHA256 verification after transfer

### Stage 2: Training (Training Server - Local)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Server                             │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load data from {data_root}/{dataset_id}/                     │
│    - train/, valid/, feature_map.json                           │
│                                                                  │
│ 2. Initialize model                                             │
│    - Create {model_root}/{dataset_id}/                          │
│    - Start TensorBoard logging                                  │
│                                                                  │
│ 3. Train with DDP (multi-GPU)                                  │
│    - Epoch 0-N                                                  │
│    - Save checkpoint each epoch                                 │
│    - Monitor metrics                                            │
│                                                                  │
│ 4. Save final model                                             │
│    - {model_root}/{dataset_id}/{model_id}.model                 │
│    - Copy to staging for transfer                               │
└─────────────────────────────────────────────────────────────────┘
```

**Checkpoint Strategy:**
- Model checkpoint saved after each epoch
- Training state: `{model_root}/{dataset_id}/{model_id}.checkpoint`
- On failure, resume from last epoch

### Stage 3: Inference (Inference Server - Local)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Inference Server                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. Receive model via SSH                                       │
│    From: Training Server:{model_root}/...                       │
│    To: {model_root}/{dataset_id}/                               │
│                                                                  │
│ 2. Load inference data                                         │
│    - {data_root}/{dataset_id}/infer_data/                       │
│    - feature_map.json, feature_processor.pkl                    │
│                                                                  │
│ 3. Run inference                                               │
│    - Multi-GPU inference with DDP                               │
│    - Output: {data_root}/{dataset_id}/{expid}_inference_result/ │
│    - Files: part_0.parquet, part_1.parquet, ...                │
│                                                                  │
│ 4. Verify output                                               │
│    - Check row counts                                           │
│    - Validate schema                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 4: Result Transport (Inference Server → Server 21)

```
┌─────────────────────────┐                    ┌─────────────────┐
│  Inference Server       │                    │   Server 21     │
├─────────────────────────┤                    ├─────────────────┤
│ 1. Prepare output       │                    │                 │
│    - Compress parquets  │                    │                 │
│    - Create manifest    │                    │                 │
│         │               │                    │                 │
│         ▼               │                    │                 │
│ 2. rsync via SSH        │ ──────────────────▶│ 3. Receive to  │
│    - Incremental sync   │   SSH Transfer     │    /hdfs/stage/ │
│    - Verify checksum    │                    │                 │
│         │               │                    │                 │
│         ▼               │                    │                 │
│ 4. Upload to Hive       │ ◀────────────────── │ 5. Trigger Hive│
│    - Load from staging  │    Command         │    load command │
│    - Verify row count   │                    │                 │
└─────────────────────────┘                    └─────────────────┘
```

### Stage 5: Monitor & Cleanup

```
┌─────────────────────────────────────────────────────────────────┐
│  Post-Processing                                                │
├─────────────────────────────────────────────────────────────────┤
│ 1. Aggregate metrics                                           │
│    - Training: loss, AUC, etc.                                  │
│    - Inference: throughput, latency                             │
│                                                                  │
│ 2. Generate report                                             │
│    - {model_root}/{dataset_id}/{model_id}_report.json           │
│                                                                  │
│ 3. Cleanup temporary files                                     │
│    - Remove staging data (optional, configurable)               │
│    - Archive old checkpoints                                    │
│                                                                  │
│ 4. Notify completion                                           │
│    - WebSocket event to frontend                                │
│    - Optional: email/webhook notification                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Transfer & Coordination Mechanism

### 5.1 SSH Transfer Manager

```python
class SSHTransferManager:
    """Handles all server-to-server data transfers"""

    async def transfer_data(
        self,
        source_host: str,
        source_path: str,
        dest_host: str,
        dest_path: str,
        task_id: int,
        step_name: str,
        chunk_size: int = 100 * 1024 * 1024  # 100MB
    ) -> TransferResult:
        """
        Transfer data with resume capability.

        Strategy:
        1. Calculate file checksums on source
        2. Check existing chunks on destination
        3. Transfer only missing chunks
        4. Verify final checksum
        5. Store progress in database
        """
```

### 5.2 Transfer Strategies

| Strategy | Tool | Use Case |
|----------|------|----------|
| **rsync** | `rsync -az --partial` | Large datasets, resume needed |
| **scp** | `scp -C` | Single files, simple transfer |
| **sftp** | Python `paramiko` | Programmatic control |
| **tar + pipe** | `tar cf - | ssh` | Directory archives |

**Recommended:** `rsync` with `--partial --progress`

### 5.3 Coordination Flow

```python
class WorkflowCoordinator:
    """Orchestrates multi-server workflow execution"""

    async def execute_workflow(self, task_id: int) -> WorkflowStatus:
        """
        Execute workflow with automatic retry and resume.

        For each stage:
        1. Check if stage already completed
        2. If completed, skip to next stage
        3. If failed, retry from checkpoint
        4. Update database state
        5. Broadcast progress via WebSocket
        """

        stages = [
            DataFetchStage(self),
            TrainingStage(self),
            InferenceStage(self),
            TransportStage(self),
            MonitorStage(self)
        ]

        for stage in stages:
            status = await stage.execute(task_id)
            if status == StageStatus.FAILED:
                return WorkflowStatus.FAILED
            elif status == StageStatus.SKIPPED:
                continue

        return WorkflowStatus.COMPLETED
```

---

## 6. Checkpoint & Resume Mechanism

### 6.1 Checkpoint Points

| Stage | Checkpoint | Resume Strategy |
|-------|-----------|-----------------|
| **Data Fetch** | After each rsync chunk | Re-transfer missing chunks only |
| **Training** | After each epoch | Resume from last saved model |
| **Inference** | After each part file | Re-process failed parts |
| **Transport** | After each file | Re-transfer missing files |

### 6.2 State Persistence

```sql
-- Database schema for checkpoint tracking

CREATE TABLE workflow_tasks (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255),
    status VARCHAR(50),  -- pending, running, completed, failed
    current_step INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE workflow_steps (
    id INTEGER PRIMARY KEY,
    task_id INTEGER,
    step_name VARCHAR(50),  -- data_fetch, train, infer, transport, monitor
    status VARCHAR(50),     -- pending, running, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    checkpoint_data JSON  -- Flexible checkpoint storage
);

CREATE TABLE transfer_chunks (
    id INTEGER PRIMARY KEY,
    task_id INTEGER,
    step_name VARCHAR(50),
    chunk_id VARCHAR(100),
    source_path VARCHAR(500),
    dest_path VARCHAR(500),
    offset INTEGER,
    size INTEGER,
    checksum VARCHAR(64),
    status VARCHAR(50),  -- pending, in_progress, completed, failed
    retry_count INTEGER DEFAULT 0,
    UNIQUE(task_id, step_name, chunk_id)
);
```

### 6.3 Resume Logic

```python
async def resume_from_checkpoint(task_id: int, step_name: str):
    """
    Resume a workflow step from its last checkpoint.

    Resume Strategy by Step:
    1. data_fetch: Query transfer_chunks for completed chunks,
                   resume with rsync --partial
    2. train: Load model.checkpoint, continue from last_epoch
    3. infer: Check output directory for existing part_*.parquet,
              skip completed files
    4. transport: Similar to data_fetch, use rsync --partial
    """
    step = db.get_step(task_id, step_name)
    checkpoint = json.loads(step.checkpoint_data)

    if step_name == "data_fetch":
        return await resume_data_fetch(task_id, checkpoint)
    elif step_name == "train":
        return await resume_training(task_id, checkpoint)
    # ... other steps
```

---

## 7. Error Recovery Strategy

### 7.1 Error Types & Handling

| Error Type | Detection | Recovery Strategy |
|------------|-----------|-------------------|
| **Network timeout** | SSH connection loss | Retry with exponential backoff |
| **SSH auth failure** | Authentication error | Alert user, pause workflow |
| **Disk full** | Write error | Cleanup temp files, retry |
| **OOM during training** | Process killed | Reduce batch size, resume |
| **Corrupted data** | Checksum mismatch | Re-transfer file |
| **Training divergence** | Loss spikes | Rollback to best checkpoint |

### 7.2 Retry Policy

```python
class RetryPolicy:
    """Configurable retry policy for each operation"""

    # Network operations
    SSH_CONNECT_MAX_RETRIES = 5
    SSH_CONNECT_BACKOFF = 2  # seconds, exponential

    # Data transfer
    TRANSFER_MAX_RETRIES = 10
    TRANSFER_CHUNK_SIZE = 100 * 1024 * 1024  # 100MB

    # Training
    TRAIN_MAX_RESTARTS = 3
    TRAIN_CHECKPOINT_INTERVAL = 1  # epoch

    # Inference
    INFER_MAX_RETRIES = 5
    INFER_BATCH_SIZE = 10000
```

### 7.3 Manual Intervention Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    Workflow Decision Points                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. Before data transfer                                         │
│    → Pause if source data not ready                             │
│    → Require user confirmation for large transfers              │
│                                                                  │
│ 2. Before training                                              │
│    → Validate data integrity                                    │
│    → Check GPU availability                                     │
│                                                                  │
│ 3. On training failure                                         │
│    → Option to resume from checkpoint                           │
│    → Option to adjust hyperparameters                           │
│                                                                  │
│ 4. Before Hive upload                                          │
│    → Preview sample results                                     │
│    → Require confirmation for production tables                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Configuration Structure

### 8.1 Updated config.yaml

```yaml
# fuxictr/workflow/config.yaml

servers:
  server_21:
    host: "your-server-21-host"
    port: 22
    username: "username"
    key_path: "/path/to/private/key"
    role: "data_source"  # data_source, training, inference

  training_server:
    host: "training-server-host"
    port: 22
    username: "username"
    key_path: "/path/to/private/key"
    role: "training"
    gpus: [0, 1, 2, 3]  # GPU device IDs

  inference_server:
    host: "inference-server-host"
    port: 22
    username: "username"
    key_path: "/path/to/private/key"
    role: "inference"
    gpus: [0, 1]

storage:
  # Local staging directories (NOT shared)
  staging_dir: "/data/staging"
  checkpoint_dir: "/data/checkpoints"
  temp_dir: "/data/tmp"

  # Remote paths (for reference)
  server_21_staging: "/tmp/staging"
  hdfs_staging: "/hdfs/staging"

# FuxiCTR paths (per server)
fuxictr_paths:
  data_root: "./data/"
  model_root: "./checkpoints/"

transfer:
  method: "rsync"  # rsync, scp, sftp
  chunk_size: 104857600  # 100MB
  max_retries: 10
  parallel_workers: 4
  compression: true
  checksum_verify: true
  bandwidth_limit: "100M"  # rsync --bwlimit

workflow:
  heartbeat_interval: 30  # seconds
  log_rotation_size: 104857600  # 100MB
  checkpoint_interval: 1  # epoch
  auto_resume: true
  notification:
    on_complete: false
    on_failure: true
    webhook_url: ""  # Optional webhook for notifications
```

### 8.2 Task Configuration

```python
# Task creation request schema
class TaskCreateRequest(BaseModel):
    # Basic info
    name: str
    experiment_id: str

    # Data source (Server 21)
    sample_sql: str
    infer_sql: str
    hdfs_path: str

    # Destination (Server 21)
    hive_table: str

    # Optional overrides
    training_server: Optional[str] = None  # Override default
    inference_server: Optional[str] = None
    gpu_count: Optional[int] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
```

---

## 9. API Endpoints

### 9.1 Workflow Management

```
POST   /api/workflow/tasks              # Create new task
GET    /api/workflow/tasks              # List all tasks
GET    /api/workflow/tasks/{id}         # Get task details
DELETE /api/workflow/tasks/{id}         # Cancel/delete task
POST   /api/workflow/tasks/{id}/retry   # Retry from failed step
POST   /api/workflow/tasks/{id}/pause   # Pause running task
POST   /api/workflow/tasks/{id}/resume  # Resume paused task
```

### 9.2 Monitoring

```
GET    /api/workflow/tasks/{id}/status      # Get current status
GET    /api/workflow/tasks/{id}/steps       # Get step details
GET    /api/workflow/tasks/{id}/logs        # Get log history
WS     /api/workflow/tasks/{id}/logs/stream # WebSocket log stream
GET    /api/workflow/tasks/{id}/metrics     # Get training metrics
GET    /api/workflow/tasks/{id}/progress    # Get transfer progress
```

---

## 10. Implementation Roadmap

### Phase 1: Core Infrastructure (Priority 1)

| Task | File | Description |
|------|------|-------------|
| SSH Transfer Manager | `fuxictr/workflow/utils/ssh_transfer.py` | rsync/scp wrapper with resume |
| Updated Database | `fuxictr/workflow/db.py` | Add checkpoint/transfer tables |
| Workflow Coordinator | `fuxictr/workflow/coordinator.py` | Main orchestration logic |

### Phase 2: Stage Executors (Priority 2)

| Task | File | Description |
|------|------|-------------|
| Data Fetch Executor | `fuxictr/workflow/executor/data_fetch.py` | SQL export + SSH transfer |
| Training Executor | `fuxictr/workflow/executor/trainer.py` | Multi-GPU training with checkpoint |
| Inference Executor | `fuxictr/workflow/executor/inference.py` | Distributed inference |
| Transport Executor | `fuxictr/workflow/executor/transport.py` | Result transfer + Hive load |

### Phase 3: API & Frontend (Priority 3)

| Task | File | Description |
|------|------|-------------|
| Update Service | `fuxictr/workflow/service.py` | New endpoints for coordinator |
| Update Frontend | `dashboard/pages/workflow.py` | UI for new workflow |
| Progress Monitor | `dashboard/components/progress.py` | Real-time progress display |

---

## 11. Key Differences from Original Design

| Aspect | Original (Shared Dir) | New (SSH Transfer) |
|--------|---------------------|-------------------|
| **Data sharing** | Shared NFS mount | SSH/rsync transfer |
| **Consistency** | Immediate (same FS) | Eventual (after transfer) |
| **Failure impact** | Affects all servers | Isolated per server |
| **Network dependency** | Low (local mount) | High (SSH required) |
| **Resume capability** | File-level | Chunk-level |
| **Scalability** | Limited by shared storage | Limited by network |

---

## 12. Security Considerations

### 12.1 SSH Authentication

- **Recommended**: SSH key authentication (no password in config)
- **Key management**: Store keys in `~/.ssh/` with proper permissions (600)
- **Key rotation**: Support multiple keys for backup

### 12.2 Data in Transit

```bash
# rsync over SSH with encryption
rsync -aze "ssh -i /path/to/key" source/ dest/

# Verify checksum after transfer
md5sum source_file | ssh user@host "cat > /tmp/checksum.md5"
```

### 12.3 Access Control

- Each server has minimal required access
- Training server: Read-only access to data source
- Inference server: Read-only access to models
- Server 21: Write access to HDFS staging

---

## 13. Monitoring & Observability

### 13.1 Metrics to Track

| Metric | Type | Description |
|--------|------|-------------|
| `workflow_duration_seconds` | Histogram | Total workflow time |
| `stage_duration_seconds` | Histogram | Per-stage duration |
| `transfer_bytes_total` | Counter | Total bytes transferred |
| `transfer_duration_seconds` | Histogram | Transfer time |
| `training_loss` | Gauge | Training loss per epoch |
| `inference_throughput` | Gauge | Rows per second |
| `retry_count_total` | Counter | Number of retries |

### 13.2 Logging Strategy

```python
# Structured logging format
{
    "timestamp": "2026-01-22T10:30:00Z",
    "task_id": 123,
    "step": "data_fetch",
    "level": "INFO",
    "message": "Transferring chunk 5/100",
    "context": {
        "chunk_id": "chunk_4",
        "bytes_transferred": 104857600,
        "percent_complete": 5
    }
}
```

---

## 14. Example Workflow Execution

```bash
# 1. Create task via API
curl -X POST http://localhost:8001/api/workflow/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "campaign_2026_01_22",
    "experiment_id": "MMoE_test_v1",
    "sample_sql": "SELECT * FROM campaign_sample WHERE date = '2026-01-21'",
    "infer_sql": "SELECT * FROM campaign_infer WHERE date = '2026-01-22'",
    "hdfs_path": "/hdfs/data/campaign",
    "hive_table": "prod.campaign_scores"
  }'

# Response: {"task_id": 123, "status": "pending"}

# 2. Monitor progress
# WebSocket: ws://localhost:8001/api/workflow/tasks/123/logs/stream

# Expected log sequence:
# [10:00:00] [data_fetch] Connecting to server_21...
# [10:00:05] [data_fetch] Executing sample SQL...
# [10:05:30] [data_fetch] Exported 1.2B rows to /tmp/staging/sample_data/
# [10:05:31] [data_fetch] Starting transfer to training_server...
# [10:15:00] [data_fetch] Transfer complete: 45.2GB in 9m29s
# [10:15:01] [train] Loading data from /data/campaign_data/
# [10:15:10] [train] Starting training with 4 GPUs...
# [11:30:00] [train] Epoch 10/10 complete - AUC: 0.8234
# [11:30:01] [train] Model saved to /checkpoints/campaign_data/MMoE_test_v1.model
# [11:30:05] [infer] Loading model...
# [11:45:00] [infer] Inference complete: 500M rows processed
# [11:45:01] [transport] Transferring results to server_21...
# [11:50:00] [transport] Loading data to Hive table prod.campaign_scores...
# [11:52:30] [monitor] Workflow complete - Total time: 1h52m30s
```

---

## 15. Troubleshooting Guide

### Issue: SSH Connection Timeout

```
Symptom: "Authentication timeout" during data transfer

Diagnosis:
1. Check network connectivity: ping server_21
2. Verify SSH key: ssh -i /path/to/key user@server_21
3. Check firewall rules

Solution:
- Add keepalive in SSH config: ServerAliveInterval 60
- Use retry logic with exponential backoff
```

### Issue: Transfer Checksum Mismatch

```
Symptom: "MD5 checksum failed" after rsync

Diagnosis:
1. Verify source file integrity
2. Check disk space on destination
3. Look for network errors

Solution:
- Re-run rsync with --checksum
- If persistent, suspect disk corruption
```

### Issue: Training OOM

```
Symptom: "CUDA out of memory" during training

Diagnosis:
1. Check GPU memory: nvidia-smi
2. Review batch size in config
3. Check model size

Solution:
- Reduce batch_size
- Enable gradient_checkpointing
- Use fewer GPUs with larger memory
- Resume from last checkpoint with new config
```

---

## 16. Future Enhancements

1. **Kubernetes Integration**: Deploy workflow pods on K8s
2. **Object Storage**: Support S3/OSS instead of SSH
3. **Multi-region**: Cross-region data transfer optimization
4. **Auto-scaling**: Dynamic GPU allocation based on workload
5. **A/B Testing**: Run multiple experiments in parallel

---

## Appendix A: File Structure

```
fuxictr/
├── workflow/
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── config.yaml                  # Configuration file
│   ├── db.py                        # Database manager
│   ├── models.py                    # Data models
│   ├── coordinator.py               # Main orchestration
│   ├── utils/
│   │   ├── ssh_client.py           # SSH connection wrapper
│   │   ├── ssh_transfer.py         # rsync/scp transfer manager
│   │   └── logger.py               # Structured logger
│   └── executor/
│       ├── data_fetch.py           # Stage 1: Data fetch
│       ├── trainer.py              # Stage 2: Training
│       ├── inference.py            # Stage 3: Inference
│       ├── transport.py            # Stage 4: Result transport
│       └── monitor.py              # Stage 5: Monitoring
│
├── dashboard/
│   └── pages/
│       └── workflow.py             # Workflow UI
│
└── docs/
    └── plans/
        └── 2026-01-22-workflow-redesign-no-shared-dir.md
```

---

## Appendix B: Database Schema

```sql
-- Complete schema for workflow tracking

CREATE TABLE workflow_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
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

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE workflow_steps (
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

    FOREIGN KEY (task_id) REFERENCES workflow_tasks(id)
);

CREATE TABLE transfer_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    step_name VARCHAR(50) NOT NULL,
    chunk_id VARCHAR(100) NOT NULL,

    -- File info
    source_path VARCHAR(500),
    dest_path VARCHAR(500),
    offset INTEGER,
    size INTEGER,
    checksum VARCHAR(64),

    -- Transfer state
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,

    FOREIGN KEY (task_id) REFERENCES workflow_tasks(id),
    UNIQUE(task_id, step_name, chunk_id)
);

CREATE INDEX idx_task_status ON workflow_tasks(status);
CREATE INDEX idx_step_status ON workflow_steps(task_id, status);
CREATE INDEX idx_transfer_status ON transfer_chunks(task_id, status);
```

---

**Document End**

For questions or clarification, please contact the development team.
