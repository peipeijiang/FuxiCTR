# Workflow Pipeline Module

## Overview

The workflow pipeline module provides a comprehensive automation system for machine learning pipelines in FuXiCTR. It orchestrates the complete ML workflow: data fetching → training → inference → Hive upload, with real-time monitoring and logging capabilities.

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

## Key Features

- **RESTful API**: FastAPI-based service for task management
- **Real-time Logs**: WebSocket support for live log streaming
- **State Management**: SQLite database for task persistence
- **6-Stage Pipeline**: Data fetch, upload, train, infer, download, upload
- **Fault Tolerance**: Retry mechanisms and error handling
- **Parallel Processing**: Multi-worker support for file transfers

## Project Structure

```
fuxictr/workflow/
├── config.yaml           # Configuration file
├── service.py            # FastAPI service
├── db.py                 # Database manager
├── models.py             # Data models
├── config.py             # Config loader
├── utils/
│   └── logger.py         # Workflow logger
├── executor/
│   ├── data_fetcher.py   # Stage 1-2: Fetch & upload
│   ├── trainer.py        # Stage 3: Training
│   ├── inferencer.py     # Stage 4-5: Inference & download
│   └── hive_uploader.py  # Stage 6: Hive upload
├── utils/
│   ├── ssh_client.py     # SSH connection manager
│   └── file_transfer.py  # SCP file transfer
└── dashboard/
    └── app.py            # Streamlit dashboard
```

## Installation

### Requirements

```bash
pip install fastapi uvicorn websockets streamlit
```

### Configuration

Edit `fuxictr/workflow/config.yaml`:

```yaml
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
  chunk_size: 10485760      # 10MB chunks
  max_retries: 3
  parallel_workers: 4
  verify_checksum: true

task:
  heartbeat_interval: 30
  log_rotation_size: 104857600  # 100MB
```

## Usage

### Starting the Service

```bash
# Start FastAPI service (port 8001)
python -m fuxictr.workflow.service

# Or with uvicorn directly
uvicorn fuxictr.workflow.service:app --host 0.0.0.0 --port 8001

# Start Streamlit dashboard
streamlit run fuxictr/workflow/dashboard/app.py
```

### Creating a Task

Via API:

```bash
curl -X POST http://localhost:8001/api/workflow/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_task",
    "experiment_id": "exp_001",
    "sample_sql": "SELECT * FROM train_data WHERE date = '\''2024-01-01'\''",
    "infer_sql": "SELECT * FROM infer_data WHERE date = '\''2024-01-02'\''",
    "hdfs_path": "/hdfs/path/to/data",
    "hive_table": "hive_db.output_table"
  }'
```

Response:

```json
{
  "task_id": 1,
  "status": "pending"
}
```

### Listing Tasks

```bash
curl http://localhost:8001/api/workflow/tasks
```

Response:

```json
[
  {
    "id": 1,
    "name": "my_task",
    "experiment_id": "exp_001",
    "sample_sql": "SELECT * FROM train_data...",
    "infer_sql": "SELECT * FROM infer_data...",
    "hdfs_path": "/hdfs/path/to/data",
    "hive_table": "hive_db.output_table",
    "status": "running",
    "current_step": 2,
    "created_at": "2024-01-20 10:00:00",
    "updated_at": "2024-01-20 10:05:00",
    "completed_at": null
  }
]
```

### Getting Task Details

```bash
curl http://localhost:8001/api/workflow/tasks/1
```

### WebSocket Logs

Connect to real-time log stream:

```javascript
const ws = new WebSocket('ws://localhost:8001/api/workflow/tasks/1/logs');

ws.onmessage = (event) => {
  const log = JSON.parse(event.data);
  console.log(`[${log.level}] ${log.step}: ${log.message}`);
  // Output: [INFO] data_fetch: Downloading sample_data.csv
};
```

Log message format:

```json
{
  "task_id": 1,
  "step": "data_fetch",
  "message": "Downloading sample_data.csv",
  "level": "INFO",
  "timestamp": "2024-01-20T10:00:00"
}
```

## Pipeline Stages

The workflow executes 6 stages sequentially:

1. **Data Fetch** (`data_fetcher`)
   - Executes SQL queries via Hive
   - Downloads sample and inference data from HDFS

2. **Data Upload** (`data_fetcher`)
   - Transfers data to training server via SCP

3. **Training** (`trainer`)
   - Runs FuXiCTR training on sample data
   - Saves model checkpoints

4. **Inference** (`inferencer`)
   - Runs inference on inference data
   - Generates predictions

5. **Result Download** (`inferencer`)
   - Downloads predictions from training server

6. **Hive Upload** (`hive_uploader`)
   - Uploads results to Hive table

## API Endpoints

| Endpoint | Method | Description | Request Body |
|----------|--------|-------------|--------------|
| `/api/workflow/tasks` | POST | Create new task | `TaskCreateRequest` |
| `/api/workflow/tasks` | GET | List all tasks | - |
| `/api/workflow/tasks/{id}` | GET | Get task details | - |
| `/api/workflow/tasks/{id}/logs` | WS | Real-time log stream | - |

### Request Schema (TaskCreateRequest)

```python
{
  "name": str,           # Task name
  "experiment_id": str,  # Experiment ID for training
  "sample_sql": str,     # SQL for sample data
  "infer_sql": str,      # SQL for inference data
  "hdfs_path": str,      # HDFS data path
  "hive_table": str      # Target Hive table
}
```

### Response Schema (Task)

```python
{
  "id": int,
  "name": str,
  "experiment_id": str,
  "sample_sql": str,
  "infer_sql": str,
  "hdfs_path": str,
  "hive_table": str,
  "status": str,         # pending/running/completed/failed
  "current_step": int,   # Current stage (0-5)
  "created_at": str,
  "updated_at": str,
  "completed_at": Optional[str]
}
```

## Database Schema

### Tasks Table

```sql
CREATE TABLE tasks (
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
```

### Task Steps Table

```sql
CREATE TABLE task_steps (
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
```

### Transfer Chunks Table

```sql
CREATE TABLE transfer_chunks (
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
```

## Error Handling

The workflow implements comprehensive error handling:

- **Retry Logic**: Failed operations are retried up to `max_retries` times
- **Chunked Transfer**: Large files are transferred in chunks with checksum verification
- **Graceful Failure**: Errors are logged and tasks marked as failed
- **WebSocket Notifications**: Real-time error updates via WebSocket

## Testing

### Unit Tests

```bash
# Test specific components
pytest tests/workflow/test_db.py -v
pytest tests/workflow/test_ssh_client.py -v
pytest tests/workflow/test_file_transfer.py -v
```

### Integration Tests

```bash
# Test full API integration
pytest tests/workflow/test_integration.py -v
```

### Run All Tests

```bash
pytest tests/workflow/ -v
```

## Monitoring

### Logs

Logs are stored per task in the `logs/` directory:

```
logs/
└── workflow/
    └── task_1/
        ├── 2024-01-20_10-00-00.log
        ├── 2024-01-20_10-01-00.log
        └── ...
```

### Dashboard

The Streamlit dashboard provides:
- Task list with status indicators
- Real-time log viewer
- Progress tracking
- Error highlighting

## Troubleshooting

### Common Issues

1. **SSH Connection Failed**
   - Verify `key_path` in config.yaml
   - Check server accessibility
   - Ensure proper permissions on private key

2. **HDFS Download Failed**
   - Verify HDFS path exists
   - Check network connectivity
   - Ensure sufficient disk space

3. **Training Failed**
   - Check experiment_id exists
   - Verify data format
   - Review training logs

## Performance Tuning

### Transfer Speed

Adjust `transfer` settings in config.yaml:

```yaml
transfer:
  chunk_size: 20971520      # Increase to 20MB
  parallel_workers: 8       # Increase parallelism
```

### Memory Usage

Monitor task memory and adjust:

```yaml
task:
  heartbeat_interval: 60    # Reduce frequency
  log_rotation_size: 52428800  # Smaller log files
```

## Contributing

When adding new features:

1. Update models in `models.py`
2. Add database migrations to `db.py`
3. Create executor in `executor/`
4. Add API endpoints in `service.py`
5. Write tests in `tests/workflow/`
6. Update this documentation

## License

This module is part of FuXiCTR and follows the same license.
