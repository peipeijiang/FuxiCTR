# FuxiCTR Workflow 部署配置指南

> **版本**: v2.0 (2026-02-02)
> **适用场景**: Dashboard + Workflow 共存架构
> **服务器**: Server 21 (数据源) + Server 142 (训练服务器)

---

## 📋 目录

1. [架构概述](#架构概述)
2. [灵活部署配置](#灵活部署配置) ⭐ 新增
3. [Server 21 配置](#server-21-配置)
4. [Server 142 配置](#server-142-配置)
5. [目录结构说明](#目录结构说明)
6. [配置文件详解](#配置文件详解)
7. [验证测试](#验证测试)
8. [常见问题](#常见问题)

---

## 架构概述

### 系统架构

```
┌─────────────────────┐         ┌─────────────────────┐
│   Server 21         │         │   Server 142        │
│   (数据源服务器)     │         │   (训练服务器)       │
│                     │         │                     │
│  ┌──────────────┐   │  SSH    │  ┌──────────────┐   │
│  │ Hive/Spark   │   │ ──────> │  │  Workflow    │   │
│  │   SQL        │───┤  rsync  │  │  Coordinator │   │
│  └──────────────┘   │         │  └──────────────┘   │
│       ↓             │         │         ↓           │
│  ┌──────────────┐   │         │  ┌──────────────┐   │
│  │  Parquet     │   │         │  │  Training    │   │
│  │  /tmp/       │   │         │  │  Inference   │   │
│  └──────────────┘   │         │  └──────────────┘   │
│                     │         │                     │
└─────────────────────┘         └─────────────────────┘
```

### 关键设计原则

| 原则 | 说明 |
|------|------|
| **数据分离** | `data/` (原始) vs `processed_data/` (处理后) |
| **模型分离** | `model_zoo/` (Dashboard) vs `workflow_models/` (Workflow) |
| **实验隔离** | 每个实验使用独立文件夹 |
| **日志分离** | Dashboard 日志 vs Workflow 日志，互不干扰 |

---

## 灵活部署配置 ⭐

### 说明

本文档中的路径（如 `/opt/fuxictr`、`/data/fuxictr`）为示例路径。您可以根据实际情况灵活调整部署位置。

### 快速配置方法

使用环境变量配置文件，一处修改全局生效：

```bash
# 1. 复制环境变量模板
cp fuxictr/fuxictr_env.sh.template fuxictr_env.sh

# 2. 编辑环境变量，修改为实际路径
nano fuxictr_env.sh

# 3. 在 ~/.bashrc 中添加
echo "source $(pwd)/fuxictr_env.sh" >> ~/.bashrc

# 4. 重新加载环境变量
source ~/.bashrc
```

### 常见部署场景

#### 场景 1：标准部署（默认）

```
/opt/fuxictr/          # 代码
/data/fuxictr/         # 数据
/opt/fuxictr_venv/     # 虚拟环境
```

#### 场景 2：单分区部署

```bash
# 修改 fuxictr_env.sh
export FUXICTR_ROOT="$HOME/fuxictr"
export FUXICTR_VENV="$HOME/fuxictr_venv"
export FUXICTR_STORAGE_BASE="$HOME/fuxictr_data"
```

#### 场景 3：多磁盘部署

```bash
# 修改 fuxictr_env.sh
export FUXICTR_ROOT="/mnt/ssd/fuxictr"               # SSD - 代码
export FUXICTR_VENV="$HOME/fuxictr_venv"               # Home - 虚拟环境
export FUXICTR_STORAGE_BASE="/mnt/hdd1/fuxictr_data"  # HDD1 - 数据
export FUXICTR_WORKFLOW_MODELS="/mnt/hdd2/fuxictr_models" # HDD2 - 模型
```

#### 场景 4：完全自定义

```bash
# 根据实际情况修改所有路径
export FUXICTR_ROOT="/your/custom/path"
export FUXICTR_VENV="/your/venv/path"
export FUXICTR_STORAGE_BASE="/your/data/path"
```

### 修改 systemd 服务使用环境变量

在 systemd 服务文件中添加 `EnvironmentFile`：

```ini
[Service]
Type=simple
User=your_username
Group=your_username
WorkingDirectory=${FUXICTR_ROOT}
EnvironmentFile=${FUXICTR_ROOT}/fuxictr_env.sh  # ← 加载环境变量
ExecStart=${FUXICTR_VENV}/bin/python -m fuxictr.workflow.service
```

### 重新配置服务后

修改环境变量后，需要重启服务：

```bash
# 重新加载环境变量
source ~/.bashrc

# 重启服务
sudo systemctl restart fuxictr-workflow
sudo systemctl restart fuxictr-dashboard
```

---

## Server 21 配置

### 1.1 创建目录结构

SSH 登录到 Server 21：

```bash
ssh username@21.xxxxxx.com

# 创建临时 staging 目录
sudo mkdir -p /tmp/fuxictr_staging
sudo chmod 755 /tmp/fuxictr_staging
sudo chown $USER:$USER /tmp/fuxictr_staging

# 验证
ls -ld /tmp/fuxictr_staging
```

### 1.2 验证 Hive/Spark 环境

```bash
# 检查 spark-sql
which spark-sql
spark-sql --version

# 检查 Hive
hive --version

# 测试查询
spark-sql -e "SELECT 1 as test"
```

### 1.3 确认数据访问

```bash
# 检查 HDFS 路径（如果使用 HDFS）
hdfs dfs -ls /your/hdfs/path

# 检查 Hive 表
spark-sql -e "SHOW DATABASES"
spark-sql -e "SELECT COUNT(*) FROM your_database.your_table LIMIT 1"
```

### 1.4 准备示例 SQL

创建 `workflow_sql_template.yaml`（供开发人员参考）：

```yaml
# Server 21 上的 SQL 模板配置

# 训练数据 SQL（从 Hive/Spark 提取）
sample_sql: |
  SELECT
    user_id,
    product_id,
    category_tag,
    price_cnt,
    click_textlist,
    label_apply,
    label_credit
  FROM your_database.source_table
  WHERE dt >= '2024-01-01'
  LIMIT 1000000

# 推理数据 SQL
infer_sql: |
  SELECT
    user_id,
    product_id,
    category_tag,
    price_cnt,
    click_textlist
  FROM your_database.inference_source_table
  WHERE dt = '2024-01-23'
  LIMIT 100000

# 推理结果写入表
hive_table: "your_database.your_result_table"
```

---

## Server 142 配置

### 2.1 系统依赖安装

```bash
# 更新系统
sudo apt-get update && sudo apt-get upgrade -y

# 安装基础依赖
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    rsync \
    openssh-client
```

### 2.2 GPU 环境

```bash
# 安装 NVIDIA 驱动
sudo apt-get install -y nvidia-driver-535

# 验证 GPU
nvidia-smi
```

### 2.3 创建虚拟环境

```bash
# 创建 Python 虚拟环境
python3.10 -m venv /opt/fuxictr_venv
source /opt/fuxictr_venv/bin/activate

# 升级 pip
pip install --upgrade pip
```

### 2.4 安装 PyTorch

```bash
# 安装 PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 验证 GPU 可用性
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 2.5 创建目录结构（重要！）

```bash
# 创建所有必需的目录
sudo mkdir -p /data/fuxictr
sudo mkdir -p /data/fuxictr/{data,processed_data,workflow_datasets,workflow_processed,workflow_models,workflow_logs}
sudo mkdir -p /data/fuxictr/dashboard_logs
sudo mkdir -p /data/fuxictr/db_backup

# 设置权限
sudo chown -R $USER:$USER /data/fuxictr

# 验证目录结构
tree -L 2 /data/fuxictr/
```

**预期目录结构**：

```
/data/fuxictr/
├── data/                      # Dashboard 原始数据（只读）
├── processed_data/            # Dashboard 处理后数据
├── workflow_datasets/         # Workflow 原始数据（从 Server 21）
├── workflow_processed/        # Workflow 处理后数据
├── workflow_models/           # Workflow 模型
├── workflow_logs/             # Workflow 日志
├── dashboard_logs/            # Dashboard 应用日志
└── db_backup/                 # 数据库备份
```

### 2.6 配置 SSH 访问到 Server 21

```bash
# 生成 SSH 密钥对（如果没有）
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# 复制公钥到 Server 21
ssh-copy-id username@21.xxxxxx.com

# 测试无密码登录
ssh username@21.xxxxxx.com "echo 'SSH connection successful'"

# 测试 rsync
echo "test" > /tmp/test.txt
rsync -avz -e "ssh" /tmp/test.txt username@21.xxxxxx.com:/tmp/
```

### 2.7 部署 FuxiCTR 代码

```bash
# 克隆代码
cd /opt
git clone https://github.com/your-org/fuxictr.git
cd fuxictr

# 激活虚拟环境
source /opt/fuxictr_venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装 FuxiCTR
pip install -e .

# 安装 Workflow 依赖
pip install \
    fastapi \
    uvicorn[standard] \
    streamlit \
    pyarrow \
    pandas \
    numpy \
    scikit-learn \
    pyyaml \
    aiofiles \
    python-multipart \
    websockets \
    aiohttp
```

---

## 目录结构说明

### 完整目录结构

```
fuxictr/                                    # 项目根目录
│
├── data/                                   # Dashboard 原始数据（用户手动上传）
│   ├── tiny_npz/
│   │   └── train.csv                      # 原始 CSV 文件
│   ├── tiny_parquet/
│   │   └── *.parquet                      # 原始 parquet 文件
│   └── jrjk_seeds_20251202/               # 原始数据（去掉 _processed 后缀）
│       └── *.csv
│
├── processed_data/                         # Dashboard 处理后数据（build_dataset 生成）
│   ├── tiny_npz/
│   │   ├── train.parquet
│   │   ├── valid.parquet
│   │   ├── feature_map.json
│   │   └── feature_processor.pkl
│   └── jrjk_seeds_20251202/
│       ├── train.parquet
│       ├── valid.parquet
│       ├── test.parquet
│       ├── feature_map.json
│       ├── feature_processor.pkl
│       └── feature_vocab.json
│
├── workflow_datasets/                      # Workflow 原始数据（从 Server 21 传输）
│   └── jrzk_seeds_20260201/
│       └── raw/
│           ├── part_0.parquet
│           └── part_1.parquet
│
├── workflow_processed/                     # Workflow 处理后数据
│   └── jrzk_seeds_20260201/
│       ├── train.parquet
│       ├── valid.parquet
│       ├── feature_map.json
│       └── feature_processor.pkl
│
├── model_zoo/                              # Dashboard 模型（保持不变）
│   ├── AutoInt/
│   │   ├── config/
│   │   │   └── model_config.yaml
│   │   ├── checkpoints/
│   │   │   └── jrjk_seeds_20251202/
│   │   │       ├── AutoInt_test/          # 实验1 独立文件夹
│   │   │       │   ├── AutoInt_test.model
│   │   │       │   ├── AutoInt_test.log
│   │   │       │   ├── checkpoints/       # Epoch checkpoints
│   │   │       │   └── tensorboard/
│   │   │       └── AutoInt_prod/          # 实验2 独立文件夹
│   │   │           ├── AutoInt_prod.model
│   │   │           └── AutoInt_prod.log
│   │   ├── config.csv                     # Dashboard 训练记录
│   │   └── run_expid.py
│   └── DeepFM/
│
├── workflow_models/                        # Workflow 模型（新增，与 model_zoo 平级）
│   ├── AutoInt/
│   │   └── jrzk_seeds_20260201/
│   │       ├── task_001_AutoInt_test/      # 按任务ID组织
│   │       │   ├── model.model
│   │       │   ├── train.log
│   │       │   ├── checkpoints/
│   │       │   └── tensorboard/
│   │       └── task_002_AutoInt_prod/
│   └── DeepFM/
│
├── dashboard/
│   ├── logs/                              # Dashboard 应用和训练日志副本
│   │   ├── streamlit.log                  # Streamlit 应用日志
│   │   ├── users/                         # 用户训练日志副本（可选）
│   │   │   ├── yeshao.log
│   │   │   └── gxwang9.log
│   │   └── training/                      # Dashboard 训练进程日志（可选）
│   │       └── AutoInt_test_20251206.log
│   ├── user_configs/                      # 用户自定义配置
│   │   ├── yeshao/
│   │   │   └── AutoInt/
│   │   │       └── model_config.yaml
│   │   └── gxwang9/
│   └── pages/
│
├── workflow_logs/                          # Workflow 日志（新增）
│   ├── task_001_data_fetch.log
│   ├── task_001_train.log
│   └── task_001_infer.log
│
├── workflow_tasks.db                       # 工作流数据库
│
└── fuxictr/
    └── workflow/
        └── config.yaml                     # Workflow 配置文件
```

### 路径对照表

| 用途 | Dashboard 路径 | Workflow 路径 |
|------|---------------|--------------|
| **原始数据** | `data/{dataset_id}/` | `workflow_datasets/{dataset_id}/raw/` |
| **处理后数据** | `processed_data/{dataset_id}/` | `workflow_processed/{dataset_id}/` |
| **模型保存** | `model_zoo/{model}/checkpoints/{dataset_id}/{exp_id}/` | `workflow_models/{model}/{dataset_id}/task_{id}_{exp_id}/` |
| **训练日志（原始）** | `{exp_id}/{exp_id}.log` | `task_{id}_{exp_id}/train.log` |
| **训练日志（副本）** | `dashboard/logs/users/{username}/{exp_id}_{timestamp}.log` | 不需要 |
| **应用日志** | `dashboard/logs/streamlit.log` | 不需要 |
| **工作流日志** | 不需要 | `workflow_logs/task_{id}_{stage}.log` |
| **训练记录 CSV** | `model_zoo/{model}/config.csv` | `workflow_models/{model}/workflow_results.csv` |

---

## 配置文件详解

### 3.1 Workflow 配置文件

**文件位置**: `/opt/fuxictr/fuxictr/workflow/config.yaml`

```yaml
# =========================================================================
# FuxiCTR Workflow Configuration v2.0
# =========================================================================

# ----------------------------------------------------------------------------
# 服务器配置
# ----------------------------------------------------------------------------
servers:
  # Server 21 - 数据源服务器
  server_21:
    host: "21.xxxxxx.com"           # ⚠️ 替换为实际主机名
    port: 22                         # SSH 端口
    username: "your_username"        # ⚠️ 替换为 SSH 用户名
    key_path: "~/.ssh/id_rsa"        # SSH 私钥路径

# ----------------------------------------------------------------------------
# 存储路径配置（新增架构）
# ----------------------------------------------------------------------------
storage:
  # Server 21 上的临时目录
  server_21_staging: "/tmp/fuxictr_staging"

  # Dashboard 数据路径
  dashboard_data_root: "/opt/fuxictr/data/"
  dashboard_processed_root: "/opt/fuxictr/processed_data/"

  # Workflow 数据路径
  workflow_datasets_root: "/data/fuxictr/workflow_datasets/"      # 原始数据（从 Server 21）
  workflow_processed_root: "/data/fuxictr/workflow_processed/"    # 处理后数据（build_dataset）

  # Dashboard 模型路径
  dashboard_model_root: "/opt/fuxictr/model_zoo/"

  # Workflow 模型路径
  workflow_model_root: "/data/fuxictr/workflow_models/"

  # 日志路径
  dashboard_log_dir: "/opt/fuxictr/dashboard/logs/"
  workflow_log_dir: "/data/fuxictr/workflow_logs/"

  # 数据库备份
  db_backup_dir: "/data/fuxictr/db_backup/"

# ----------------------------------------------------------------------------
# 数据传输配置
# ----------------------------------------------------------------------------
transfer:
  chunk_size: 104857600              # 100MB
  max_retries: 10
  compression: true
  verify_checksum: true
  parallel_workers: 4
  timeout: 300
  bandwidth_limit: null              # 可选：限制带宽，如 "10M"

# ----------------------------------------------------------------------------
# Workflow 任务配置
# ----------------------------------------------------------------------------
workflow:
  heartbeat_interval: 30             # 心跳间隔（秒）
  log_rotation_size: 104857600       # 日志轮转大小（100MB）
  task_timeout: 86400                # 任务超时（24小时）

# ----------------------------------------------------------------------------
# 数据库配置
# ----------------------------------------------------------------------------
database:
  path: "/opt/fuxictr/workflow_tasks.db"
  backup_enabled: true
  backup_retention_days: 30

# ----------------------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------------------
logging:
  level: "INFO"                      # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  console_output: true
```

### 3.2 配置文件检查清单

在部署前，请确认以下配置项已正确设置：

| 配置项 | 位置 | 示例值 | 说明 |
|--------|------|--------|------|
| **Server 21 主机** | `servers.server_21.host` | `21.xxxxxx.com` | ⚠️ 必须替换 |
| **SSH 用户名** | `servers.server_21.username` | `your_username` | ⚠️ 必须替换 |
| **SSH 密钥路径** | `servers.server_21.key_path` | `~/.ssh/id_rsa` | ⚠️ 必须存在 |
| **数据根目录** | `storage.*_root` | `/data/fuxictr/...` | ⚠️ 必须存在 |
| **模型根目录** | `storage.*_model_root` | `/data/fuxictr/...` | ⚠️ 必须存在 |

---

## 验证测试

### 4.1 测试 SSH 连接

```bash
# 从 Server 142 测试到 Server 21 的连接
ssh -i ~/.ssh/id_rsa username@21.xxxxxx.com "hostname && date"

# 预期输出：Server 21 的主机名和当前时间
```

### 4.2 测试 rsync 传输

```bash
# 创建测试文件
echo "test data" > /tmp/test_file.txt

# 测试 rsync 到 Server 21
rsync -avz -e "ssh -i ~/.ssh/id_rsa" \
    /tmp/test_file.txt \
    username@21.xxxxxx.com:/tmp/fuxictr_staging/

# 验证文件已传输
ssh username@21.xxxxxx.com "ls -lh /tmp/fuxictr_staging/test_file.txt"
```

### 4.3 测试目录权限

```bash
# 检查所有目录是否存在且有写权限
dirs=(
    "/data/fuxictr/data"
    "/data/fuxictr/processed_data"
    "/data/fuxictr/workflow_datasets"
    "/data/fuxictr/workflow_processed"
    "/data/fuxictr/workflow_models"
    "/data/fuxictr/workflow_logs"
    "/opt/fuxictr/model_zoo"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir exists"
        if [ -w "$dir" ]; then
            echo "  └─ writable"
        else
            echo "  └─ ✗ NOT writable"
        fi
    else
        echo "✗ $dir does NOT exist"
    fi
done
```

### 4.4 测试 Python 环境

```bash
# 激活虚拟环境
source /opt/fuxictr_venv/bin/activate

# 测试 PyTorch GPU
python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

# 测试 FuxiCTR 导入
python << 'EOF'
import fuxictr
print(f"FuxiCTR version: {fuxictr.__version__}")
EOF

# 测试依赖包
python << 'EOF'
import fastapi, streamlit, pyarrow, pandas, yaml
print("All dependencies installed successfully!")
EOF
```

---

## 启动服务

### 5.1 启动 Workflow 后端

```bash
cd /opt/fuxictr
source /opt/fuxictr_venv/bin/activate

# 设置配置文件路径
export WORKFLOW_CONFIG_PATH=/opt/fuxictr/fuxictr/workflow/config.yaml

# 启动后端服务（端口 8001）
nohup python -m fuxictr.workflow.service \
    > /data/fuxictr/workflow_logs/service.log 2>&1 &

# 查看日志
tail -f /data/fuxictr/workflow_logs/service.log

# 验证服务运行
curl http://localhost:8001/api/health
```

### 5.2 启动 Dashboard 前端

```bash
cd /opt/fuxictr
source /opt/fuxictr_venv/bin/activate

# 启动 Streamlit Dashboard（端口 8501）
nohup streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    > /opt/fuxictr/dashboard/logs/streamlit.log 2>&1 &

# 查看日志
tail -f /opt/fuxictr/dashboard/logs/streamlit.log

# 验证服务运行
curl http://localhost:8501
```

### 5.3 使用 systemd 管理（推荐）

**创建 workflow 服务** (`/etc/systemd/system/fuxictr-workflow.service`):

```ini
[Unit]
Description=FuxiCTR Workflow Service
After=network.target

[Service]
Type=simple
User=your_username
Group=your_username
WorkingDirectory=/opt/fuxictr
Environment="PATH=/opt/fuxictr_venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="WORKFLOW_CONFIG_PATH=/opt/fuxictr/fuxictr/workflow/config.yaml"
ExecStart=/opt/fuxictr_venv/bin/python -m fuxictr.workflow.service
Restart=always
RestartSec=10
StandardOutput=append:/data/fuxictr/workflow_logs/service.log
StandardError=append:/data/fuxictr/workflow_logs/service.log

[Install]
WantedBy=multi-user.target
```

**创建 dashboard 服务** (`/etc/systemd/system/fuxictr-dashboard.service`):

```ini
[Unit]
Description=FuxiCTR Dashboard
After=network.target fuxictr-workflow.service

[Service]
Type=simple
User=your_username
Group=your_username
WorkingDirectory=/opt/fuxictr
Environment="PATH=/opt/fuxictr_venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/opt/fuxictr_venv/bin/streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false
Restart=always
RestartSec=10
StandardOutput=append:/opt/fuxictr/dashboard/logs/streamlit.log
StandardError=append:/opt/fuxictr/dashboard/logs/streamlit.log

[Install]
WantedBy=multi-user.target
```

**启用并启动服务**:

```bash
# 重新加载 systemd 配置
sudo systemctl daemon-reload

# 启用服务（开机自启）
sudo systemctl enable fuxictr-workflow
sudo systemctl enable fuxictr-dashboard

# 启动服务
sudo systemctl start fuxictr-workflow
sudo systemctl start fuxictr-dashboard

# 查看服务状态
sudo systemctl status fuxictr-workflow
sudo systemctl status fuxictr-dashboard

# 查看日志
sudo journalctl -u fuxictr-workflow -f
sudo journalctl -u fuxictr-dashboard -f
```

---

## 常见问题

### Q1: SSH 连接失败

**问题**: `Permission denied (publickey)`

**解决**:
```bash
# 检查密钥是否存在
ls -la ~/.ssh/id_rsa*

# 如果不存在，生成密钥
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# 复制公钥到 Server 21
ssh-copy-id -i ~/.ssh/id_rsa.pub username@21.xxxxxx.com

# 测试连接
ssh -i ~/.ssh/id_rsa username@21.xxxxxx.com
```

### Q2: 目录权限问题

**问题**: `Permission denied` when writing to directories

**解决**:
```bash
# 修改目录所有者
sudo chown -R $USER:$USER /data/fuxictr

# 修改目录权限
sudo chmod -R 755 /data/fuxictr
```

### Q3: PyTorch CUDA 不可用

**问题**: `CUDA available: False`

**解决**:
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 重新安装 PyTorch（匹配 CUDA 版本）
pip uninstall torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

### Q4: 服务启动失败

**问题**: 服务无法启动或立即退出

**解决**:
```bash
# 查看详细日志
tail -100 /data/fuxictr/workflow_logs/service.log

# 手动启动查看错误
cd /opt/fuxictr
source /opt/fuxictr_venv/bin/activate
export WORKFLOW_CONFIG_PATH=/opt/fuxictr/fuxictr/workflow/config.yaml
python -m fuxictr.workflow.service
```

### Q5: Dashboard 和 Workflow 数据混淆

**问题**: 不确定数据应该放在哪个目录

**解决**: 参考 [路径对照表](#路径对照表)

- **Dashboard 手动训练**: 原始数据放 `data/`，处理后放 `processed_data/`
- **Workflow 自动流程**: 原始数据自动从 Server 21 获取，放 `workflow_datasets/`，处理后放 `workflow_processed/`

---

## 附录

### A. 配置文件模板

完整的 `config.yaml` 模板文件，请参考：`/opt/fuxictr/fuxictr/workflow/config.yaml`

### B. 监控和维护

```bash
# 查看磁盘使用
df -h /data/fuxictr

# 清理旧的 Workflow 日志（保留最近 30 天）
find /data/fuxictr/workflow_logs -name "*.log" -mtime +30 -delete

# 清理旧的数据库备份（保留最近 30 天）
find /data/fuxictr/db_backup -name "*.db.bak" -mtime +30 -delete

# 查看服务资源占用
ps aux | grep fuxictr
```

### C. 联系方式

如有问题，请联系：
- 开发人员: [your-name]
- 技术支持: [support-email]
- 文档: [documentation-url]

---

**文档版本**: v2.0
**最后更新**: 2026-02-02
**维护者**: FuxiCTR Team
