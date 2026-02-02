# FuxiCTR Workflow 双服务器部署指南

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              整体架构                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐         ┌──────────────────────────────────────┐ │
│  │   Server 21          │         │   Server 142 (主服务器)               │ │
│  │   数据源服务器        │         │   FuxiCTR 框架部署                    │ │
│  │                      │         │                                      │ │
│  │  - Hive/Spark        │         │  - FuxiCTR 框架                     │ │
│  │  - HDFS存储          │  SSH    │  - Workflow Service (FastAPI)       │ │
│  │  - 原始数据          │ ────────│  - Dashboard (Streamlit)            │ │
│  │  - SQL执行           │  rsync  │  - 训练 + 推理                       │ │
│  │  - SSH服务           │         │  - GPU 训练环境                      │ │
│  └──────────────────────┘         │  - 数据处理 (build_dataset)         │ │
│                                   │  - SQLite 任务数据库                 │ │
│                                   └──────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

数据流向:
Server 21 (SQL执行) → Parquet文件 → SSH/rsync传输 → Server 142 → build_dataset → 训练/推理

前端 → Dashboard → Workflow API → 协调器 → 执行器阶段 → 数据库
         ↓                    ↓          ↓            ↓         ↓
      Streamlit           FastAPI   Orchestrator   5个Stage    SQLite
```

---

## 工作流程执行详解

### 完整执行流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         前端操作 (Dashboard)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 用户访问 http://server_142:8501                                         │
│  2. 点击 "全流程管理" → "+ 新建"                                            │
│  3. 填写任务配置:                                                           │
│     - 用户名 (user): yeshao                                                 │
│     - 模型 (model): MMoE                                                    │
│     - 实验ID (experiment_id): MMoE_default                                  │
│     - 样本SQL (sample_sql): SELECT ... FROM your_table                     │
│     - 推理SQL (infer_sql): SELECT ... FROM your_infer_table                 │
│     - HDFS路径 (hdfs_path): /data/your/path                                 │
│     - Hive表 (hive_table): your_db.result_table                             │
│  4. 点击 "保存并运行"                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Workflow API (FastAPI)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  POST /api/workflow/tasks                                                   │
│                                                                             │
│  1. 接收请求，创建任务记录                                                   │
│  2. 生成 task_id                                                           │
│  3. 创建 WorkflowLogger 并注册广播回调                                      │
│  4. 在后台启动工作流执行                                                     │
│  5. 立即返回 task_id 给前端                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Workflow Orchestrator (协调器)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  提交任务 → 执行5个Stage → 监控进度 → 错误重试 → 完成                        │
│                                                                             │
│  WebSocket连接: ws://server_142:8001/api/workflow/tasks/{task_id}/logs     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        5个执行阶段 (Executors)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Stage 1: data_fetch (DataFetchExecutor)                                    │
│    ├── SSH到Server 21                                                       │
│    ├── 执行Spark SQL导出Parquet                                              │
│    ├── rsync传输到Server 142的staging目录                                   │
│    ├── 自动特征检测 (_tag=categorical, _cnt=numeric, _textlist=sequence)    │
│    └── 运行 build_dataset 生成 parquet切片文件 (大数据存储格式)              │
│                                                                             │
│  Stage 2: train (TrainingExecutor)                                          │
│    ├── 读取 data_fetch checkpoint 获取处理后的数据路径                       │
│    ├── 合并配置: model_config.yaml + 自动检测的特征                          │
│    ├── 生成临时配置: {experiment_id}_task{task_id}.yaml                     │
│    ├── 运行 run_expid.py 启动训练                                          │
│    ├── 实时解析日志: epoch, loss, auc                                      │
│    └── 保存模型checkpoint                                                   │
│                                                                             │
│  Stage 3: infer (InferenceExecutor)                                         │
│    ├── 加载训练好的模型                                                     │
│    ├── 读取推理数据 (从data_fetch阶段的infer_data路径)                       │
│    ├── 运行分布式推理                                                       │
│    └── 保存推理结果到parquet                                                │
│                                                                             │
│  Stage 4: transport (TransportExecutor)                                     │
│    ├── rsync传输推理结果回Server 21                                         │
│    └── 执行Hive LOAD DATA命令                                               │
│                                                                             │
│  Stage 5: monitor (MonitorExecutor)                                        │
│    └── 生成最终报告和汇总指标                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          前端实时监控 (WebSocket)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  前端建立WebSocket连接 → 接收实时日志 → 显示进度和指标                       │
│                                                                             │
│  日志类型:                                                                  │
│    - log: 普通日志消息                                                      │
│    - progress: 进度更新 (current/total/percent)                             │
│    - metric: 指标更新 (loss, auc)                                           │
│    - error: 错误消息                                                        │
│    - complete: 阶段完成                                                     │
│    - status: 状态更新                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 配置文件查找和合并逻辑

```
用户在Dashboard中选择:
├── 用户名: yeshao
├── 模型: MMoE
└── Experiment ID: MMoE_default

↓ config_merge.py 查找配置 (find_original_config)

优先级1: 用户个人配置
├── dashboard/user_configs/yeshao/MMoE/config/model_config.yaml
├── dashboard/user_configs/yeshao/multitask/MMoE/config/model_config.yaml
├── ../../dashboard/user_configs/yeshao/MMoE/config/model_config.yaml
├── ../../dashboard/user_configs/yeshao/multitask/MMoE/config/model_config.yaml
├── /opt/fuxictr/dashboard/user_configs/yeshao/MMoE/config/model_config.yaml
└── /opt/fuxictr/dashboard/user_configs/yeshao/multitask/MMoE/config/model_config.yaml

优先级2: 模型默认配置 (仅当用户配置不存在时)
├── model_zoo/multitask/MMoE/config/model_config.yaml
├── ../../model_zoo/multitask/MMoE/config/model_config.yaml
├── /opt/fuxictr/model_zoo/multitask/MMoE/config/model_config.yaml
└── ...

↓ 找到配置后

1. 加载 model_config.yaml，获取 "MMoE_default" 实验配置
2. 如果存在 "Base" 配置，先合并 (Base + MMoE_default)
3. 保留所有模型和训练参数 (embedding_dim, batch_size, epochs, loss_weight等)

↓ data_fetch 阶段完成后

4. 从 data_fetch checkpoint 获取:
   ├── dataset_id: {experiment_id}.{timestamp} (例如: MMoE_default.20240123_143052)
   ├── dataset_dir: /data/fuxictr/datasets/{dataset_id}/
   ├── train_data: {dataset_dir}/processed/train*.parquet (切片文件)
   ├── valid_data: {dataset_dir}/processed/valid*.parquet (切片文件)
   ├── test_data: {dataset_dir}/processed/test*.parquet (切片文件)
   ├── infer_data: {dataset_dir}/raw/infer/ (原始推理数据)
   ├── feature_cols: 自动检测的特征列表 (基于列名: _tag, _cnt, _textlist)
   └── label_col: 用户配置的标签列 (从 dataset_config.yaml 读取)

↓ 生成合并配置 (prepare_training_config)

5. 创建临时配置文件:
   ├── 路径: fuxictr/workflow/config/{experiment_id}_task{task_id}.yaml
   ├── 内容: 原始配置 + 新的数据路径和特征
   └── 仅替换: data_root, train_data, valid_data, test_data, feature_cols, label_col

↓ 训练阶段

6. 运行: run_expid.py --config config/ --expid {experiment_id}_task{task_id}
   └── 使用合并后的配置进行训练
```

**重要说明**:
- **feature_cols**: 由workflow自动检测 (根据列名后缀: _tag, _cnt, _textlist)
- **label_col**: 由用户在dataset_config.yaml中预配置 (不自动检测)
- **dataset_id**: 格式为 `{experiment_id}.{timestamp}`，确保每次运行唯一
- **目录结构**: 使用标准化结构 `datasets_root/{exp_id.dataset_id}/raw|processed|inference_output/`

### HDFS和Hive表路径配置

**在Dashboard创建任务时配置**:

```yaml
# HDFS路径 - 用于指定HDFS上的数据位置
hdfs_path: "/data/your_project/raw_data"

# Hive表 - 用于指定推理结果写入的目标表
hive_table: "your_database.your_result_table"

# 样本数据SQL - 从Hive/HDFS提取训练/验证/测试数据
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

# 推理数据SQL - 从Hive/HDFS提取待推理数据
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
```

**路径使用说明**:

| 配置项 | 用途 | 示例 |
|--------|------|------|
| `hdfs_path` | HDFS上的原始数据路径 | `/data/project/raw/` |
| `sample_sql` | 训练数据提取SQL | `SELECT * FROM db.table LIMIT 1M` |
| `infer_sql` | 推理数据提取SQL | `SELECT * FROM db.infer_table` |
| `hive_table` | 推理结果写入表 | `db.result_table` |

**数据流转**:
```
1. Server 21: Spark SQL → sample_sql → Parquet → /tmp/fuxictr_staging/{dataset_id}/train/
2. SSH/rsync → Server 142: /data/fuxictr/datasets/{dataset_id}/raw/train/
3. build_dataset → Server 142: /data/fuxictr/datasets/{dataset_id}/processed/train*.parquet (切片)

4. 训练阶段 (train):
   ├── 读取: /data/fuxictr/datasets/{dataset_id}/processed/train*.parquet
   ├── 模型保存: model_zoo/{model}/checkpoints/{dataset_id}/{experiment_id}.model
   └── checkpoint: /data/fuxictr/datasets/{dataset_id}/

5. 推理阶段 (infer):
   ├── 读取推理数据: /data/fuxictr/datasets/{dataset_id}/raw/infer/*.parquet
   ├── 加载模型: model_zoo/{model}/checkpoints/{dataset_id}/{experiment_id}.model
   └── 输出结果: /data/fuxictr/datasets/{dataset_id}/inference_output/*.parquet

6. 传输阶段 (transport):
   ├── rsync → Server 21: /tmp/staging/{dataset_id}/
   └── Hive LOAD DATA → {hive_table}
```

**推理结果详细说明**:

| 项目 | 路径/格式 | 说明 |
|------|----------|------|
| **输出路径** | `/data/fuxictr/datasets/{dataset_id}/inference_output/` | 推理结果目录 |
| **文件格式** | `*.parquet` | Parquet文件 |
| **内容** | 原始特征 + 预测结果 | 包含输入特征和预测列 |
| **最终存储** | `{hive_table}` | 通过LOAD DATA写入Hive表 |

**推理结果示例**:
```
/data/fuxictr/datasets/MMoE_default.20240123_143052/inference_output/
├── part-00000.parquet
├── part-00001.parquet
└── part-00002.parquet

parquet文件包含列:
- user_id, product_id, ...  (原始特征列)
- prediction_0 (任务1预测: 如CTR概率)
- prediction_1 (任务2预测: 如CVR概率)
```

**目录结构总览**:
```
/data/fuxictr/
├── datasets/                        # Workflow生成的标准化数据目录
│   └── {exp_id}.{timestamp}/       # 例如: MMoE_default.20240123_143052
│       ├── raw/                     # 从Server 21传输过来的原始Parquet
│       │   ├── train/              # 训练集原始数据
│       │   └── infer/              # 推理集原始数据
│       ├── processed/              # build_dataset生成的Parquet切片
│       │   ├── train*.parquet     # 训练集
│       │   ├── valid*.parquet     # 验证集
│       │   ├── test*.parquet      # 测试集
│       │   └── feature_map.json   # 特征映射文件
│       └── inference_output/      # 推理结果
│           └── *.parquet          # 推理结果文件
│
├── data/                           # App使用的用户手动上传数据目录
│   └── raw/                       # 用户拖拽上传的原始文件
│
└── model_zoo/                      # 模型仓库 (App和Workflow共用)
    └── {model_name}/              # 例如: multitask/MMoE
        ├── config/                # 模型配置文件
        └── checkpoints/           # 训练checkpoint目录
            └── {exp_id}.{timestamp}/  # 按exp_id.dataset_id组织
                ├── *.model        # 模型文件
                ├── events.out.tfevents.*  # TensorBoard日志
                └── log/           # 训练日志
```

---

### App与Workflow路径差异说明

**App** 和 **Workflow** 使用不同的目录结构：

| 项目 | App路径 | Workflow路径 |
|------|--------|--------|
| **训练数据** | `data/{dataset_id}/train*.parquet` | `datasets/{exp_id.dataset_id}/processed/train*.parquet` |
| **模型存储** | `model_zoo/{model}/checkpoints/{dataset_id}/` | `model_zoo/{model}/checkpoints/{exp_id.dataset_id}/` |
| **推理结果** | `data/{dataset_id}/{expid}_inference_result/` | `datasets/{exp_id.dataset_id}/inference_output/` |

**注意**: `run_expid.py` 通过检测 `data_root` 是否包含 `/processed/` 来自动区分使用哪种路径结构。

---

## 第一部分：Server 21 配置

### 1.1 创建目录结构

```bash
# SSH登录到Server 21
ssh username@21.xxxxxx.com

# 创建staging目录
mkdir -p /tmp/fuxictr_staging
chmod 755 /tmp/fuxictr_staging
```

### 1.2 验证Hive/Spark环境

```bash
# 检查spark-sql是否可用
which spark-sql
spark-sql --version

# 检查Hive环境
hive --version

# 测试SQL执行（可选）
spark-sql -e "SELECT 1"
```

### 1.3 配置SSH访问（从Server 142到Server 21）

```bash
# 在Server 142上执行：
# 1. 生成SSH密钥对（如果还没有）
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# 2. 将公钥复制到Server 21
ssh-copy-id username@21.xxxxxx.com

# 3. 测试无密码登录
ssh username@21.xxxxxx.com
```

### 1.4 验证数据访问

```bash
# 确认HDFS路径可访问
hdfs dfs -ls /your/hdfs/path

# 确认Hive表可查询
spark-sql -e "SELECT COUNT(*) FROM your_database.your_table LIMIT 1"
```

---

## 第二部分：Server 142 配置（主服务器）

### 2.1 系统依赖

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
    python3-dev

# 安装NVIDIA驱动（如果还没有）
sudo apt-get install -y nvidia-driver-535

# 验证GPU
nvidia-smi
```

### 2.2 安装CUDA和cuDNN

```bash
# CUDA 11.8（示例，根据实际版本调整）
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 添加到PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证CUDA
nvcc --version
```

### 2.3 安装PyTorch（GPU版本）

```bash
# 创建虚拟环境
python3.10 -m venv /opt/fuxictr_venv
source /opt/fuxictr_venv/bin/activate

# 安装PyTorch（CUDA 11.8）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 验证GPU可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 2.4 安装FuxiCTR框架

```bash
# 克隆代码仓库
cd /opt
git clone https://github.com/your-org/fuxictr.git
cd fuxictr

# 安装依赖
pip install -r requirements.txt

# 安装FuxiCTR
pip install -e .

# 验证安装
python -c "import fuxictr; print(fuxictr.__version__)"
```

### 2.5 安装Workflow依赖

```bash
cd /opt/fuxictr

# 安装workflow相关依赖
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
    python-multipart

# 安装前端依赖
pip install \
    streamlit-extras \
    plotly \
    altair
```

### 2.6 创建目录结构

```bash
# 创建数据目录 - 新的标准化结构
sudo mkdir -p /data/fuxictr/{datasets,data,model_zoo}
sudo mkdir -p /data/fuxictr/data/raw
sudo mkdir -p /data/fuxictr/datasets
sudo mkdir -p /data/fuxictr/db_backup
sudo chown -R $USER:$USER /data/fuxictr

# 目录结构说明
ls -la /data/fuxictr/
```

**新目录结构说明**:
```
/data/fuxictr/
├── datasets/                        # Workflow标准化数据目录
│   └── {exp_id}.{timestamp}/       # 自动生成的数据目录
│       ├── raw/                     # 原始数据
│       ├── processed/              # 处理后的切片
│       └── inference_output/      # 推理结果
│
├── data/                           # App手动上传数据目录
│   └── raw/                       # 用户拖拽的原始文件
│
├── model_zoo/                      # 模型仓库（App和Workflow共用）
│   └── {model}/checkpoints/{exp_id}.{timestamp}/
│
└── db_backup/                      # 数据库备份
```

### 2.7 配置workflow配置文件

```bash
cd /opt/fuxictr/fuxictr/workflow

# 编辑配置文件
vim config.yaml
```

**config.yaml 配置示例**:
```yaml
# 服务器配置
servers:
  server_21:
    host: "21.xxxxxx.com"           # TODO: 替换为实际主机名
    port: 22
    username: "your_username"        # TODO: SSH用户名
    key_path: "~/.ssh/id_rsa"       # TODO: SSH私钥路径

# 存储路径配置
storage:
  server_21_staging: "/tmp/fuxictr_staging"  # Server 21上的临时目录
  staging_dir: "/data/fuxictr/staging"       # 已废弃，使用datasets_root
  checkpoint_dir: "/data/fuxictr/checkpoints"

# FuxiCTR 框架路径配置
fuxictr_paths:
  # App使用的数据目录（用户手动上传）
  data_root: "../../../data/"

  # Workflow使用的数据目录（标准化结构）
  datasets_root: "/data/fuxictr/datasets"

  # 统一的模型保存目录
  model_root: "../../../model_zoo"

# 数据传输配置
transfer:
  chunk_size: 104857600              # 100MB
  max_retries: 10
  compression: true
  verify_checksum: true
  parallel_workers: 4
  timeout: 300

# Workflow任务配置
workflow:
  heartbeat_interval: 30
  log_rotation_size: 104857600       # 100MB
  task_timeout: 86400                # 24小时

# 数据库配置
database:
  path: "workflow_tasks.db"
  backup_dir: "/data/fuxictr/db_backup"

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "workflow.log"
```

### 2.8 测试SSH连接到Server 21

```bash
# 测试SSH连接
ssh -i ~/.ssh/id_rsa username@21.xxxxxx.com "echo 'Connection successful'"

# 测试rsync
echo "test" > /tmp/test.txt
rsync -avz -e "ssh -i ~/.ssh/id_rsa" /tmp/test.txt username@21.xxxxxx.com:/tmp/
```

---

## 第三部分：FuxiCTR 框架配置

### 3.1 配置文件位置和优先级

FuxiCTR支持两种配置来源：

1. **用户个人配置** (优先级最高): `dashboard/user_configs/{user}/{model}/config/{experiment_id}.yaml`
2. **模型默认配置** (回退): `model_zoo/{model}/config/{experiment_id}.yaml`

### 3.2 创建用户配置目录结构

```bash
cd /opt/fuxictr

# 创建用户配置目录
mkdir -p dashboard/user_configs/yeshao/MMoE/config

# 复制模型默认配置作为模板
cp model_zoo/multitask/MMoE/config/*.yaml dashboard/user_configs/yeshao/MMoE/config/

# 目录结构
dashboard/
├── user_configs/
│   └── yeshao/                    # 用户名
│       └── MMoE/                  # 模型名
│           └── config/            # 配置目录
│               ├── mmoe_exp_001.yaml
│               ├── mmoe_exp_002.yaml
│               └── model_config.yaml
```

### 3.3 FuxiCTR配置文件结构

FuxiCTR使用**两个配置文件**来管理模型和数据：

```
dashboard/user_configs/{user}/{model}/config/
├── model_config.yaml       # 模型参数 + 训练参数 ⭐
└── dataset_config.yaml      # 数据路径 + 特征定义 ⭐
```

**关键点**：
- **model_config.yaml**: 定义模型结构、超参数、训练配置（用户在dashboard中配置）
- **dataset_config.yaml**: 定义数据路径和特征（会被workflow自动替换）

---

### 3.4 model_config.yaml 示例

**这是用户在Dashboard中配置的主要文件**，包含模型和训练参数：

```yaml
# =========================================================================
# model_config.yaml - 模型参数 + 训练参数
# =========================================================================

### Base: 所有实验的默认配置（会被继承）
Base:
    model_root: './checkpoints/'
    num_workers: 0
    verbose: 1
    early_stop_patience: 2
    pickle_feature_encoder: True
    save_best_only: True
    eval_steps: null
    debug_mode: False

### MMoE_default: 默认实验配置
MMoE_default:
    model: MMoE
    dataset_id: sample_dataset_001  # ⚠️ 会被workflow中的experiment_id替换

    # ========== 模型参数 ==========
    num_tasks: 2                   # 任务数量
    num_experts: 8                 # Expert数量
    expert_hidden_units: [512,256,128]  # Expert隐藏层
    gate_hidden_units: [128, 64]   # Gate隐藏层
    tower_hidden_units: [128, 64]  # Tower隐藏层
    hidden_activations: relu       # 激活函数
    embedding_dim: 64              # Embedding维度
    net_dropout: 0.1               # Dropout率
    embedding_regularizer: 1.e-6    # Embedding正则化
    batch_norm: False              # 是否使用BatchNorm

    # ========== 训练参数 ==========
    loss: ['binary_crossentropy','binary_crossentropy']  # 损失函数
    metrics: ['logloss', 'AUC']    # 评估指标
    task: ['binary_classification','binary_classification']
    optimizer: adam                # 优化器
    learning_rate: 1.e-3           # 学习率 ⭐
    batch_size: 1024               # Batch大小 ⭐
    epochs: 10                     # 训练轮数 ⭐
    shuffle: True
    seed: 2024
    monitor: 'AUC'                 # 监控指标
    monitor_mode: 'max'

    # ========== 多任务损失平衡（可选）==========
    # 方式1: Uncertainty Weighting (UW) - 自动学习损失权重
    # loss_weight: 'UW'

    # 方式2: GradNorm (GN) - 基于梯度的损失平衡
    # loss_weight: 'GN'
    # gradnorm_alpha: 1.5           # GradNorm超参数

    # 方式3: 手动指定权重
    # loss_weight: [0.4, 0.6]        # 任务1和任务2的权重

### MMoE_with_UW: 使用不确定性加权
MMoE_with_UW:
    model: MMoE
    dataset_id: sample_dataset_001

    # 模型和训练参数同上
    num_experts: 8
    embedding_dim: 64
    batch_size: 1024
    epochs: 10
    learning_rate: 1.e-3

    # ========== 启用不确定性加权 ==========
    loss: ['binary_crossentropy','binary_crossentropy']
    metrics: ['logloss', 'AUC']
    task: ['binary_classification','binary_classification']
    optimizer: adam

    # 自动学习任务权重（推荐用于多任务不平衡场景）
    loss_weight: 'UW'  # ⭐ Uncertainty Weighting

### MMoE_with_GradNorm: 使用GradNorm
MMoE_with_GradNorm:
    model: MMoE
    dataset_id: sample_dataset_001

    num_experts: 8
    embedding_dim: 64
    batch_size: 1024
    epochs: 10
    learning_rate: 1.e-3

    loss: ['binary_crossentropy','binary_crossentropy']
    metrics: ['logloss', 'AUC']
    task: ['binary_classification','binary_classification']
    optimizer: adam

    # ========== 启用GradNorm ==========
    loss_weight: 'GN'  # ⭐ GradNorm
    gradnorm_alpha: 1.5  # GradNorm超参数（默认1.5）

### MMoE_with_hash_embedding: 使用Hash Embedding
MMoE_with_hash:
    model: MMoE
    dataset_id: sample_dataset_001

    num_experts: 8
    embedding_dim: 64
    batch_size: 1024
    epochs: 10
    learning_rate: 1.e-3

    loss: ['binary_crossentropy','binary_crossentropy']
    metrics: ['logloss', 'AUC']
    task: ['binary_classification','binary_classification']
    optimizer: adam

    # ========== Hash Embedding配置 ==========
    # 对于高基数特征（如user_id、product_id），使用hash embedding可以减少内存
    feature_encoder: "hash"  # ⭐ 使用hash编码器
    hash_bits: 24              # hash空间大小（2^24个桶）

### MMoE_large_batch: 大batch实验
MMoE_large_batch:
    # 继承MMoE_default的配置
    # 只覆盖需要修改的参数
    batch_size: 2048               # 更大的batch
    learning_rate: 2.e-3           # 相应提高学习率
    embedding_dim: 128              # 更大的embedding

### MMoE_more_experts: 更多Expert
MMoE_more_experts:
    num_experts: 16                # 更多expert
    expert_hidden_units: [1024,512,256]
    gate_hidden_units: [256, 128]

### MMoE_4gpu: 4卡训练配置
MMoE_4gpu:
    # GPU配置也在这里设置
    batch_size: 2048               # 4卡可以用更大batch
    learning_rate: 2.e-3
    embedding_dim: 128
```

**⚠️ 重要**:
- **所有训练参数**（GPU、batch_size、epochs、learning_rate等）都在 `model_config.yaml` 中配置
- **Workflow的 `config.yaml`** 只包含服务器连接、存储路径、传输配置等基础设施参数
- 用户在Dashboard中选择不同的 Experiment ID（如 `MMoE_default`、`MMoE_4gpu`、`MMoE_with_UW`）来使用不同的训练配置

---

### 3.5 高级配置详解

#### 3.5.1 多任务损失权重配置

多任务学习需要平衡不同任务的损失，FuxiCTR支持三种方式：

**方式1: Uncertainty Weighting (UW)** ⭐ 推荐
```yaml
MMoE_with_UW:
    loss_weight: 'UW'  # 自动学习不确定性权重
    # 优点：自适应，无需手动调参
    # 适用：任务权重不确定的场景
```

**方式2: GradNorm (GN)**
```yaml
MMoE_with_GN:
    loss_weight: 'GN'           # 基于梯度的归一化
    gradnorm_alpha: 1.5         # 超参数（默认1.5）
    # 优点：考虑梯度大小，平衡学习速度
    # 适用：需要更精细控制的任务平衡
```

**方式3: 手动权重**
```yaml
MMoE_manual:
    loss_weight: [0.4, 0.6]    # 手动指定各任务权重
    # 适用：已明确知道最优权重的场景
```

#### 3.5.2 Hash Embedding 配置

对于高基数特征（如user_id可能有数百万个不同值），使用hash embedding可以显著减少内存：

```yaml
MMoE_with_hash:
    feature_encoder: "hash"    # 启用hash编码器
    hash_bits: 24               # hash空间（2^24 = 16M个桶）

    # 特征列仍需配置，但编码方式会使用hash
    feature_cols:
      - name: [user_id_hash]     # 高基数特征
        type: categorical
        dtype: str
        active: True
        embedding_size: 64       # 较小的embedding即可
```

**hash_bits 选择建议**:
- `hash_bits=16`: 65536个桶（内存最小，适合小规模）
- `hash_bits=24`: 16M个桶（推荐，平衡内存和冲突）
- `hash_bits=32`: 4G个桶（大规模数据，内存充足）

#### 3.5.3 完整配置示例对比

```yaml
# ============ 基础配置 ============
MMoE_basic:
    num_experts: 8
    embedding_dim: 64
    batch_size: 1024
    epochs: 10
    learning_rate: 1.e-3
    loss: ['binary_crossentropy','binary_crossentropy']
    # 使用默认等权重（两个任务权重相同）

# ============ UW配置 ============
MMoE_uw:
    num_experts: 8
    embedding_dim: 64
    batch_size: 1024
    epochs: 10
    learning_rate: 1.e-3
    loss: ['binary_crossentropy','binary_crossentropy']
    loss_weight: 'UW'          # ⭐ 自动学习权重

# ============ GN配置 ============
MMoE_gn:
    num_experts: 8
    embedding_dim: 64
    batch_size: 1024
    epochs: 10
    learning_rate: 1.e-3
    loss: ['binary_crossentropy','binary_crossentropy']
    loss_weight: 'GN'          # ⭐ GradNorm
    gradnorm_alpha: 1.5

# ============ Hash + UW配置 ============
MMoE_hash_uw:
    num_experts: 8
    embedding_dim: 32            # hash可以用更小embedding
    batch_size: 2048             # 节省内存可以用更大batch
    epochs: 10
    learning_rate: 2.e-3
    loss: ['binary_crossentropy','binary_crossentropy']
    loss_weight: 'UW'
    feature_encoder: "hash"     # ⭐ Hash编码
    hash_bits: 24

# ============ 4卡 + GN配置 ============
MMoE_4gpu_gn:
    num_experts: 16              # 4卡可以用更多expert
    embedding_dim: 128
    batch_size: 4096             # 4卡可以用更大batch
    epochs: 10
    learning_rate: 2.e-3
    loss: ['binary_crossentropy','binary_crossentropy']
    loss_weight: 'GN'
    gradnorm_alpha: 1.5
```

---

### 3.6 配置最佳实践

#### 3.6.1 命名规范

建议使用描述性的Experiment ID名称：

```yaml
# 好的命名
MMoE_default:           # 默认配置
MMoE_uw:                # 使用UW
MMoE_gn:                # 使用GradNorm
MMoE_hash:              # 使用Hash Embedding
MMoE_4gpu:              # 4卡配置
MMoE_large_batch:        # 大batch
MMoE_more_experts:       # 更多Expert

# 不好的命名（不够描述性）
exp1:
test:
config_v2:
```

#### 3.6.2 配置继承

利用Base配置减少重复：

```yaml
Base:
    # 公共配置
    model_root: './checkpoints/'
    early_stop_patience: 2
    save_best_only: True
    optimizer: adam
    learning_rate: 1.e-3
    embedding_dim: 64
    batch_norm: False

MMoE_small:
    # 小模型实验 - 只覆盖不同的参数
    num_experts: 4
    expert_hidden_units: [256,128]

MMoE_large:
    # 大模型实验
    num_experts: 16
    expert_hidden_units: [1024,512,256]
```

#### 3.6.3 参数调优建议

| 场景 | 推荐配置 |
|------|---------|
| **任务不平衡** | `loss_weight: 'UW'` |
| **需要精细控制** | `loss_weight: 'GN', gradnorm_alpha: 1.5` |
| **高基数特征** | `feature_encoder: "hash", hash_bits: 24` |
| **内存不足** | `embedding_dim: 32, batch_size: 512` |
| **多卡训练** | `batch_size: 4096` (4卡) |
| **快速实验** | `epochs: 5, num_experts: 4` |
| **生产训练** | `epochs: 20, early_stop_patience: 3` |

---

### 3.5 dataset_config.yaml 示例

**这个文件会被workflow自动替换**，但需要提供默认模板：

```yaml
# =========================================================================
# dataset_config.yaml - 数据路径 + 特征定义
# =========================================================================

sample_dataset_001:  # ⚠️ dataset_id需要与model_config.yaml中一致
  # ========== 数据路径（会被workflow自动替换）==========
  data_root: ../../../data/
  data_format: parquet
  train_data: ../../../data/sample_dataset_001  # 原始parquet路径
  valid_data: ../../../data/sample_dataset_001
  test_data: ../../../data/sample_dataset_001
  infer_data: ../../../data/sample_dataset_001

  # 数据分割（如果需要）
  split_type: random
  train_size: 0.8
  valid_size: 0.1
  test_size: 0.1

  # ========== 特征定义（会被workflow自动检测替换）==========
  min_categr_count: 1
  feature_cols:
    # 分类特征（以_tag结尾）
    - name:
      - area_tag
      - age_tag
      - gender_tag
      - manufacture_tag
      active: true
      dtype: str
      type: categorical

    # 数值特征（以_cnt结尾）
    - name:
      - call_30_days_max_cnt
      - call_90_days_max_cnt
      - sms_total_click_cnt
      active: true
      dtype: float
      type: numeric
      normalizer: StandardScaler

    # 序列特征（以_textlist结尾或特殊名称）
    - name:
      - appInstalls              # 特殊序列特征
      - outerBizSorted           # 特殊序列特征
      - click_180_days_product_textlist
      active: true
      dtype: list
      type: sequence
      max_len: 15
      encoder: MaskedAveragePooling

  # 标签列
  label_col:
    - name: label_apply          # 任务1的标签
      dtype: float
      threshold: 0.5
    - name: label_credit          # 任务2的标签
      dtype: float
      threshold: 0.3
```

---

### 3.6 Workflow中的配置合并逻辑

**重要**: 理解workflow如何处理这两个配置文件：

```
用户在Dashboard选择:
├── 用户名: yeshao
├── 模型: MMoE
└── Experiment ID: MMoE_default

↓ workflow执行流程

1. 读取原始配置:
   ├── model_config.yaml (MMoE_default)
   │    → 获取: 模型参数 + 训练参数
   └── dataset_config.yaml (sample_dataset_001)
        → 获取: 默认数据路径和特征模板

2. 执行数据获取:
   ├── SQL取数据 → Parquet
   ├── SSH传输到staging目录
   └── 自动特征检测 → 新的feature_cols

3. 配置合并 (config_merge.py):
   ├── 保留: model_config.yaml中的所有参数
   └── 替换:
        ├── data_root → /data/fuxictr/data
        ├── train_data → /data/fuxictr/data/{dataset_id}/train/*.parquet
        ├── valid_data → /data/fuxictr/data/{dataset_id}/valid/*.parquet
        ├── test_data → /data/fuxictr/data/{dataset_id}/test/*.parquet
        ├── feature_cols → 自动检测的特征
        └── label_col → 自动检测的标签

4. 生成merged_config: {experiment_id}_task{task_id}.yaml

5. 启动训练: run_expid.py --expid {experiment_id}_task{task_id}
```

### 3.5 Workflow配置文件（/opt/fuxictr/fuxictr/workflow/config.yaml）

```yaml
# =========================================================================
# FuxiCTR Workflow Configuration - 双服务器部署方案
# =========================================================================

servers:
  server_21:
    host: "21.xxxxxx.com"           # ⚠️ 替换为实际值
    port: 22
    username: "your_username"        # ⚠️ 替换为实际值
    key_path: "~/.ssh/id_rsa"        # ⚠️ 替换为实际值

storage:
  server_21_staging: "/tmp/fuxictr_staging"
  staging_dir: "/data/fuxictr/staging"
  checkpoint_dir: "/data/fuxictr/checkpoints"

fuxictr_paths:
  data_root: "/data/fuxictr/data"
  model_root: "/data/fuxictr/models"

transfer:
  chunk_size: 104857600              # 100MB
  max_retries: 10
  compression: true
  verify_checksum: true
  parallel_workers: 4
  timeout: 300

workflow:
  heartbeat_interval: 30
  log_rotation_size: 104857600
  task_timeout: 86400

training:
  gpus: "0,1,2,3"
  default_batch_size: 1024
  default_epochs: 10

database:
  path: "workflow_tasks.db"
  backup_dir: "/data/fuxictr/db_backup"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "workflow.log"
```

---

## 第四部分：启动服务

### 4.1 启动Workflow后端服务

```bash
cd /opt/fuxictr

# 激活虚拟环境
source /opt/fuxictr_venv/bin/activate

# 设置配置文件路径
export WORKFLOW_CONFIG_PATH=/opt/fuxictr/fuxictr/workflow/config.yaml

# 启动后端服务（端口8001）
nohup python -m fuxictr.workflow.service > logs/workflow.log 2>&1 &

# 或使用uvicorn直接启动
uvicorn fuxictr.workflow.service:app \
    --host 0.0.0.0 \
    --port 8001 \
    --reload \
    --log-level info
```

### 4.2 启动Dashboard前端

```bash
cd /opt/fuxictr

# 启动Streamlit Dashboard（端口8501）
nohup streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    > logs/dashboard.log 2>&1 &
```

### 4.3 使用systemd管理服务（推荐）

**创建workflow服务** (`/etc/systemd/system/fuxictr-workflow.service`):

```ini
[Unit]
Description=FuxiCTR Workflow Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/opt/fuxictr
Environment="PATH=/opt/fuxictr_venv/bin"
Environment="WORKFLOW_CONFIG_PATH=/opt/fuxictr/fuxictr/workflow/config.yaml"
ExecStart=/opt/fuxictr_venv/bin/python -m fuxictr.workflow.service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**创建dashboard服务** (`/etc/systemd/system/fuxictr-dashboard.service`):

```ini
[Unit]
Description=FuxiCTR Dashboard
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/opt/fuxictr
Environment="PATH=/opt/fuxictr_venv/bin"
ExecStart=/opt/fuxictr_venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**启用服务**:

```bash
# 重载systemd配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start fuxictr-workflow
sudo systemctl start fuxictr-dashboard

# 设置开机自启
sudo systemctl enable fuxictr-workflow
sudo systemctl enable fuxictr-dashboard

# 查看状态
sudo systemctl status fuxictr-workflow
sudo systemctl status fuxictr-dashboard
```

---

## 第五部分：验证部署

### 5.1 检查服务状态

```bash
# 检查后端API
curl http://localhost:8001/
# 预期输出: {"name":"FuxiCTR Workflow API","version":"2.0.0",...}

# 检查任务列表
curl http://localhost:8001/api/workflow/tasks

# 检查Dashboard
curl http://localhost:8501
```

### 5.2 测试完整流程

#### 步骤1: 在Dashboard中创建任务

1. 访问 `http://server_142:8501`
2. 点击"全流程管理"
3. 点击"+ 新建"创建任务
4. 填写任务配置：
   - 用户名: `yeshao`
   - 模型: `MMoE`
   - Experiment ID: `mmoe_exp_001`
   - 样本数据SQL:
     ```sql
     SELECT
       user_id,
       product_id,
       category,
       price,
       quantity,
       label
     FROM your_database.your_table
     LIMIT 100000
     ```
   - 推理数据SQL:
     ```sql
     SELECT
       user_id,
       product_id,
       category,
       price,
       quantity
     FROM your_database.your_inference_table
     LIMIT 10000
     ```

#### 步骤2: 启动任务并观察日志

1. 点击"保存并运行"
2. 观察"实时日志"区域，应该看到：
   - `[data_fetch]` 连接Server 21，执行SQL
   - `[data_fetch]` 传输数据
   - `[train]` 开始训练，显示epoch进度
   - `[train]` 实时显示loss和auc指标

### 5.3 检查数据流

```bash
# 检查staging目录
ls -lh /data/fuxictr/staging/

# 检查处理后的数据
ls -lh /data/fuxictr/data/

# 检查模型保存
ls -lh /data/fuxictr/models/

# 检查数据库
sqlite3 /opt/fuxictr/workflow_tasks.db "SELECT * FROM tasks ORDER BY id DESC LIMIT 1;"
```

---

## 第六部分：常见问题排查

### 问题1: SSH连接失败

```bash
# 检查SSH配置
cat ~/.ssh/config

# 手动测试SSH连接
ssh -vvv username@21.xxxxxx.com

# 检查密钥权限
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
```

### 问题2: Workflow服务启动失败

```bash
# 查看日志
tail -f /opt/fuxictr/logs/workflow.log

# 检查端口占用
netstat -tuln | grep 8001

# 检查Python环境
which python
python --version
pip list | grep fuxictr
```

### 问题3: 数据传输失败

```bash
# 检查staging目录权限
ls -la /data/fuxictr/staging/

# 手动测试rsync
rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_rsa" \
  /tmp/test.txt \
  username@21.xxxxxx.com:/tmp/fuxictr_staging/
```

### 问题4: GPU训练失败

```bash
# 检查GPU状态
nvidia-smi

# 检查CUDA
nvcc --version

# 测试PyTorch GPU
python -c "import torch; x = torch.cuda.get_device_properties(0); print(x)"
```

### 问题5: Dashboard无法连接后端

```bash
# 检查API_BASE配置
grep API_BASE dashboard/pages/workflow.py

# 修改为正确的地址
# API_BASE = "http://localhost:8001"  # 或实际IP
```

---

## 附录A: 快速部署脚本

### A.1 Server 142 一键部署脚本

```bash
#!/bin/bash
# deploy_fuxictr.sh

set -e

echo "=========================================="
echo "FuxiCTR Workflow 部署脚本"
echo "=========================================="

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then
    echo "请使用sudo运行此脚本"
    exit 1
fi

# 安装系统依赖
apt-get update
apt-get install -y python3.10 python3.10-venv python3-pip git build-essential

# 创建目录
mkdir -p /data/fuxictr/{staging,data,models,checkpoints,db_backup}
chown -R $SUDO_USER:$SUDO_USER /data/fuxictr

# 安装Python虚拟环境
sudo -u $SUDO_USER python3.10 -m venv /opt/fuxictr_venv
source /opt/fuxictr_venv/bin/activate

# 安装PyTorch
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 安装FuxiCTR（假设代码已克隆）
cd /opt/fuxictr
pip install -e .
pip install -r requirements.txt
pip install fastapi uvicorn streamlit pyarrow pandas

echo "部署完成！"
echo "请手动配置: fuxictr/workflow/config.yaml"
```

### A.2 启动脚本

```bash
#!/bin/bash
# start_all.sh

cd /opt/fuxictr
source /opt/fuxictr_venv/bin/activate

# 创建日志目录
mkdir -p logs

# 启动后端
echo "启动Workflow服务..."
nohup python -m fuxictr.workflow.service > logs/workflow.log 2>&1 &
echo $! > workflow.pid

# 启动前端
echo "启动Dashboard..."
nohup streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    > logs/dashboard.log 2>&1 &
echo $! > dashboard.pid

echo "所有服务已启动"
echo "后端: http://localhost:8001"
echo "前端: http://localhost:8501"
```

---

## 附录B: 环境变量配置

### B.1 ~/.bashrc 添加

```bash
# FuxiCTR 环境变量
export FUXICTR_HOME=/opt/fuxictr
export FUXICTR_VENV=/opt/fuxictr_venv
export WORKFLOW_CONFIG_PATH=/opt/fuxictr/fuxictr/workflow/config.yaml

# CUDA
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 快捷命令
alias fuxictr-start="cd /opt/fuxictr && source start_all.sh"
alias fuxictr-stop="cd /opt/fuxictr && pkill -f 'fuxictr.workflow.service' && pkill -f 'streamlit run dashboard/app.py'"
alias fuxictr-logs="tail -f /opt/fuxictr/logs/workflow.log"
```

---

## 附录C: 监控和日志

### C.1 日志位置

```
/opt/fuxictr/
├── logs/
│   ├── workflow.log       # Workflow服务日志
│   ├── dashboard.log      # Dashboard日志
│   └── tasks/             # 任务日志
│       └── task_123/      # 每个任务的日志目录
├── workflow_tasks.db      # 任务数据库
└── workflow_config.yaml   # 配置文件
```

### C.2 监控命令

```bash
# 查看服务状态
ps aux | grep -E "fuxictr|streamlit"

# 查看GPU使用
watch -n 1 nvidia-smi

# 查看磁盘使用
df -h /data/fuxictr

# 查看网络连接
netstat -tuln | grep -E "8001|8501"

# 实时查看日志
tail -f /opt/fuxictr/logs/workflow.log
```

---

## 部署完成检查清单

### Server 21 - 数据源服务器

**基本配置**
- [ ] SSH服务运行在端口22
- [ ] spark-sql可用 (`spark-sql --version`)
- [ ] hive可用 (`hive --version`)
- [ ] 有权限访问HDFS路径 (`hdfs dfs -ls /your/path`)
- [ ] 有权限查询Hive表

**目录和权限**
- [ ] 创建staging目录: `/tmp/fuxictr_staging`
- [ ] 目录权限755: `chmod 755 /tmp/fuxictr_staging`
- [ ] 磁盘空间充足 (至少100GB可用)

**SSH访问配置**
- [ ] Server 142可以无密码SSH登录到Server 21
- [ ] SSH密钥权限正确: `~/.ssh/id_rsa` (600), `~/.ssh/id_rsa.pub` (644)

---

### Server 142 - 主服务器 (训练服务器)

**系统环境**
- [ ] Python 3.10+已安装
- [ ] NVIDIA驱动已安装: `nvidia-smi`
- [ ] CUDA 11.8+已安装: `nvcc --version`
- [ ] PyTorch GPU版本可用: `python -c "import torch; print(torch.cuda.is_available())"`

**目录结构**
- [ ] `/data/fuxictr/staging` - 数据传输临时目录
- [ ] `/data/fuxictr/data` - 处理后的数据目录
- [ ] `/data/fuxictr/models` - 模型保存目录
- [ ] `/data/fuxictr/checkpoints` - 模型checkpoint目录
- [ ] `/data/fuxictr/db_backup` - 数据库备份目录
- [ ] `/opt/fuxictr` - FuxiCTR代码目录

**FuxiCTR安装**
- [ ] FuxiCTR已安装: `python -c "import fuxictr; print(fuxictr.__version__)"`
- [ ] 依赖已安装: `pip list | grep -E "fastapi|streamlit|pyarrow"`

**Workflow配置**
- [ ] `/opt/fuxictr/fuxictr/workflow/config.yaml` 已配置
- [ ] Server 21连接信息正确:
  ```yaml
  servers:
    server_21:
      host: "21.xxxxxx.com"      # ⚠️ 替换为实际值
      username: "your_username"   # ⚠️ 替换为实际值
      key_path: "~/.ssh/id_rsa"   # ⚠️ 替换为实际值
  ```
- [ ] 路径配置正确:
  ```yaml
  fuxictr_paths:
    data_root: "/data/fuxictr/data"
    model_root: "/data/fuxictr/models"
  storage:
    staging_dir: "/data/fuxictr/staging"
    server_21_staging: "/tmp/fuxictr_staging"
  ```

**用户配置目录**
- [ ] 用户配置目录存在: `dashboard/user_configs/{user}/{model}/config/`
- [ ] model_config.yaml已配置 (包含模型+训练参数)
- [ ] Experiment ID存在 (如: MMoE_default, MMoE_with_UW)

---

### 服务启动检查

**后端服务 (Workflow API)**
- [ ] Workflow服务运行在端口8001: `curl http://localhost:8001/`
- [ ] API返回正确信息:
  ```json
  {"name":"FuxiCTR Workflow API","version":"2.0.0"}
  ```
- [ ] 任务列表接口可用: `curl http://localhost:8001/api/workflow/tasks`

**前端服务 (Dashboard)**
- [ ] Dashboard运行在端口8501: `curl http://localhost:8501`
- [ ] 可以在浏览器访问: `http://server_142:8501`

**WebSocket连接**
- [ ] WebSocket端点可用: `ws://server_142:8001/api/workflow/tasks/{task_id}/logs`

---

### 功能测试检查

**1. 配置文件读取测试**
```bash
# 测试配置读取
cd /opt/fuxictr
python -c "
from fuxictr.workflow.utils.config_merge import find_original_config
path = find_original_config('yeshao', 'MMoE', 'MMoE_default')
print(f'Config found: {path}')
"
```

**2. SSH连接测试**
```bash
# 测试SSH到Server 21
ssh -i ~/.ssh/id_rsa username@21.xxxxxx.com "echo 'Connection OK'"

# 测试rsync
echo "test" > /tmp/test.txt
rsync -avz -e "ssh -i ~/.ssh/id_rsa" /tmp/test.txt \
  username@21.xxxxxx.com:/tmp/fuxictr_staging/
```

**3. 完整流程测试**
通过Dashboard创建测试任务:
1. 访问 `http://server_142:8501`
2. 点击"全流程管理"
3. 填写任务配置:
   - 用户名: your_username
   - 模型: MMoE
   - Experiment ID: MMoE_default
   - sample_sql: 简单的SELECT语句
   - infer_sql: 简单的SELECT语句
4. 点击"保存并运行"
5. 观察"实时日志"区域，应该看到:
   - `[data_fetch]` 连接Server 21
   - `[data_fetch]` 特征检测结果
   - `[train]` 训练开始
   - 实时loss和auc指标

---

### 常见问题排查

**问题1: Workflow服务启动失败**
```bash
# 查看详细日志
tail -f /opt/fuxictr/logs/workflow.log

# 检查配置文件语法
python -c "import yaml; yaml.safe_load(open('/opt/fuxictr/fuxictr/workflow/config.yaml'))"

# 检查端口占用
netstat -tuln | grep 8001
```

**问题2: 配置文件找不到**
```bash
# 检查用户配置目录
ls -la dashboard/user_configs/{user}/{model}/config/

# 检查model_config.yaml是否存在
cat dashboard/user_configs/{user}/{model}/config/model_config.yaml

# 确认Experiment ID存在
grep "^{experiment_id}:" dashboard/user_configs/{user}/{model}/config/model_config.yaml
```

**问题3: 前端无法连接后端**
```bash
# 检查API_BASE配置
grep "API_BASE" dashboard/pages/workflow.py
# 应该是: API_BASE = "http://localhost:8001"

# 测试API连接
curl http://localhost:8001/api/workflow/servers
```

**问题4: 数据传输失败**
```bash
# 检查SSH连接
ssh -i ~/.ssh/id_rsa username@21.xxxxxx.com

# 检查staging目录权限
ls -la /data/fuxictr/staging/
ls -la /tmp/fuxictr_staging/  # Server 21上

# 手动测试rsync
rsync -avz --progress -e "ssh -i ~/.ssh/id_rsa" \
  /tmp/test.txt username@21.xxxxxx.com:/tmp/fuxictr_staging/
```

**问题5: GPU训练失败**
```bash
# 检查GPU状态
nvidia-smi

# 检查CUDA
nvcc --version

# 测试PyTorch GPU
python -c "import torch; print(torch.cuda.device_count())"

# 检查环境变量
echo $CUDA_VISIBLE_DEVICES
```

---

### 两台服务器运行程序对照表

| 组件 | Server 21 | Server 142 |
|------|-----------|------------|
| **Hive/Spark** | ✅ 运行 | ❌ 不需要 |
| **HDFS** | ✅ 存储数据 | ❌ 不需要 |
| **SSH服务** | ✅ 监听端口22 | ❌ 作为客户端连接 |
| **Workflow API** | ❌ 不需要 | ✅ 端口8001 |
| **Dashboard** | ❌ 不需要 | ✅ 端口8501 |
| **FuxiCTR框架** | ❌ 不需要 | ✅ 完整部署 |
| **GPU训练** | ❌ 不需要 | ✅ CUDA+PyTorch |
| **build_dataset** | ❌ 不需要 | ✅ 数据处理 |
| **run_expid.py** | ❌ 不需要 | ✅ 训练/推理 |
| **SQLite数据库** | ❌ 不需要 | ✅ 任务状态 |

---

### 前端监控位置说明

**监控位置**: Dashboard → 全流程管理 → 任务详情 → 实时日志

```
http://server_142:8501
├── 全流程管理 (点击左侧菜单)
│   ├── 任务列表 (显示所有任务)
│   └── 任务详情 (点击任务卡片进入)
│       ├── 基本信息 (状态、时间、配置)
│       ├── 步骤状态 (5个Stage的进度)
│       ├── 实时日志 (WebSocket实时流)
│       │   ├── [data_fetch] - 数据获取和特征检测
│       │   ├── [train] - 训练进度和指标
│       │   ├── [infer] - 推理进度
│       │   ├── [transport] - 数据传输
│       │   └── [monitor] - 最终报告
│       └── 操作按钮 (重试、取消、删除)
```

**日志颜色说明**:
- 🟡 `[INFO]` - 普通信息
- 🟢 `[progress]` - 进度更新
- 🟠 `[metric]` - 指标更新 (loss, auc)
- 🔴 `[ERROR]` - 错误信息
- 🔵 `[complete]` - 阶段完成

---

---

## 总结

完成以上配置后，您将拥有：

1. **Server 21**: 提供Hive/Spark数据源
2. **Server 142**: 运行完整的FuxiCTR框架
   - Workflow后端服务 (端口8001)
   - Dashboard前端界面 (端口8501)
   - GPU训练环境
   - 自动化数据处理和训练流程

现在您可以通过Dashboard界面创建workflow任务，系统将自动完成数据获取、特征处理、模型训练和推理的全流程。

**核心工作流**:
```
Dashboard创建任务 → Workflow API接收 → Orchestrator协调 → 5个Stage执行 → 实时日志推送
     ↓                ↓                   ↓              ↓            ↓
  用户选择配置    生成task_id      按顺序执行      数据获取+训练   WebSocket推送
                  后台执行         checkpoint保
                                 存断点续传
```
