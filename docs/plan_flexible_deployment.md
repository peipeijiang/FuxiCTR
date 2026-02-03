# 灵活部署配置优化计划

## 问题重述

**现状**：
- 部署文档 `docs/workflow_deployment_2026.md` 中的路径是硬编码的假设值
- 例如：`/opt/fuxictr`、`/data/fuxictr`、`/opt/fuxictr_venv` 等
- 实际部署时，用户可能需要将代码放在不同的位置

**目标**：
- 将硬编码路径改为可配置
- 提供灵活的部署选项
- 支持多种常见的部署路径
- 简化配置过程

---

## 当前硬编码路径分析

### 文档中的假设路径

| 路径类型 | 假设值 | 说明 |
|---------|--------|------|
| **代码目录** | `/opt/fuxictr` | Git 仓库位置 |
| **虚拟环境** | `/opt/fuxictr_venv` | Python 虚拟环境 |
| **数据目录** | `/data/fuxictr/data` | Dashboard 原始数据 |
| **处理后数据** | `/data/fuxictr/processed_data` | Dashboard 处理后数据 |
| **Workflow 数据** | `/data/fuxictr/workflow_datasets` | Workflow 原始数据 |
| **Workflow 处理** | `/data/fuxictr/workflow_processed` | Workflow 处理后数据 |
| **模型目录** | `/data/fuxictr/workflow_models` | Workflow 模型 |
| **日志目录** | `/data/fuxictr/workflow_logs` | Workflow 日志 |
| **Dashboard 日志** | `/opt/fuxictr/dashboard/logs` | Dashboard 日志 |

### 问题

1. ❌ 用户必须按照这些路径部署，不灵活
2. ❌ 如果磁盘分区不同，需要修改大量配置
3. ❌ 没有提供常见场景的部署示例

---

## 解决方案

### 方案 A：使用环境变量（推荐）✅

**核心思想**：所有路径通过环境变量配置，文档提供默认值示例

**优点**：
- ✅ 完全灵活，用户可以自由选择路径
- ✅ 只需修改 `.bashrc` 或 systemd 服务的 `Environment` 行
- ✅ 不需要修改代码或配置文件

**实施**：

1. **创建环境变量配置文件** `/opt/fuxictr/fuxictr_env.sh`：

```bash
#!/bin/bash
# FuxiCTR 部署环境变量配置
# 根据实际情况修改以下路径

# ============================================================================
# 基础路径
# ============================================================================

# 代码根目录（Git 仓库位置）
export FUXICTR_ROOT="/opt/fuxictr"                    # ⚠️ 修改为实际路径

# Python 虚拟环境
export FUXICTR_VENV="$FUXICTR_ROOT/../fuxictr_venv"   # ⚠️ 修改为实际路径

# ============================================================================
# 数据存储路径（Server 142）
# ============================================================================

# 基础存储目录（所有数据的根目录）
export FUXICTR_STORAGE_BASE="/data/fuxictr"         # ⚠️ 修改为实际路径

# Dashboard 数据路径
export FUXICTR_DATA_ROOT="$FUXICTR_STORAGE_BASE/data"
export FUXICTR_PROCESSED_ROOT="$FUXICTR_STORAGE_BASE/processed_data"

# Workflow 数据路径
export FUXICTR_WORKFLOW_DATASETS="$FUXICTR_STORAGE_BASE/workflow_datasets"
export FUXICTR_WORKFLOW_PROCESSED="$FUXICTR_STORAGE_BASE/workflow_processed"
export FUXICTR_WORKFLOW_MODELS="$FUXICTR_STORAGE_BASE/workflow_models"
export FUXICTR_WORKFLOW_LOGS="$FUXICTR_STORAGE_BASE/workflow_logs"

# 日志路径
export FUXICTR_DASHBOARD_LOG_DIR="$FUXICTR_ROOT/dashboard/logs"

# ============================================================================
# Workflow 配置
# ============================================================================

export FUXICTR_CONFIG_PATH="$FUXICTR_ROOT/fuxictr/workflow/config.yaml"

# ============================================================================
# Server 21 配置（数据源服务器）
# ============================================================================

export FUXICTR_SERVER_21_HOST="21.xxxxxx.com"          # ⚠️ 修改为实际主机名
export FUXICTR_SERVER_21_USER="your_username"          # ⚠️ 修改为实际用户名
export FUXICTR_SERVER_21_STAGING="/tmp/fuxictr_staging"

echo "✅ FuxiCTR 环境变量已加载"
echo "📂 代码目录: $FUXICTR_ROOT"
echo "🐍 虚拟环境: $FUXICTR_VENV"
echo "💾 存储目录: $FUXICTR_STORAGE_BASE"
```

2. **在 `~/.bashrc` 中加载**：

```bash
# 在 ~/.bashrc 末尾添加
source /opt/fuxictr/fuxictr_env.sh
```

3. **修改 systemd 服务使用环境变量**：

`/etc/systemd/system/fuxictr-workflow.service`：
```ini
[Unit]
Description=FuxiCTR Workflow Service
After=network.target

[Service]
Type=simple
User=your_username
Group=your_username
WorkingDirectory=/opt/fuxictr
EnvironmentFile=/opt/fuxictr/fuxictr_env.sh  # ← 加载环境变量
ExecStart=/opt/fuxictr_venv/bin/python -m fuxictr.workflow.service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

### 方案 B：提供多种部署场景模板

**场景 1：标准部署（默认）**
```
/opt/fuxictr/          # 代码
/data/fuxictr/         # 数据
/opt/fuxictr_venv/     # 虚拟环境
```

**场景 2：单分区部署**
```
/home/username/fuxictr/        # 代码
/home/username/fuxictr_data/   # 数据
/home/username/fuxictr_venv/     # 虚拟环境
```

**场景 3：多磁盘部署**
```
/mnt/ssd/fuxictr/              # 代码（SSD）
/mnt/hdd1/fuxictr_data/        # 数据（HDD1）
/mnt/hdd2/fuxictr_models/       # 模型（HDD2）
/home/username/fuxictr_venv/     # 虚拟环境
```

**场景 4：自定义路径**
```
/path/to/custom/fuxictr/        # 用户自定义
```

---

### 方案 C：创建部署配置向导脚本

**脚本功能**：
1. 询问用户部署场景
2. 自动生成环境变量文件
3. 自动创建目录结构
4. 自动生成 systemd 服务文件
5. 自动修改配置文件

**脚本示例**：

```bash
#!/bin/bash
# 交互式部署配置脚本

echo "🚀 FuxiCTR 部署配置向导"
echo ""

# 选择部署场景
echo "请选择部署场景："
echo "1) 标准部署 (/opt/fuxictr + /data/fuxictr)"
echo "2) 单分区部署 (~/fuxictr)"
echo "3) 多磁盘部署（自定义）"
echo "4) 完全自定义"
echo ""
read -p "请输入选择 [1-4]: " choice

case $choice in
    1)
        FUXICTR_ROOT="/opt/fuxictr"
        FUXICTR_VENV="/opt/fuxictr_venv"
        FUXICTR_STORAGE_BASE="/data/fuxictr"
        ;;
    2)
        FUXICTR_ROOT="$HOME/fuxictr"
        FUXICTR_VENV="$HOME/fuxictr_venv"
        FUXICTR_STORAGE_BASE="$HOME/fuxictr_data"
        ;;
    3)
        read -p "请输入代码目录: " FUXICTR_ROOT
        read -p "请输入虚拟环境路径: " FUXICTR_VENV
        read -p "请输入数据目录: " FUXICTR_STORAGE_BASE
        ;;
    4)
        read -p "请输入代码目录: " FUXICTR_ROOT
        read -p "请输入虚拟环境路径: " FUXICTR_VENV
        read -p "请输入数据基础目录: " FUXICTR_STORAGE_BASE
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

# 生成环境变量文件
cat > /opt/fuxictr/fuxictr_env.sh <<EOF
export FUXICTR_ROOT="$FUXICTR_ROOT"
export FUXICTR_VENV="$FUXICTR_VENV"
export FUXICTR_STORAGE_BASE="$FUXICTR_STORAGE_BASE"
# ... 其他变量
EOF

echo "✅ 配置文件已生成: /opt/fuxictr/fuxictr_env.sh"
echo "请运行 'source ~/.bashrc' 使其生效"
```

---

## 实施计划

### Phase 1: 创建环境变量配置文件

**文件**: `fuxictr/fuxictr_env.sh.template`

**内容**: 包含所有路径的环境变量模板

### Phase 2: 更新部署文档

**修改 `docs/workflow_deployment_2026.md`**：

1. **添加"灵活部署"章节**
2. **提供环境变量配置方法**
3. **提供多种场景示例**
4. **强调关键点：用户只需修改环境变量文件**

### Phase 3: 更新配置文件

**修改 `fuxictr/workflow/config.yaml`**：

将硬编码路径改为使用环境变量：

```yaml
# 之前（硬编码）
storage:
  workflow_datasets_root: "/data/fuxictr/workflow_datasets/"

# 之后（使用环境变量）
storage:
  workflow_datasets_root: "${FUXICTR_WORKFLOW_DATASETS}"
```

**注意**：Python 需要在加载 YAML 时扩展环境变量

### Phase 4: 创建部署配置向导脚本

**文件**: `scripts/configure_deployment.sh`

**功能**：
- 交互式配置向导
- 自动生成所有配置文件
- 自动创建目录结构

---

## 关键修改点

### 1. 环境变量加载

**在 Python 代码中加载环境变量**：

```python
import os
from pathlib import Path

# 使用环境变量，提供默认值
FUXICTR_ROOT = Path(os.environ.get('FUXICTR_ROOT', '/opt/fuxictr'))
FUXICTR_STORAGE_BASE = Path(os.environ.get('FUXICTR_STORAGE_BASE', '/data/fuxictr'))

# 计算派生路径
WORKFLOW_DATASETS = FUXICTR_STORAGE_BASE / 'workflow_datasets'
```

### 2. YAML 配置文件扩展

**方法 1：在 Python 中扩展环境变量**

```python
import os
import yaml

config_path = os.environ.get('FUXICTR_CONFIG_PATH', 'config.yaml')

def expand_env_vars(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value

with open(config_path) as f:
    config = yaml.safe_load(f)
    # 扩展环境变量
    config = {k: expand_env_vars(v) for k, v in config.items()}
```

**方法 2：使用 yamlls 扩展**

```bash
# 安装 yq
sudo apt-get install yq

# 在配置文件中使用 ${VAR}
# 在启动前扩展
export FUXICTR_STORAGE_BASE="/data/fuxictr"
envsubst < config.yaml.template > config.yaml
```

### 3. systemd 服务使用环境变量

**方法**：使用 `EnvironmentFile`

```ini
[Service]
EnvironmentFile=/opt/fuxictr/fuxictr_env.sh
ExecStart=/opt/fuxictr_venv/bin/python -m fuxictr.workflow.service
```

---

## 风险评估

| 风险 | 级别 | 缓解措施 |
|------|------|----------|
| 环境变量未设置 | 中 | 提供默认值，文档中明确说明 |
| 路径不一致 | 中 | 使用绝对路径，避免相对路径 |
| 权限问题 | 低 | 创建目录时自动设置正确权限 |

---

## 示例：多磁盘部署配置

**场景**：
- SSD（空间小）：存放代码和虚拟环境
- HDD1（大容量）：存放数据
- HDD2（大容量）：存放模型和日志

**环境变量配置**：

```bash
# fuxictr_env.sh
export FUXICTR_ROOT="/mnt/ssd/fuxictr"
export FUXICTR_VENV="/home/username/fuxictr_venv"
export FUXICTR_STORAGE_BASE="/mnt/hdd1/fuxictr_data"

# 覆盖模型路径（可选）
export FUXICTR_WORKFLOW_MODELS="/mnt/hdd2/fuxictr_models"
```

**目录结构**：
```
/mnt/ssd/fuxictr/           # 代码（SSD，快速访问）
├── fuxictr/
├── dashboard/
└── workflow/

/mnt/hdd1/fuxictr_data/    # 数据（HDD，大容量）
├── data/
├── processed_data/
├── workflow_datasets/
└── workflow_processed/

/mnt/hdd2/fuxictr_models/   # 模型（HDD，大容量）
└── workflow_models/
```

---

## 是否执行此计划？

**选项**：
1. **"是，创建灵活部署配置"** - 实施环境变量方案
2. **"先看看效果"** - 先创建配置向导脚本，您测试后再决定
3. **"不需要"** - 现有硬编码路径已够用

请告诉我您的决定。
