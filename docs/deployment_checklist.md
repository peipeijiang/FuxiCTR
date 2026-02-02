# Workflow 部署配置检查清单

> **使用说明**: 请逐项检查并填写，确认所有配置项正确后，交给开发人员部署。

---

## Server 21 配置清单

### 基础信息

| 配置项 | 值 | 状态 | 备注 |
|--------|-----|------|------|
| 主机地址/IP | `21.xxxxxx.com` | ⬜ 确认 | 需要替换为实际值 |
| SSH 端口 | `22` | ⬜ 确认 | 默认 22 |
| SSH 用户名 | `__________` | ⬜ 确认 | 需要填写 |
| Staging 目录 | `/tmp/fuxictr_staging` | ⬜ 创建 | `sudo mkdir -p /tmp/fuxictr_staging` |

### 环境验证

| 检查项 | 命令 | 预期结果 | 状态 |
|--------|------|---------|------|
| spark-sql 可用 | `which spark-sql` | 显示路径 | ⬜ 通过 |
| Hive 可用 | `hive --version` | 显示版本 | ⬜ 通过 |
| HDFS 访问 | `hdfs dfs -ls /path` | 显示文件列表 | ⬜ 通过 |
| 临时目录 | `ls -ld /tmp/fuxictr_staging` | 目录存在且有权限 | ⬜ 通过 |

### 数据库信息

| 配置项 | 值 | 状态 | 备注 |
|--------|-----|------|------|
| Hive 数据库名 | `__________` | ⬜ 确认 | |
| 训练数据表 | `__________` | ⬜ 确认 | |
| 推理数据表 | `__________` | ⬜ 确认 | |
| 结果写入表 | `__________` | ⬜ 确认 | |

---

## Server 142 配置清单

### 系统信息

| 配置项 | 值 | 状态 | 备注 |
|--------|-----|------|------|
| 主机地址/IP | `__________` | ⬜ 确认 | |
| GPU 数量 | `__` 块 | ⬜ 确认 | `nvidia-smi` |
| GPU 型号 | `__________` | ⬜ 确认 | |
| Python 版本 | `3.10` | ⬜ 确认 | 需要 3.10+ |

### 目录结构

| 目录 | 路径 | 创建命令 | 状态 |
|------|------|---------|------|
| Dashboard 原始数据 | `/data/fuxictr/data` | `sudo mkdir -p` | ⬜ 创建 |
| Dashboard 处理数据 | `/data/fuxictr/processed_data` | `sudo mkdir -p` | ⬜ 创建 |
| Workflow 原始数据 | `/data/fuxictr/workflow_datasets` | `sudo mkdir -p` | ⬜ 创建 |
| Workflow 处理数据 | `/data/fuxictr/workflow_processed` | `sudo mkdir -p` | ⬜ 创建 |
| Dashboard 模型 | `/opt/fuxictr/model_zoo` | `sudo mkdir -p` | ⬜ 创建 |
| Workflow 模型 | `/data/fuxictr/workflow_models` | `sudo mkdir -p` | ⬜ 创建 |
| Workflow 日志 | `/data/fuxictr/workflow_logs` | `sudo mkdir -p` | ⬜ 创建 |
| Dashboard 日志 | `/opt/fuxictr/dashboard/logs` | `sudo mkdir -p` | ⬜ 创建 |
| 数据库备份 | `/data/fuxictr/db_backup` | `sudo mkdir -p` | ⬜ 创建 |

### Python 环境

| 检查项 | 命令 | 预期结果 | 状态 |
|--------|------|---------|------|
| 虚拟环境 | `ls /opt/fuxictr_venv` | 目录存在 | ⬜ 通过 |
| PyTorch 版本 | `python -c "import torch; print(torch.__version__)"` | `2.1.0` | ⬜ 通过 |
| CUDA 可用 | `python -c "import torch; print(torch.cuda.is_available())"` | `True` | ⬜ 通过 |
| GPU 数量 | `python -c "import torch; print(torch.cuda.device_count())"` | `>0` | ⬜ 通过 |
| FuxiCTR 安装 | `python -c "import fuxictr; print(fuxictr.__version__)"` | 显示版本 | ⬜ 通过 |

### SSH 连接测试

| 检查项 | 命令 | 预期结果 | 状态 |
|--------|------|---------|------|
| SSH 密钥存在 | `ls -la ~/.ssh/id_rsa` | 文件存在 | ⬜ 通过 |
| SSH 连接 Server 21 | `ssh username@21.xxxxxx.com "echo ok"` | `ok` | ⬜ 通过 |
| rsync 测试 | `rsync -avz /tmp/test.txt username@21.xxxxxx.com:/tmp/` | 传输成功 | ⬜ 通过 |

---

## 配置文件清单

### workflow/config.yaml

| 配置项 | 配置路径 | 示例值 | 实际值 | 状态 |
|--------|---------|--------|--------|------|
| Server 21 主机 | `servers.server_21.host` | `21.xxxxxx.com` | `__________` | ⬜ 配置 |
| SSH 端口 | `servers.server_21.port` | `22` | `__` | ⬜ 配置 |
| SSH 用户名 | `servers.server_21.username` | `your_username` | `__________` | ⬜ 配置 |
| SSH 密钥 | `servers.server_21.key_path` | `~/.ssh/id_rsa` | `__________` | ⬜ 配置 |
| Server 21 Staging | `storage.server_21_staging` | `/tmp/fuxictr_staging` | `__________` | ⬜ 配置 |
| Dashboard 数据根 | `storage.dashboard_data_root` | `/opt/fuxictr/data/` | `__________` | ⬜ 配置 |
| Dashboard 处理数据 | `storage.dashboard_processed_root` | `/opt/fuxictr/processed_data/` | `__________` | ⬜ 配置 |
| Workflow 数据集 | `storage.workflow_datasets_root` | `/data/fuxictr/workflow_datasets/` | `__________` | ⬜ 配置 |
| Workflow 处理数据 | `storage.workflow_processed_root` | `/data/fuxictr/workflow_processed/` | `__________` | ⬜ 配置 |
| Dashboard 模型 | `storage.dashboard_model_root` | `/opt/fuxictr/model_zoo/` | `__________` | ⬜ 配置 |
| Workflow 模型 | `storage.workflow_model_root` | `/data/fuxictr/workflow_models/` | `__________` | ⬜ 配置 |

---

## 服务启动清单

### Workflow 后端服务

| 步骤 | 命令 | 预期结果 | 状态 |
|------|------|---------|------|
| 1. 激活虚拟环境 | `source /opt/fuxictr_venv/bin/activate` | 提示符变化 | ⬜ 完成 |
| 2. 设置环境变量 | `export WORKFLOW_CONFIG_PATH=...` | 无输出 | ⬜ 完成 |
| 3. 启动服务 | `python -m fuxictr.workflow.service` | 服务运行 | ⬜ 完成 |
| 4. 检查健康 | `curl http://localhost:8001/api/health` | 返回 200 | ⬜ 通过 |

### Dashboard 前端服务

| 步骤 | 命令 | 预期结果 | 状态 |
|------|------|---------|------|
| 1. 激活虚拟环境 | `source /opt/fuxictr_venv/bin/activate` | 提示符变化 | ⬜ 完成 |
| 2. 启动 Dashboard | `streamlit run dashboard/app.py --server.port 8501` | 服务运行 | ⬜ 完成 |
| 3. 检查访问 | `curl http://localhost:8501` | 返回 HTML | ⬜ 通过 |

---

## 最终验证

| 验证项 | 验证方法 | 预期结果 | 状态 |
|--------|---------|---------|------|
| Workflow API 可用 | `curl http://localhost:8001/api/v1/tasks` | 返回任务列表 | ⬜ 通过 |
| Dashboard 可访问 | 浏览器打开 `http://server:8501` | 显示 Dashboard | ⬜ 通过 |
| 数据目录可写 | `touch /data/fuxictr/workflow_datasets/test && rm ...` | 成功创建/删除 | ⬜ 通过 |
| 模型目录可写 | `touch /data/fuxictr/workflow_models/test && rm ...` | 成功创建/删除 | ⬜ 通过 |
| 日志目录可写 | `touch /data/fuxictr/workflow_logs/test && rm ...` | 成功创建/删除 | ⬜ 通过 |

---

## 问题记录

| 序号 | 问题描述 | 解决方案 | 状态 |
|------|---------|---------|------|
| 1 |  |  | ⬜ 待解决 |
| 2 |  |  | ⬜ 待解决 |
| 3 |  |  | ⬜ 待解决 |

---

## 签字确认

| 角色 | 姓名 | 签字 | 日期 |
|------|------|------|------|
| 配置人员 |  |  |  |
| 验证人员 |  |  |  |
| 开发负责人 |  |  |  |

---

**检查清单版本**: v1.0
**最后更新**: 2026-02-02
