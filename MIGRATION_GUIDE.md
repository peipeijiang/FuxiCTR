# FuxiCTR 架构升级指南 (v2.0)

## 概述

本次更新实现了 Dashboard 与 Workflow 的完全分离，包括：
- 数据目录分离（`data/` vs `processed_data/` vs `workflow_data/`）
- 模型目录分离（`model_zoo/` vs `workflow_models/`）
- 实验隔离（每个实验独立文件夹）
- 日志分离（Dashboard 日志 vs Workflow 日志）

## 迁移步骤

### 1. 备份现有数据

```bash
# 备份 data 目录
cp -r data data_backup_$(date +%Y%m%d)

# 备份 model_zoo 目录（可选）
# cp -r model_zoo model_zoo_backup_$(date +%Y%m%d)
```

### 2. 迁移 processed data

```bash
# 运行迁移脚本
python scripts/migrate_processed_data.py

# 验证迁移结果
python scripts/migrate_processed_data.py --verify
```

**说明**:
- 将 `data/{dataset_id}_processed/` 迁移到 `processed_data/{dataset_id}/`
- `data/` 目录保留原始数据文件

### 3. 迁移模型结构（可选）

```bash
# 运行模型结构迁移脚本
python scripts/migrate_model_structure.py

# 验证迁移结果
python scripts/migrate_model_structure.py --verify
```

**说明**:
- 将模型从 `{dataset_id}/{exp_id}.model` 迁移到 `{dataset_id}/{exp_id}/{exp_id}.model`
- 每个实验使用独立文件夹

### 4. 验证功能

#### Dashboard 测试
```bash
# 启动 Dashboard
streamlit run dashboard/app.py

# 测试训练功能
# 确认数据加载正常
# 确认模型保存路径
```

#### Workflow 测试
```bash
# 启动 Workflow 服务
python -m fuxictr.workflow.service

# 测试 API
curl http://localhost:8001/api/v1/tasks
```

## 目录结构

### 数据目录
```
data/                          # Dashboard 原始数据
├── tiny_npz/
├── tiny_parquet/
└── jrjk_seeds_20251202/

processed_data/                # Dashboard 处理后数据
├── tiny_npz/
│   ├── train.parquet
│   ├── valid.parquet
│   └── feature_map.json
└── jrjk_seeds_20251202/

workflow_data/                 # Workflow 数据
├── datasets/                   # 原始数据（从 Server 21）
│   └── {dataset_id}/raw/
└── processed/                  # 处理后数据
    └── {dataset_id}/
```

### 模型目录
```
model_zoo/                     # Dashboard 模型
└── {model}/checkpoints/
    └── {dataset_id}/
        └── {exp_id}/          # 实验独立文件夹
            ├── {exp_id}.model
            ├── {exp_id}.log
            ├── checkpoints/
            └── tensorboard/

workflow_models/               # Workflow 模型
└── {model}/
    └── {dataset_id}/
        └── task_{id}_{exp_id}/ # 按任务 ID 组织
            ├── model.model
            ├── train.log
            ├── checkpoints/
            └── tensorboard/
```

### 日志目录
```
dashboard/logs/                # Dashboard 日志
├── streamlit.log              # 应用日志
├── sse_server.log             # SSE Server 日志
└── users/                     # 训练日志副本
    └── {username}/
        └── {exp_id}_{timestamp}.log

workflow_logs/                 # Workflow 日志
├── task_{id}_data_fetch.log
├── task_{id}_train.log
└── task_{id}_infer.log
```

## 向后兼容性

- ✅ Dashboard 完全兼容旧路径
- ✅ 模型加载支持新旧两种结构
- ✅ 训练脚本自动检测运行模式

## 回滚方案

如果需要回滚：
```bash
# 恢复数据目录
rm -rf processed_data/
mv data_backup_YYYYMMDD/jrjk_seeds_20251202_processed data/

# 恢复模型（如果备份了）
# rm -rf model_zoo
# mv model_zoo_backup_YYYYMMDD model_zoo
```

## 配置文件

### workflow/config.yaml
新增了 `storage` 配置节：
```yaml
storage:
  dashboard_data_root: "../../../data/"
  dashboard_processed_root: "../../../processed_data/"
  dashboard_model_root: "../../../model_zoo/"
  workflow_datasets_root: "./workflow_data/datasets/"
  workflow_processed_root: "./workflow_data/processed/"
  workflow_model_root: "./workflow_models/"
  workflow_log_root: "./workflow_logs/"
```

## 常见问题

### Q: 迁移后数据丢失？
A: 检查 `data_backup_*` 目录，数据已备份

### Q: Dashboard 找不到数据？
A: 确认使用的是 `processed_data/` 而不是 `data/`

### Q: 模型加载失败？
A: 检查模型路径是否正确，支持新旧两种结构

## 联系方式

- 文档: `docs/workflow_deployment_2026.md`
- 检查清单: `docs/deployment_checklist.md`
- 问题反馈: GitHub Issues

---

**版本**: v2.0
**日期**: 2026-02-02
**维护者**: FuxiCTR Team
