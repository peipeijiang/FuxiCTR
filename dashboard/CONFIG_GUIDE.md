# XFDL 实验平台使用指南

---

# 🧪 XFDL 实验平台

*专业的深度学习模型训练与推理平台*

---

## 📑 目录

1. [平台简介](#1-平台简介)
2. [前端使用指南](#2-前端使用指南)
3. [配置文件说明](#3-配置文件说明)
4. [任务执行](#4-任务执行)
5. [全流程管理](#5-全流程管理)
6. [常见问题](#6-常见问题)
7. [进阶功能](#7-进阶功能)
   - [7.1 数据集覆盖模式](#71-数据集覆盖模式)
   - [7.2 多任务学习](#72-多任务学习)
   - [7.3 多任务损失权重配置](#73-多任务损失权重配置)
   - [7.4 Hash Embedding 配置](#74-hash-embedding-配置)
   - [7.5 参数调优](#75-参数调优)

---

## 1. 平台简介

XFDL（eXperimental Framework for Deep Learning）实验平台是一个面向深度学习模型训练与推理的专业平台，支持：

- **多用户管理**：支持团队协作，独立的工作空间
- **丰富的模型库**：集成 50+ CTR 预测模型
- **可视化配置**：友好的 Web 界面配置管理
- **全流程监控**：实时日志、资源监控、任务追踪
- **灵活的部署**：支持 CPU/GPU、单卡/多卡训练

---

## 2. 前端使用指南

### 2.1 页面布局

```
┌────────────────────────────────────────────────────────────┐
│  🧪 XFDL 实验平台                    📘 使用教程            │
├──────────┬─────────────────────────────────────────────────┤
│          │                                                 │
│ 侧边栏   │           主工作区                              │
│          │                                                 │
│ • 用户   │  ┌─────────────────────────────────────────┐   │
│ • 模型   │  │  任务执行                               │   │
│ • GPU    │  │                                         │   │
│ • 配置   │  │  选择实验ID、设备、参数                 │   │
│          │  │                                         │   │
│ • 覆盖   │  └─────────────────────────────────────────┘   │
│ • 监控   │                                                 │
│          │  ┌─────────────────────────────────────────┐   │
│          │  │  实时日志                               │   │
│          │  │                                         │   │
│          │  │  显示训练/推理的实时输出                 │   │
│          │  │                                         │   │
│          │  └─────────────────────────────────────────┘   │
└──────────┴─────────────────────────────────────────────────┘
```

### 2.2 基本操作流程

#### 步骤 1：选择用户和模型

在左侧侧边栏：

1. **选择用户名**
   - 从下拉列表选择你的用户名
   - 每个用户有独立的配置空间

2. **选择模型**
   - 选择要使用的模型（如 DeepFM、DCN 等）
   - 系统会自动加载对应的配置文件

#### 步骤 2：配置管理（可选）

点击 **"🛠️ 配置管理"** 展开：

| 配置文件 | 说明 | 编辑方式 |
|---------|------|---------|
| `dataset_config.yaml` | 数据集配置 | 代码编辑器 |
| `model_config.yaml` | 模型配置 | 代码编辑器 |
| `run_expid.py` | 运行脚本 | 代码编辑器 |

**操作步骤**：
1. 点击配置卡片，打开代码编辑器
2. 编辑 YAML 配置
3. 点击 **"保存配置"** 按钮

#### 步骤 3：执行任务

在 **"▶️ 任务执行"** 区域：

1. **填写实验ID**（expid）
   - 必须在 `model_config.yaml` 中定义
   - 例如：`DeepFM_test`

2. **选择设备**
   - `CPU (-1)`：使用 CPU 训练
   - `GPU 0, 1, ...`：选择指定 GPU
   - 多选支持多卡训练

3. **设置 num_workers**
   - 数据加载的进程数
   - 建议：2-4

4. **选择运行模式**
   - **🔥 开始训练**：训练模型
   - **🔮 开始推理**：模型推理

5. **查看实时日志**
   - 下方日志窗口实时显示输出

#### 步骤 4：监控任务

点击 **"📡 服务器活动与任务监控"** 查看：

| 监控项 | 说明 |
|--------|------|
| 运行任务 | 当前正在运行的任务列表 |
| 个人配额 | 个人最大并发任务数（3） |
| 全局配额 | 全局最大并发任务数（10） |

#### 步骤 5：查看结果

点击 **"📊 模型权重"**：

- 浏览训练好的模型权重
- 下载模型文件
- 查看训练日志

---

## 3. 配置文件说明

### 3.1 dataset_config.yaml

数据集配置定义数据来源和特征工程：

```yaml
# 数据集配置示例
tiny_csv:
    # ===== 数据路径配置 =====
    data_root: ../data/              # 数据根目录
    data_format: csv                 # 数据格式：csv/parquet/h5
    train_data: tiny_csv/train.csv  # 训练集路径
    valid_data: tiny_csv/valid.csv  # 验证集路径
    test_data: tiny_csv/test.csv    # 测试集路径（可选）

    # ===== 特征列配置 =====
    feature_cols:
        # 稀疏分类特征
        -   name: user_id
            active: True
            dtype: str
            type: categorical        # 特征类型
            embedding_dim: 16        # 嵌入维度

        # 数值特征
        -   name: price
            active: True
            dtype: float
            type: numeric            # 数值类型

        # 序列特征
        -   name: hist_item_id
            active: True
            dtype: str
            type: sequence           # 序列类型
            embedding_dim: 16
            max_len: 10              # 最大序列长度
            padding: post            # 填充方式
            share_embedding: item_id # 共享嵌入

    # ===== 标签配置 =====
    label_col:
        name: click                 # 标签列名
        dtype: float                # 数据类型
```

**主要字段说明**：

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `data_root` | str | 数据根目录 | `../data/` |
| `data_format` | str | 数据格式 | `csv` / `parquet` / `h5` |
| `train_data` | str | 训练集路径 | `train.csv` |
| `valid_data` | str | 验证集路径 | `valid.csv` |
| `feature_cols` | list | 特征列配置 | 见上方 |
| `label_col` | dict | 标签配置 | `{name: click, dtype: float}` |

**特征类型说明**：

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `categorical` | 稀疏分类特征 | 用户ID、商品ID |
| `numeric` | 数值特征 | 价格、评分 |
| `sequence` | 序列特征 | 历史点击序列 |
| `meta` | 元特征 | 时间、设备类型 |

### 3.2 model_config.yaml

模型配置定义模型结构和训练参数：

```yaml
# ===== 基础配置块（所有实验共享）=====
Base:
    model_root: ./checkpoints/      # 模型保存目录
    workers: 3                      # 数据加载进程数
    verbose: 1                      # 日志详细程度
    patience: 2                     # 早停耐心值
    pickle_feature_encoder: True    # 保存特征编码器
    save_best_only: True            # 仅保存最佳模型
    debug: False                    # 调试模式

# ===== 实验配置块 =====
DeepFM_test:
    # 模型配置
    model: DeepFM                   # 模型类名
    dataset_id: tiny_csv            # 数据集ID
    task: binary_classification     # 任务类型

    # 训练配置
    loss: binary_crossentropy       # 损失函数
    metrics: [logloss, AUC]        # 评估指标
    optimizer: adam                # 优化器
    learning_rate: 1.e-3           # 学习率
    batch_size: 128                # 批大小
    epochs: 10                     # 训练轮数
    shuffle: True                  # 是否打乱数据

    # 正则化配置
    embedding_regularizer: 1.e-8    # Embedding正则
    net_regularizer: 0             # 网络正则

    # 早停配置
    monitor: AUC                   # 监控指标
    monitor_mode: max              # 监控模式
    early_stop_patience: 2         # 早停耐心

    # 模型特定参数
    embedding_dim: 16              # 嵌入维度
    hidden_units: [64, 32]         # 隐藏层大小
    dropout_rates: [0.2, 0.1]      # Dropout率
```

**主要参数说明**：

| 参数类别 | 参数名 | 说明 | 默认值 |
|---------|--------|------|-------|
| **模型** | `model` | 模型类名 | - |
| | `dataset_id` | 数据集ID | - |
| | `task` | 任务类型 | `binary_classification` |
| **训练** | `learning_rate` | 学习率 | `1.e-3` |
| | `batch_size` | 批大小 | `128` |
| | `epochs` | 训练轮数 | `10` |
| | `optimizer` | 优化器 | `adam` |
| **正则** | `embedding_regularizer` | Embedding正则 | `1.e-8` |
| | `net_regularizer` | 网络正则 | `0` |
| **早停** | `monitor` | 监控指标 | `AUC` |
| | `patience` | 耐心值 | `2` |

---

## 4. 任务执行

### 4.1 训练任务

**操作步骤**：

1. 配置 `dataset_config.yaml` 和 `model_config.yaml`
2. 在 **"任务执行"** 区域填写实验ID
3. 选择 GPU 设备（可选多卡）
4. 设置 `num_workers`（建议 2-4）
5. 点击 **"🔥 开始训练"**
6. 查看下方实时日志

**输出文件**：

```
checkpoints/
└── DeepFM_test/
    ├── DeepFM_test.model          # 模型权重
    ├── DeepFM_test.pkl            # 特征编码器
    ├── DeepFM_test.json           # 训练配置
    └── DeepFM_test.log            # 训练日志
```

### 4.2 推理任务

**操作步骤**：

1. 确保已有训练好的模型
2. 在 `dataset_config.yaml` 配置 `infer_data`
3. 点击 **"🔮 开始推理"**
4. 查看推理日志

**输出文件**：

```
checkpoints/
└── DeepFM_test/
    └── DeepFM_test.inference.csv  # 推理结果
```

### 4.3 多GPU训练

**操作步骤**：

1. 在设备选择中勾选多个 GPU
2. 系统自动使用 `torchrun` 启动分布式训练
3. 每个GPU会分配一个训练进程

**示例**：

```
选择: GPU 0, GPU 1, GPU 2, GPU 3
启动: torchrun --nproc_per_node=4 run_expid.py ...
```

---

## 5. 全流程管理

### 5.1 创建任务

点击 **"+ 新建"** 按钮创建新的训练任务：

1. 输入任务名称
2. 系统自动创建任务记录

### 5.2 配置任务

点击任务行的 **"配置"** 按钮：

1. 配置用户和模型
2. 配置数据集和超参数
3. 点击 **"保存并运行"** 启动任务

### 5.3 查看日志

在任务详情页面查看 **"实时日志"**：

- ⏳ 待处理：任务等待开始
- 🔄 运行中：任务正在执行
- ✅ 已完成：任务成功完成
- ❌ 失败：任务执行失败

### 5.4 删除任务

点击 **"🗑"** 按钮删除任务：

1. 点击删除按钮
2. 确认删除（✓）
3. 任务及其关联数据将被清除

---

## 6. 常见问题

### Q1: 如何选择合适的模型？

**答**: 根据数据特点选择：

- **数据量小** (< 100万): LR, FM, DNN
- **中等数据量** (100万-1000万): DeepFM, DCN, xDeepFM
- **大数据量** (> 1000万): DIN, DIEN, SIM
- **有行为序列**: DIN, DIEN, BST
- **需要实时性**: DCN, DeepCrossing

### Q2: 任务无法启动怎么办？

**答**: 检查以下几点：

1. **用户名是否选择**：必须先选择用户
2. **配额是否超限**：个人3个，全局10个
3. **实验ID是否存在**：必须在 `model_config.yaml` 中定义
4. **设备是否可用**：检查GPU是否被占用

### Q3: 如何查看GPU使用情况？

**答**：在 **"📡 服务器活动与任务监控"** 查看：

- 当前运行任务
- GPU占用情况
- 剩余可用资源

### Q4: 如何调整超参数？

**答**: 通过 **"🛠️ 配置管理"** 修改 `model_config.yaml`：

```yaml
# 调整学习率
learning_rate: 5.e-4  # 从 1.e-3 降低

# 调整批大小
batch_size: 256       # 从 128 增大

# 调整嵌入维度
embedding_dim: 32     # 从 16 增大
```

### Q5: 如何处理内存不足？

**答**: 优化方案：

```yaml
# 减少批大小
batch_size: 64

# 减少worker数量
workers: 2

# 使用parquet格式
data_format: parquet  # 比csv更省内存
```

### Q6: 配置修改后不生效？

**答**: 确认操作：

1. 修改配置后点击 **"保存配置"**
2. 确认保存成功提示
3. 重新启动任务

### Q7: 如何查看历史任务？

**答**: 点击 **"全流程管理"** 查看：

- 所有任务列表
- 任务状态和进度
- 任务详情和日志

### Q8: 推理结果在哪里？

**答**: 推理结果保存在：

```
checkpoints/
└── {实验ID}/
    └── {实验ID}.inference.csv
```

可以通过 **"📊 模型权重"** 页面下载。

---

## 7. 进阶功能

### 7.1 数据集覆盖模式

在侧边栏勾选 **"✅ 启用数据集覆盖"**：

- 动态生成临时数据集配置
- 不修改原始配置文件
- 仅对当前任务生效

**适用场景**：
- 快速测试不同数据集
- 不想修改原有配置
- 临时切换数据源

### 7.2 多任务学习

配置多任务模型：

```yaml
# 模型选择
model: MMoE  # 或 PLE, ShareBottom

# 任务定义
task: [ctr, cvr]

# 多标签配置
label_col:
    - {name: click, dtype: float}
    - {name: conversion, dtype: float}

# 损失和指标
loss: [binary_crossentropy, binary_crossentropy]
metrics: [[logloss, AUC], [logloss, AUC]]
```

### 7.3 多任务损失权重配置

在多任务学习中，不同任务的重要性可能不同，需要配置损失权重来平衡各任务的训练。FuxiCTR 支持三种权重配置方式：

#### 方式一：手动权重

手动指定每个任务的权重系数：

```yaml
MMoE_manual:
    model: MMoE
    task: [ctr, cvr]
    loss: [binary_crossentropy, binary_crossentropy]

    # 手动设置权重：ctr 40%, cvr 60%
    loss_weight: [0.4, 0.6]
```

**适用场景**：
- 对任务重要性有明确认知
- 业务目标有优先级差异
- 需要精细控制各任务贡献

#### 方式二：不确定性加权（Uncertainty Weighting, UW）

自动学习任务权重，基于任务的不确定性动态调整：

```yaml
MMoE_with_UW:
    model: MMoE
    task: [ctr, cvr]
    loss: [binary_crossentropy, binary_crossentropy]

    # 使用不确定性加权
    loss_weight: 'UW'

    # UW相关参数（可选）
    uw_init_log_sigma: [0.0, 0.0]  # 初始log sigma值
```

**原理说明**：
- UW 方法为每个任务学习一个可训练的参数 σ（不确定性）
- 权重计算：`weight = 1 / (2 * σ^2)`
- 不确定性高的任务（难学的任务）会被赋予较低权重
- 不确定性低的任务（容易学的任务）会被赋予较高权重

**优点**：
- 自动适应任务难度
- 无需人工调参
- 对噪声任务鲁棒

**适用场景**：
- 任务难度差异较大
- 不确定最佳权重配比
- 希望自动平衡多任务

#### 方式三：GradNorm

基于梯度平衡的动态权重调整：

```yaml
MMoE_with_GradNorm:
    model: MMoE
    task: [ctr, cvr]
    loss: [binary_crossentropy, binary_crossentropy]

    # 使用GradNorm
    loss_weight: 'GN'

    # GradNorm参数（可选）
    gradnorm_alpha: 1.5    # 平衡系数，通常0.5-2.0
    gradnorm_steps: 1      # 每N步更新一次权重
```

**原理说明**：
- GradNorm 通过监控各任务损失的学习速率来调整权重
- 确保所有任务以相近的速率学习
- 避免某个任务主导训练过程

**优点**：
- 平衡各任务学习进度
- 防止简单任务过拟合
- 提升整体性能

**适用场景**：
- 各任务学习速度差异大
- 某些任务收敛过快
- 需要平衡所有任务表现

#### 权重配置对比

| 方法 | 配置复杂度 | 自动调整 | 适用场景 |
|------|-----------|---------|---------|
| 手动权重 | 低 | 否 | 权重明确、业务优先级清晰 |
| UW | 低 | 是 | 任务难度差异大、不确定权重 |
| GradNorm | 中 | 是 | 学习速度不均、需平衡进度 |

### 7.4 Hash Embedding 配置

对于高基数（高基数类别）特征（如用户ID、商品ID），传统 Embedding 可能消耗大量内存。Hash Embedding 通过哈希函数将特征映射到固定大小的向量空间，大幅减少内存占用。

#### 配置示例

```yaml
# 全局Hash Embedding配置
feature_encoder: "hash"       # 使用hash编码器
hash_bits: 24                # hash空间大小：2^24 = 16,777,216

MMoE_hash:
    model: MMoE
    dataset_id: large_dataset

    # 特征列配置
    feature_cols:
        # 高基数特征使用hash
        -   name: user_id
            active: True
            dtype: str
            type: categorical
            embedding_dim: 16
            feature_encoder: "hash"     # 单独指定hash
            hash_bits: 24               # 单独指定hash空间

        # 低基数特征使用正常embedding
        -   name: gender
            active: True
            dtype: str
            type: categorical
            embedding_dim: 8
            # 不指定feature_encoder，使用默认
```

#### Hash Bits 选择指南

| hash_bits | 表空间 | 内存占用 (16维) | 适用场景 |
|-----------|--------|----------------|---------|
| 18 | 262K | ~8MB | 小规模、特征基教适中 |
| 20 | 1M | ~32MB | 中等规模 |
| 22 | 4M | ~128MB | 较大规模 |
| 24 | 16M | ~512MB | 大规模、高基数特征 |
| 26 | 67M | ~2GB | 超大规模（谨慎使用） |

**内存计算公式**：
```
内存(MB) ≈ (2^hash_bits) × embedding_dim × 4字节 / (1024×1024)
```

#### Hash Embedding 优缺点

**优点**：
- 内存占用固定，不受特征基数影响
- 支持未知特征值（测试集新特征）
- 减少过拟合风险

**缺点**：
- 存在哈希冲突（不同特征可能映射到同一向量）
- 可能损失部分特征信息
- 需要权衡 hash_bits 和模型效果

#### 最佳实践

```yaml
# 推荐配置：选择性使用Hash
MMoE_mixed:
    model: MMoE

    # 全局不启用hash
    # feature_encoder: "normal"

    feature_cols:
        # 高基数特征：使用hash
        -   name: user_id           # 假设1000万+不同值
            type: categorical
            embedding_dim: 16
            feature_encoder: "hash"
            hash_bits: 24

        -   name: item_id           # 假设500万+不同值
            type: categorical
            embedding_dim: 16
            feature_encoder: "hash"
            hash_bits: 22

        # 低基数特征：使用正常embedding
        -   name: gender            # 只有2-10个不同值
            type: categorical
            embedding_dim: 8
            # 不指定feature_encoder

        -   name: age_group         # 只有5-10个不同值
            type: categorical
            embedding_dim: 8
```

### 7.5 参数调优

使用参数调优功能：

```bash
# 多实验网格搜索
cd experiment
python run_param_tuner.py \
    --config config/DCN_tuner_config.yaml \
    --gpu 0 1 2 3
```

---

## 8. 快捷操作

### 快捷键

| 操作 | 快捷键 |
|------|--------|
| 保存配置 | `Ctrl + S` / `Cmd + S` |
| 刷新页面 | `Ctrl + R` / `Cmd + R` |
| 返回顶部 | `Home` |
| 跳到底部 | `End` |

### 页面功能

| 功能 | 位置 |
|------|------|
| 配置管理 | 🛠️ 配置管理 |
| 任务执行 | ▶️ 任务执行 |
| 资源监控 | 📡 服务器活动与任务监控 |
| 模型权重 | 📊 模型权重 |
| 可视化 | 📈 可视化 |
| 使用教程 | 📘 使用教程 |
| 全流程管理 | 全流程管理 |

---

## 9. 技术支持

### 获取帮助

- **查看教程**：点击 📘 使用教程
- **查看配置**：点击 🛠️ 配置管理
- **监控任务**：点击 📡 服务器活动与任务监控
- **反馈问题**：联系平台管理员

### 最佳实践

1. **开始前**：先选择用户和模型
2. **配置时**：仔细检查 YAML 语法
3. **训练中**：关注实时日志输出
4. **完成后**：及时下载模型权重
5. **遇到问题**：查看 FAQ 和日志

---

***

**XFDL 实验平台** - 让深度学习更简单

版本: v2.0 | 更新: 2024-01