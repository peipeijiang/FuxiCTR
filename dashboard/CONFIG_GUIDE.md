## XFDL 配置全解：dataset_config.yaml & model_config.yaml

> 适用场景：在前端“🛠️ 配置管理”编辑或在侧栏开启“✅ 启用数据集覆盖”时，快速了解每个字段的含义与常见填法。保存后会写入 `dashboard/user_configs/<user>/<model>/` 形成个人副本，不影响原始文件。

### 1. dataset_config.yaml

一份或多份数据集定义，每个顶层 key 是一个 `dataset_id`。常见字段：

| 字段 | 作用 | 示例与说明 |
| ---- | ---- | ---------- |
| `data_root` | 数据根目录，其他相对路径的基准 | `../data/` |
| `data_format` | 数据格式 | `csv` / `parquet` / `h5` |
| `train_data` | 训练集路径（可相对 `data_root` 或绝对路径） | `../data/tiny_csv/train.csv` |
| `valid_data` | 验证集路径 | `../data/tiny_csv/valid.csv` |
| `test_data` | 测试集路径（可为空） | `../data/tiny_csv/test.csv` |
| `infer_data` | 推理用数据路径（可为文件或目录） | `../data/tiny_csv/infer.csv` |
| `min_categr_count` | 稀有类别截断阈值，低于此频次的类别会映射为 OOV | `1` |
| `feature_cols` | 特征列定义列表 | 见下方详解 |
| `label_col` | 标签列定义 | `{name: click, dtype: float}` |
| 其它常见 | `sequence_features` / `max_seq_len` / `padding` 等序列相关设置；`pickle_feature_encoder` 用于保存特征编码器 |

**feature_cols 常用键：**

- `name`：列名或列名列表（如多值序列）。
- `active`：是否启用该特征，`True/False`。
- `dtype`：`str`/`int`/`float`，或 `json`、`list`（多值）。
- `type`：特征类型，如 `categorical`、`numeric`、`sequence`、`meta`。
- `embedding_dim`：嵌入维度（离散/序列特征）。
- `share_embedding`：与其他列共享 embedding 表，填共享组名。
- `vocab_size` / `padding_idx` / `max_len`：序列或离散特征的词表/填充/长度控制。

**label_col：**

- `name`：标签列名，单任务时通常为 `click` 或 `label`。
- `dtype`：数值类型，通常 `float`。
- 多任务：在 `feature_map.labels` 中会有多个 label，模型输出也对应多列。

**如何修改：**

- 前端“🛠️ 配置管理”直接编辑 YAML，保存即落盘个人副本。
- 侧栏“覆盖模式”：勾选后选择数据集模板或自定义路径，系统会生成临时 `dataset_config.yaml`，不改原文件。

### 2. model_config.yaml

顶层包含一个 `Base` 通用配置块 + 多个以实验 ID 命名的配置块。常见字段：

| 字段 | 作用 | 示例 |
| ---- | ---- | ---- |
| `model` | 模型类名（与 `src` 目录下定义对应） | `DeepFM` |
| `dataset_id` | 绑定的数据集 ID（对应 dataset_config 的 key） | `tiny_csv` |
| `task` | 任务类型 | `binary_classification` / `regression` |
| `loss` | 损失函数 | `binary_crossentropy` / `bpr` 等 |
| `metrics` | 评估指标列表 | `['logloss', 'AUC']` |
| `optimizer` | 优化器 | `adam` / `sgd` / `adamw` |
| `learning_rate` | 学习率 | `1.e-3` |
| `batch_size` | batch 大小 | `128` |
| `epochs` | 训练轮数 | `10` |
| `shuffle` | 是否打乱训练数据 | `True` |
| `monitor` / `monitor_mode` | 早停/保存依据 | `AUC` / `max` |
| `patience` / `early_stop_patience` | 早停耐心 | `2` |
| `workers` | DataLoader `num_workers`（每个进程/每个 rank 的 CPU loader 数） | `3` |
| `embedding_regularizer` / `net_regularizer` | 正则系数 | `1.e-8` / `0` |
| `model_root` | 模型保存根目录 | `./checkpoints/` |
| `seed` | 随机种子 | `2024` |
| 其它模型特定 | 如 `embedding_dim`、层数、hidden_size、dropout、attention 相关参数等，取决于具体模型 |

**Base 块：**

- 为所有实验提供默认值；实验块缺省时继承 `Base`。常见字段：`model_root`、`workers`、`verbose`、`patience`、`pickle_feature_encoder`、`save_best_only`、`debug` 等。

**多任务配置：**

- `task` 可为列表；`label_col` 需在 dataset_config 中定义多个标签。
- 模型名选择 multitask 目录下的模型（如 PLE/MMoE/APG_*），推理/训练均支持多卡。

**如何修改/新增实验：**

- 在“🛠️ 配置管理”选中 `model_config.yaml`，直接编辑。
- 复制现有实验块，改名为新 `expid`，调整参数即可；前端“▶️ 任务执行”使用该 `expid`。

### 3. 覆盖模式（dataset 覆盖）与多卡/num_workers

**覆盖模式是什么？**  
在侧栏勾选 `✅ 启用数据集覆盖` 后，前端会根据你选择的模板或填写的路径，动态生成一份临时 `dataset_config.yaml`，仅供当前任务使用，不修改模型目录下的原始配置。

**覆盖模式会改哪些字段？**  
- `dataset_id`、`data_root`、`train_data`、`valid_data`、`test_data`、`infer_data` 等路径相关字段会被更新为你在侧栏填入/选择的值。  
- `feature_cols`、`label_col` 等特征/标签定义保持原样（来自默认或你的个人副本），除非你在“🛠️ 配置管理”里手动编辑。  
- 生成的临时文件保存在 `dashboard/logs/configs/<expid>_<timestamp>/dataset_config.yaml`，任务启动时会传给 `run_expid.py`。

**覆盖模式的作用域与恢复**  
- 只作用于当前任务（当次点击“开始训练/推理”），不会写回模型目录或你的个人副本。  
- 关闭覆盖模式后，重新使用默认/个人配置。  
- 若需清理，直接不再使用该临时目录或手动删除 `dashboard/logs/configs/` 下的对应子目录即可。

**多卡与 num_workers**  
- `num_workers`(或配置中的 `workers`) 仅控制 DataLoader 的 CPU 进程数；多卡时总进程 ≈ `workers × 卡数`，不要超过可用 CPU 核心/内存承受。  
- 多卡：在“▶️ 任务执行”选择多张 GPU，前端自动构造 `torchrun --nproc_per_node=<卡数>`，训练/推理均支持；多任务模型推理已做 rank 分片输出避免文件名冲突。

### 4. 训练 / 推理步骤回顾

1. 侧栏选用户、模型；按需启用覆盖模式。
2. “🛠️ 配置管理”检查并保存三文件（dataset/model/run_expid）。
3. “▶️ 任务执行”填 `expid`、选设备（CPU/单卡/多卡）、设 `num_workers`。
4. 点击 `🔥 开始训练` 或 `🔮 开始推理`，下方查看实时日志。
5. “📡 服务器活动与任务监控”查看资源/配额（个人 3，全局 10）。
6. “📊 模型权重”浏览/下载历史权重与日志；“📈 可视化”可启动 TensorBoard。

### 5. FAQ 快查

- 不能启动？确认用户名已选，未超配额 (个人 3 / 全局 10)。
- 想多卡推理？设备选择多 GPU，系统自动 torchrun，输出按 rank 分片。
- 配置乱了？配置卡片点击“重置”，或关闭覆盖模式恢复默认。
