# 实施计划：训练和推理日志显示优化

## 需求重述

优化训练和推理过程中的日志显示，重点关注：
1. **推理时文件的推理进度** - 显示当前文件和总体进度
2. **多卡进度的显示** - 所有 GPU 的工作状态应该可见
3. **Dashboard 中的实时日志** - 通过 WebSocket 实时显示进度条和指标

## 当前问题分析

### 训练日志 (`fuxictr/pytorch/models/rank_model.py`)
- ✅ 有 tqdm 进度条（单卡时）
- ❌ 多卡时只显示 rank 0 的进度条
- ❌ 缺少 ETA（预计剩余时间）
- ❌ tqdm 输出到 stdout，Dashboard 通过读取日志文件显示（有延迟）

### 推理日志 (`model_zoo/common/run_expid.py`)
- ✅ 有 tqdm 双进度条（总体 + 单文件）
- ❌ 多卡时各 rank 的进度条独立显示，分散
- ❌ 缺少全局汇总（所有 rank 的总进度）
- ❌ 缺少文件级别的速度（rows/s）

### Dashboard 实时日志系统 (`dashboard/pages/workflow.py`)
- ✅ 完整的 WebSocket 实时日志系统（第 229-480 行）
- ✅ 支持日志类型：log, progress, metric, error, complete, status
- ✅ 自定义 HTML 进度条组件（第 513-540 行）
- ❌ 训练/推理的 tqdm 进度条没有连接到 WebSocket 系统

## 技术方案

### 核心思想：双模式设计

```
命令行模式：保持 tqdm 原有体验（stdout 输出进度条）
Dashboard 模式：通过自定义 TqdmWebSocketAdapter 将 tqdm 输出转换为 WebSocket 消息
```

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Training/Inference Process                │
│                                                               │
│  ┌──────────────┐         ┌─────────────────────────────┐   │
│  │  tqdm loops  │────────>│  TqdmWebSocketAdapter       │   │
│  │              │         │  - 拦截 tqdm 更新           │   │
│  └──────────────┘         │  - 转换为 WS 消息           │   │
│                           │  - 保持 stdout 输出          │   │
│                           └──────────┬──────────────────┘   │
└──────────────────────────────────────┼────────────────────────┘
                                       │
                                       ▼
                           ┌───────────────────────┐
                           │  WorkflowLogger       │
                           │  - progress()         │
                           │  - metric()           │
                           │  - log()              │
                           └───────────┬───────────┘
                                       │
                                       ▼
                           ┌───────────────────────┐
                           │  Dashboard Frontend   │
                           │  - 实时进度条更新     │
                           │  - 指标可视化         │
                           └───────────────────────┘
```

## 实施步骤

### Phase 1: 创建 TqdmWebSocketAdapter（核心组件）

**新建文件**: `fuxictr/pytorch/utils/tqdm_adapter.py`

**核心类**:
```python
class TqdmWebSocketAdapter(tqdm):
    """tqdm 适配器，将进度广播到 WebSocket"""

    def __init__(self, iterable=None, logger=None, step_name="unknown",
                 rank=None, world_size=1, **kwargs):
        self._ws_logger = logger
        self._step_name = step_name
        self._rank = rank
        self._world_size = world_size
        self._last_broadcast = 0
        super().__init__(iterable, **kwargs)

    def update(self, n=1):
        """覆盖 update 方法以广播进度"""
        super().update(n)
        # 每 1% 或至少每 N 个 batch 广播一次
        if self._ws_logger and self._should_broadcast():
            self._broadcast_progress()
```

**多卡聚合**:
```python
class DistributedTqdmAdapter(TqdmWebSocketAdapter):
    """聚合所有 rank 的进度"""

    def update(self, n=1):
        super().update(n)
        # 使用 torch.distributed.all_gather 收集所有 rank 的进度
        # 只有 rank 0 广播到 WebSocket
```

### Phase 2: 修改 rank_model.py（训练）

**文件**: `fuxictr/pytorch/models/rank_model.py`

**修改点**:
1. 添加可选的 `workflow_logger` 参数到 `__init__`
2. 修改 `train_epoch()` 使用 `TqdmWebSocketAdapter`（当 logger 可用时）
3. 修改 `evaluate()` 使用 adapter
4. 添加指标广播到 WebSocket

**向后兼容**:
```python
# 检测 logger 是否可用
if self._workflow_logger:
    batch_iterator = TqdmWebSocketAdapter(data_generator, logger=...)
else:
    batch_iterator = tqdm(data_generator, ...)  # 原有行为
```

### Phase 3: 修改 run_expid.py（推理）

**文件**: `model_zoo/common/run_expid.py`

**修改点**:
1. 添加 `workflow_logger` 参数到 `run_inference()`
2. 修改双进度条使用 `TqdmWebSocketAdapter`
3. 添加文件级别的速度显示（rows/s）
4. 改进多卡进度汇总日志

**进度显示示例**:
```
Rank 0: Total: [##########] 50% 500k/1M rows 12.5k rows/s
Rank 0: File data_202401.parquet: [##########] 75% 75k/100k rows
```

### Phase 4: Workflow Executor 集成

**文件**: `fuxictr/workflow/executor/trainer.py`

**修改点**:
1. 传递 `task_id` 到训练进程（通过环境变量或命令行参数）
2. 设置 `FUXICTR_WORKFLOW_MODE` 环境变量
3. 训练进程中检测环境变量并初始化 WorkflowLogger

**实现方式**:
```python
# trainer.py
env = os.environ.copy()
env['FUXICTR_WORKFLOW_TASK_ID'] = str(task_id)
env['FUXICTR_WORKFLOW_MODE'] = 'dashboard'

# run_expid.py
if os.environ.get('FUXICTR_WORKFLOW_MODE') == 'dashboard':
    # 初始化 WorkflowLogger 并传递给 model
```

### Phase 5: Dashboard 显示优化（可选）

**文件**: `dashboard/pages/workflow.py`

**改进点**:
1. 优化进度条显示（添加多卡信息）
2. 显示训练指标图表（使用现有的 metric 类型）
3. 添加 ETA 显示

## 修改文件清单

### 必须修改的文件

| 文件 | 修改内容 | 优先级 |
|------|---------|--------|
| `fuxictr/pytorch/utils/tqdm_adapter.py` | **新建** | 高 |
| `fuxictr/pytorch/models/rank_model.py` | 集成 adapter | 高 |
| `model_zoo/common/run_expid.py` | 推理进度优化 | 高 |
| `fuxictr/workflow/executor/trainer.py` | 传递 task_id | 中 |
| `fuxictr/workflow/executor/inference.py` | 类似 trainer | 中 |
| `dashboard/pages/workflow.py` | 进度显示优化 | 低 |

## 向后兼容性保证

1. **命令行模式**（无 Dashboard）：
   - `workflow_logger=None` 时使用原生 tqdm
   - 完全保持原有体验

2. **自动检测**：
   ```python
   def _detect_dashboard_mode():
       return 'FUXICTR_WORKFLOW_TASK_ID' in os.environ
   ```

3. **降级处理**：
   - WebSocket 连接失败时自动禁用广播
   - 不影响训练/推理进程

## 风险评估

| 风险 | 级别 | 缓解措施 |
|------|------|----------|
| WebSocket 广播影响训练性能 | 中 | 批量发送（每 1% 或每 N 个 batch），使用异步发送 |
| 多卡同步延迟 | 中 | 只在 rank 0 执行聚合，降低广播频率 |
| WebSocket 连接断开 | 低 | 异常捕获 + 自动禁用，不影响训练 |
| 多进程 Logger 共享 | 低 | 只有 rank 0 初始化 WorkflowLogger |

## 估计工作量

| 阶段 | 内容 | 预估时间 |
|------|------|---------|
| Phase 1 | TqdmWebSocketAdapter 开发 | 1-2 天 |
| Phase 2 | 训练集成 | 0.5-1 天 |
| Phase 3 | 推理集成 | 0.5-1 天 |
| Phase 4 | Workflow 集成 | 1-2 天 |
| Phase 5 | Dashboard 优化（可选） | 0.5-1 天 |
| 测试与调试 | | 1-2 天 |
| **总计** | | **5-9 天** |

## 验收标准

### 命令行模式（无变化）
- ✅ tqdm 进度条正常显示
- ✅ 日志输出到 stdout
- ✅ 多卡训练/推理正常工作

### Dashboard 模式（新功能）
- ✅ 实时显示训练/推理进度
- ✅ 显示指标（loss, AUC 等）
- ✅ 多卡进度聚合显示
- ✅ ETA（预计剩余时间）
- ✅ 文件级别的推理速度（rows/s）

### 性能要求
- ✅ WebSocket 广播开销 < 1% 训练时间
- ✅ 多卡同步延迟 < 100ms
