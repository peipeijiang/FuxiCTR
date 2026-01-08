# Multi-Task Loss Weighting Methods

## 简介

FuxiCTR支持多种多任务学习的损失函数加权方法：

1. **Equal Weighting (EQ)** - 等权重（默认）
2. **Manual Weights** - 手动指定权重
3. **Uncertainty Weighting (UW)** - 基于不确定性的自动加权
4. **GradNorm (GN)** - 基于梯度归一化的自动加权

**注意**：这四种方法是**互斥**的，每次只能选择一种。

---

## 0. Manual Weights (手动权重)

### 原理

如果你已经通过实验或领域知识了解了各任务的重要性，可以直接指定固定的权重。

损失函数：
```
L_total = w1 * L1 + w2 * L2 + ... + wn * Ln
```

其中 w1, w2, ..., wn 是你手动指定的常量权重。

### 使用方法

在配置文件中将 `loss_weight` 设置为权重列表：

```yaml
MMoE_with_manual:
    model: MMoE
    num_tasks: 2
    loss_weight: [0.4, 0.6]  # 任务1权重0.4，任务2权重0.6
    # ... 其他参数
```

或者使用整数权重：
```yaml
loss_weight: [1, 2]  # 任务1权重1，任务2权重2（相当于[0.33, 0.67]）
```

### 适用场景

✅ 已知任务重要性差异  
✅ 需要强调特定任务  
✅ 作为自动方法的baseline对比  

⚠️ 注意：
- 权重列表长度必须等于任务数量
- 权重会保持固定，不会在训练过程中调整
- 需要通过实验调优找到最佳权重

---

## 1. Uncertainty Weighting (UW)

### 原理

Uncertainty Weighting是一种自动学习多任务损失函数权重的方法，基于论文：
**"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"** (CVPR 2018)

该方法通过学习每个任务的不确定性（uncertainty）参数，自动平衡多个任务的损失函数，无需手动调整权重。

对于多任务学习，传统的损失函数为：
```
L_total = w1 * L1 + w2 * L2 + ... + wn * Ln
```

Uncertainty Weighting将权重建模为可学习的参数：
```
L_total = Σ (0.5 * exp(-log_σi²) * Li + 0.5 * log_σi²)
```

其中：
- `σi²` 是第i个任务的方差参数（不确定性）
- 该方法自动学习每个任务的权重
- 高不确定性的任务会获得较小的权重
- 低不确定性的任务会获得较大的权重

### 使用方法

在配置文件中设置 `loss_weight: 'UW'`：

```yaml
MMoE_with_UW:
    model: MMoE
    loss_weight: 'UW'  # 启用 Uncertainty Weighting
    num_tasks: 2
    # ... 其他参数
```

---

## 2. GradNorm (GN)

### 原理

GradNorm是一种基于梯度归一化的自动损失平衡方法，基于论文：
**"GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"** (ICML 2018)

该方法通过以下步骤自动调整任务权重：

1. **计算梯度范数**：对每个任务计算其对共享层的梯度范数
2. **相对训练速率**：根据初始损失计算每个任务的相对训练速率
3. **梯度平衡**：使用不对称参数α来平衡各任务的梯度

损失函数：
```
L_total = Σ wi * Li
```

GradNorm目标：
```
L_grad = Σ |G_i(t) - Ḡ(t) * [r_i(t)]^α|
```

其中：
- `G_i(t)` 是任务i在时间t的梯度范数
- `Ḡ(t)` 是所有任务的平均梯度范数
- `r_i(t)` 是任务i的相对逆训练速率
- `α` 是不对称参数，控制适应速度（默认1.5）

### 使用方法

在配置文件中设置 `loss_weight: 'GN'` 和 `gradnorm_alpha`：

```yaml
MMoE_with_GN:
    model: MMoE
    loss_weight: 'GN'  # 启用 GradNorm
    gradnorm_alpha: 1.5  # GradNorm超参数 (默认: 1.5)
    num_tasks: 2
    # ... 其他参数
```

### GradNorm超参数

- **alpha (α)**: 控制恢复力的强度
  - `α < 1.0`: 偏好难任务（训练慢的任务）
  - `α = 1.0`: 平衡所有任务
  - `α > 1.0`: 偏好简单任务（训练快的任务）
  - 推荐值: 1.5

---

## 支持的loss_weight选项

| 选项 | 描述 | 超参数 | 是否互斥 |
|------|------|--------|----------|
| `'EQ'` | Equal Weight（默认） | 无 | ✅ |
| `[w1, w2, ...]` | Manual Weights | 无 | ✅ |
| `'UW'` | Uncertainty Weighting | 无 | ✅ |
| `'GN'` | GradNorm | `gradnorm_alpha` (默认1.5) | ✅ |

**重要**：以上四种方法是**互斥**的，配置时只能选择其中一种：

```yaml
# ✅ 正确：使用一种方法
loss_weight: 'UW'

# ✅ 正确：使用手动权重
loss_weight: [0.3, 0.7]

# ❌ 错误：不能混用
loss_weight: 'UW'  # 设置了UW后，手动权重会被忽略
manual_weights: [0.3, 0.7]
```

---

## 运行示例

### 使用Equal Weighting (默认)
```bash
# 不设置loss_weight或设置为'EQ'
python run_expid.py --config ./config/ --expid MMoE_default --gpu 0
```

### 使用Manual Weights
```bash
python run_expid.py --config ./config/ --expid MMoE_with_manual --gpu 0
```

### 使用Uncertainty Weighting
```bash
python run_expid.py --config ./config/ --expid MMoE_with_UW --gpu 0
```

### 使用GradNorm
```bash
cd model_zoo/multitask/MMoE
python run_expid.py --config ./config/ --expid MMoE_with_GN --gpu 0
```

---

## 适用模型

所有继承自 `MultiTaskModel` 的多任务模型都支持这些加权方法，包括：
- MMoE
- PLE
- ShareBottom
- APG_MMOE
- APG_AITM
- M3oE

---

## 方法对比

### Manual Weights (手动权重)
✅ **优势**：
- 简单直观：直接控制每个任务的重要性
- 可解释性强：权重含义清晰
- 确定性：结果可复现

⚠️ **注意**：
- 需要手动调优
- 固定权重，不能自适应
- 需要领域知识或大量实验

### Uncertainty Weighting (UW)
✅ **优势**：
- 理论基础：基于贝叶斯不确定性
- 简单高效：无需额外超参数
- 自适应：根据任务难度自动调整

⚠️ **注意**：
- 依赖损失值大小
- 训练初期可能波动

### GradNorm (GN)
✅ **优势**：
- 基于梯度：直接优化训练动态
- 可控性强：通过α控制平衡策略
- 适应性好：根据训练进度调整

⚠️ **注意**：
- 需要识别共享层
- 计算开销略高
- 需要调整α参数

### 选择建议

| 场景 | 推荐方法 | 配置示例 |
|------|---------|----------|
| 已知任务权重 | Manual Weights | `loss_weight: [0.3, 0.7]` |
| 任务难度差异大 | GradNorm | `loss_weight: 'GN'` |
| 快速实验/无先验知识 | Uncertainty Weighting | `loss_weight: 'UW'` |
| 需要精细控制 | GradNorm (调整α) | `loss_weight: 'GN', gradnorm_alpha: 1.5` |
| 默认/任务难度相近 | Equal Weight | `loss_weight: 'EQ'` 或不设置 |

### 优先级选择流程

```
是否已知最优权重？
├─ 是 → Manual Weights [0.3, 0.7]
└─ 否 → 任务难度差异是否很大？
       ├─ 是 → GradNorm (α=1.5)
       ├─ 否 → 是否需要快速实验？
       │      ├─ 是 → Uncertainty Weighting
       │      └─ 否 → Equal Weight (baseline)
       └─ 不确定 → 建议对比多种方法
```

---

## 代码实现

### Uncertainty Weighting
```python
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num))
    
    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.params[i])
            loss_sum += 0.5 * precision * loss + 0.5 * self.params[i]
        return loss_sum
```

### GradNorm
```python
class GradNorm(nn.Module):
    def __init__(self, num_tasks=2, alpha=1.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.loss_scale = nn.Parameter(torch.ones(num_tasks))
```

---

## 参考文献

### Uncertainty Weighting
```bibtex
@inproceedings{kendall2018multi,
  title={Multi-task learning using uncertainty to weigh losses for scene geometry and semantics},
  author={Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  booktitle={CVPR},
  pages={7482--7491},
  year={2018}
}
```

### GradNorm
```bibtex
@inproceedings{chen2018gradnorm,
  title={GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks},
  author={Chen, Zhao and Badrinarayanan, Vijay and Lee, Chen-Yu and Rabinovich, Andrew},
  booktitle={ICML},
  pages={794--803},
  year={2018}
}
```
