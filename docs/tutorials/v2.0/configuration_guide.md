# Configuration Guide

Complete guide to configuring XFDL experiments using YAML files.

**Reading Time:** ~10 minutes
**Last Updated:** 2026-01-21
**Prerequisites:** Basic knowledge of YAML format

---

## Overview

XFDL uses two main configuration files to manage experiments:

- **`dataset_config.yaml`** - Dataset settings and feature definitions
- **`model_config.yaml`** - Model hyperparameters and training settings

<callout type="info">
**About Configuration Files**

Configuration files use YAML format with key-value pairs. Always use spaces for indentation, never tabs. We recommend 2 or 4 spaces for consistent formatting.
</callout>

---

## Dataset Configuration

### Basic Structure

The dataset configuration defines your data sources and how features should be processed.

```yaml
# dataset_config.yaml
taobao_tiny:  # dataset_id
    data_root: ../data/
    data_format: csv
    train_data: ../data/tiny_data/train_sample.csv
    valid_data: ../data/tiny_data/valid_sample.csv
    test_data: ../data/tiny_data/test_sample.csv
    min_categr_count: 1
    feature_cols:
        - {name: "userid", active: True, dtype: str, type: categorical}
        - {name: "adgroup_id", active: True, dtype: str, type: categorical}
    label_col: {name: clk, dtype: float}
```

### Key Parameters

| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `dataset_id` | string | Unique identifier for the dataset | Yes | - |
| `data_root` | string | Root directory for data files | Yes | - |
| `data_format` | string | Data format: `csv` or `h5` | Yes | - |
| `train_data` | string | Path to training data | Yes | - |
| `valid_data` | string | Path to validation data | Yes | - |
| `test_data` | string | Path to test data | No | - |
| `min_categr_count` | int | Minimum count for feature filtering | No | 1 |

### Feature Columns

Feature columns define how each feature in your dataset should be processed.

#### Basic Feature Configuration

```yaml
feature_cols:
    - name: userid              # Feature name (column header)
      active: True              # Whether to use this feature
      dtype: str                # Data type
      type: categorical         # Feature type
```

#### Feature Types

XFDL supports three main feature types:

**1. Categorical Features**

For discrete values with a finite set of possible values.

```yaml
- name: user_id
  active: True
  dtype: str
  type: categorical
  embedding_dim: 16            # Optional: Override default embedding dimension
```

**2. Numeric Features**

For continuous numerical values.

```yaml
- name: price
  active: True
  dtype: float
  type: numeric
  normalizer: StandardScaler   # Optional: Normalize numeric values
```

**3. Sequence Features**

For variable-length sequences (e.g., user behavior history).

```yaml
- name: click_history
  active: True
  dtype: str
  type: sequence
  splitter: " "                # Split sequence by space
  max_len: 50                  # Maximum sequence length
  padding: pre                 # Padding strategy: pre or post
  encoder: MaskedAveragePooling  # How to aggregate sequence
```

<callout type="tip">
**Working with Sequences**

Sequence features require special handling during preprocessing. The `splitter` parameter defines how to split individual items in your sequence string. For example, if your sequence is "item1,item2,item3", use `splitter: ","`.
</callout>

### Advanced Feature Options

#### Embedding Sharing

Share embeddings between related features to reduce model size.

```yaml
- name: user_id
  active: True
  dtype: str
  type: categorical
  share_embedding: user_id     # Share with another feature

- name: user_id_alt
  active: True
  dtype: str
  type: categorical
  share_embedding: user_id     # Same embedding layer
```

#### Pretrained Embeddings

Use pre-trained embeddings for features.

```yaml
- name: item_id
  active: True
  dtype: str
  type: categorical
  pretrained_emb: path/to/embeddings.h5  # HDF5 file with embeddings
  freeze_emb: False                        # Whether to freeze embeddings
```

#### Custom Preprocessing

Define custom preprocessing functions for complex features.

```yaml
- name: timestamp
  active: True
  dtype: str
  type: categorical
  preprocess: extract_hour  # Custom function name
```

The preprocessing function should be defined in your dataset class:

```python
def extract_hour(self, df_col):
    """Extract hour from timestamp."""
    return pd.to_datetime(df_col).dt.hour.astype(str)
```

### Label Configuration

Define the target variable for your model.

```yaml
label_col:
    name: clk          # Column name for label
    dtype: float       # Data type of label
```

<callout type="warning">
**Important Notes**

- The label column must exist in all data files (train, valid, test)
- For binary classification, use dtype `float` with values 0.0 and 1.0
- For multi-class classification, use dtype `int` with class indices
</callout>

---

## Model Configuration

### Basic Structure

Model configuration defines hyperparameters and training settings.

```yaml
# model_config.yaml
Base:                              # Shared settings (optional)
    model_root: '../checkpoints/'
    workers: 3
    verbose: 1
    patience: 2
    learning_rate: 1.e-3
    batch_size: 128
    epochs: 10

DeepFM_test:                      # Experiment ID
    model: DeepFM                  # Model class name
    dataset_id: taobao_tiny        # Dataset to use
    loss: 'binary_crossentropy'
    metrics: ['logloss', 'AUC']
    task: binary_classification
    hidden_units: [64, 32]         # Model-specific params
    embedding_dim: 10
    embedding_regularizer: 1.e-8
```

### Configuration Inheritance

Use the `Base` section to define shared settings across multiple experiments.

**With Base:**

```yaml
Base:
    model_root: '../checkpoints/'
    batch_size: 128
    learning_rate: 1.e-3

Exp1:
    model: DeepFM
    learning_rate: 5.e-4  # Override Base

Exp2:
    model: DeepFM
    # Inherits learning_rate from Base
```

**Without Base:**

```yaml
Exp1:
    model_root: '../checkpoints/'
    batch_size: 128
    learning_rate: 1.e-3
    model: DeepFM

Exp2:
    model_root: '../checkpoints/'
    batch_size: 128
    learning_rate: 1.e-3
    model: DeepFM
```

### Common Training Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `learning_rate` | float | Initial learning rate | 0.001 |
| `batch_size` | int | Training batch size | 128 |
| `epochs` | int | Maximum number of epochs | 10 |
| `optimizer` | string | Optimizer: adam, adamw | adam |
| `loss` | string | Loss function | binary_crossentropy |
| `metrics` | list | Evaluation metrics | ['logloss', 'AUC'] |
| `seed` | int | Random seed for reproducibility | - |

### Early Stopping

Configure early stopping to prevent overfitting.

```yaml
patience: 2                    # Stop after 2 epochs without improvement
monitor: 'AUC'                 # Metric to monitor
monitor_mode: 'max'            # 'max' for metrics to maximize
every_x_epochs: 1              # Evaluate every N epochs
save_best_only: True           # Only save best model
```

<callout type="info">
**Custom Monitor Metrics**

You can combine multiple metrics:

```yaml
monitor: {'AUC': 2, 'logloss': -1}  # 2*AUC - logloss
monitor_mode: 'max'
```
</callout>

### Regularization

Control overfitting with regularization techniques.

```yaml
# Weight regularization
net_regularizer: l2(1.e-5)      # L2 regularization for MLP
embedding_regularizer: 1.e-8    # L2 regularization for embeddings

# Dropout
net_dropout: 0.1                # Dropout rate for MLP layers

# Batch Normalization
batch_norm: True                # Apply batch normalization
```

---

## Model-Specific Parameters

Different models have different hyperparameters. Below are common examples.

### DeepFM

```yaml
DeepFM_exp:
    model: DeepFM
    hidden_units: [64, 32]          # MLP hidden layers
    hidden_activations: relu        # Activation function
    embedding_dim: 10
    net_dropout: 0.1
    batch_norm: False
```

### DCN (Deep & Cross Network)

```yaml
DCN_exp:
    model: DCN
    num_cross_layers: 3             # Number of cross layers
    hidden_units: [64, 32]
    embedding_dim: 10
```

### DIN (Deep Interest Network)

```yaml
DIN_exp:
    model: DIN
    attention_hidden_units: [64, 32]  # Attention MLP
    attention_activation: relu
    embedding_dim: 16
```

<details>
<summary>View All Model Parameters</summary>

For a complete list of model-specific parameters, refer to the model implementation in `fuxictr/models/`. Each model class documents its available parameters in the constructor.

```python
class DeepFM(tf.keras.Model):
    """
    DeepFM Model.

    Parameters:
    -----------
    feature_encoder : FeatureEncoder
        The feature encoder instance.

    embedding_dim : int
        Dimension of feature embeddings.

    hidden_units : list
        List of hidden units for MLP layers.

    hidden_activations : str or list
        Activation function(s) for MLP.

    net_regularizer : float or str
        Regularization for MLP weights.

    net_dropout : float
        Dropout rate for MLP layers.

    batch_norm : bool
        Whether to use batch normalization.
    """
```

</details>

---

## Best Practices

### File Organization

Organize your configuration files logically:

```
project/
├── config/
│   ├── dataset_config.yaml
│   └── model_config.yaml
├── data/
│   └── ...
└── checkpoints/
    └── ...
```

### Naming Conventions

Use descriptive, consistent names:

```yaml
# Good
taobao_ad_ctr_v1
criteo_x4_test

# Avoid
exp1
test
config2
```

### Parameter Tuning

Start with default parameters, then tune systematically:

```yaml
# Stage 1: Defaults
learning_rate: 1.e-3
batch_size: 128
embedding_dim: 10

# Stage 2: Tune learning rate
learning_rate: 5.e-4  # Better validation performance?

# Stage 3: Tune batch size
batch_size: 256  # Faster training?

# Stage 4: Tune model-specific params
hidden_units: [128, 64]  # Larger model?
```

<callout type="success">
**Recommended Workflow**

1. Start with default parameters
2. Tune learning rate first (most impactful)
3. Adjust batch size based on memory
4. Tune model-specific parameters
5. Apply regularization if overfitting
</callout>

---

## Common Issues

### Issue: FileNotFoundError

**Problem:** Data file not found at specified path.

**Solution:**
- Check that paths are relative to `data_root`
- Verify files exist at specified locations
- Use absolute paths if relative paths don't work

```yaml
# Correct
data_root: ../data/
train_data: train.csv  # Resolves to ../data/train.csv

# Incorrect
train_data: /absolute/path/train.csv  # Ignores data_root
```

### Issue: Poor Performance

**Problem:** Model performance is unexpectedly low.

**Common Causes:**

| Symptom | Cause | Solution |
|---------|-------|----------|
| Training loss not decreasing | Learning rate too low | Increase `learning_rate` |
| Training loss oscillating | Learning rate too high | Decrease `learning_rate` |
| Large train-valid gap | Overfitting | Add regularization, reduce model size |
| Both losses high | Underfitting | Increase model capacity, train longer |

### Issue: Memory Errors

**Problem:** CUDA out of memory during training.

**Solutions:**

1. Reduce batch size:
   ```yaml
   batch_size: 64  # Reduce from 128
   ```

2. Reduce embedding dimension:
   ```yaml
   embedding_dim: 8  # Reduce from 16
   ```

3. Reduce model size:
   ```yaml
   hidden_units: [32, 16]  # Smaller MLP
   ```

---

## Quick Reference

### Dataset Config Template

```yaml
dataset_id:
    data_root: ../data/
    data_format: csv
    train_data: path/to/train.csv
    valid_data: path/to/valid.csv
    min_categr_count: 1
    feature_cols:
        - {name: feature1, active: True, dtype: str, type: categorical}
        - {name: feature2, active: True, dtype: float, type: numeric}
    label_col: {name: label, dtype: float}
```

### Model Config Template

```yaml
Base:
    model_root: ../checkpoints/
    batch_size: 128
    learning_rate: 1.e-3
    epochs: 10

experiment_id:
    model: ModelName
    dataset_id: dataset_id
    loss: binary_crossentropy
    metrics: ['logloss', 'AUC']
    task: binary_classification
```

---

## Next Steps

- [ ] Create your `dataset_config.yaml`
- [ ] Create your `model_config.yaml`
- [ ] Verify YAML syntax with a linter
- [ ] Run your first experiment
- [ ] Monitor training with TensorBoard

For more information, see:
- [Installation Guide](installation.md)
- [Quick Start](quickstart.md)
- [API Reference](api_reference.md)

---

## Additional Resources

- [YAML Specification](https://yaml.org/spec/)
- [Model Zoo](../../model_zoo/) - Example configurations
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

<callout type="tip">
**Need Help?**

Join our community forum or open an issue on GitHub for specific questions.
</callout>
