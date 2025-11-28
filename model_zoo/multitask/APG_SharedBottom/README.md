# APG_SharedBottom

This is an implementation of APG_SharedBottom, which combines APG (Adaptive Parameter Generation) with Shared-Bottom multi-task learning architecture.

## Configuration

The model configuration is located in `config/model_config.yaml`.

Key parameters:
- `condition_mode`: "self-wise", "group-wise", or "mix-wise".
- `condition_features`: List of feature names used for condition if mode is not "self-wise".
- `rank_k`: Rank for low-rank decomposition in APG.
- `overparam_p`: Over-parameterization dimension.

## Usage

### Training

```bash
python run_expid.py --config ./config/ --expid APG_SharedBottom_test --mode train
```

### Inference

```bash
python run_expid.py --config ./config/ --expid APG_SharedBottom_test --mode inference
```

## Configuration Tutorial

### Dataset Configuration (`dataset_config.yaml`)

The `dataset_config.yaml` file defines the dataset parameters.

- **data_root**: The root directory of the data.
- **data_format**: The format of the data files (e.g., `parquet`, `csv`).
- **train_data**, **valid_data**, **test_data**: Paths to the training, validation, and test data files or directories.
- **infer_data**: Path to the inference data file or directory.
- **feature_cols**: A list of feature definitions. Each feature definition includes:
    - `name`: The name of the feature column(s).
    - `active`: Whether the feature is active.
    - `dtype`: The data type of the feature (e.g., `str`, `float`).
    - `type`: The type of the feature (`categorical`, `numeric`, `sequence`).
- **label_col**: A list of label definitions. Each label definition includes:
    - `name`: The name of the label column.
    - `dtype`: The data type of the label.
    - `threshold`: The threshold for binary classification (optional).

### Model Configuration (`model_config.yaml`)

The `model_config.yaml` file defines the model hyperparameters and training settings.

- **Base**: Common configurations shared across experiments.
- **Experiment ID** (e.g., `APG_SharedBottom_test`): Specific configurations for an experiment.
    - `model`: The model class name.
    - `dataset_id`: The dataset ID defined in `dataset_config.yaml`.
    - `loss`: A list of loss functions for each task.
    - `metrics`: A list of evaluation metrics.
    - `task`: A list of task types (e.g., `binary_classification`).
    - `optimizer`: The optimizer to use (e.g., `adam`).
    - `learning_rate`: The learning rate.
    - `batch_size`: The batch size.
    - `epochs`: The number of training epochs.
    - **Model-specific parameters**: Parameters specific to the model architecture (e.g., `condition_mode`, `rank_k`).
