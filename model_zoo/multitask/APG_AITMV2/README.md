# APG_AITMV2

APG_AITMV2 is a cascade-friendly multi-task model that combines:

- **APG** (Adaptive Parameter Generation): condition-aware parameter generation via `condition_z`.
- **PLE/CGC bottom**: controllable shared & task-specific experts.
- **Cascaded transfer**: **unidirectional** task information transfer (recommended for funnel tasks).

## Configuration

The model configuration is located in `config/model_config.yaml`.

Key parameters:
- `condition_mode`: "self-wise", "group-wise", or "mix-wise".
- `condition_features`: list of feature names used for condition if mode is not "self-wise".
- `num_layers`, `num_shared_experts`, `num_specific_experts`: PLE/CGC bottom structure.
- `transfer_type`: `gated_residual` (default) or `attn` (baseline).
- `use_prev_logit`: whether to inject upstream logit into the transfer gate.
- `detach_prev_rep`, `detach_prev_logit`: enforce one-way gradient flow for cascade tasks.
- `tower_type`: `dnn` (default) or `apg`.

## Usage

### Training

```bash
python run_expid.py --config ./config/ --expid APG_AITMV2_test --mode train
```

### Inference

```bash
python run_expid.py --config ./config/ --expid APG_AITMV2_test --mode inference
```

### Sweep Inference

```bash
python run_expid.py --config ./config/ --expid APG_AITMV2_test --mode inference --sweep
```

This mode sweeps the domain feature (e.g., `product`) over all valid values and outputs partitioned Parquet files.
