# APG_AITM

This is an implementation of APG_AITM, which combines APG (Adaptive Parameter Generation) with AITM (Adaptive Information Transfer Multi-task).

## Configuration

The model configuration is located in `config/model_config.yaml`.

Key parameters:
- `condition_mode`: "self-wise", "group-wise", or "mix-wise".
- `condition_features`: List of feature names used for condition if mode is not "self-wise".
- `rank_k`: Rank for low-rank decomposition in APG.
- `overparam_p`: Over-parameterization dimension.

## Usage

```bash
python run_expid.py --expid APG_AITM_test
```
