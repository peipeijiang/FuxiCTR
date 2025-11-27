# M3oE

This is an implementation of M3oE (Multi-Domain Multi-Task Mixture-of-Experts).

## Configuration

The model configuration is located in `config/model_config.yaml`.

Key parameters:
- `num_domains`: Number of domains.
- `domain_feature`: Feature name used as domain indicator.
- `expert_num`: Number of experts.
- `expert_hidden_units`: Hidden units for experts.
- `tower_hidden_units`: Hidden units for towers.

## Usage

```bash
python run_expid.py --expid M3oE_test
```
