# =========================================================================
# Copyright (C) 2026. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
Config Merge Utility for Workflow.

Merges original experiment config with new data paths and auto-detected features.
Only replaces data paths and feature_cols, keeps other parameters from original config.

FuxiCTR uses two config files:
- model_config.yaml: Model + training parameters (experiment configs like "MMoE_default")
- dataset_config.yaml: Data paths + features + label_col (user-configured)

Key Points:
- feature_cols: Auto-detected by workflow from column names (_tag, _cnt, _textlist)
- label_col: User-configured in dataset_config.yaml (NOT auto-detected)
- data_root, train_data, valid_data, test_data: Replaced with workflow paths
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


def find_original_config(user: str, model: str, experiment_id: str) -> Optional[str]:
    """
    Find original model_config.yaml from dashboard user configs.

    FuxiCTR uses two config files:
    - model_config.yaml: Model + training parameters (this is what we need)
    - dataset_config.yaml: Data paths + features (will be auto-replaced)

    Search order (two-tier priority):
    1. Priority 1: User's personal config directory (most specific)
       - dashboard/user_configs/{user}/{model}/config/model_config.yaml
       - dashboard/user_configs/{user}/multitask/{model}/config/model_config.yaml

    2. Priority 2: Model default config (fallback only if user config not found)
       - model_zoo/multitask/{model}/config/model_config.yaml
       - model_zoo/{model}/config/model_config.yaml

    Args:
        user: Username (e.g., "yeshao")
        model: Model name (e.g., "MMoE")
        experiment_id: Experiment ID (e.g., "MMoE_default") - key in model_config.yaml

    Returns:
        Path to model_config.yaml file, or None if not found
    """
    # Priority 1: User's personal config directory (most specific)
    user_config_paths = [
        # Standard user config path
        f"dashboard/user_configs/{user}/{model}/config/model_config.yaml",
        # User config with multitask subdirectory
        f"dashboard/user_configs/{user}/multitask/{model}/config/model_config.yaml",
        # Relative paths from workflow directory
        f"../../dashboard/user_configs/{user}/{model}/config/model_config.yaml",
        f"../../dashboard/user_configs/{user}/multitask/{model}/config/model_config.yaml",
        # Absolute paths (common deployments)
        f"/opt/fuxictr/dashboard/user_configs/{user}/{model}/config/model_config.yaml",
        f"/opt/fuxictr/dashboard/user_configs/{user}/multitask/{model}/config/model_config.yaml",
    ]

    # First, try user's personal config
    for path in user_config_paths:
        if os.path.exists(path):
            logger.info(f"Found user model_config.yaml at: {path}")
            return path

    # Priority 2: Model default config (fallback only if user config not found)
    model_config_paths = [
        # Standard model zoo paths
        f"model_zoo/multitask/{model}/config/model_config.yaml",
        f"model_zoo/{model}/config/model_config.yaml",
        # Relative paths from workflow directory
        f"../../model_zoo/multitask/{model}/config/model_config.yaml",
        f"../../model_zoo/{model}/config/model_config.yaml",
        # Absolute paths (common deployments)
        f"/opt/fuxictr/model_zoo/multitask/{model}/config/model_config.yaml",
        f"/opt/fuxictr/model_zoo/{model}/config/model_config.yaml",
    ]

    # If user config not found, try model default config
    for path in model_config_paths:
        if os.path.exists(path):
            logger.info(f"User config not found, using model default model_config.yaml at: {path}")
            return path

    logger.error(f"model_config.yaml not found for {user}/{model}")
    return None


def find_dataset_config(user: str, model: str) -> Optional[str]:
    """
    Find original dataset_config.yaml from dashboard user configs.

    Args:
        user: Username (e.g., "yeshao")
        model: Model name (e.g., "MMoE")

    Returns:
        Path to dataset_config.yaml file, or None if not found
    """
    # Priority 1: User's personal config directory
    user_config_paths = [
        f"dashboard/user_configs/{user}/{model}/config/dataset_config.yaml",
        f"dashboard/user_configs/{user}/multitask/{model}/config/dataset_config.yaml",
        f"../../dashboard/user_configs/{user}/{model}/config/dataset_config.yaml",
        f"/opt/fuxictr/dashboard/user_configs/{user}/{model}/config/dataset_config.yaml",
    ]

    for path in user_config_paths:
        if os.path.exists(path):
            logger.info(f"Found user dataset_config.yaml at: {path}")
            return path

    # Priority 2: Model default config
    model_config_paths = [
        f"model_zoo/multitask/{model}/config/dataset_config.yaml",
        f"../../model_zoo/multitask/{model}/config/dataset_config.yaml",
        f"/opt/fuxictr/model_zoo/multitask/{model}/config/dataset_config.yaml",
    ]

    for path in model_config_paths:
        if os.path.exists(path):
            logger.info(f"User dataset_config not found, using model default at: {path}")
            return path

    logger.warning(f"dataset_config.yaml not found for {user}/{model}, will use auto-detected features only")
    return None


def load_original_config(config_path: str) -> Dict[str, Any]:
    """Load original experiment config from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_merged_config(
    original_config: Dict[str, Any],
    data_root: str,
    train_data: str,
    valid_data: str,
    test_data: str,
    feature_cols: list,
    label_col: list,
    task_id: int,
    dataset_id: Optional[str] = None,
    model_root: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create merged config by replacing only data paths and features.

    Original config parameters are preserved except:
    - data_root → workflow data_root (datasets_root/{exp_id.dataset_id}/processed/)
    - train_data, valid_data, test_data → processed data paths
    - feature_cols → auto-detected feature_cols (from column names)
    - label_col → user-configured label_col (from dataset_config.yaml)
    - dataset_id → exp_id.timestamp (for FuxiCTR to create correct subdirectories)
    - model_root → model_zoo/{model}/checkpoints/

    Args:
        original_config: Original experiment config
        data_root: Workflow data root directory (processed/)
        train_data: Processed training data path
        valid_data: Processed validation data path
        test_data: Processed test data path
        feature_cols: Auto-detected feature columns
        label_col: User-configured label column (NOT auto-detected)
        task_id: Task ID for temp config naming
        dataset_id: Dataset ID (exp_id.timestamp) for FuxiCTR
        model_root: Model root directory (model_zoo/{model}/checkpoints/)

    Returns:
        Merged configuration dictionary
    """
    # Start with original config
    merged = original_config.copy()

    # Replace only these fields
    merged["data_root"] = data_root
    merged["train_data"] = train_data
    merged["valid_data"] = valid_data
    merged["test_data"] = test_data
    merged["feature_cols"] = feature_cols
    merged["label_col"] = label_col

    # Set dataset_id for FuxiCTR to create correct subdirectories
    # FuxiCTR uses: model_dir = os.path.join(model_root, feature_map.dataset_id)
    if dataset_id:
        merged["dataset_id"] = dataset_id

    # Set model_root for FuxiCTR
    if model_root:
        merged["model_root"] = model_root

    # Mark as workflow-generated config
    merged["_workflow_generated"] = True
    merged["_workflow_task_id"] = task_id

    return merged


def save_merged_config(
    merged_config: Dict[str, Any],
    experiment_id: str,
    task_id: int,
    output_dir: Optional[str] = None
) -> str:
    """
    Save merged config to temporary file for training.

    Args:
        merged_config: Merged configuration
        experiment_id: Original experiment ID
        task_id: Workflow task ID
        output_dir: Output directory (default: workflow/config/)

    Returns:
        Path to saved config file
    """
    if output_dir is None:
        # Default to workflow config directory
        workflow_dir = os.path.dirname(os.path.dirname(__file__))
        output_dir = os.path.join(workflow_dir, "config")

    os.makedirs(output_dir, exist_ok=True)

    # Use task-specific name to avoid conflicts
    config_filename = f"{experiment_id}_task{task_id}.yaml"
    config_path = os.path.join(output_dir, config_filename)

    with open(config_path, 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Saved merged config to: {config_path}")
    return config_path


def prepare_training_config(
    user: str,
    model: str,
    experiment_id: str,
    data_root: str,
    train_data: str,
    valid_data: str,
    test_data: str,
    feature_cols: list,
    label_col: list,
    task_id: int,
    dataset_id: Optional[str] = None,
    model_root: Optional[str] = None
) -> tuple:
    """
    Complete workflow to prepare training config.

    Workflow:
    1. Find and load model_config.yaml (user config → model default)
    2. Extract experiment config from model_config.yaml using experiment_id as key
    3. Merge with auto-detected data paths and features
    4. Save merged config

    Args:
        user: Username
        model: Model name
        experiment_id: Experiment ID (key in model_config.yaml, e.g., "MMoE_default")
        data_root: Data root directory (processed/ directory)
        train_data: Training data path
        valid_data: Validation data path
        test_data: Test data path
        feature_cols: Auto-detected feature columns
        label_col: User-configured label column (loaded from dataset_config.yaml)
        task_id: Task ID
        dataset_id: Dataset ID (exp_id.timestamp) for FuxiCTR
        model_root: Model root directory (model_zoo/{model}/checkpoints/)

    Returns:
        Tuple of (config_path, merged_config, original_config_path)

    Raises:
        FileNotFoundError: If model_config.yaml not found or experiment_id not found in it
    """
    # Step 1: Find model_config.yaml
    model_config_path = find_original_config(user, model, experiment_id)
    if not model_config_path:
        raise FileNotFoundError(
            f"model_config.yaml not found for {user}/{model}. "
            f"Please ensure the model exists in the dashboard."
        )

    # Step 2: Load model_config.yaml
    all_model_configs = load_original_config(model_config_path)

    # Step 3: Extract the specific experiment config
    if experiment_id not in all_model_configs:
        # Try Base config as fallback
        if "Base" in all_model_configs:
            logger.warning(f"Experiment ID '{experiment_id}' not found in model_config.yaml, using Base config")
            experiment_config = all_model_configs["Base"].copy()
        else:
            raise FileNotFoundError(
                f"Experiment ID '{experiment_id}' not found in model_config.yaml. "
                f"Available keys: {list(all_model_configs.keys())}"
            )
    else:
        experiment_config = all_model_configs[experiment_id].copy()

    # Merge with Base config if exists (for inheritance)
    if "Base" in all_model_configs and experiment_id != "Base":
        base_config = all_model_configs["Base"]
        # Apply Base defaults, then override with experiment_config
        merged_from_base = base_config.copy()
        merged_from_base.update(experiment_config)
        experiment_config = merged_from_base

    logger.info(f"Loaded experiment config '{experiment_id}' with {len(experiment_config)} fields")

    # Step 4: Create merged config
    merged_config = create_merged_config(
        original_config=experiment_config,
        data_root=data_root,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        feature_cols=feature_cols,
        label_col=label_col,
        task_id=task_id,
        dataset_id=dataset_id,  # Pass dataset_id for FuxiCTR
        model_root=model_root    # Pass model_root for FuxiCTR
    )

    # Log what's being replaced
    logger.info("Config merge - Replaced fields:")
    logger.info(f"  data_root: {experiment_config.get('data_root')} → {data_root}")
    logger.info(f"  feature_cols: {len(experiment_config.get('feature_cols', []))} groups → {len(feature_cols)} groups (auto-detected)")
    logger.info(f"  label_col: {experiment_config.get('label_col')} → {label_col} (user-configured)")
    logger.info(f"  data_paths: processed data from workflow")

    # Step 5: Save merged config
    config_path = save_merged_config(merged_config, experiment_id, task_id)

    return config_path, merged_config, model_config_path
