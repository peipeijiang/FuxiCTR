#!/usr/bin/env python3
"""Sync run_expid.py modifications to all model directories."""

import os
import re
from pathlib import Path

# 查找所有 run_expid.py 文件
root_dir = Path("/Users/shane/fuxictr/model_zoo")
exclude_dirs = {"common"}  # 排除 common 目录（已经是最新版本）

run_expid_files = []
for run_expid_file in root_dir.rglob("run_expid.py"):
    rel_path = run_expid_file.relative_to(root_dir)
    if rel_path.parts[0] not in exclude_dirs:
        run_expid_files.append(run_expid_file)

print(f"Found {len(run_expid_files)} run_expid.py files to update")
for f in sorted(run_expid_files):
    print(f"  - {f.relative_to(root_dir)}")

# 修改模式
modifications = [
    {
        "name": "run_train signature",
        "pattern": r"def run_train\(model, feature_map, params, args\):",
        "replacement": r"def run_train(model, feature_map, params, args, workflow_logger=None):\n    \"\"\"Training function.\n\n    Args:\n        model: Model instance for training\n        feature_map: Feature map for data processing\n        params: Parameters dictionary\n        args: Arguments dictionary\n        workflow_logger: Optional WorkflowLogger for Dashboard WebSocket broadcasting\n    \"\"\"",
        "run_after": [
            ("    rank = params.get('distributed_rank', 0)\n    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()",
             "    rank = params.get('distributed_rank', 0)\n    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()\n\n    # Set workflow_logger on model if available\n    if workflow_logger:\n        model._workflow_logger = workflow_logger")
        ]
    },
    {
        "name": "run_inference signature",
        "pattern": r"def run_inference\(model, feature_map, params, args\):",
        "replacement": r"def run_inference(model, feature_map, params, args, workflow_logger=None):\n    \"\"\"Inference function.\n\n    Args:\n        model: Model instance for inference\n        feature_map: Feature map for data processing\n        params: Parameters dictionary\n        args: Arguments dictionary\n        workflow_logger: Optional WorkflowLogger for Dashboard WebSocket broadcasting\n    \"\"\"",
    },
    {
        "name": "main function - before model training",
        "pattern": r"    if args\['mode'\] == 'train':\n        run_train\(model, feature_map, params, args\)\n    elif args\['mode'\] == 'inference':\n        run_inference\(model, feature_map, params, args\)",
        "replacement": r"    # Initialize workflow_logger if in Dashboard mode\n    workflow_logger = None\n    if os.environ.get('FUXICTR_WORKFLOW_MODE') == 'dashboard':\n        try:\n            from fuxictr.workflow.utils.logger import get_workflow_logger\n            task_id = os.environ.get('FUXICTR_TASK_ID')\n            if task_id:\n                workflow_logger = get_workflow_logger(int(task_id))\n                if rank == 0:\n                    logging.info(f\"Workflow logger initialized for task {task_id}\")\n        except Exception as e:\n            logging.warning(f\"Failed to initialize workflow logger: {e}\")\n\n    if args['mode'] == 'train':\n        run_train(model, feature_map, params, args, workflow_logger=workflow_logger)\n    elif args['mode'] == 'inference':\n        run_inference(model, feature_map, params, args, workflow_logger=workflow_logger)",
    }
]


def apply_modifications(file_path):
    """Apply all modifications to a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modified = False

    for mod in modifications:
        pattern = mod["pattern"]
        replacement = mod["replacement"]

        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            print(f"  ✓ Applied: {mod['name']}")
            content = new_content
            modified = True

        # Apply run_after modifications
        if "run_after" in mod:
            for after_pattern, after_replacement in mod["run_after"]:
                if after_pattern in content:
                    content = content.replace(after_pattern, after_replacement)
                    print(f"  ✓ Applied: {mod['name']} (run_after)")
                    modified = True

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


# 处理所有文件
print("\n" + "="*60)
print("Starting synchronization...")
print("="*60 + "\n")

updated_count = 0
skipped_count = 0

for file_path in run_expid_files:
    rel_path = file_path.relative_to(root_dir)
    print(f"Processing: {rel_path}")

    try:
        if apply_modifications(file_path):
            updated_count += 1
            print(f"  → Updated\n")
        else:
            skipped_count += 1
            print(f"  → Skipped (no changes needed)\n")
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

print("="*60)
print(f"Summary: {updated_count} updated, {skipped_count} skipped")
print("="*60)
