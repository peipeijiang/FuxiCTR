#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型结构迁移脚本：重组模型结构，每个实验使用独立文件夹

旧结构:
    model_zoo/{model}/checkpoints/{dataset_id}/{exp_id}.model
    model_zoo/{model}/checkpoints/{dataset_id}/{exp_id}.log

新结构:
    model_zoo/{model}/checkpoints/{dataset_id}/{exp_id}/{exp_id}.model
    model_zoo/{model}/checkpoints/{dataset_id}/{exp_id}/{exp_id}.log
    model_zoo/{model}/checkpoints/{dataset_id}/{exp_id}/checkpoints/  (epoch checkpoints)
    model_zoo/{model}/checkpoints/{dataset_id}/{exp_id}/tensorboard/ (TensorBoard logs)

使用方法:
    python scripts/migrate_model_structure.py [--verify]

Author: FuxiCTR Team
Date: 2026-02-02
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_model_directories() -> List[Path]:
    """查找所有模型目录"""
    model_zoo_root = project_root / "model_zoo"
    model_dirs = []

    if not model_zoo_root.exists():
        print(f"❌ model_zoo 目录不存在: {model_zoo_root}")
        return model_dirs

    for item in model_zoo_root.iterdir():
        if item.is_dir() and not item.name.startswith("_") and not item.name.startswith("."):
            # 检查是否有 checkpoints 目录
            checkpoints_dir = item / "checkpoints"
            if checkpoints_dir.exists():
                model_dirs.append(item)

    return sorted(model_dirs)


def find_experiments_in_dataset(dataset_dir: Path) -> List[Tuple[str, Path]]:
    """
    查找数据集中的所有实验

    Returns:
        List of (exp_id, file_path) tuples
    """
    experiments = []

    # 查找所有 .model 和 .log 文件
    for item in dataset_dir.iterdir():
        if item.is_file():
            if item.suffix in [".model", ".log"]:
                exp_id = item.stem  # 去掉扩展名
                experiments.append((exp_id, item))

    return experiments


def migrate_experiments(dataset_dir: Path) -> Tuple[int, int, int]:
    """
    迁移实验到独立文件夹

    Returns:
        (migrated_count, skipped_count, error_count)
    """
    migrated_count = 0
    skipped_count = 0
    error_count = 0

    # 查找所有实验
    experiments = find_experiments_in_dataset(dataset_dir)

    # 按实验 ID 分组
    exp_groups = {}
    for exp_id, file_path in experiments:
        if exp_id not in exp_groups:
            exp_groups[exp_id] = []
        exp_groups[exp_id].append(file_path)

    # 为每个实验创建独立文件夹
    for exp_id, files in exp_groups.items():
        exp_dir = dataset_dir / exp_id

        # 检查是否已经是新结构（实验文件夹已存在）
        if exp_dir.exists() and exp_dir.is_dir():
            # 检查文件夹内是否已经有文件
            existing_files = list(exp_dir.iterdir())
            if existing_files:
                print(f"  ⏭️  跳过（实验文件夹已存在且非空）: {exp_id}")
                skipped_count += 1
                continue

        # 创建实验文件夹
        try:
            os.makedirs(exp_dir, exist_ok=True)
        except Exception as e:
            print(f"  ❌ 创建实验文件夹失败: {exp_id} - {e}")
            error_count += 1
            continue

        # 移动文件到实验文件夹
        moved_files = []
        for file_path in files:
            try:
                new_path = exp_dir / file_path.name
                if not new_path.exists():
                    shutil.move(str(file_path), str(new_path))
                    moved_files.append(file_path.name)
            except Exception as e:
                print(f"  ⚠️  移动文件失败: {file_path.name} - {e}")
                error_count += 1

        if moved_files:
            print(f"  ✅ 迁移: {exp_id} ({len(moved_files)} 个文件)")
            migrated_count += 1

    return migrated_count, skipped_count, error_count


def migrate_model_structure():
    """迁移所有模型的结构"""
    model_dirs = find_model_directories()

    if not model_dirs:
        print("❌ 没有找到任何模型目录")
        return False

    print("=" * 80)
    print("开始迁移模型结构...")
    print("=" * 80)
    print(f"找到 {len(model_dirs)} 个模型")
    print()

    total_migrated = 0
    total_skipped = 0
    total_error = 0

    for model_dir in model_dirs:
        model_name = model_dir.name
        checkpoints_dir = model_dir / "checkpoints"

        print(f"处理模型: {model_name}")

        if not checkpoints_dir.exists():
            print(f"  ⏭️  跳过（无 checkpoints 目录）")
            print()
            continue

        # 遍历所有数据集目录
        dataset_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]

        for dataset_dir in dataset_dirs:
            dataset_id = dataset_dir.name
            print(f"  数据集: {dataset_id}")

            migrated, skipped, errors = migrate_experiments(dataset_dir)
            total_migrated += migrated
            total_skipped += skipped
            total_error += errors

        print()

    print("=" * 80)
    print("迁移完成！")
    print("=" * 80)
    print(f"✅ 成功迁移: {total_migrated} 个实验")
    print(f"⏭️  跳过: {total_skipped} 个实验")
    print(f"❌ 错误: {total_error} 个")
    print()

    return total_error == 0


def verify_migration():
    """验证迁移结果"""
    model_dirs = find_model_directories()

    print("=" * 80)
    print("验证迁移结果...")
    print("=" * 80)
    print()

    for model_dir in model_dirs:
        model_name = model_dir.name
        checkpoints_dir = model_dir / "checkpoints"

        print(f"模型: {model_name}")

        if not checkpoints_dir.exists():
            print("  无 checkpoints 目录")
            print()
            continue

        # 统计
        old_structure_count = 0
        new_structure_count = 0

        for dataset_dir in checkpoints_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_id = dataset_dir.name

            # 检查是否还有旧的 .model/.log 文件在数据集目录下
            for item in dataset_dir.iterdir():
                if item.is_file() and item.suffix in [".model", ".log"]:
                    old_structure_count += 1

            # 检查新结构的实验文件夹
            for exp_dir in dataset_dir.iterdir():
                if exp_dir.is_dir():
                    new_structure_count += 1

        print(f"  旧结构文件数: {old_structure_count}")
        print(f"  新结构实验数: {new_structure_count}")

        if old_structure_count == 0:
            print("  ✅ 已完全迁移到新结构")
        else:
            print("  ⚠️  仍有旧结构文件")

        print()


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_migration()
        return

    print("⚠️  此操作将重组模型结构，每个实验使用独立文件夹")
    print("   建议：在执行前先备份 model_zoo/ 目录")
    print()

    response = input("确认继续？(yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("❌ 操作已取消")
        return

    print()

    # 执行迁移
    success = migrate_model_structure()

    # 验证结果
    if success:
        verify_migration()


if __name__ == "__main__":
    main()
