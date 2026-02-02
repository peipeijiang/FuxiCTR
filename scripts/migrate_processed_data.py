#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据迁移脚本：将 processed data 从 data/ 迁移到 processed_data/

使用方法:
    python scripts/migrate_processed_data.py

迁移规则:
    - data/{dataset_id}_processed/ -> processed_data/{dataset_id}/
    - 保留 data/ 目录中的原始数据文件
"""

import os
import sys
import shutil
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def migrate_processed_data():
    """迁移处理后的数据从 data/ 到 processed_data/"""
    data_dir = project_root / "data"
    processed_dir = project_root / "processed_data"

    # 确保 processed_data 目录存在
    processed_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("开始迁移 processed data...")
    print("=" * 80)
    print(f"源目录: {data_dir}")
    print(f"目标目录: {processed_dir}")
    print()

    migrated_count = 0
    skipped_count = 0
    error_count = 0

    # 遍历 data/ 目录
    if not data_dir.exists():
        print(f"❌ 源目录不存在: {data_dir}")
        return False

    for item in data_dir.iterdir():
        if not item.is_dir():
            continue

        dataset_name = item.name

        # 只迁移以 _processed 结尾的目录
        if not dataset_name.endswith("_processed"):
            print(f"⏭️  跳过（非 processed 目录）: {dataset_name}")
            skipped_count += 1
            continue

        # 新目录名（去掉 _processed 后缀）
        new_name = dataset_name.replace("_processed", "")
        src = item
        dst = processed_dir / new_name

        # 检查目标目录是否已存在
        if dst.exists():
            print(f"⚠️  跳过（目标已存在）: {src} -> {dst}")
            skipped_count += 1
            continue

        try:
            # 移动目录
            shutil.move(str(src), str(dst))
            print(f"✅ 迁移: {dataset_name} -> {new_name}")
            migrated_count += 1
        except Exception as e:
            print(f"❌ 迁移失败: {dataset_name} -> {new_name}")
            print(f"   错误: {e}")
            error_count += 1

    print()
    print("=" * 80)
    print("迁移完成！")
    print("=" * 80)
    print(f"✅ 成功迁移: {migrated_count} 个目录")
    print(f"⏭️  跳过: {skipped_count} 个目录")
    print(f"❌ 失败: {error_count} 个目录")
    print()

    return error_count == 0


def verify_migration():
    """验证迁移结果"""
    data_dir = project_root / "data"
    processed_dir = project_root / "processed_data"

    print("=" * 80)
    print("验证迁移结果...")
    print("=" * 80)
    print()

    # 检查是否还有 _processed 目录
    remaining_processed = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.endswith("_processed"):
            remaining_processed.append(item.name)

    if remaining_processed:
        print(f"⚠️  data/ 中仍有 {len(remaining_processed)} 个 _processed 目录:")
        for name in remaining_processed:
            print(f"   - {name}")
    else:
        print("✅ data/ 中没有 _processed 目录")

    print()

    # 统计 processed_data/ 中的目录
    if processed_dir.exists():
        datasets = [d.name for d in processed_dir.iterdir() if d.is_dir()]
        print(f"✅ processed_data/ 中有 {len(datasets)} 个数据集:")
        for name in sorted(datasets):
            dataset_path = processed_dir / name
            file_count = sum(1 for _ in dataset_path.rglob("*") if _.is_file())
            print(f"   - {name} ({file_count} 个文件)")
    else:
        print("❌ processed_data/ 目录不存在")

    print()


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_migration()
        return

    # 确认操作
    print("⚠️  此操作将移动 data/ 中以 _processed 结尾的目录到 processed_data/")
    print("   建议：在执行前先备份 data/ 目录")
    print()

    response = input("确认继续？(yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("❌ 操作已取消")
        return

    print()

    # 执行迁移
    success = migrate_processed_data()

    # 验证结果
    if success:
        verify_migration()


if __name__ == "__main__":
    main()
