#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PathManager - 路径管理器

管理 Dashboard 和 Workflow 的路径隔离，确保两个系统使用不同的存储空间。

Author: FuxiCTR Team
Date: 2026-02-02
"""

import os
from pathlib import Path
from typing import Dict, Optional


class PathManager:
    """
    路径管理器 - 管理 Dashboard 和 Workflow 的路径隔离

    设计原则:
    - Dashboard 和 Workflow 使用完全独立的存储空间
    - 原始数据和处理后数据分离
    - 模型按 dataset_id/exp_id 组织，每个实验独立文件夹
    - 日志按用途分离，互不干扰

    目录结构:
    fuxictr/
    ├── data/                          # Dashboard 原始数据
    ├── processed_data/                # Dashboard 处理后数据
    ├── workflow_data/
    │   ├── datasets/                  # Workflow 原始数据
    │   └── processed/                 # Workflow 处理后数据
    ├── model_zoo/                     # Dashboard 模型
    └── workflow_models/               # Workflow 模型
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化路径管理器

        Args:
            config: 配置字典，包含 storage 路径配置
        """
        self.config = config or {}

        # 获取 storage 配置
        storage = self.config.get("storage", {})

        # ========== Dashboard 路径 ==========
        self.dashboard_data_root = storage.get(
            "dashboard_data_root", "../../../data/"
        )
        self.dashboard_processed_root = storage.get(
            "dashboard_processed_root", "../../../processed_data/"
        )
        self.dashboard_model_root = storage.get(
            "dashboard_model_root", "../../../model_zoo/"
        )
        self.dashboard_log_dir = storage.get(
            "dashboard_log_dir", "../../../dashboard/logs/"
        )

        # ========== Workflow 路径 ==========
        self.workflow_datasets_root = storage.get(
            "workflow_datasets_root", "./workflow_data/datasets/"
        )
        self.workflow_processed_root = storage.get(
            "workflow_processed_root", "./workflow_data/processed/"
        )
        self.workflow_model_root = storage.get(
            "workflow_model_root", "./workflow_models/"
        )
        self.workflow_log_root = storage.get(
            "workflow_log_root", "./workflow_logs/"
        )

        # ========== 通用路径 ==========
        self.db_backup_dir = storage.get(
            "db_backup_dir", "./db_backup/"
        )

        # 确保路径以 / 结尾（方便拼接）
        self._normalize_paths()

    def _normalize_paths(self):
        """标准化路径格式，确保以 / 结尾"""
        for attr_name in dir(self):
            if attr_name.endswith("_root") or attr_name.endswith("_dir"):
                path = getattr(self, attr_name)
                if path and not path.endswith("/"):
                    setattr(self, attr_name, path + "/")

    # ========================================================================
    # Dashboard 路径方法
    # ========================================================================

    def get_dashboard_data_dir(self, dataset_id: str) -> str:
        """
        获取 Dashboard 原始数据目录

        Args:
            dataset_id: 数据集 ID

        Returns:
            Dashboard 原始数据目录路径

        Example:
            >>> pm.get_dashboard_data_dir("tiny_npz")
            '../../../data/tiny_npz/'
        """
        return os.path.join(self.dashboard_data_root, dataset_id)

    def get_dashboard_processed_dir(self, dataset_id: str) -> str:
        """
        获取 Dashboard 处理后数据目录

        Args:
            dataset_id: 数据集 ID

        Returns:
            Dashboard 处理后数据目录路径

        Example:
            >>> pm.get_dashboard_processed_dir("tiny_npz")
            '../../../processed_data/tiny_npz/'
        """
        return os.path.join(self.dashboard_processed_root, dataset_id)

    def get_dashboard_model_dir(self, model_name: str, dataset_id: str, exp_id: str) -> str:
        """
        获取 Dashboard 模型目录（按实验组织）

        Args:
            model_name: 模型名称（如 AutoInt, DeepFM）
            dataset_id: 数据集 ID
            exp_id: 实验 ID

        Returns:
            Dashboard 模型目录路径

        Example:
            >>> pm.get_dashboard_model_dir("AutoInt", "tiny_npz", "test")
            '../../../model_zoo/AutoInt/checkpoints/tiny_npz/test/'
        """
        return os.path.join(
            self.dashboard_model_root,
            model_name,
            "checkpoints",
            dataset_id,
            exp_id
        )

    def get_dashboard_log_file(self, username: str, exp_id: str, timestamp: str) -> str:
        """
        获取 Dashboard 训练日志副本路径

        Args:
            username: 用户名
            exp_id: 实验 ID
            timestamp: 时间戳

        Returns:
            Dashboard 日志文件路径

        Note:
            这是训练日志的副本（stdout/stderr 重定向），原始日志在模型文件夹内
        """
        return os.path.join(
            self.dashboard_log_dir,
            "users" if username else "training",
            f"{exp_id}_{username or 'common'}_train_{timestamp}.log"
        )

    # ========================================================================
    # Workflow 路径方法
    # ========================================================================

    def get_workflow_dataset_dir(self, dataset_id: str) -> str:
        """
        获取 Workflow 原始数据目录

        Args:
            dataset_id: 数据集 ID

        Returns:
            Workflow 原始数据目录路径

        Example:
            >>> pm.get_workflow_dataset_dir("jrzk_seeds_20260201")
            './workflow_data/datasets/jrzk_seeds_20260201/raw/'
        """
        return os.path.join(self.workflow_datasets_root, dataset_id, "raw")

    def get_workflow_processed_dir(self, dataset_id: str) -> str:
        """
        获取 Workflow 处理后数据目录

        Args:
            dataset_id: 数据集 ID

        Returns:
            Workflow 处理后数据目录路径

        Example:
            >>> pm.get_workflow_processed_dir("jrzk_seeds_20260201")
            './workflow_data/processed/jrzk_seeds_20260201/'
        """
        return os.path.join(self.workflow_processed_root, dataset_id)

    def get_workflow_model_dir(self, model_name: str, dataset_id: str, task_id: int, exp_id: str) -> str:
        """
        获取 Workflow 模型目录（按任务 ID 组织）

        Args:
            model_name: 模型名称
            dataset_id: 数据集 ID
            task_id: 任务 ID
            exp_id: 实验 ID

        Returns:
            Workflow 模型目录路径

        Example:
            >>> pm.get_workflow_model_dir("AutoInt", "jrzk_seeds_20260201", 1, "test")
            './workflow_models/AutoInt/jrzk_seeds_20260201/task_1_test/'
        """
        exp_full_id = f"task_{task_id}_{exp_id}"
        return os.path.join(
            self.workflow_model_root,
            model_name,
            dataset_id,
            exp_full_id
        )

    def get_workflow_log_file(self, task_id: int, stage_name: str) -> str:
        """
        获取 Workflow 工作流日志文件路径

        Args:
            task_id: 任务 ID
            stage_name: 阶段名称（data_fetch, train, infer, transport, monitor）

        Returns:
            Workflow 日志文件路径

        Example:
            >>> pm.get_workflow_log_file(1, "train")
            './workflow_logs/task_1_train.log'
        """
        return os.path.join(
            self.workflow_log_root,
            f"task_{task_id}_{stage_name}.log"
        )

    # ========================================================================
    # 工具方法
    # ========================================================================

    def ensure_dir(self, dir_path: str) -> str:
        """
        确保目录存在，不存在则创建

        Args:
            dir_path: 目录路径

        Returns:
            目录路径
        """
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def get_relative_path(self, abs_path: str, base_dir: Optional[str] = None) -> str:
        """
        获取相对路径

        Args:
            abs_path: 绝对路径
            base_dir: 基础目录（默认为项目根目录）

        Returns:
            相对路径
        """
        if base_dir is None:
            base_dir = os.getcwd()
        return os.path.relpath(abs_path, base_dir)

    def resolve_path(self, path: str) -> str:
        """
        解析路径，处理相对路径和 ~

        Args:
            path: 路径

        Returns:
            解析后的绝对路径
        """
        return os.path.abspath(os.path.expanduser(path))

    def to_dict(self) -> Dict:
        """
        转换为字典（用于调试）

        Returns:
            所有路径的字典
        """
        return {
            "dashboard_data_root": self.dashboard_data_root,
            "dashboard_processed_root": self.dashboard_processed_root,
            "dashboard_model_root": self.dashboard_model_root,
            "dashboard_log_dir": self.dashboard_log_dir,
            "workflow_datasets_root": self.workflow_datasets_root,
            "workflow_processed_root": self.workflow_processed_root,
            "workflow_model_root": self.workflow_model_root,
            "workflow_log_root": self.workflow_log_root,
            "db_backup_dir": self.db_backup_dir,
        }

    def __repr__(self) -> str:
        """字符串表示"""
        return f"PathManager({self.to_dict()})"


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试 PathManager
    config = {
        "storage": {
            "dashboard_data_root": "../../../data/",
            "dashboard_processed_root": "../../../processed_data/",
            "workflow_datasets_root": "./workflow_data/datasets/",
            "workflow_processed_root": "./workflow_data/processed/",
        }
    }

    pm = PathManager(config)

    print("Dashboard 路径:")
    print(f"  数据: {pm.get_dashboard_data_dir('tiny_npz')}")
    print(f"  处理后: {pm.get_dashboard_processed_dir('tiny_npz')}")
    print()

    print("Workflow 路径:")
    print(f"  数据集: {pm.get_workflow_dataset_dir('jrzk_seeds_20260201')}")
    print(f"  处理后: {pm.get_workflow_processed_dir('jrzk_seeds_20260201')}")
    print()

    print("模型路径:")
    print(f"  Dashboard: {pm.get_dashboard_model_dir('AutoInt', 'tiny_npz', 'test')}")
    print(f"  Workflow: {pm.get_workflow_model_dir('AutoInt', 'jrzk_seeds_20260201', 1, 'test')}")
    print()

    print("日志路径:")
    print(f"  Dashboard: {pm.get_dashboard_log_file('yeshao', 'AutoInt_test', '20260202_120000')}")
    print(f"  Workflow: {pm.get_workflow_log_file(1, 'train')}")
