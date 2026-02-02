#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ConfigLoader - 配置智能加载器

支持从两个位置加载配置：
1. 用户配置: dashboard/user_configs/{username}/{model}/model_config.yaml (优先)
2. 默认配置: model_zoo/{model}/config/model_config.yaml (回退)

Author: FuxiCTR Team
Date: 2026-02-02
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional


class ConfigLoader:
    """
    配置加载器 - 智能加载用户配置或默认配置

    配置优先级:
        1. 用户配置 (dashboard/user_configs/{username}/{model}/model_config.yaml)
        2. 默认配置 (model_zoo/{model}/config/model_config.yaml)

    配置合并策略:
        - Base 配置作为基础
        - experiment_id 配置覆盖 Base 中的对应字段
    """

    def __init__(
        self,
        user_config_dir: str = "dashboard/user_configs",
        model_zoo_dir: str = "model_zoo"
    ):
        """
        初始化配置加载器

        Args:
            user_config_dir: 用户配置目录
            model_zoo_dir: 模型库目录
        """
        self.user_config_dir = user_config_dir
        self.model_zoo_dir = model_zoo_dir

    def load_model_config(
        self,
        username: str,
        model_name: str,
        exp_id: str
    ) -> Dict:
        """
        加载模型配置

        Args:
            username: 用户名
            model_name: 模型名称（如 AutoInt, DeepFM）
            exp_id: 实验 ID（如 AutoInt_test, default）

        Returns:
            合并后的配置字典 (Base + exp_id)

        Raises:
            ValueError: 如果 exp_id 不存在于配置中
            FileNotFoundError: 如果配置文件不存在

        Example:
            >>> loader = ConfigLoader()
            >>> config = loader.load_model_config("yeshao", "AutoInt", "AutoInt_test")
            >>> print(config["learning_rate"])
        """
        # 1. 尝试加载用户配置
        user_config = self._try_load_user_config(username, model_name, exp_id)
        if user_config is not None:
            return user_config

        # 2. 回退到默认配置
        default_config = self._load_default_config(model_name, exp_id)
        return default_config

    def _try_load_user_config(
        self,
        username: str,
        model_name: str,
        exp_id: str
    ) -> Optional[Dict]:
        """
        尝试加载用户配置

        Returns:
            配置字典，如果用户配置不存在则返回 None
        """
        config_path = os.path.join(
            self.user_config_dir,
            username,
            model_name,
            "model_config.yaml"
        )

        if not os.path.exists(config_path):
            return None

        return self._load_config_file(config_path, exp_id)

    def _load_default_config(self, model_name: str, exp_id: str) -> Dict:
        """
        加载默认配置

        Returns:
            配置字典

        Raises:
            FileNotFoundError: 如果默认配置文件不存在
            ValueError: 如果 exp_id 不存在于配置中
        """
        config_path = os.path.join(
            self.model_zoo_dir,
            model_name,
            "config",
            "model_config.yaml"
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Default config not found: {config_path}\n"
                f"Please ensure model exists in model_zoo/{model_name}/"
            )

        return self._load_config_file(config_path, exp_id)

    def _load_config_file(self, config_path: str, exp_id: str) -> Dict:
        """
        从配置文件加载并合并配置

        Args:
            config_path: 配置文件路径
            exp_id: 实验 ID

        Returns:
            合并后的配置字典

        Raises:
            ValueError: 如果 exp_id 不存在于配置中
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 合并 Base + exp_id
        base = config.get("Base", {})
        exp_config = config.get(exp_id)

        if exp_config is None:
            # 尝试查找所有可用的 exp_id
            available_exp_ids = [k for k in config.keys() if k != "Base"]
            raise ValueError(
                f"Experiment ID '{exp_id}' not found in config: {config_path}\n"
                f"Available experiment IDs: {available_exp_ids}"
            )

        # 合并配置（exp_config 覆盖 base）
        merged_config = {**base, **exp_config}

        # 保留原始配置的引用（用于调试）
        merged_config["_config_source"] = config_path
        merged_config["_base_config"] = base
        merged_config["_exp_config"] = exp_config

        return merged_config

    def list_available_experiments(
        self,
        username: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, list]:
        """
        列出可用的实验配置

        Args:
            username: 用户名（如果指定，只查看该用户的配置）
            model_name: 模型名称（如果指定，只查看该模型的配置）

        Returns:
            字典，键为模型名，值为可用实验 ID 列表

        Example:
            >>> loader = ConfigLoader()
            >>> experiments = loader.list_available_experiments()
            >>> print(experiments["AutoInt"])
            ['AutoInt_test', 'AutoInt_prod', 'default']
        """
        experiments = {}

        # 如果指定了用户名，只查看用户配置
        if username:
            config_roots = [os.path.join(self.user_config_dir, username)]
        else:
            # 否则查看用户配置和默认配置
            config_roots = [self.user_config_dir, self.model_zoo_dir]

        for config_root in config_roots:
            if not os.path.exists(config_root):
                continue

            for model_dir in os.listdir(config_root):
                model_path = os.path.join(config_root, model_dir)
                if not os.path.isdir(model_path):
                    continue

                # 查找 model_config.yaml
                config_file = os.path.join(model_path, "model_config.yaml")
                if not os.path.exists(config_file):
                    continue

                # 读取配置
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)

                    # 获取所有实验 ID（排除 Base）
                    exp_ids = [k for k in config.keys() if k != "Base"]

                    if model_name:
                        # 如果指定了模型名，只添加该模型
                        if model_dir == model_name:
                            experiments[model_dir] = exp_ids
                    else:
                        # 否则添加所有模型
                        if model_dir not in experiments:
                            experiments[model_dir] = []
                        experiments[model_dir].extend(exp_ids)

                except Exception as e:
                    # 跳过无法解析的配置文件
                    continue

        return experiments

    def get_config_path(
        self,
        username: str,
        model_name: str,
        use_user_config: bool = True
    ) -> str:
        """
        获取配置文件路径

        Args:
            username: 用户名
            model_name: 模型名称
            use_user_config: 是否使用用户配置

        Returns:
            配置文件路径
        """
        if use_user_config:
            return os.path.join(
                self.user_config_dir,
                username,
                model_name,
                "model_config.yaml"
            )
        else:
            return os.path.join(
                self.model_zoo_dir,
                model_name,
                "config",
                "model_config.yaml"
            )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    loader = ConfigLoader()

    # 测试列出可用的实验
    print("可用的实验配置:")
    experiments = loader.list_available_experiments()
    for model_name, exp_ids in sorted(experiments.items()):
        print(f"  {model_name}:")
        for exp_id in sorted(exp_ids):
            print(f"    - {exp_id}")
    print()

    # 测试加载配置
    try:
        print("测试加载配置:")
        config = loader.load_model_config("yeshao", "AutoInt", "AutoInt_test")
        print(f"  learning_rate: {config.get('learning_rate')}")
        print(f"  batch_size: {config.get('batch_size')}")
        print(f"  epochs: {config.get('epochs')}")
        print(f"  配置来源: {config.get('_config_source')}")
    except Exception as e:
        print(f"  加载失败: {e}")
