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
Feature Auto-Detection and Processing Module for Workflow.

Automatically detects feature types from parquet files and generates
feature_cols configuration, then runs build_dataset.
"""

import os
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureAutoDetector:
    """
    Automatically detect feature types from parquet files.

    Mirrors the frontend "更新特征" functionality in dashboard/app.py
    """

    # Special sequence columns
    SPECIAL_SEQUENCES = ["appInstalls", "outerBizSorted", "outerModelCleanSorted"]

    # Column suffix patterns
    CATEGORICAL_SUFFIXES = ["_tag"]
    NUMERIC_SUFFIXES = ["_cnt"]
    SEQUENCE_SUFFIXES = ["_textlist"]

    def __init__(self):
        pass

    def detect_from_parquet(self, parquet_path: str) -> List[Dict[str, Any]]:
        """
        Detect feature columns from a parquet file.

        Args:
            parquet_path: Path to parquet file (file or directory)

        Returns:
            List of feature column dicts with format:
            [
                {"name": [...], "type": "categorical", "dtype": "int", "active": True},
                {"name": [...], "type": "numeric", "dtype": "float", "active": True, "normalizer": "StandardScaler"},
                {"name": [...], "type": "sequence", "dtype": "str", "active": True, "max_len": 15, "encoder": "MaskedAveragePooling"}
            ]
        """
        # Find parquet file
        target_file = self._find_parquet_file(parquet_path)
        if not target_file:
            raise ValueError(f"No parquet file found at {parquet_path}")

        logger.info(f"Reading parquet file: {target_file}")
        try:
            df = pd.read_parquet(target_file)
        except Exception as e:
            raise ValueError(f"Failed to read parquet {target_file}: {e}")

        cols = list(df.columns)
        logger.info(f"Detected columns: {cols}")

        # Detect features by suffix
        categorical = self._detect_categorical(cols)
        numeric = self._detect_numeric(cols)
        sequence = self._detect_sequence(cols)

        # Build feature_cols list
        feature_cols = []
        if categorical:
            feature_cols.append(OrderedDict({
                "name": categorical,
                "type": "categorical",
                "dtype": "int",
                "active": True
            }))
            logger.info(f"Detected {len(categorical)} categorical features")

        if numeric:
            feature_cols.append(OrderedDict({
                "name": numeric,
                "type": "numeric",
                "dtype": "float",
                "active": True,
                "normalizer": "StandardScaler"
            }))
            logger.info(f"Detected {len(numeric)} numeric features")

        if sequence:
            feature_cols.append(OrderedDict({
                "name": sequence,
                "type": "sequence",
                "dtype": "str",
                "active": True,
                "max_len": 15,
                "encoder": "MaskedAveragePooling"
            }))
            logger.info(f"Detected {len(sequence)} sequence features")

        if not feature_cols:
            logger.warning(f"No features detected from columns: {cols}")
            return []

        return [dict(fc) for fc in feature_cols]

    def _find_parquet_file(self, path: str) -> Optional[str]:
        """Find the first parquet file in a path."""
        if not path:
            return None

        abs_path = os.path.abspath(path)

        # Direct file
        if os.path.isfile(abs_path) and abs_path.endswith('.parquet'):
            return abs_path

        # Directory
        if os.path.isdir(abs_path):
            parquet_files = sorted([f for f in os.listdir(abs_path) if f.endswith('.parquet')])
            if parquet_files:
                return os.path.join(abs_path, parquet_files[0])

        return None

    def _detect_categorical(self, cols: List[str]) -> List[str]:
        """Detect categorical features by suffix."""
        cat = [c for c in cols if any(c.endswith(suf) for suf in self.CATEGORICAL_SUFFIXES)]
        # Special case: "product" column
        if "product" in cols and "product" not in cat:
            cat.append("product")
        return self._unique_keep_order(cat)

    def _detect_numeric(self, cols: List[str]) -> List[str]:
        """Detect numeric features by suffix."""
        num = [c for c in cols if any(c.endswith(suf) for suf in self.NUMERIC_SUFFIXES)]
        return self._unique_keep_order(num)

    def _detect_sequence(self, cols: List[str]) -> List[str]:
        """Detect sequence features by suffix and special names."""
        seq = [c for c in cols if any(c.endswith(suf) for suf in self.SEQUENCE_SUFFIXES)]
        # Add special sequences if present
        for special in self.SPECIAL_SEQUENCES:
            if special in cols and special not in seq:
                seq.append(special)
        return self._unique_keep_order(seq)

    @staticmethod
    def _unique_keep_order(lst: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        out = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return sorted(out)


class DatasetBuilder:
    """
    Build dataset with feature processing for workflow.

    Integrates FeatureProcessor and build_dataset from fuxictr.preprocess
    """

    def __init__(self, data_root: str, dataset_id: str):
        """
        Initialize dataset builder.

        Args:
            data_root: Root data directory
            dataset_id: Dataset identifier (used as subdirectory name)
        """
        self.data_root = data_root
        self.dataset_id = dataset_id
        self.data_dir = os.path.join(data_root, dataset_id)

    def build(
        self,
        train_data_path: str,
        valid_data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        feature_cols: Optional[List[Dict[str, Any]]] = None,
        label_col: Optional[List[str]] = None,
        feature_cols_only: bool = False
    ) -> Dict[str, str]:
        """
        Build dataset with feature processing.

        Args:
            train_data_path: Path to training data (parquet)
            valid_data_path: Path to validation data (optional)
            test_data_path: Path to test data (optional)
            feature_cols: Feature column definitions (auto-detected if None)
            label_col: Label column name (auto-detected if None)
            feature_cols_only: If True, only detect features without building

        Returns:
            Dictionary with processed data paths:
            {
                "train_data": "...",
                "valid_data": "...",
                "test_data": "...",
                "feature_map": "...",
                "feature_cols": [...]
            }
        """
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Auto-detect feature_cols if not provided
        if feature_cols is None:
            detector = FeatureAutoDetector()
            feature_cols = detector.detect_from_parquet(train_data_path)

            if not feature_cols:
                raise ValueError(f"Failed to auto-detect features from {train_data_path}")

            logger.info(f"Auto-detected feature_cols with {len(feature_cols)} groups")

        # Auto-detect label_col if not provided
        if label_col is None:
            label_col = self._detect_label_col(train_data_path)
            logger.info(f"Auto-detected label_col: {label_col}")

        if feature_cols_only:
            return {
                "feature_cols": feature_cols,
                "label_col": label_col
            }

        # Import fuxictr preprocess modules
        try:
            from fuxictr.preprocess import FeatureProcessor, build_dataset as fuxi_build_dataset
        except ImportError:
            raise ImportError("Failed to import fuxictr.preprocess modules")

        # Create feature processor
        feature_processor = FeatureProcessor(
            feature_cols=feature_cols,
            label_col=label_col,
            dataset_id=self.dataset_id,
            data_root=self.data_root
        )

        # Build dataset
        logger.info(f"Building dataset with train_data={train_data_path}")

        # Map data paths (handle relative paths)
        train_data = self._resolve_data_path(train_data_path)
        valid_data = self._resolve_data_path(valid_data_path) if valid_data_path else None
        test_data = self._resolve_data_path(test_data_path) if test_data_path else None

        # Run build_dataset
        processed_train, processed_valid, processed_test = fuxi_build_dataset(
            feature_processor,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data
        )

        feature_map_path = os.path.join(self.data_dir, "feature_map.json")

        logger.info(f"Dataset built successfully:")
        logger.info(f"  train_data -> {processed_train}")
        logger.info(f"  valid_data -> {processed_valid}")
        logger.info(f"  test_data -> {processed_test}")
        logger.info(f"  feature_map -> {feature_map_path}")

        return {
            "train_data": processed_train,
            "valid_data": processed_valid,
            "test_data": processed_test,
            "feature_map": feature_map_path,
            "feature_cols": feature_cols,
            "label_col": label_col
        }

    def _resolve_data_path(self, path: str) -> str:
        """Resolve data path (handle relative paths)."""
        if not path:
            return path

        if os.path.isabs(path):
            return path

        # Try relative to data_root
        candidate = os.path.join(self.data_root, path)
        if os.path.exists(candidate):
            return candidate

        return path

    def _detect_label_col(self, data_path: str) -> List[str]:
        """Auto-detect label column from data."""
        target_file = FeatureAutoDetector()._find_parquet_file(data_path)
        if not target_file:
            return ["label"]  # Default fallback

        df = pd.read_parquet(target_file)
        cols = list(df.columns)

        # Common label column names
        label_candidates = ["label", "target", "y", "click", "conversion"]
        for candidate in label_candidates:
            if candidate in cols:
                return [candidate]

        # Fallback: use last column
        return [cols[-1]] if cols else ["label"]


async def auto_process_dataset(
    data_root: str,
    dataset_id: str,
    train_data_path: str,
    valid_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    feature_cols: Optional[List[Dict[str, Any]]] = None,
    label_col: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Async wrapper for dataset building.

    This function can be called from workflow executors.
    """
    import asyncio

    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    builder = DatasetBuilder(data_root, dataset_id)

    return await loop.run_in_executor(
        None,
        builder.build,
        train_data_path,
        valid_data_path,
        test_data_path,
        feature_cols,
        label_col,
        False
    )
