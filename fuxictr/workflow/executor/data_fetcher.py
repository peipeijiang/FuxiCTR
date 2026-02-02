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
Data Fetch Executor - Stage 1

Fetches data from Server 21 (HDFS/Spark) and transfers via SSH to training server.

Features:
- SQL execution on Server 21
- Parquet export to local staging
- SSH/rsync transfer to training server
- Auto-detect feature_cols (not label_col - label_col is user-configured)
- Checkpoint-based resume

Directory Structure (exp_id.dataset_id naming):
  datasets_root/{exp_id.dataset_id}/
    ├── raw/              # Raw parquet from SQL export
    │   ├── train/        # Training set SQL export
    │   └── infer/        # Inference set SQL export
    ├── processed/        # Parquet切片 from build_dataset
    │   ├── train*.parquet
    │   ├── valid*.parquet
    │   ├── test*.parquet
    │   └── feature_map.json
    └── inference_output/ # Inference results (generated in Stage 3)
"""

import asyncio
import subprocess
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from fuxictr.workflow.executor.base import BaseExecutor
from fuxictr.workflow.utils.ssh_transfer import SSHTransferManager
from fuxictr.workflow.utils.feature_processor import (
    FeatureAutoDetector,
    DatasetBuilder,
    auto_process_dataset
)


logger = logging.getLogger(__name__)


class DataFetchExecutor(BaseExecutor):
    """
    Executor for Stage 1: Data Fetch.

    Workflow:
    1. Execute SQL on Server 21 (Hive/Spark)
    2. Export to local staging on Server 21
    3. Transfer via SSH/rsync to training server
    4. Verify data integrity
    5. Save checkpoint
    """

    def __init__(
        self,
        db_manager,
        transfer_manager: SSHTransferManager,
        config: Dict[str, Any],
        logger
    ):
        """
        Initialize data fetch executor.

        Args:
            db_manager: Database manager
            transfer_manager: SSH transfer manager
            config: Configuration dictionary
            logger: Workflow logger
        """
        super().__init__(db_manager, config, logger)
        self.transfer_manager = transfer_manager

        # Server configuration
        self.servers = config.get("servers", {})
        self.storage = config.get("storage", {})

    async def execute(
        self,
        task_id: int,
        task: Dict,
        step: Dict,
        checkpoint: Optional[Dict[str, Any]],
        cancel_event: Optional[asyncio.Event]
    ) -> Dict[str, Any]:
        """
        Execute data fetch stage.

        Args:
            task_id: Task ID
            task: Task dictionary with SQL queries
            step: Step dictionary
            checkpoint: Optional checkpoint data
            cancel_event: Optional cancellation event

        Returns:
            Result dictionary
        """
        step_name = "data_fetch"
        result = {
            "success": False,
            "error": None,
            "bytes_transferred": 0,
            "files_transferred": 0
        }

        try:
            self.logger.log(
                step_name,
                f"Starting data fetch for task {task_id}"
            )

            # Get server configurations
            source_server = self.servers.get("server_21", {})
            if not source_server:
                raise ValueError("server_21 not configured")

            # Get task parameters
            sample_sql = task.get("sample_sql", "")
            infer_sql = task.get("infer_sql", "")
            hdfs_path = task.get("hdfs_path", "")
            experiment_id = task.get("experiment_id", "")
            user = task.get("user", "")
            model = task.get("model", "")

            # Generate unique dataset_id: exp_id.dataset_id
            # Using timestamp to ensure uniqueness for each workflow run
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_id = f"{experiment_id}.{timestamp}"

            # Stage paths - using new directory structure
            server_21_staging = self.storage.get("server_21_staging", "/tmp/staging")
            fuxictr_paths = self.config.get("fuxictr_paths", {})
            datasets_root = fuxictr_paths.get("datasets_root", "/data/fuxictr/datasets")

            # Standardized directory structure: datasets_root/{exp_id.dataset_id}/
            dataset_dir = f"{datasets_root}/{dataset_id}"
            raw_dir = f"{dataset_dir}/raw"
            processed_dir = f"{dataset_dir}/processed"
            inference_output_dir = f"{dataset_dir}/inference_output"

            # Create directories
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            os.makedirs(inference_output_dir, exist_ok=True)

            self.logger.log(
                step_name,
                f"Using dataset_id: {dataset_id}"
            )
            self.logger.log(
                step_name,
                f"Dataset directory: {dataset_dir}"
            )

            # Step 1: Execute SQL on Server 21
            if cancel_event and cancel_event.is_set():
                return result

            await self._execute_sql_export(
                source_server, sample_sql, infer_sql,
                server_21_staging, dataset_id, step_name
            )

            # Step 2: Transfer data to training server (to raw/ directory)
            if cancel_event and cancel_event.is_set():
                return result

            transfer_result = await self._transfer_data(
                source_server, server_21_staging,
                raw_dir, dataset_id,
                task_id, step_name, checkpoint
            )

            result["bytes_transferred"] = transfer_result.get("bytes_transferred", 0)
            result["files_transferred"] = transfer_result.get("files_transferred", 0)

            # Step 3: Verify data integrity
            if cancel_event and cancel_event.is_set():
                return result

            await self._verify_data(
                raw_dir, dataset_id, step_name
            )

            # Step 4: Load user's label_col from dataset_config.yaml
            if cancel_event and cancel_event.is_set():
                return result

            label_col = await self._load_user_label_col(
                user, model, experiment_id, step_name
            )

            # Step 5: Auto-process dataset (feature detection + build_dataset)
            if cancel_event and cancel_event.is_set():
                return result

            dataset_result = await self._auto_process_dataset(
                raw_dir, processed_dir, dataset_id, step_name, task, label_col
            )

            # Step 6: Save checkpoint with processed data paths
            checkpoint_data = {
                "dataset_id": dataset_id,
                "sample_sql_executed": True,
                "infer_sql_executed": True,
                "data_transferred": True,
                "verified": True,
                "transfer_bytes": result["bytes_transferred"],
                "dataset_processed": True,
                "processed_data_paths": {
                    "dataset_id": dataset_id,
                    "dataset_dir": dataset_dir,
                    "train_data": dataset_result.get("train_data"),
                    "valid_data": dataset_result.get("valid_data"),
                    "test_data": dataset_result.get("test_data"),
                    "infer_data": f"{raw_dir}/infer/",  # Raw inference data
                    "feature_map": dataset_result.get("feature_map"),
                    "feature_cols": dataset_result.get("feature_cols"),
                    "label_col": label_col  # User-configured label_col
                },
                "raw_data_paths": {
                    "train_raw": f"{raw_dir}/train/",
                    "infer_raw": f"{raw_dir}/infer/"
                },
                "inference_output_path": inference_output_dir,
                "timestamp": datetime.now().isoformat()
            }
            self._save_checkpoint(task_id, step_name, checkpoint_data)

            # Update result with dataset info
            result["dataset_processed"] = True
            result["dataset_id"] = dataset_id
            result["processed_data_paths"] = dataset_result

            result["success"] = True
            return result

        except Exception as e:
            logger.exception(f"Data fetch failed for task {task_id}")
            result["error"] = str(e)
            return result

    async def _execute_sql_export(
        self,
        server: Dict[str, Any],
        sample_sql: str,
        infer_sql: str,
        staging_path: str,
        dataset_id: str,
        step_name: str
    ):
        """
        Execute SQL queries on Server 21 and export to parquet.

        Args:
            server: Server configuration
            sample_sql: Sample data SQL
            infer_sql: Inference data SQL
            staging_path: Staging directory on Server 21
            dataset_id: Dataset identifier
            step_name: Step name for logging
        """
        self.logger.log(
            step_name,
            "Executing SQL export on Server 21..."
        )

        host = server.get("host")
        port = server.get("port", 22)
        username = server.get("username")
        key_path = server.get("key_path")

        # Create staging directories
        sample_output = f"{staging_path}/{dataset_id}/train/"
        infer_output = f"{staging_path}/{dataset_id}/infer/"

        # Build SSH commands for directory creation
        mkdir_commands = [
            f"ssh -i {key_path} -p {port} {username}@{host} "
            f"'mkdir -p {sample_output} {infer_output}'"
        ]

        for cmd in mkdir_commands:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.wait()

        # Build Spark SQL commands
        # Note: Adjust based on your actual Spark/Hive setup
        sample_cmd = (
            f"ssh -i {key_path} -p {port} {username}@{host} "
            f"'" + f'spark-sql --master yarn -e "{sample_sql}" '
            f'--output-format parquet '
            f'--output {sample_output}' + "'"
        )

        infer_cmd = (
            f"ssh -i {key_path} -p {port} {username}@{host} "
            f"'" + f'spark-sql --master yarn -e "{infer_sql}" '
            f'--output-format parquet '
            f'--output {infer_output}' + "'"
        )

        # Execute sample export
        self.logger.log(step_name, "Exporting sample data...")
        proc = await asyncio.create_subprocess_shell(
            sample_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.wait()

        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            raise Exception(f"Sample export failed: {stderr.decode()}")

        # Execute inference export
        self.logger.log(step_name, "Exporting inference data...")
        proc = await asyncio.create_subprocess_shell(
            infer_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.wait()

        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            raise Exception(f"Inference export failed: {stderr.decode()}")

        self.logger.log(
            step_name,
            "SQL export completed successfully"
        )

    async def _transfer_data(
        self,
        source_server: Dict[str, Any],
        source_staging: str,
        dest_staging: str,
        dataset_id: str,
        task_id: int,
        step_name: str,
        checkpoint: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transfer data from Server 21 to training server via SSH.

        Args:
            source_server: Source server configuration
            source_staging: Source staging path
            dest_staging: Destination staging path
            dataset_id: Dataset identifier
            task_id: Task ID
            step_name: Step name
            checkpoint: Optional checkpoint for resume

        Returns:
            Transfer result dictionary
        """
        self.logger.log(
            step_name,
            "Transferring data via SSH/rsync..."
        )

        host = source_server.get("host")
        username = source_server.get("username")
        key_path = source_server.get("key_path")
        port = source_server.get("port", 22)

        # Create destination directory
        os.makedirs(f"{dest_staging}/{dataset_id}", exist_ok=True)

        # Transfer training data
        train_source = f"{source_staging}/{dataset_id}/train/"
        train_dest = f"{dest_staging}/{dataset_id}/train/"

        train_result = await self.transfer_manager.transfer_directory(
            source_host=host,
            source_path=train_source,
            dest_host="localhost",  # Local destination
            dest_path=train_dest,
            ssh_key=key_path,
            ssh_user=username,
            ssh_port=port,
            task_id=task_id,
            step_name=f"{step_name}_train"
        )

        if not train_result.success:
            raise Exception(f"Training data transfer failed: {train_result.error_message}")

        # Transfer inference data
        infer_source = f"{source_staging}/{dataset_id}/infer/"
        infer_dest = f"{dest_staging}/{dataset_id}/infer/"

        infer_result = await self.transfer_manager.transfer_directory(
            source_host=host,
            source_path=infer_source,
            dest_host="localhost",
            dest_path=infer_dest,
            ssh_key=key_path,
            ssh_user=username,
            ssh_port=port,
            task_id=task_id,
            step_name=f"{step_name}_infer"
        )

        if not infer_result.success:
            raise Exception(f"Inference data transfer failed: {infer_result.error_message}")

        total_bytes = train_result.bytes_transferred + infer_result.bytes_transferred
        total_files = train_result.files_transferred + infer_result.files_transferred

        self.logger.log(
            step_name,
            f"Transfer complete: {total_bytes} bytes, {total_files} files"
        )

        return {
            "bytes_transferred": total_bytes,
            "files_transferred": total_files
        }

    async def _verify_data(
        self,
        staging_path: str,
        dataset_id: str,
        step_name: str
    ):
        """
        Verify transferred data integrity.

        Args:
            staging_path: Staging directory (raw/ directory)
            dataset_id: Dataset identifier
            step_name: Step name for logging
        """
        self.logger.log(step_name, "Verifying data integrity...")

        try:
            import pyarrow.parquet as pq
            import pyarrow as pa

            # Check training data
            train_path = f"{staging_path}/train/"
            if os.path.exists(train_path):
                # Find parquet files
                import glob
                parquet_files = glob.glob(os.path.join(train_path, "*.parquet"))
                if parquet_files:
                    # Read first file to verify
                    table = pq.read_table(parquet_files[0])
                    self.logger.log(
                        step_name,
                        f"Training data verified: {len(table)} rows, {len(table.columns)} columns"
                    )
                else:
                    # Check if it's a directory with subdirectories
                    subdirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
                    if subdirs:
                        for subdir in subdirs:
                            subdir_path = os.path.join(train_path, subdir)
                            sub_files = glob.glob(os.path.join(subdir_path, "*.parquet"))
                            if sub_files:
                                table = pq.read_table(sub_files[0])
                                self.logger.log(
                                    step_name,
                                    f"Training data [{subdir}] verified: {len(table)} rows"
                                )

            # Check inference data
            infer_path = f"{staging_path}/infer/"
            if os.path.exists(infer_path):
                parquet_files = glob.glob(os.path.join(infer_path, "*.parquet"))
                if parquet_files:
                    table = pq.read_table(parquet_files[0])
                    self.logger.log(
                        step_name,
                        f"Inference data verified: {len(table)} rows, {len(table.columns)} columns"
                    )

            self.logger.log(step_name, "Data verification complete")

        except ImportError:
            # pyarrow not available, skip detailed verification
            self.logger.log(
                step_name,
                "Warning: pyarrow not available, skipping detailed verification"
            )
        except Exception as e:
            logger.warning(f"Data verification warning: {e}")

    async def _load_user_label_col(
        self,
        user: str,
        model: str,
        experiment_id: str,
        step_name: str
    ) -> List[str]:
        """
        Load user's label_col from dataset_config.yaml.

        The label_col is user-configured (not auto-detected).

        Search order:
        1. User's personal config: dashboard/user_configs/{user}/{model}/config/dataset_config.yaml
        2. Model default config: model_zoo/multitask/{model}/config/dataset_config.yaml

        Args:
            user: Username
            model: Model name
            experiment_id: Experiment ID (for logging)
            step_name: Step name for logging

        Returns:
            List of label column names, e.g., ["label"]
        """
        from fuxictr.workflow.utils.config_merge import find_dataset_config
        import yaml

        self.logger.log(
            step_name,
            f"Loading user's label_col from dataset_config.yaml..."
        )

        # Find dataset_config.yaml
        dataset_config_path = find_dataset_config(user, model)
        if not dataset_config_path:
            self.logger.log(
                step_name,
                f"Warning: dataset_config.yaml not found for {user}/{model}, using default ['label']"
            )
            return ["label"]

        try:
            with open(dataset_config_path, 'r') as f:
                dataset_config = yaml.safe_load(f)

            label_col = dataset_config.get("label_col", ["label"])
            self.logger.log(
                step_name,
                f"Loaded user's label_col: {label_col} from {dataset_config_path}"
            )
            return label_col

        except Exception as e:
            logger.warning(f"Failed to load label_col from dataset_config: {e}")
            self.logger.log(
                step_name,
                f"Warning: Failed to load label_col, using default ['label']"
            )
            return ["label"]

    async def _auto_process_dataset(
        self,
        raw_dir: str,
        processed_dir: str,
        dataset_id: str,
        step_name: str,
        task: Dict[str, Any],
        label_col: List[str]
    ) -> Dict[str, Any]:
        """
        Automatically process dataset: detect features and build_dataset.

        This replaces the manual "更新特征" step in the frontend dashboard.
        - feature_cols: Auto-detected from column names (_tag, _cnt, _textlist)
        - label_col: User-configured (passed as parameter)

        Args:
            raw_dir: Raw data directory (contains train/ and infer/ subdirs)
            processed_dir: Processed data directory (output of build_dataset)
            dataset_id: Dataset identifier (exp_id.timestamp)
            step_name: Step name for logging
            task: Task dictionary
            label_col: User-configured label column

        Returns:
            Dictionary with processed data paths
        """
        self.logger.log(
            step_name,
            "Auto-processing dataset: detecting features and building..."
        )

        train_raw = f"{raw_dir}/train/"
        infer_raw = f"{raw_dir}/infer/"

        # Check if train data exists
        if not os.path.exists(train_raw):
            raise FileNotFoundError(f"Training data not found at {train_raw}")

        # Find actual parquet file
        detector = FeatureAutoDetector()
        train_parquet = detector._find_parquet_file(train_raw)
        if not train_parquet:
            raise ValueError(f"No parquet file found in {train_raw}")

        self.logger.log(
            step_name,
            f"Found training data: {train_parquet}"
        )

        # Auto-detect features first (for logging)
        try:
            feature_cols = detector.detect_from_parquet(train_parquet)
            cat_count = sum(len(g.get("name", [])) for g in feature_cols if g.get("type") == "categorical")
            num_count = sum(len(g.get("name", [])) for g in feature_cols if g.get("type") == "numeric")
            seq_count = sum(len(g.get("name", [])) for g in feature_cols if g.get("type") == "sequence")
            self.logger.log(
                step_name,
                f"Auto-detected features: {cat_count} categorical, {num_count} numeric, {seq_count} sequence"
            )
            self.logger.log(
                step_name,
                f"User-configured label_col: {label_col}"
            )
        except Exception as e:
            self.logger.log(
                step_name,
                f"Warning: Feature detection had issues: {e}"
            )

        # Run build_dataset with user's label_col
        try:
            # Use processed_dir as the output directory
            build_result = await auto_process_dataset(
                data_root=processed_dir,  # Output to processed/ directory
                dataset_id=dataset_id,
                train_data_path=train_raw,
                valid_data_path=None,  # Will be split from train
                test_data_path=None,   # Will be split from train
                feature_cols=None,     # Auto-detect
                label_col=label_col    # User-configured
            )

            self.logger.log(
                step_name,
                f"Dataset built successfully:"
            )
            self.logger.log(
                step_name,
                f"  train_data -> {build_result.get('train_data')}"
            )
            self.logger.log(
                step_name,
                f"  valid_data -> {build_result.get('valid_data')}"
            )
            self.logger.log(
                step_name,
                f"  test_data -> {build_result.get('test_data')}"
            )
            self.logger.log(
                step_name,
                f"  feature_map -> {build_result.get('feature_map')}"
            )

            # Save inference data path for later use
            if os.path.exists(infer_raw):
                build_result["infer_data"] = infer_raw

            # Store info for config merge (will be done in training stage)
            build_result["original_config_needed"] = True
            build_result["user"] = task.get("user", "")
            build_result["model"] = task.get("model", "")

            self.logger.log(
                step_name,
                f"Feature processing complete. Config merge will be done in training stage."
            )

            return build_result

        except Exception as e:
            logger.exception(f"Dataset building failed: {e}")
            self.logger.log(
                step_name,
                f"Dataset building failed: {e}"
            )
            raise


# Backward compatibility alias
DataFetcher = DataFetchExecutor
