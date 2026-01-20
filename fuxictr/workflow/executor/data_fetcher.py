import asyncio
from typing import Dict, Any
from fuxictr.workflow.utils.ssh_client import SSHClient
from fuxictr.workflow.config import Config
from fuxictr.workflow.db import DatabaseManager
from fuxictr.workflow.utils.logger import WorkflowLogger
from fuxictr.workflow.utils.file_transfer import FileTransferManager
from fuxictr.workflow.models import StepStatus

class DataFetcher:
    def __init__(self, ssh_client: SSHClient, config: Config,
                 db_manager: DatabaseManager, logger: WorkflowLogger,
                 task_id: int):
        self.ssh_client = ssh_client
        self.config = config
        self.db = db_manager
        self.logger = logger
        self.task_id = task_id
        self.transfer_manager = FileTransferManager(
            chunk_size=config.transfer.chunk_size,
            max_retries=config.transfer.max_retries
        )

    async def execute(self, sample_sql: str, infer_sql: str,
                     hdfs_path: str) -> bool:
        """Execute data fetching stage"""
        step_name = "data_fetch"

        self.db.update_step_status(self.task_id, step_name, StepStatus.RUNNING)
        self.logger.log(step_name, "Starting data fetch stage")

        try:
            # Export from HDFS
            self.logger.log(step_name, "Exporting data from HDFS")
            await self._export_from_hdfs(sample_sql, infer_sql, hdfs_path)

            # Transfer data
            self.logger.log(step_name, "Transferring data to server 142")
            await self._transfer_data()

            # Verify data
            self.logger.log(step_name, "Verifying data integrity")
            await self._verify_data()

            self.db.update_step_status(self.task_id, step_name, StepStatus.COMPLETED)
            self.logger.complete(step_name)
            return True

        except Exception as e:
            self.db.update_step_status(
                self.task_id, step_name, StepStatus.FAILED,
                error_message=str(e)
            )
            self.logger.error(step_name, str(e))
            return False

    async def _export_from_hdfs(self, sample_sql: str, infer_sql: str, hdfs_path: str):
        """Export data from HDFS to shared directory"""
        sample_cmd = f"spark.sql --master yarn -e \"{sample_sql}\" --output-format parquet --output {self.config.storage.shared_dir}/sample_data"
        infer_cmd = f"spark.sql --master yarn -e \"{infer_sql}\" --output-format parquet --output {self.config.storage.shared_dir}/infer_data"

        await self.ssh_client.execute(sample_cmd)
        await self.ssh_client.execute(infer_cmd)

    async def _transfer_data(self):
        """Transfer data from shared dir to local"""
        sample_src = f"{self.config.storage.shared_dir}/sample_data"
        infer_src = f"{self.config.storage.shared_dir}/infer_data"
        sample_dst = f"{self.config.storage.staging_dir}/sample_data"
        infer_dst = f"{self.config.storage.staging_dir}/infer_data"

        await self.transfer_manager.download_with_resume(
            self.ssh_client, sample_src, sample_dst,
            self.db, self.task_id, "data_fetch", self.logger
        )

        await self.transfer_manager.download_with_resume(
            self.ssh_client, infer_src, infer_dst,
            self.db, self.task_id, "data_fetch", self.logger
        )

    async def _verify_data(self):
        """Verify downloaded data integrity"""
        import pyarrow.parquet as pq

        sample_path = f"{self.config.storage.staging_dir}/sample_data"
        infer_path = f"{self.config.storage.staging_dir}/infer_data"

        sample_df = pq.read_table(sample_path).to_pandas()
        assert len(sample_df) > 0, "Sample data is empty"

        infer_df = pq.read_table(infer_path).to_pandas()
        assert len(infer_df) > 0, "Infer data is empty"

        self.logger.log("data_fetch", f"Verified: {len(sample_df)} samples, {len(infer_df)} infer rows")
