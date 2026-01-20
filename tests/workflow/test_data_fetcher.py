import pytest
from unittest.mock import Mock
from fuxictr.workflow.executor.data_fetcher import DataFetcher

@pytest.mark.asyncio
async def test_data_fetcher_initialization():
    """Test data fetcher initialization"""
    # Create mock config
    mock_config = Mock()
    mock_config.transfer.chunk_size = 1024 * 1024
    mock_config.transfer.max_retries = 3

    fetcher = DataFetcher(
        ssh_client=Mock(),
        config=mock_config,
        db_manager=Mock(),
        logger=Mock(),
        task_id=1
    )
    assert fetcher is not None
    assert fetcher.task_id == 1
