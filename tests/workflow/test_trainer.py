import pytest
from fuxictr.workflow.executor.trainer import Trainer

@pytest.mark.asyncio
async def test_trainer_initialization():
    """Test trainer initialization"""
    trainer = Trainer(
        config=None,
        db_manager=None,
        logger=None,
        task_id=1
    )
    assert trainer is not None
