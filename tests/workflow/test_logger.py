import pytest
import asyncio
from fuxictr.workflow.utils.logger import WorkflowLogger

@pytest.mark.asyncio
async def test_logger_creates_log_message():
    """Test that logger creates properly formatted log messages"""
    logger = WorkflowLogger(task_id=1)
    message = logger.log("test_step", "Test message", level="INFO")

    assert message["type"] == "log"
    assert message["task_id"] == 1
    assert message["step"] == "test_step"
    assert message["data"]["message"] == "Test message"

@pytest.mark.asyncio
async def test_logger_progress():
    """Test progress logging"""
    logger = WorkflowLogger(task_id=1)
    message = logger.progress("test_step", 50, 100)

    assert message["type"] == "progress"
    assert message["data"]["percent"] == 50

@pytest.mark.asyncio
async def test_logger_error():
    """Test error logging"""
    logger = WorkflowLogger(task_id=1)
    message = logger.error("test_step", "Error occurred")

    assert message["type"] == "error"
    assert message["data"]["message"] == "Error occurred"

@pytest.mark.asyncio
async def test_logger_complete():
    """Test complete logging"""
    logger = WorkflowLogger(task_id=1)
    message = logger.complete("test_step")

    assert message["type"] == "complete"
    assert message["step"] == "test_step"
