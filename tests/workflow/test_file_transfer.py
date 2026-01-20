import pytest
import tempfile
import os
from fuxictr.workflow.utils.file_transfer import FileTransferManager

@pytest.mark.asyncio
async def test_chunk_calculation():
    """Test file chunking calculation"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"0" * (20 * 1024 * 1024))  # 20MB file
        temp_path = f.name

    try:
        manager = FileTransferManager(chunk_size=10*1024*1024)
        chunks = manager.calculate_chunks(temp_path)
        assert len(chunks) == 2  # Should be 2 chunks
        assert chunks[0]["size"] == 10*1024*1024
    finally:
        os.unlink(temp_path)
