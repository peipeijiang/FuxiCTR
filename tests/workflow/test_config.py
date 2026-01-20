import pytest
import os
from fuxictr.workflow.config import Config

def test_load_config():
    """Test loading configuration from file"""
    config = Config()
    assert config.servers.server_21.host is not None
    assert config.storage.shared_dir is not None
    assert config.transfer.chunk_size == 10485760

def test_get_nested_config():
    """Test accessing nested configuration"""
    config = Config()
    assert hasattr(config, 'servers')
    assert hasattr(config, 'storage')
    assert hasattr(config, 'transfer')
