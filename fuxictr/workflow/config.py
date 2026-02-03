import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
from yaml import YAMLError


def _expand_env_vars(obj):
    """Recursively expand environment variables in a configuration object."""
    if isinstance(obj, str):
        # 扩展 ${VAR} 格式的环境变量
        return os.path.expandvars(obj)
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    else:
        return obj

@dataclass
class ServerConfig:
    """Configuration for a single server connection."""
    host: str
    port: int
    username: str
    key_path: Optional[str] = None
    password: Optional[str] = None

@dataclass
class ServersConfig:
    """Configuration container for all servers."""
    server_21: ServerConfig

@dataclass
class StorageConfig:
    """Configuration for storage directories."""
    shared_dir: str
    staging_dir: str
    checkpoint_dir: str

@dataclass
class TransferConfig:
    """Configuration for file transfer settings."""
    chunk_size: int
    max_retries: int
    parallel_workers: int
    verify_checksum: bool

@dataclass
class TaskConfig:
    """Configuration for task execution settings."""
    heartbeat_interval: int
    log_rotation_size: int

@dataclass
class Config:
    """Main configuration class for workflow settings."""
    servers: ServersConfig
    storage: StorageConfig
    transfer: TransferConfig
    task: TaskConfig

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration from a YAML file.

        Args:
            config_path: Path to the configuration file. Can be string or Path object.
                        Defaults to workflow/config.yaml in project root.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            YAMLError: If the YAML syntax is invalid.
            ValueError: If required configuration sections are missing.
        """
        if config_path is None:
            # Default to workflow/config.yaml in project root
            current_dir = Path(__file__).parent
            config_path = current_dir / "config.yaml"

        # Convert to Path object if string is provided
        config_path = Path(config_path) if isinstance(config_path, str) else config_path

        # Handle FileNotFoundError
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Handle YAMLError and missing sections
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        except YAMLError as e:
            raise YAMLError(f"Invalid YAML syntax in configuration file {config_path}: {e}")

        # 扩展环境变量
        data = _expand_env_vars(data)

        # 检查缺失的必需配置项
        required_sections = ['servers', 'storage', 'transfer', 'task']
        missing_sections = [section for section in required_sections if section not in data]
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {', '.join(missing_sections)}")

        self.servers = ServersConfig(**{k: ServerConfig(**v) for k, v in data['servers'].items()})
        self.storage = StorageConfig(**data['storage'])
        self.transfer = TransferConfig(**data['transfer'])
        self.task = TaskConfig(**data['task'])
