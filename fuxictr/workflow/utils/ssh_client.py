import asyncio
import paramiko
from typing import Optional


class SSHCommandError(Exception):
    """Custom exception for SSH command execution errors"""
    def __init__(self, message: str, exit_code: Optional[int] = None, stderr: Optional[str] = None):
        self.exit_code = exit_code
        self.stderr = stderr
        super().__init__(message)


class SSHClient:
    DEFAULT_CONNECTION_TIMEOUT = 30
    DEFAULT_COMMAND_TIMEOUT = 300

    def __init__(self, host: str, port: int, username: str,
                 key_path: Optional[str] = None,
                 password: Optional[str] = None,
                 connection_timeout: Optional[int] = None,
                 command_timeout: Optional[int] = None):
        self._validate_host(host)
        self._validate_port(port)
        self._validate_auth(key_path, password)

        self.host = host
        self.port = port
        self.username = username
        self.key_path = key_path
        self.password = password
        self.connection_timeout = connection_timeout or self.DEFAULT_CONNECTION_TIMEOUT
        self.command_timeout = command_timeout or self.DEFAULT_COMMAND_TIMEOUT
        self.client: Optional[paramiko.SSHClient] = None

    def _validate_host(self, host: str):
        """Validate host parameter"""
        if not host or not isinstance(host, str):
            raise ValueError("Host must be a non-empty string")

    def _validate_port(self, port: int):
        """Validate port parameter"""
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError("Port must be an integer between 1 and 65535")

    def _validate_auth(self, key_path: Optional[str], password: Optional[str]):
        """Validate that at least one auth method is provided"""
        if not key_path and not password:
            raise ValueError("Either key_path or password must be provided")

    def _validate_command(self, command: str):
        """Validate command parameter"""
        if not command or not isinstance(command, str):
            raise ValueError("Command must be a non-empty string")

    async def connect(self):
        """Establish SSH connection"""
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, self._connect_sync),
            timeout=self.connection_timeout
        )

    def _connect_sync(self):
        """Synchronous connection"""
        connect_kwargs = {
            'hostname': self.host,
            'port': self.port,
            'username': self.username,
            'timeout': self.connection_timeout
        }

        if self.key_path:
            connect_kwargs['key_filename'] = self.key_path
        else:
            connect_kwargs['password'] = self.password

        self.client.connect(**connect_kwargs)

    async def execute(self, command: str) -> str:
        """Execute command and return output"""
        self._validate_command(command)

        if not self.client:
            await self.connect()

        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self._execute_sync, command),
            timeout=self.command_timeout
        )

    def _execute_sync(self, command: str) -> str:
        """Synchronous command execution"""
        stdin, stdout, stderr = self.client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()

        output = stdout.read().decode()
        error = stderr.read().decode()

        if exit_code != 0:
            raise SSHCommandError(
                f"Command failed with exit code {exit_code}",
                exit_code=exit_code,
                stderr=error if error else None
            )

        return output

    async def close(self):
        """Close SSH connection"""
        if self.client:
            self.client.close()
            self.client = None
