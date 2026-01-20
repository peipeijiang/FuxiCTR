import pytest
from fuxictr.workflow.utils.ssh_client import SSHClient, SSHCommandError
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.asyncio
async def test_ssh_client_connects():
    """Test SSH client connection"""
    client = SSHClient(host="testhost", port=22, username="testuser", password="testpass")
    assert client is not None


@pytest.mark.asyncio
async def test_ssh_execute_command():
    """Test executing command via SSH"""
    client = SSHClient(host="testhost", port=22, username="testuser", password="testpass")

    with patch.object(client, '_connect_sync'):
        with patch.object(client, '_execute_sync') as mock_exec:
            mock_exec.return_value = "success"
            result = await client.execute("ls /tmp")
            assert result == "success"


def test_ssh_client_invalid_host():
    """Test SSH client rejects invalid host"""
    with pytest.raises(ValueError, match="Host must be a non-empty string"):
        SSHClient(host="", port=22, username="testuser", password="testpass")

    with pytest.raises(ValueError, match="Host must be a non-empty string"):
        SSHClient(host=None, port=22, username="testuser", password="testpass")


def test_ssh_client_invalid_port():
    """Test SSH client rejects invalid port"""
    with pytest.raises(ValueError, match="Port must be an integer between 1 and 65535"):
        SSHClient(host="testhost", port=0, username="testuser", password="testpass")

    with pytest.raises(ValueError, match="Port must be an integer between 1 and 65535"):
        SSHClient(host="testhost", port=65536, username="testuser", password="testpass")

    with pytest.raises(ValueError, match="Port must be an integer between 1 and 65535"):
        SSHClient(host="testhost", port="invalid", username="testuser", password="testpass")


def test_ssh_client_no_auth():
    """Test SSH client requires authentication method"""
    with pytest.raises(ValueError, match="Either key_path or password must be provided"):
        SSHClient(host="testhost", port=22, username="testuser")


def test_ssh_client_invalid_command():
    """Test SSH client validates commands"""
    client = SSHClient(host="testhost", port=22, username="testuser", password="testpass")

    with pytest.raises(ValueError, match="Command must be a non-empty string"):
        import asyncio
        asyncio.run(client.execute(""))


def test_ssh_command_error():
    """Test custom SSHCommandError exception"""
    error = SSHCommandError("Command failed", exit_code=1, stderr="Error output")
    assert error.exit_code == 1
    assert error.stderr == "Error output"
    assert str(error) == "Command failed"


@pytest.mark.asyncio
async def test_ssh_close_clears_client():
    """Test that close() properly clears the client reference"""
    client = SSHClient(host="testhost", port=22, username="testuser", password="testpass")

    # Mock the client
    mock_ssh_client = MagicMock()
    client.client = mock_ssh_client

    await client.close()

    # Verify close was called and client was cleared
    mock_ssh_client.close.assert_called_once()
    assert client.client is None


@pytest.mark.asyncio
async def test_ssh_client_timeouts():
    """Test SSH client with custom timeouts"""
    client = SSHClient(
        host="testhost",
        port=22,
        username="testuser",
        password="testpass",
        connection_timeout=60,
        command_timeout=600
    )

    assert client.connection_timeout == 60
    assert client.command_timeout == 600
