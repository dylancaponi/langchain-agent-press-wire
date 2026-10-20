import pytest
from pytest_socket import disable_socket, enable_socket


@pytest.fixture(autouse=True)
def _no_network():
    """Unit tests never touch the network; HTTP is mocked with respx.
    Unix sockets stay allowed because asyncio's event loop needs them."""
    disable_socket(allow_unix_socket=True)
    yield
    enable_socket()
