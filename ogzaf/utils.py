import requests
import urllib3
import ssl
import socket


class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    """
    The UN Data Portal server doesn't support "RFC 5746 secure renegotiation". This causes and error when the client is using OpenSSL 3, which enforces that standard by default.
    The fix is to create a custom SSL context that allows for legacy connections. This defines a function get_legacy_session() that should be used instead of requests().
    """

    # "Transport adapter" that allows us to use custom ssl_context.
    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=self.ssl_context,
        )


def get_legacy_session():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT  #in Python 3.12 you will be able to switch from 0x4 to ssl.OP_LEGACY_SERVER_CONNECT.
    session = requests.session()
    session.mount("https://", CustomHttpAdapter(ctx))
    return session


# Function to check if connected to internet
def is_connected():
    try:
        # connect to the host -- tells us if the host is actually
        # reachable
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False
