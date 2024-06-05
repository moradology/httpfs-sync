import urllib3
from urllib3 import Retry, Timeout
from urllib3.exceptions import HTTPError

POOL_MANAGER = None

def get_pool_manager(connect_timeout=5, read_timeout=45, retries=3):
    """
    Creates a urllib3 PoolManager instance configured for downloading large files.

    :param connect_timeout: Timeout for establishing a connection.
    :param read_timeout: Timeout for reading data from an open connection.
    :param retries: Number of retries on failed requests.
    :return: Configured instance of urllib3.PoolManager.
    """
    global POOL_MANAGER # We need to reuse this, so we'll stash it globally
    if POOL_MANAGER is None:
        timeout_settings = Timeout(connect=connect_timeout, read=read_timeout)
        retry_strategy = Retry(total=retries, backoff_factor=1, status_forcelist=[500, 502, 503, 504])

        http_manager = urllib3.PoolManager(maxsize=50, timeout=timeout_settings, retries=retry_strategy)
        POOL_MANAGER = http_manager
    return POOL_MANAGER


def raise_for_status(response, url):
    """
    Raises FileNotFoundError for 404s, otherwise mirror raise_for_status behavior of e.g. aiohttp.
    """
    if response.status == 404:
        raise FileNotFoundError(url)

    if 400 <= response.status < 600:
        raise HTTPError(f"Request failed with status code {response.status}")
