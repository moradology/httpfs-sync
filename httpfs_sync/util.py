import urllib3
from urllib3 import Retry, Timeout
from urllib3.exceptions import HTTPError


def get_conn_pool(connect_timeout=5, read_timeout=45, retries=3):
    """
    Creates a urllib3 PoolManager instance configured for downloading large files.

    :param connect_timeout: Timeout for establishing a connection.
    :param read_timeout: Timeout for reading data from an open connection.
    :param retries: Number of retries on failed requests.
    :return: Configured instance of urllib3.PoolManager.
    """
    timeout_settings = Timeout(connect=connect_timeout, read=read_timeout)
    retry_strategy = Retry(total=retries, backoff_factor=1, status_forcelist=[500, 502, 503, 504])

    http_pool = urllib3.PoolManager(maxsize=1, timeout=timeout_settings, retries=retry_strategy)

    return http_pool


def raise_for_status(response, url):
    """
    Raises FileNotFoundError for 404s, otherwise mirror raise_for_status behavior of e.g. aiohttp.
    """
    if response.status == 404:
        raise FileNotFoundError(url)

    if 400 <= response.status < 600:
        raise HTTPError(f"Request failed with status code {response.status}")
