import datetime
import io
import logging
import numbers
import os
import os.path as osp
import shutil
import stat
import tempfile
from urllib.parse import urlparse

import urllib3
from urllib3 import Timeout, Retry
from urllib3.exceptions import HTTPError

from fsspec import AbstractFileSystem
from fsspec.compression import compr
from fsspec.core import get_compression
from fsspec.implementations.http import ex, ex2
from fsspec.utils import isfilelike, stringify_path
from fsspec.utils import (
    DEFAULT_BLOCK_SIZE,
    glob_translate,
    isfilelike,
    nullcontext,
    tokenize,
)
import yarl

logger = logging.getLogger("fsspec.local")

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

    http_pool = urllib3.PoolManager(
        maxsize=1,
        timeout=timeout_settings,
        retries=retry_strategy
    )

    return http_pool

def raise_for_status(response):
    """
    Raises an HTTPError if the response contains an HTTP error status code.

    :param response: The HTTP response object from urllib3.
    :raises: HTTPError for error status codes (4XX, 5XX).
    """
    if 400 <= response.status < 600:
        raise HTTPError(f"Request failed with status code {response.status}", response=response)


class SynchronousHttpFileSystem(AbstractFileSystem):
    """Synchronous fs-like interface to http resources"""

    def __init__(
        self,
        simple_links=True,
        block_size=DEFAULT_BLOCK_SIZE,
        same_scheme=True,
        size_policy=None,
        cache_type="bytes",
        cache_options=None,
        get_conn_pool=get_conn_pool,
        encoded=False,
        **storage_options,
    ):
        """
        NB: if this is called async, you must await set_client

        Parameters
        ----------
        block_size: int
            Blocks to read bytes; if 0, will default to raw requests file-like
            objects instead of HTTPFile instances
        simple_links: bool
            If True, will consider both HTML <a> tags and anything that looks
            like a URL; if False, will consider only the former.
        same_scheme: True
            When doing ls/glob, if this is True, only consider paths that have
            http/https matching the input URLs.
        size_policy: this argument is deprecated
        client_kwargs: dict
            Passed to aiohttp.ClientSession, see
            https://docs.aiohttp.org/en/stable/client_reference.html
            For example, ``{'auth': aiohttp.BasicAuth('user', 'pass')}``
        get_client: Callable[..., aiohttp.ClientSession]
            A callable which takes keyword arguments and constructs
            an aiohttp.ClientSession. It's state will be managed by
            the HTTPFileSystem class.
        storage_options: key-value
            Any other parameters passed on to requests
        cache_type, cache_options: defaults used in open
        """
        super().__init__(self, **storage_options)
        self.block_size = block_size if block_size is not None else DEFAULT_BLOCK_SIZE
        self.simple_links = simple_links
        self.same_schema = same_scheme
        self.cache_type = cache_type
        self.cache_options = cache_options
        self.get_conn_pool = get_conn_pool
        self.encoded = encoded
        self.kwargs = storage_options

    @property
    def fsid(self):
        return "http-sync"

    def encode_url(self, url):
        return yarl.URL(url, encoded=self.encoded)

    @classmethod
    def _strip_protocol(cls, path):
        """For HTTP, we always want to keep the full URL"""
        return path
    
    def _raise_not_found_for_status(self, response, url):
        """
        Raises FileNotFoundError for 404s, otherwise uses raise_for_status.
        """
        if response.status == 404:
            raise FileNotFoundError(url)
        raise_for_status(response)

    def ls(self, path, detail=True, **kwargs):
        if self.use_listings_cache and path in self.dircache:
            out = self.dircache[path]
        else:
            out = self._ls_real(path, detail=detail, **kwargs)
            self.dircache[path] = out
        return out

    def _ls_real(self, path, detail=False, **kwargs):
        # ignoring URL-encoded arguments
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(path)
        pool = self.get_conn_pool()

        with pool as http:
            response = http.request("GET", path)
        self._raise_not_found_for_status(response, path)

        try:
            text = response.data.decode('utf-8')
            if self.simple_links:
                links = ex2.findall(text) + [u[2] for u in ex.findall(text)]
            else:
                links = [u[2] for u in ex.findall(text)]
        except UnicodeDecodeError:
            # Handle binary response or decoding error
            links = []
        
        out = set()
        parts = urlparse(path)
        for l in links:
            if isinstance(l, tuple):
                l = l[1]
            if l.startswith("/") and len(l) > 1:
                # absolute URL on this server
                l = f"{parts.scheme}://{parts.netloc}{l}"
            if l.startswith("http"):
                if self.same_schema and l.startswith(path.rstrip("/") + "/"):
                    out.add(l)
                elif l.replace("https", "http").startswith(
                    path.replace("https", "http").rstrip("/") + "/"
                ):
                    # allowed to cross http <-> https
                    out.add(l)
            else:
                if l not in ["..", "../"]:
                    # Ignore FTP-like "parent"
                    out.add("/".join([path.rstrip("/"), l.lstrip("/")]))
        if not out and path.endswith("/"):
            out = self._ls_real(path.rstrip("/"), detail=False)
        if detail:
            return [
                {
                    "name": u,
                    "size": None,
                    "type": "directory" if u.endswith("/") else "file",
                }
                for u in out
            ]
        else:
            return sorted(out)

    def info(self, path, **kwargs):
        """Get info of URL

        Tries to access location via HEAD, and then GET methods, but does
        not fetch the data.

        It is possible that the server does not supply any size information, in
        which case size will be given as None (and certain operations on the
        corresponding file will not work).
        """
        info = {}
        pool = self.get_conn_pool()

        for policy in ["head", "get"]:
            try:
                info.update(
                    file_info(
                        self.encode_url(path),
                        pool=pool,
                        size_policy=policy,
                        **self.kwargs,
                        **kwargs,
                    )
                )
                if info.get("size") is not None:
                    break
            except Exception as exc:
                if policy == "get":
                    # If get failed, then raise a FileNotFoundError
                    raise FileNotFoundError(path) from exc
                logger.debug("", exc_info=exc)

        return {"name": path, "size": None, **info, "type": "file"}


    def cat_file(self, url, start=None, end=None, **kwargs):
        kw = self.kwargs.copy()
        logger.debug(url)

        if start is not None or end is not None:
            if start == end:
                return b""
            headers = kw.pop("headers", {}).copy()

            headers["Range"] = self.process_limits(url, start, end)
            kw["headers"] = headers
        pool = self.get_conn_pool()
        with pool as http:
            response = http.request("GET", self.encode_url(url), **kw)
        self._raise_not_found_for_status(response, url)
        out = response.data
        return out

    def process_limits(self, url, start, end):
        """Helper for "Range"-based SYNC cat_file"""
        size = None
        suff = False
        if start is not None and start < 0:
            # if start is negative and end None, end is the "suffix length"
            if end is None:
                end = -start
                start = ""
                suff = True
            else:
                size = size or (self.info(url))["size"]
                start = size + start
        elif start is None:
            start = 0
        if not suff:
            if end is not None and end < 0:
                if start is not None:
                    size = size or (self.info(url))["size"]
                    end = size + end
            elif end is None:
                end = ""
            if isinstance(end, numbers.Integral):
                end -= 1  # bytes range is inclusive
        return f"bytes={start}-{end}"


# ========== HELPERS ===============
# ========== HELPERS ===============
# ========== HELPERS ===============
# ========== HELPERS ===============
def file_info(url, pool, size_policy="head", **kwargs):
    """Call HEAD on the server to get details about the file (size/checksum etc.)

    Default operation is to explicitly allow redirects and use encoding
    'identity' (no compression) to get the true size of the target.
    """
    logger.debug("Retrieve file size for %s", url)
    kwargs = kwargs.copy()
    ar = kwargs.pop("allow_redirects", True)
    head = kwargs.get("headers", {}).copy()
    head["Accept-Encoding"] = "identity"
    kwargs["headers"] = head

    info = {}
    with pool as http:
        if size_policy == "head":
            response = http.request("HEAD", url, allow_redirects=ar, **kwargs)
        elif size_policy == "get":
            response = http.request("GET", url, allow_redirects=ar, **kwargs)
        else:
            raise TypeError(f'size_policy must be "head" or "get", got {size_policy}')
    raise_for_status(response)

    headers = response.getheaders()

    # Check for 'Accept-Ranges' header
    accept_ranges = headers.get('Accept-Ranges', '')
    if accept_ranges.lower() == 'none' or 'Accept-Ranges' not in headers:
        # No random access supported, return None
        info["random_access"] = False
    else:
        info["random_access"] = True

    if "Content-Length" in headers:
        # Some servers may choose to ignore Accept-Encoding and return
        # compressed content, in which case the returned size is unreliable.
        if "Content-Encoding" not in headers or headers["Content-Encoding"] in [
            "identity",
            "",
        ]:
            info["size"] = int(headers["Content-Length"])
    elif "Content-Range" in headers:
        info["size"] = int(headers["Content-Range"].split("/")[1])

    if "Content-Type" in headers:
        info["mimetype"] = headers["Content-Type"].partition(";")[0]

    info["url"] = str(response.geturl())

    for checksum_field in ["ETag", "Content-MD5", "Digest"]:
        if headers.get(checksum_field):
            info[checksum_field] = headers[checksum_field]

    return info


def file_size(url, pool=None, *args, **kwargs):
    if pool is None:
        pool = get_conn_pool()
    info = file_info(url, pool, *args, **kwargs)
    return info.get("size")

