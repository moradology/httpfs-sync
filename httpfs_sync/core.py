import io
import logging
import numbers
import re
from copy import copy
from urllib.parse import urlparse

import yarl
from fsspec import AbstractFileSystem
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.registry import register_implementation
from fsspec.utils import DEFAULT_BLOCK_SIZE, isfilelike, nullcontext

from .file import SyncHTTPFile, SyncHTTPStreamFile
from .util import get_conn_pool, raise_for_status

# TODO: determine any compr/compression behaviors missing


# Reimplementing these instead of importing from fsspec's http.py to avoid aiohttp dependency
ex = re.compile(r"""<(a|A)\s+(?:[^>]*?\s+)?(href|HREF)=["'](?P<url>[^"']+)""")
ex2 = re.compile(r"""(?P<url>http[s]?://[-a-zA-Z0-9@:%_+.~#?&/=]+)""")
logger = logging.getLogger("httpfs-sync")


class SyncHTTPFileSystem(AbstractFileSystem):
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
            objects instead of SyncHTTPFile instances
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
            the SyncHTTPFileSystem class.
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

        # Clean caching-related parameters from `storage_options`
        # before propagating them as `request_options` through `self.kwargs`.
        # TODO: Maybe rename `self.kwargs` to `self.request_options` to make
        #       it clearer.
        request_options = copy(storage_options)
        self.use_listings_cache = request_options.pop("use_listings_cache", False)
        request_options.pop("listings_expiry_time", None)
        request_options.pop("max_paths", None)
        request_options.pop("skip_instance_cache", None)
        self.kwargs = request_options

    @property
    def fsid(self):
        return "http-sync"

    def encode_url(self, url):
        # Maintained to ensure consistency with upstream
        return str(yarl.URL(url, encoded=self.encoded))

    @classmethod
    def _strip_protocol(cls, path):
        """For HTTP, we always want to keep the full URL"""
        return path

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
        pool = self.get_conn_pool()

        response = pool.request("GET", path, preload_content=False, **kwargs)
        raise_for_status(response, path)

        try:
            text = response.data.decode("utf-8")
            if self.simple_links:
                links = ex2.findall(text) + [u[2] for u in ex.findall(text)]
            else:
                links = [u[2] for u in ex.findall(text)]
        except UnicodeDecodeError:
            # Handle binary response or decoding error
            links = []

        out = set()
        parts = urlparse(path)
        for link in links:
            if isinstance(link, tuple):
                link = link[1]
            if link.startswith("/") and len(link) > 1:
                # absolute URL on this server
                link = f"{parts.scheme}://{parts.netloc}{link}"
            if link.startswith("http"):
                if self.same_schema and link.startswith(path.rstrip("/") + "/"):
                    out.add(link)
                elif link.replace("https", "http").startswith(
                    path.replace("https", "http").rstrip("/") + "/"
                ):
                    # allowed to cross http <-> https
                    out.add(link)
            else:
                if link not in ["..", "../"]:
                    # Ignore FTP-like "parent"
                    out.add("/".join([path.rstrip("/"), link.lstrip("/")]))
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

    def cat_file_generator(
        self, url, start=None, end=None, chunk_size=DEFAULT_BLOCK_SIZE, **kwargs
    ):
        kw = self.kwargs.copy()
        kw.update(kwargs)

        if start is not None or end is not None:
            if start == end:
                return b""
            headers = kw.pop("headers", {}).copy()

            headers["Range"] = self.process_limits(url, start, end)
            kw["headers"] = headers

        pool = self.get_conn_pool()
        response = pool.request("GET", self.encode_url(url), preload_content=False, **kw)
        raise_for_status(response, url)

        for chunk in response.stream(chunk_size):
            yield chunk

    def cat_file(self, url, start=None, end=None, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)

        if start is not None or end is not None:
            if start == end:
                return b""
            headers = kw.pop("headers", {}).copy()

            headers["Range"] = self.process_limits(url, start, end)
            kw["headers"] = headers
        pool = self.get_conn_pool()
        response = pool.request("GET", self.encode_url(url), **kw)
        raise_for_status(response, url)
        out = response.data
        return out

    def get_file(
        self, rpath, lpath, chunk_size=DEFAULT_BLOCK_SIZE, callback=DEFAULT_CALLBACK, **kwargs
    ):
        kw = self.kwargs.copy()
        kw.update(kwargs)

        pool = self.get_conn_pool()
        response = pool.request("GET", self.encode_url(rpath), preload_content=False, **kw)
        raise_for_status(response, rpath)
        headers = response.getheaders()

        try:
            size = int(headers["content-length"])
        except (ValueError, KeyError):
            size = None

        callback.set_size(size)
        if isfilelike(lpath):
            outfile = lpath
        else:
            outfile = open(lpath, "wb")

        try:
            for chunk in response.stream(chunk_size):
                outfile.write(chunk)
                callback.relative_update(len(chunk))
        finally:
            if not isfilelike(lpath):
                outfile.close()

    def put_file(
        self,
        lpath,
        rpath,
        chunk_size=DEFAULT_BLOCK_SIZE,
        callback=DEFAULT_CALLBACK,
        method="POST",
        **kwargs,
    ):
        def gen_chunks():
            # Support passing arbitrary file-like objects
            # and use them instead of streams.
            if isinstance(lpath, io.IOBase):
                context = nullcontext(lpath)
                use_seek = False  # might not support seeking
            else:
                context = open(lpath, "rb")
                use_seek = True

            with context as f:
                if use_seek:
                    callback.set_size(f.seek(0, 2))
                    f.seek(0)
                else:
                    callback.set_size(getattr(f, "size", None))

                chunk = f.read(chunk_size)
                while chunk:
                    yield chunk
                    callback.relative_update(len(chunk))
                    chunk = f.read(chunk_size)

        kw = self.kwargs.copy()
        kw.update(kwargs)
        pool = self.get_conn_pool()

        method = method.to_upper()
        if method not in ("POST", "PUT"):
            raise ValueError(f"method has to be either 'POST' or 'PUT', not: {method!r}")

        response = pool.request(method, self.encode_url(rpath), body=gen_chunks(), **kw)
        raise_for_status(response, rpath)
        return response

    def exists(self, path, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        try:
            pool = self.get_conn_pool()
            response = pool.request("GET", self.encode_url(path), **kw)
            return response.status < 400
        except Exception:
            return False

    def isfile(self, path, **kwargs):
        return self.exists(path, **kwargs)

    def open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=None,  # XXX: This differs from the base class.
        cache_type="bytes",
        cache_options=None,
        size=None,
        **kwargs,
    ):
        """Make a file-like object

        Parameters
        ----------
        path: str
            Full URL with protocol
        mode: string
            must be "rb"
        block_size: int or None
            Bytes to download in one request; use instance value if None. If
            zero, will return a streaming Requests file-like instance.
        kwargs: key-value
            Any other parameters, passed to requests calls
        """
        if mode != "rb":
            raise NotImplementedError
        block_size = block_size if block_size is not None else self.block_size
        kw = self.kwargs.copy()
        kw.update(kwargs)
        size = size or self.info(path, **kwargs)["size"]
        if block_size and size:
            return SyncHTTPFile(
                self,
                path,
                get_conn_pool=self.get_conn_pool,
                block_size=block_size or self.block_size,
                mode=mode,
                size=size,
                cache_type=cache_type or self.cache_type,
                cache_options=cache_options or self.cache_options,
                **kw,
            )
        else:
            return SyncHTTPStreamFile(
                self,
                path,
                get_conn_pool=self.get_conn_pool,
                mode=mode,
                **kw,
            )

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

    @classmethod
    def overwrite_async_registration(cls):
        register_implementation("http", cls, clobber=True)
        register_implementation("https", cls, clobber=True)


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
    if size_policy == "head":
        response = pool.request("HEAD", url, redirect=ar, preload_content=False, **kwargs)
    elif size_policy == "get":
        response = pool.request("GET", url, redirect=ar, preload_content=False, **kwargs)
    else:
        raise TypeError(f'size_policy must be "head" or "get", got {size_policy}')
    raise_for_status(response, url)

    headers = response.getheaders()

    # Check for 'Accept-Ranges' header
    accept_ranges = headers.get("Accept-Ranges", "")
    if accept_ranges.lower() == "none" or "Accept-Ranges" not in headers:
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
