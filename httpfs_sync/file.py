import logging
import re

from fsspec.caching import AllBytes
from fsspec.spec import AbstractBufferedFile
from fsspec.utils import DEFAULT_BLOCK_SIZE

from .util import raise_for_status


logger = logging.getLogger("httpfs-sync")


class SyncHTTPFile(AbstractBufferedFile):
    """
    A file-like object pointing to a remove HTTP(S) resource

    Supports only reading, with read-ahead of a predermined block-size.

    In the case that the server does not supply the filesize, only reading of
    the complete file in one go is supported.

    Parameters
    ----------
    url: str
        Full URL of the remote resource, including the protocol
    get_conn_pool: a function that returns a urllib3 poolmanager
    block_size: int or None
        The amount of read-ahead to do, in bytes. Default is 5MB, or the value
        configured for the FileSystem creating this file
    size: None or int
        If given, this is the size of the file in bytes, and we don't attempt
        to call the server to find the value.
    kwargs: all other key-values are passed to requests calls.
    """

    def __init__(
        self,
        fs,
        url,
        get_conn_pool=None,
        block_size=None,
        mode="rb",
        cache_type="bytes",
        cache_options=None,
        size=None,
        **kwargs,
    ):
        if mode != "rb":
            raise NotImplementedError("File mode not supported")
        self.url = url
        self.get_conn_pool = get_conn_pool
        self.conn_pool = self.get_conn_pool()
        self.details = {"name": url, "size": size, "type": "file"}
        super().__init__(
            fs=fs,
            path=url,
            mode=mode,
            block_size=block_size,
            cache_type=cache_type,
            cache_options=cache_options,
            **kwargs,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['conn_pool']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn_pool = self.get_conn_pool()

    def read(self, length=-1):
        """Read bytes from file

        Parameters
        ----------
        length: int
            Read up to this many bytes. If negative, read all content to end of
            file. If the server has not supplied the filesize, attempting to
            read only part of the data will raise a ValueError.
        """
        if (
            (length < 0 and self.loc == 0)  # explicit read all
            # but not when the size is known and fits into a block anyways
            and not (self.size is not None and self.size <= self.blocksize)
        ):
            self.fetch_all()
        if self.size is None:
            if length < 0:
                self.fetch_all()
        else:
            length = min(self.size - self.loc, length)
        return super().read(length)

    def fetch_all(self):
        """Read whole file in one shot, without caching

        This is only called when position is still at zero,
        and read() is called without a byte-count.
        """
        logger.debug(f"Fetch all for {self}")
        if not isinstance(self.cache, AllBytes):
            with self.conn_pool as http:
                response = http.request("GET", self.fs.encode_url(self.url), **self.kwargs)
                raise_for_status(response, self.url)
                out = response.read()

            self.cache = AllBytes(size=len(out), fetcher=None, blocksize=None, data=out)
            self.size = len(out)

    def _parse_content_range(self, headers):
        """Parse the Content-Range header"""
        s = headers.get("Content-Range", "")
        m = re.match(r"bytes (\d+-\d+|\*)/(\d+|\*)", s)
        if not m:
            return None, None, None

        if m[1] == "*":
            start = end = None
        else:
            start, end = [int(x) for x in m[1].split("-")]
        total = None if m[2] == "*" else int(m[2])
        return start, end, total

    def _fetch_range(self, start, end):
        """Download a block of data

        The expectation is that the server returns only the requested bytes,
        with HTTP code 206. If this is not the case, we first check the headers,
        and then stream the output - if the data size is bigger than we
        requested, an exception is raised.
        """
        logger.debug(f"Fetch range for {self}: {start}-{end}")
        kwargs = self.kwargs.copy()
        headers = kwargs.pop("headers", {}).copy()
        headers["Range"] = f"bytes={start}-{end - 1}"
        with self.conn_pool as http:
            response = http.request("GET", self.url, headers=headers, preload_content=False, **kwargs)
            resp_headers = response.getheaders()

            # If the server has handled the range request, it should reply
            # with status 206 (partial content). But we'll guess that a suitable
            # Content-Range header or a Content-Length no more than the
            # requested range also mean we have got the desired range.
            response_is_range = (
                response.status == 206
                or self._parse_content_range(resp_headers)[0] == start
                or int(resp_headers.get("Content-Length", end + 1)) <= end - start
            )

            if response_is_range:
                # partial content, as expected
                out = response.read()
            elif start > 0:
                raise ValueError(
                    "The HTTP server doesn't appear to support range requests. "
                    "Only reading this file from the beginning is supported. "
                    "Open with block_size=0 for a streaming file interface."
                )
            else:
                # Response is not a range, but we want the start of the file,
                # so we can read the required amount anyway.
                cl = 0
                out = []
                for chunk in response.stream(1024**5):
                    out.append(chunk)
                    cl += len(chunk)
                    if cl > end - start:
                        break
                out = b"".join(out)[: end - start]

        return out
    

class SyncHTTPStreamFile(AbstractBufferedFile):
    def __init__(self, fs, url, get_conn_pool, mode="rb", **kwargs):
        self.url = url
        self.get_conn_pool = get_conn_pool
        self.conn_pool = self.get_conn_pool()
        if mode != "rb":
            raise ValueError
        self.details = {"name": url, "size": None}
        self.kwargs = kwargs
        super().__init__(fs=fs, path=url, mode=mode, cache_type="none", **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['conn_pool']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn_pool = self.get_conn_pool()

    def seek(self, loc, whence=0):
        if loc == 0 and whence == 1:
            return
        if loc == self.loc and whence == 0:
            return
        raise ValueError("Cannot seek streaming HTTP file")

    def read(self, num=-1):
        with self.conn_pool as http:
            response = http.request("GET", self.url, preload_content=False, **self.kwargs)
            if num < 0:
                out = response.read()
            else:
                out = response.read(num)
            self.loc += len(out)

        return out

    def close(self):
        super().close()
