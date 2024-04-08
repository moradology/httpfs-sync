from fsspec.implementations.http import HTTPFileSystem

from httpfs_sync.core import SyncHTTPFileSystem


def test_find():
    filesystem, host, test_path = (
        HTTPFileSystem,
        "https://ftp.fau.de",
        "https://ftp.fau.de/debian-cd/current/amd64/list-dvd",
    )
    test_fs = filesystem(host)
    filenames_ftp = test_fs.find(test_path)
    assert filenames_ftp

    """Test .find() method on debian server (compare sync-https to async https) with constant folder"""
    filesystem, host, test_path = (
        SyncHTTPFileSystem,
        "https://ftp.fau.de",
        "https://ftp.fau.de/debian-cd/current/amd64/list-dvd",
    )
    test_fs = filesystem()
    filenames_http = test_fs.find(test_path)
    roots = [f.rsplit("/", 1)[-1] for f in filenames_http]

    assert all(f.rsplit("/", 1)[-1] in roots for f in filenames_ftp)


def test_byte_range_read():
    # read just the first kb of leibniz' monadology in german
    b = SyncHTTPFileSystem().read_bytes(
        "https://www.gutenberg.org/cache/epub/39441/pg39441-images.html", 0, 4 * 2**8
    )
    assert len(b) == 4 * 2**8


def test_fs_registration():
    from fsspec import filesystem

    SyncHTTPFileSystem.overwrite_async_registration()
    httpsfs = filesystem("https")
    assert isinstance(httpsfs, SyncHTTPFileSystem)

    httpfs = filesystem("http")
    assert isinstance(httpfs, SyncHTTPFileSystem)
