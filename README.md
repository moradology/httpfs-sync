# httpfs-sync

## Overview
`httpfs-sync` is an implementation of a filesystem interface for the Python `fsspec`
library that uses synchronous requests based on `urllib3`. This serves as an alternative
to the default asynchronous (based on `aiohttp`) filesystem implementation provided by
`fsspec`. This project is ideal for environments where asynchronous operations are not
feasible or desired, allowing users to interact with HTTP-based filesystems using a
synchronous API.

If you're dealing with deadlocks potentially involving the default HTTP 'filesystem' in
`fsspec`, the implementation here may help with verification of behavior: just use the classes
provided here as a drop-in replacement for `HTTPFileSystem` and compare behaviors.

## Features
- Synchronous HTTP requests for file system operations.
- Fully compatible with the `fsspec` API, making it easy to drop in as a replacement.
- Uses `urllib3` for robust, production-ready HTTP communications.

## Installation

To install `httpfs-sync`, you can use pip:

```bash
pip install httpfs-sync
```

## Usage
Here's a simple example of how to use httpfs-sync to read a file from an HTTP server:
```python
import fsspec
from httpfs_sync.core import SyncHTTPFileSystem

# enable use of SyncHTTPFileSystem and associated files for both http and https
SyncHTTPFileSystem.overwrite_async_registration()

# Use the registered file system to access a remote file
with fsspec.open('http://example.com/data.txt', 'r') as f:
    data = f.read()
    print(data)
```

## License
httpfs-sync is distributed under the MIT License, allowing the code to be used in commercial applications as well as open-source projects.