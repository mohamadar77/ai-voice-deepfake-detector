"""Directory hashing shared by evaluation and serving.

A leaf module with no project imports, so the serving process can verify a model
bundle without importing the evaluation or metrics code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_directory(directory: Path) -> str:
    """SHA-256 over every file's relative path and bytes, in sorted order."""

    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
            digest.update(b"|")
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    digest.update(chunk)
            digest.update(b"\n")
    return digest.hexdigest()
