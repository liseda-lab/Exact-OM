"""Server-owned identities for public, content-addressed frontend build assets.

The configured export is trusted build input, never a request-supplied directory.
Snapshot bytes at mounting and bind subsequent delivery to the same file identity.
A suffix or response Cache-Control header alone cannot authorize public caching.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

PUBLIC_ASSET_SCOPE = "exact.verified_public_build_asset"
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
_HASHED_NAME = re.compile(r"(?:^|[-.])[a-f0-9]{16,64}(?:[-.]|$)")
_PUBLIC_SUFFIXES = frozenset(
    {
        ".css",
        ".js",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".svg",
        ".avif",
        ".ico",
    }
)


def file_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class PublicBuildAsset:
    path: Path
    sha256: str
    stat: os.stat_result

    def unchanged(self):
        try:
            return file_identity(self.path.stat()) == file_identity(self.stat)
        except OSError:
            return False


def build_public_asset_registry(root: Path) -> dict[str, PublicBuildAsset]:
    """Hash allowed exported build files in bounded chunks; no body buffering at delivery."""
    assets = {}
    static = root / "_next/static"
    for path in sorted(static.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or len(relative.parts) < 4
            or relative.parts[2] not in {"css", "chunks", "media"}
            or path.suffix not in _PUBLIC_SUFFIXES
            or not _HASHED_NAME.search(path.name)
        ):
            continue
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                digest.update(block)
        if file_identity(before) != file_identity(path.stat()):
            raise ValueError("Frontend export changed while its public assets were admitted")
        assets[str(relative)] = PublicBuildAsset(path, digest.hexdigest(), before)
    return assets


def public_asset(scope) -> PublicBuildAsset | None:
    """Only the static route can install this server-owned identity, never client headers."""
    value = scope.get(PUBLIC_ASSET_SCOPE)
    return value if isinstance(value, PublicBuildAsset) else None


def public_asset_cache(scope, status: int) -> str | None:
    asset = public_asset(scope)
    if scope.get("method") in {"GET", "HEAD"} and status in {200, 206} and asset is not None:
        return IMMUTABLE_CACHE
    return None
