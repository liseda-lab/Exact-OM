"""Generic HTTP provider for YAML-described URL and archive tracks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import unquote, urlparse

from .archive import safe_extract_archive
from .base import BaseDescriptorProvider, FetchResult
from .descriptor import TrackDescriptor, safe_relative_path
from .lockfile import sha256_file
from .provider import DescriptorError, IntegrityError


class HttpTransport(Protocol):
    """Injectable, streaming HTTP transport used by hermetic provider tests."""

    def download(self, url: str, destination: Path) -> Mapping[str, str]:
        """Download ``url`` into ``destination`` and return response headers."""

    def head(self, url: str) -> Mapping[str, str]:
        """Return response headers without downloading the response body."""


class UrllibHttpTransport:
    """Standard-library HTTP transport with atomic downloads and bounded timeouts."""

    def __init__(self, *, timeout: float = 60.0):
        self.timeout = timeout

    @staticmethod
    def _headers(value: Any) -> dict[str, str]:
        return {str(key).lower(): str(item) for key, item in value.items()}

    def download(self, url: str, destination: Path) -> Mapping[str, str]:
        """Stream an HTTP response to an atomically renamed local file."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        request = urllib.request.Request(url, headers={"User-Agent": "Exact-OM track provider"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response, temporary.open(
                "wb"
            ) as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
                headers = self._headers(response.headers)
            os.replace(temporary, destination)
            return headers
        except (OSError, urllib.error.URLError) as exc:
            raise IntegrityError(f"Could not download {url}: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)

    def head(self, url: str) -> Mapping[str, str]:
        """Issue an HTTP HEAD request for upstream change detection."""

        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "Exact-OM track provider"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return self._headers(response.headers)
        except (OSError, urllib.error.URLError) as exc:
            raise IntegrityError(f"Could not inspect {url}: {exc}") from exc


class DeclarativeHttpProvider(BaseDescriptorProvider):
    """Materialize a track from declarative HTTP resources and safe unpack rules."""

    backend = "http"

    def __init__(
        self, descriptor: TrackDescriptor, *, transport: HttpTransport | None = None
    ) -> None:
        if descriptor.provider != self.backend:
            raise DescriptorError(
                f"Descriptor {descriptor.name!r} selects {descriptor.provider!r}, not HTTP"
            )
        super().__init__(descriptor)
        self.transport = transport or UrllibHttpTransport()
        allowed = {"urls", "revision", "checksum_manifest"}
        unknown = set(descriptor.upstream) - allowed
        if unknown:
            raise DescriptorError(f"Unknown HTTP upstream key(s): {', '.join(sorted(unknown))}")
        self._resources = self._validate_resources(descriptor.upstream)
        for resource in self._resources:
            if resource["tasks"] is None:
                continue
            missing = sorted(set(resource["tasks"]) - set(descriptor.tasks))
            if missing:
                raise DescriptorError(
                    "HTTP resource references unknown task(s): " + ", ".join(missing)
                )

    @staticmethod
    def _validate_resources(upstream: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        resources = upstream.get("urls")
        if not isinstance(resources, list) or not resources:
            raise DescriptorError("HTTP descriptor upstream.urls must be a non-empty list")
        validated: list[dict[str, Any]] = []
        for index, item in enumerate(resources):
            location = f"upstream.urls[{index}]"
            if not isinstance(item, Mapping):
                raise DescriptorError(f"{location} must be a mapping")
            unknown = set(item) - {
                "url",
                "filename",
                "sha256",
                "extract",
                "destination",
                "tasks",
            }
            if unknown:
                raise DescriptorError(f"Unknown key(s) at {location}: {', '.join(sorted(unknown))}")
            url = str(item.get("url", "")).strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise DescriptorError(f"{location}.url must be an HTTP(S) URL")
            filename = item.get("filename")
            if filename is None:
                filename = Path(unquote(parsed.path)).name
            filename = safe_relative_path(str(filename), f"{location}.filename")
            digest = item.get("sha256")
            if digest is not None:
                digest = str(digest).lower()
                if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                    raise DescriptorError(f"{location}.sha256 must be a 64-character hex digest")
            extract = item.get("extract", False)
            if not isinstance(extract, bool):
                raise DescriptorError(f"{location}.extract must be a boolean")
            destination = item.get("destination")
            if destination is not None:
                destination = safe_relative_path(str(destination), f"{location}.destination")
            tasks = item.get("tasks")
            if tasks is not None and (
                not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks)
            ):
                raise DescriptorError(f"{location}.tasks must be a list of task names")
            validated.append(
                {
                    "url": url,
                    "filename": filename,
                    "sha256": digest,
                    "extract": extract,
                    "destination": destination,
                    "tasks": tuple(tasks) if tasks is not None else None,
                }
            )
        return tuple(validated)

    def _fetch(
        self, destination: Path, revision: str | None, task: str, update: bool
    ) -> FetchResult:
        configured_revision = self.descriptor.upstream.get("revision")
        if (
            revision is not None
            and configured_revision is not None
            and revision != configured_revision
        ):
            raise DescriptorError(
                f"HTTP track {self.name!r} has fixed descriptor revision {configured_revision!r}; "
                f"it cannot fetch revision {revision!r}"
            )
        validators: list[dict[str, Any]] = []
        resources = tuple(
            resource
            for resource in self._resources
            if resource["tasks"] is None or task in resource["tasks"]
        )
        if not resources:
            raise DescriptorError(f"HTTP descriptor has no upstream resources for task {task!r}")
        fetch_warnings: list[str] = []
        for resource in resources:
            download_path = destination / resource["filename"]
            headers = {
                str(key).lower(): str(value)
                for key, value in self.transport.download(resource["url"], download_path).items()
            }
            actual = sha256_file(download_path)
            expected = resource["sha256"]
            if expected and actual != expected:
                message = (
                    f"Downloaded checksum mismatch for {resource['url']}: expected "
                    f"{expected}, found {actual}"
                )
                if not update:
                    raise IntegrityError(message)
                fetch_warnings.append(
                    message + "; accepted because this was an explicit update repin"
                )
            validators.append(
                {
                    "url": resource["url"],
                    "etag": headers.get("etag"),
                    "last_modified": headers.get("last-modified"),
                    "sha256": actual,
                    "filename": resource["filename"],
                }
            )
            if resource["extract"]:
                extract_destination = destination / (
                    resource["destination"] or self._archive_stem(resource["filename"])
                )
                safe_extract_archive(download_path, extract_destination)
            elif resource["destination"]:
                output = destination / resource["destination"]
                output.parent.mkdir(parents=True, exist_ok=True)
                os.replace(download_path, output)
                validators[-1]["filename"] = resource["destination"]

        canonical = json.dumps(validators, sort_keys=True, separators=(",", ":"))
        pinned_revision = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return FetchResult(
            root=destination,
            upstream_id=",".join(resource["url"] for resource in resources),
            revision=pinned_revision,
            revision_ref=(
                str(revision or configured_revision) if revision or configured_revision else None
            ),
            validators=tuple(validators),
            warnings=tuple(fetch_warnings),
        )

    @staticmethod
    def _archive_stem(filename: str) -> str:
        name = Path(filename).name
        for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip", ".tar"):
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return Path(name).stem

    def _check_upstream(self, entry: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
        moved = False
        report_warnings: list[str] = []
        for validator in entry.get("validators", []):
            url = validator.get("url")
            if not url:
                continue
            try:
                current = {
                    str(key).lower(): str(value)
                    for key, value in self.transport.head(str(url)).items()
                }
            except IntegrityError as exc:
                report_warnings.append(
                    f"Could not check whether HTTP upstream moved for {url}: {exc}"
                )
                continue
            comparisons = (
                ("etag", "etag"),
                ("last_modified", "last-modified"),
            )
            for locked_key, header_key in comparisons:
                locked = validator.get(locked_key)
                observed = current.get(header_key)
                if locked and observed and str(locked) != observed:
                    moved = True
        return moved, tuple(report_warnings)
