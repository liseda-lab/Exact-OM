"""Installed-package identity checks for the shared ontology stack."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution as installed_distribution
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path


def distribution_version(module: object, distribution: str) -> str:
    """Return one exact public version and reject shadowed distribution drift."""

    module_version = getattr(module, "__version__", None)
    if not isinstance(module_version, str) or not module_version:
        module_version = None
    try:
        installed_version = version(distribution)
    except PackageNotFoundError:
        installed_version = None
    if (
        module_version is not None
        and installed_version is not None
        and module_version != installed_version
    ):
        raise RuntimeError(
            f"installed {distribution} module/distribution version mismatch: "
            f"{module_version!r} != {installed_version!r}"
        )
    return module_version or installed_version or "unknown"


@lru_cache(maxsize=4)
def distribution_code_fingerprint(name: str) -> str | None:
    """Hash installed wheel code once per process, independent of its install path.

    Installed dependencies must remain immutable for a running experiment, as with
    the existing package-version cache. Local candidate wheels can share a version.
    """
    try:
        metadata = installed_distribution(name)
    except PackageNotFoundError:
        return None
    module_name = {
        "pyowl-core": "pyowl_core",
        "pyowl2vec-star-projector": "pyowl2vec_star_projector",
        "pyelk-reasoner": "pyelk",
        "pyhermit": "pyhermit",
    }.get(name)
    if module_name is not None:
        spec = find_spec(module_name)
        expected = Path(str(metadata.locate_file(f"{module_name}/__init__.py"))).resolve()
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
            raise RuntimeError(
                f"{name} import does not resolve to its fingerprinted installed wheel"
            )
    files = {}
    for entry in metadata.files or ():
        relative = Path(str(entry))
        if relative.is_absolute() or ".." in relative.parts:
            continue
        if relative.suffix not in {".py", ".so", ".dll", ".dylib"}:
            continue
        path = Path(str(metadata.locate_file(entry)))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[relative.as_posix()] = digest.hexdigest()
    if not files:
        raise RuntimeError(
            f"{name} requires an installed wheel code inventory for execution identity"
        )
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def ontology_execution_identity(reasoner: str = "asserted") -> dict[str, str | None]:
    """Bind only the ontology dependencies selected by an execution path."""
    names = ["pyowl-core", "pyowl2vec-star-projector"]
    optional = {"elk": "pyelk-reasoner", "hermit": "pyhermit"}.get(reasoner.strip().lower())
    if optional is not None:
        names.append(optional)
    return {name: distribution_code_fingerprint(name) for name in names}


__all__ = ["distribution_version", "distribution_code_fingerprint", "ontology_execution_identity"]
