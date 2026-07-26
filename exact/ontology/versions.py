"""Installed-package identity checks for the shared ontology stack."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


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


__all__ = ["distribution_version"]
