"""Local wheels with equal versions must not share execution cache identities."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.ontology import projection, reasoning, versions


@pytest.fixture(autouse=True)
def clear_inventory_cache():
    versions.distribution_code_fingerprint.cache_clear()
    yield
    versions.distribution_code_fingerprint.cache_clear()


def test_installed_code_identity_is_path_free_and_changes_with_rebuilt_binary(
    tmp_path, monkeypatch
):
    roots = [tmp_path / name for name in ("first", "relocated")]
    for root in roots:
        (root / "package").mkdir(parents=True)
        (root / "package/__init__.py").write_text("same_interface = True\n")
        (root / "package/native.so").write_bytes(b"native build one")
        (root / "package/README.md").write_text("not execution code")
    selected = roots[0]

    def metadata(name):
        return SimpleNamespace(
            version="0.2.0",
            files=[
                Path("package/__init__.py"),
                Path("package/native.so"),
                Path("package/README.md"),
            ],
            locate_file=lambda entry: selected / entry,
        )

    monkeypatch.setattr(versions, "installed_distribution", metadata)
    first = versions.distribution_code_fingerprint("fixture")
    assert len(first) == 64
    selected = roots[1]
    versions.distribution_code_fingerprint.cache_clear()
    assert versions.distribution_code_fingerprint("fixture") == first
    (selected / "package/native.so").write_bytes(b"native build two")
    # A fresh execution process reads the new installed immutable artifacts.
    versions.distribution_code_fingerprint.cache_clear()
    assert versions.distribution_code_fingerprint("fixture") != first


def test_dataset_cache_identity_tracks_only_selected_native_dependencies(monkeypatch):
    revisions = {
        name: name
        for name in ("pyowl-core", "pyowl2vec-star-projector", "pyelk-reasoner", "pyhermit")
    }
    monkeypatch.setattr(versions, "distribution_code_fingerprint", revisions.get)
    projector = projection.projector_cache_identity(projection.ProjectorSettings())
    asserted = reasoning.reasoner_cache_identity("asserted")
    hermit = reasoning.reasoner_cache_identity("hermit")
    revisions["pyhermit"] = "rebuilt-hermit"
    assert projection.projector_cache_identity(projection.ProjectorSettings()) == projector
    assert reasoning.reasoner_cache_identity("asserted") == asserted
    assert reasoning.reasoner_cache_identity("hermit") != hermit
    revisions["pyowl-core"] = "rebuilt-core"
    assert projection.projector_cache_identity(projection.ProjectorSettings()) != projector


def test_same_version_source_shadow_cannot_claim_installed_wheel_identity(tmp_path, monkeypatch):
    installed = tmp_path / "installed" / "pyhermit"
    shadow = tmp_path / "source" / "pyhermit"
    for path in (installed, shadow):
        path.mkdir(parents=True)
        (path / "__init__.py").write_text('__version__ = "0.2.0"\n')
    monkeypatch.setattr(
        versions,
        "installed_distribution",
        lambda name: SimpleNamespace(
            version="0.2.0",
            files=[Path("pyhermit/__init__.py")],
            locate_file=lambda entry: installed.parent / entry,
        ),
    )
    monkeypatch.setattr(
        versions, "find_spec", lambda name: SimpleNamespace(origin=str(shadow / "__init__.py"))
    )
    with pytest.raises(RuntimeError, match="fingerprinted installed wheel"):
        versions.distribution_code_fingerprint("pyhermit")
