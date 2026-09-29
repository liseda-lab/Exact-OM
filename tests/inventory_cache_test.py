"""Native inventory reuse binds bytes and IO semantics, never experiment labels."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.kinds import EntityKind
from exact.experiments import reporting
from exact.utils.provenance import sha256_file


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    path = tmp_path / "source.owl"
    path.write_text("native fixture")
    calls = []

    def load(path, **kwargs):
        calls.append((Path(path), kwargs))
        return SimpleNamespace(entities=lambda kind: ["s", "t"] if kind == EntityKind.CLASS else [])

    monkeypatch.setattr(reporting, "resolve_source", load)
    monkeypatch.setattr(reporting, "ontology_execution_identity", lambda: {"native": "pinned"})
    return path, tmp_path / "cache", calls


def inspect(path, root, options=None):
    return reporting._source_inventory(
        path, source_format="owl", options=options or {}, cache_root=root
    )


def test_inventory_replay_does_not_load_native_source(inventory):
    path, root, calls = inventory
    first = inspect(path, root)
    assert inspect(path, root) == first
    assert first["entities"]["class"] == 2
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["ontology", "options", "native", "code"])
def test_inventory_invalidates_changed_inputs_and_execution(inventory, monkeypatch, change):
    path, root, calls = inventory
    inspect(path, root)
    options = {}
    if change == "ontology":
        path.write_text("changed native ontology")
    elif change == "options":
        options = {"include_abox": False}
    elif change == "native":
        monkeypatch.setattr(reporting, "ontology_execution_identity", lambda: {"native": "changed"})
    else:
        original = reporting.sha256_file
        monkeypatch.setattr(
            reporting,
            "sha256_file",
            lambda p: "changed-code" if Path(p) == Path(reporting.__file__) else original(p),
        )
    inspect(path, root, options)
    assert len(calls) == 2


def test_inventory_checks_import_bytes_even_on_cache_hit(inventory, tmp_path):
    path, root, calls = inventory
    imported = tmp_path / "import.owl"
    imported.write_text("original imported ontology")
    options = {"imports": {"urn:import": {"path": str(imported), "sha256": sha256_file(imported)}}}
    inspect(path, root, options)
    imported.write_text("changed imported ontology")
    with pytest.raises(ValueError, match="import checksum"):
        inspect(path, root, options)
    options["imports"]["urn:import"]["sha256"] = sha256_file(imported)
    inspect(path, root, options)
    assert len(calls) == 2


@pytest.mark.parametrize("corruption", ["invalid_json", "wrong_counts"])
def test_corrupt_inventory_is_recomputed(inventory, corruption):
    path, root, calls = inventory
    expected = inspect(path, root)
    saved = next(root.glob("*.json"))
    if corruption == "invalid_json":
        saved.write_text("truncated{")
    else:
        value = json.loads(saved.read_text())
        value["value"]["entities"]["class"] = 999
        saved.write_text(json.dumps(value))
    assert inspect(path, root) == expected
    assert inspect(path, root) == expected
    assert len(calls) == 2


def test_inventory_task_metadata_is_materialized_fresh(inventory, monkeypatch):
    path, root, calls = inventory
    monkeypatch.setattr(
        reporting,
        "resolve_alignment_inputs",
        lambda **kwargs: SimpleNamespace(
            source=path, target=path, full_reference=None, candidates=None
        ),
    )
    options = dict(
        config=ConfigModel(),
        experiment_id="E03",
        task_id="D0",
        stage="screen",
        split_role="development",
        reference_role="valid",
        reference_completeness="complete",
        capabilities=[],
        cache_root=root,
    )
    first = reporting.inspect_dataset_task(**options)
    options.update(
        experiment_id="E15",
        task_id="D0-transfer",
        capabilities=["calibration"],
        reference_completeness="partial",
    )
    second = reporting.inspect_dataset_task(**options)
    assert second["entity_counts"] == first["entity_counts"]
    assert second["experiment_id"] == "E15" and second["task_id"] == "D0-transfer"
    assert second["capabilities"] == ["calibration"]
    assert second["reference_completeness"] == "partial"
    assert len(calls) == 1


def test_unavailable_inventory_storage_does_not_block_native_inspection(inventory, monkeypatch):
    path, root, calls = inventory

    def unavailable(*args, **kwargs):
        raise OSError("cache storage unavailable")

    monkeypatch.setattr(reporting, "freeze_json", unavailable)
    assert inspect(path, root)["entities"]["class"] == 2
    assert len(calls) == 1
