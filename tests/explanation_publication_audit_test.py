"""Independent audit sidecars retain the frozen publication selection binding."""

import hashlib
import json

import pytest

from tools import audit_review_publication as audit


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_explicit_selection_sidecar_is_required_and_hash_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "file_hash", digest, raising=False)
    runtime_input = tmp_path / "input"
    runtime_input.mkdir()
    sidecar = tmp_path / "verification" / "selection.json"
    sidecar.parent.mkdir()
    selected = {"cases": [{"case_id": "review-01", "targets": []}]}
    sidecar.write_text(json.dumps(selected))
    definition = {"review_provenance": {"selection_manifest_sha256": digest(sidecar)[7:]}}

    with pytest.raises(FileNotFoundError):
        audit.read_selection(runtime_input, definition)
    path, result = audit.read_selection(runtime_input, definition, sidecar)
    assert path == sidecar
    assert result == selected
    assert list(runtime_input.iterdir()) == []

    sidecar.write_text(json.dumps({"cases": []}))
    with pytest.raises(AssertionError, match="does not match the frozen publication"):
        audit.read_selection(runtime_input, definition, sidecar)
    assert list(runtime_input.iterdir()) == []


def test_default_selection_location_preserves_existing_audit_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "file_hash", digest, raising=False)
    selection = tmp_path / "selection.json"
    selection.write_text('{"cases": []}\n')
    definition = {"review_provenance": {"selection_manifest_sha256": digest(selection)[7:]}}
    assert audit.read_selection(tmp_path, definition) == (selection, {"cases": []})
