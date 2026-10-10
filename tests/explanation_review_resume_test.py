"""Completed preparation reuses only its exact publication and immutable resources."""

import pytest

from exact_inspect.artifacts import atomic_json, publish_bundle
from exact_inspect.contracts import canonical_hash, file_hash
from exact_inspect.study.models import Publish
from tests.explanation_review_publication_test import review_fixture
from tools.prepare_review_bundle import prepare
from tools.review_environment import ReviewError


def completed_review(root):
    run = root / "input-run"
    atomic_json(run / "run_manifest.json", {"fixture": "resume-boundary"})
    atomic_json(run / "source_decisions.json", {"records": []})
    raw = review_fixture(root)
    raw["definition"]["review_provenance"]["run_manifest_sha256"] = file_hash(
        run / "run_manifest.json"
    )[7:]
    publication = Publish.model_validate(raw)
    atomic_json(root / "publication.json", publication)
    atomic_json(root / "catalogue.json", {"publication_sha256": canonical_hash(publication)})
    package = root / "package"
    atomic_json(package / "prepared.json", {"fixture": "inert validated bundle"})
    publish_bundle(package)
    return run, raw


def test_completed_review_resume_preserves_publication_and_all_resource_bytes(tmp_path):
    run, _ = completed_review(tmp_path)
    frozen = {path: file_hash(path) for path in tmp_path.rglob("*") if path.is_file()}
    assert prepare(tmp_path, run) == tmp_path / "publication.json"
    assert prepare(tmp_path, run, inputs_only=True) == tmp_path / "publication.json"
    with pytest.raises(ValueError, match="another generation attempt"):
        prepare(tmp_path, run, generation_attempt=2)
    assert {path: file_hash(path) for path in frozen} == frozen


@pytest.mark.parametrize(
    "changed",
    ["publication", "catalogue", "asset", "receipt", "index", "context_manifest", "context_db"],
)
def test_completed_review_refuses_changed_publication_or_root_resources(tmp_path, changed):
    run, raw = completed_review(tmp_path)
    if changed == "publication":
        raw["definition"]["information_text"] += " Changed after completion."
        atomic_json(tmp_path / "publication.json", raw)
    elif changed == "catalogue":
        atomic_json(tmp_path / "catalogue.json", {"publication_sha256": "sha256:" + "0" * 64})
    else:
        if changed in {"asset", "receipt"}:
            asset = next(row for row in raw["definition"]["assets"] if row["kind"] == "ontology")
            relative = asset["path" if changed == "asset" else "admission_receipt_path"]
        else:
            scope = raw["definition"]["workspace_scopes"][0]
            relative = (
                scope["index_path"]
                if changed == "index"
                else scope["contexts"][0]["path"]
                + ("/manifest.json" if changed == "context_manifest" else "/context.sqlite")
            )
        target = tmp_path / relative
        target.write_bytes(target.read_bytes() + b"changed")
    before = {path: file_hash(path) for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises((ValueError, ReviewError), match="differs|changed"):
        prepare(tmp_path, run)
    assert {path: file_hash(path) for path in before} == before
