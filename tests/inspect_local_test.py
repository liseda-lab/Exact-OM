"""The documented local workflow preserves saved matcher data and deploys in isolation."""

import json
from pathlib import Path

import pytest

import sys

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Exact Explain requires Python 3.11+ SQLite limits"
)

from fastapi.testclient import TestClient

from exact.runs import ExplanationStore, RunLayout, refresh_manifest
from exact.runs.decisions import append_event, event, observe_selection, write_candidate_decisions
from exact_inspect.app import create_app, resolve_frontend_dir
from exact_inspect.artifacts import BundleLibrary, export_archive, validate_bundle
from exact_inspect.cli import main
from exact_inspect.contracts import file_hash
from exact_inspect.service import create_prepared_app
from exact_inspect.settings import InspectSettings


@pytest.fixture
def local_bundle(tmp_path):
    from tests.inspect_ui_fixture import create_fixture

    return create_fixture(tmp_path)


def test_preparation_trace_evidence_navigation_and_portable_import(local_bundle, tmp_path):
    client = TestClient(create_prepared_app(local_bundle, library_dir=tmp_path / "library"))
    run = client.get("/api/v1/runs").json()["items"][0]
    prefix = "/api/v1/runs/" + run["run_id"]
    source = client.get(prefix + "/sources").json()["items"][0]["entity"]
    candidates = client.get(prefix + "/candidates", params={"source": source["iri"]}).json()[
        "items"
    ]
    assert len(candidates) == 1
    assert any(s["name"] == "S_final" and s["value"] == 0.81 for s in candidates[0]["scores"])
    pair = client.get(prefix + "/pair", params={"pair_id": candidates[0]["pair_id"]}).json()
    assert any(
        e["stage"] == "retrieval"
        and e["status"] == "completed"
        and e["reason_code"] == "provided_candidate_pool"
        for e in pair["events"]
    )
    evidence = client.get(
        prefix + "/pair-evidence", params={"pair_id": candidates[0]["pair_id"]}
    ).json()
    assert evidence["items"] and all(e["fact_ids"] for e in evidence["items"])
    for ontology in client.get("/api/v1/ontologies").json()["items"]:
        roots = client.get(
            "/api/v1/ontologies/" + ontology["ontology_version_id"] + "/entry-points",
            params={"kind": "class", "basis": "literal_asserted"},
        )
        assert roots.status_code == 200 and roots.json()["items"]
    archive = tmp_path / "bundle.zip"
    export_archive(local_bundle, archive)
    receipt = client.post(
        "/api/v1/bundles/import",
        content=archive.read_bytes(),
        headers={"Content-Type": "application/zip"},
    )
    assert receipt.status_code == 200, receipt.text
    package = receipt.json()["package_id"]
    assert client.post("/api/v1/bundles/" + package + "/select").status_code == 200
    assert client.get("/api/v1/health").json()["package_id"] == package
    assert (
        client.post(
            "/api/v1/bundles/import-jobs", headers={"Origin": "https://other.example"}
        ).status_code
        == 403
    )
    assert validate_bundle(local_bundle).package_id == package


def test_staged_score_directory_is_not_silently_imported(tmp_path):
    from exact_inspect.contracts import DomainError
    from exact_inspect.decisions import import_run

    (tmp_path / "pairs.jsonl").write_text("{}\n")
    with pytest.raises(DomainError, match="Expected an Exact alignment run"):
        import_run(
            tmp_path,
            tmp_path / "view",
            source_ontology_version_id="s",
            target_ontology_version_id="t",
        )


def test_source_candidate_rank_score_is_retained():
    from exact_inspect.decisions import _source_candidates

    document = {
        "records": [
            {
                "Src": "urn:s",
                "SrcKind": "class",
                "candidates": [
                    {
                        "target": "urn:t",
                        "S_rank": 0.72,
                        "gold_target": True,
                        "S_private": {"secret": 1},
                    }
                ],
            }
        ]
    }
    candidates = list(_source_candidates(document, "sha256:source").values())
    assert len(candidates) == 1
    assert candidates[0]["values"] == {"S_rank": 0.72}
    assert candidates[0]["reference"]["scores"][0]["name"] == "S_rank"
    assert candidates[0]["reference"]["scores"][0]["value"] == 0.72


def test_local_pages_and_host_boundary(tmp_path):
    export = tmp_path / "static"
    for page in ("", "browse", "library"):
        directory = export / page
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.html").write_text("<html><script>self.boot=1</script></html>")
    settings = InspectSettings(frontend_dir=export, library_dir=tmp_path / "library")
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        for page in ("/", "/browse/", "/library/"):
            response = client.get(page)
            assert response.status_code == 200
            assert "script-src 'self' 'sha256-" in response.headers["Content-Security-Policy"]
        for page in ("/participate/", "/admin/", "/legacy/", "/api/v1/study/state"):
            assert client.get(page).status_code == 404
        assert client.get("/", headers={"Host": "other.example"}).status_code == 400
    with pytest.raises(ValueError, match="Frontend export not found"):
        resolve_frontend_dir(InspectSettings(frontend_dir=tmp_path / "missing"))


def test_empty_ontology_is_rejected_but_empty_run_auxiliary_is_bound(tmp_path):
    from exact_inspect.preparation import InputBinding, OntologyBinding

    empty = tmp_path / "empty"
    empty.touch()
    binding = InputBinding(path=str(empty), size=0, sha256=file_hash(empty))
    assert binding.verify(tmp_path) == empty
    with pytest.raises(ValueError, match="must be nonempty"):
        OntologyBinding(name="source", root=binding)


def test_unsupported_viewer_python_has_actionable_error(monkeypatch, capsys):
    import exact_inspect.cli as cli

    monkeypatch.setattr(cli.sys, "version_info", (3, 10))
    with pytest.raises(SystemExit) as result:
        cli.main(["serve"])
    assert result.value.code == 2
    assert "requires Python 3.11" in capsys.readouterr().err
