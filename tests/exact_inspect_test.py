from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from exact.runs import ExplanationStore, RunLayout
from exact_inspect.app import create_app, resolve_frontend_dir
from exact_inspect.bundles import StudyVisualizerService, export_bundle
from exact_inspect.cli import build_parser, main
from exact_inspect.settings import InspectSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "deploy" / "render" / "study_bundles" / "omim-ordo"


def _record(source: str = "source", target: str = "target") -> dict:
    return {
        "src_iri": source,
        "tgt_iri": target,
        "selected_labels": {"source": "Source label", "target": "Target label"},
        "confidences": {"S_final": 0.91},
        "prediction": {"global_match": True, "threshold_positive": True},
    }


def _v1_run(root: Path) -> Path:
    alignment = root / "model" / "alignment"
    alignment.mkdir(parents=True)
    (alignment / "src2tgt.maps_local.tsv").write_text(
        "SrcEntity\tTgtEntity\tScore\nsource\ttarget\t0.91\n", encoding="utf-8"
    )
    explanations = alignment / "default" / "full_explanations.json"
    explanations.parent.mkdir()
    explanations.write_text(json.dumps([_record()]), encoding="utf-8")
    return root


def _v2_run(root: Path) -> Path:
    layout = RunLayout.create(root)
    layout.mapping_path("local").write_text(
        "SrcEntity\tTgtEntity\tScore\nsource\ttarget\t0.91\n", encoding="utf-8"
    )
    ExplanationStore(layout.explanations_dir, run_id="inspect-test").append([_record()])
    return root


def _multi_shard_v2_run(root: Path) -> Path:
    layout = RunLayout.create(root)
    layout.mapping_path("local").write_text(
        "SrcEntity\tTgtEntity\tScore\n" "source-a\ttarget-a\t0.91\n" "source-b\ttarget-b\t0.82\n",
        encoding="utf-8",
    )
    store = ExplanationStore(
        layout.explanations_dir,
        run_id="inspect-test",
        shard_mb=0.000001,
    )
    first = _record("source-a", "target-a")
    first["selected_labels"] = {"source": "Source A", "target": "Target A"}
    second = _record("source-b", "target-b")
    second["selected_labels"] = {"source": "Source B", "target": "Target B"}
    store.append([first, second])
    index = json.loads(layout.explanation_index_path.read_text(encoding="utf-8"))
    assert len(index["shards"]) == 2
    return root


def test_legacy_settings_warn_and_new_environment_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    legacy.mkdir()
    current.mkdir()
    monkeypatch.setenv("EXACT_STUDY_RUN_DIR", str(legacy))
    with pytest.warns(DeprecationWarning, match="EXACT_STUDY"):
        assert InspectSettings().run_dir == legacy.resolve()

    monkeypatch.setenv("EXACT_INSPECT_RUN_DIR", str(current))
    with pytest.warns(DeprecationWarning, match="EXACT_STUDY"):
        settings = InspectSettings()
    assert settings.run_dir == current.resolve()


def test_frontend_resolution_order_and_api_only_fallback(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    package = tmp_path / "package"
    project = tmp_path / "project"
    packaged = package / "static"
    development = project / "explanations_visualizer" / "out"
    for directory in (explicit, packaged, development):
        directory.mkdir(parents=True)
        (directory / "index.html").write_text("ok", encoding="utf-8")

    settings = InspectSettings(frontend_dir=explicit)
    assert resolve_frontend_dir(settings, package_dir=package, project_root=project) == explicit
    (explicit / "index.html").unlink()
    assert resolve_frontend_dir(settings, package_dir=package, project_root=project) == packaged
    (packaged / "index.html").unlink()
    assert resolve_frontend_dir(settings, package_dir=package, project_root=project) == development
    (development / "index.html").unlink()
    assert resolve_frontend_dir(settings, package_dir=package, project_root=project) is None


@pytest.mark.parametrize("layout", ["v1", "v2"])
def test_open_mode_serves_runreader_artifacts(tmp_path: Path, layout: str) -> None:
    run_dir = _v1_run(tmp_path / layout) if layout == "v1" else _v2_run(tmp_path / layout)
    app = create_app(InspectSettings(run_dir=run_dir, enable_ontology_info=False))
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["mode"] == "open"
    assert health.json()["layout_version"] == (1 if layout == "v1" else 2)

    sources = client.get("/api/study/sources").json()
    assert sources == [
        {
            "source_id": "source",
            "source_label": "Source label" if layout == "v1" else "source",
        }
    ]
    bundle = client.get("/api/study/source", params={"source": "source"})
    assert bundle.status_code == 200
    assert bundle.json()["source_label"] == "Source label"
    assert bundle.json()["targets"][0]["target_id"] == "target"
    assert bundle.json()["targets"][0]["score"] == pytest.approx(0.91)
    assert client.get("/api/study/sources").json()[0]["source_label"] == "Source label"

    banner = client.get("/")
    assert banner.status_code == 200
    assert "API-only mode" in banner.text


def test_v2_open_reads_at_most_one_shard_per_requested_source(tmp_path: Path) -> None:
    run_dir = _multi_shard_v2_run(tmp_path / "v2-multi")
    app = create_app(InspectSettings(run_dir=run_dir, enable_ontology_info=False))
    service = app.state.study_service
    client = TestClient(app)

    assert service.reader is not None
    assert service.reader.explanation_shard_reads == 0
    assert [row["source_id"] for row in client.get("/api/study/sources").json()] == [
        "source-a",
        "source-b",
    ]
    assert service.reader.explanation_shard_reads == 0

    first = client.get("/api/study/source", params={"source": "source-a"})
    assert first.status_code == 200
    assert first.json()["source_label"] == "Source A"
    assert service.reader.explanation_shard_reads == 1

    assert client.get("/api/study/source", params={"source": "source-a"}).status_code == 200
    assert service.reader.explanation_shard_reads == 1

    second = client.get("/api/study/source", params={"source": "source-b"})
    assert second.status_code == 200
    assert second.json()["source_label"] == "Source B"
    assert service.reader.explanation_shard_reads == 2


def test_committed_bundle_endpoints_are_unchanged(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<!doctype html><p>viewer</p>", encoding="utf-8")
    app = create_app(
        InspectSettings(
            run_dir=BUNDLE_DIR,
            frontend_dir=frontend,
            enable_ontology_info=False,
        )
    )
    client = TestClient(app)

    response = client.get("/api/study/sources")
    assert response.status_code == 200
    assert response.json()[0]["source_id"] == app.state.study_service.available_sources()[0]
    assert client.get("/?source=http://example.org/source").status_code == 200


def test_bundle_exports_plain_run_and_can_be_served(tmp_path: Path) -> None:
    run_dir = _v1_run(tmp_path / "run")
    destination = export_bundle(run_dir, tmp_path / "bundle")

    assert (destination / "study_bundle.json").is_file()
    assert (destination / "analysis" / "user_study" / "ontology_cache.json").is_file()
    service = StudyVisualizerService(destination, enable_ontology_info=False)
    assert service.available_sources() == ["source"]


def test_cli_contract_has_three_modes() -> None:
    parser = build_parser()
    assert parser.parse_args(["open", "/tmp/run", "--no-browser"]).command == "open"
    assert parser.parse_args(["serve", "--run-dir", "/tmp/run"]).command == "serve"
    assert parser.parse_args(["bundle", "/tmp/run", "/tmp/bundle"]).command == "bundle"


def test_bundle_yaml_job_mode_is_preserved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "bundle-job.yaml"
    job.write_text(
        "bundle:\n"
        f"  run_dir: {tmp_path / 'run'}\n"
        f"  bundle_dir: {tmp_path / 'bundle'}\n"
        "  overwrite: true\n"
        "job:\n"
        "  logging_level: DEBUG\n",
        encoding="utf-8",
    )

    assert main(["bundle", "--job-config", str(job), "--dry-run"]) == 0
    command = capsys.readouterr().out
    assert "exact-inspect bundle" in command
    assert "--overwrite" in command
    assert "--log-level DEBUG" in command


def test_missing_fastapi_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    from exact_inspect import app as app_module

    original_import = builtins.__import__

    def reject_fastapi(name: str, *args: object, **kwargs: object) -> object:
        if name == "fastapi" or name.startswith("fastapi."):
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_fastapi)
    with pytest.raises(RuntimeError, match=r"exact-om\[viz\]"):
        app_module._fastapi_components()
