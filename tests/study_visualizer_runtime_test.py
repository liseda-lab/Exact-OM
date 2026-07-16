import json
from pathlib import Path

from fastapi.testclient import TestClient

from study_visualizer_runtime.app import (
    StudyVisualizerService,
    create_study_visualizer_app,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "deploy" / "render" / "study_bundles" / "omim-ordo"


def test_source_options_include_labels_and_preserve_bundle_order():
    service = StudyVisualizerService(run_dir=BUNDLE_DIR, enable_ontology_info=False)
    options = service.source_options()
    mapping = json.loads(
        (BUNDLE_DIR / "analysis" / "user_study" / "study_mapping.json").read_text(encoding="utf-8")
    )

    assert options
    assert options[0]["source_id"] == mapping["pairs"][0]["id"]
    assert options[0]["source_label"]
    assert len(options) == len(service.available_sources())


def test_api_study_sources_returns_bundle_local_source_options(tmp_path):
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    (frontend_dir / "index.html").write_text(
        "<!doctype html><html><body>ok</body></html>", encoding="utf-8"
    )

    app = create_study_visualizer_app(
        run_dir=BUNDLE_DIR,
        enable_ontology_info=False,
        frontend_build_dir=frontend_dir,
    )
    client = TestClient(app)

    response = client.get("/api/study/sources")
    assert response.status_code == 200

    payload = response.json()
    assert payload
    assert set(payload[0].keys()) == {"source_id", "source_label"}
    assert payload[0]["source_id"] == app.state.study_service.available_sources()[0]
    assert payload[0]["source_label"]
