"""Public review metadata is bounded, explicit and never a raw private manifest."""

from fastapi.testclient import TestClient

from exact_inspect.artifacts import publish_bundle
from exact_inspect.contracts import VisibilityPolicy
from exact_inspect.review_metadata import ReviewBundleSummary
from exact_inspect.service import create_prepared_app


def test_about_endpoint_has_same_strict_public_shape_used_by_study(tmp_path):
    about = ReviewBundleSummary(
        title="Development review",
        prepared_at="2026-10-10T12:00:00Z",
        source_run_id="recorded-dev-run",
        run_manifest_sha256="sha256:" + "a" * 64,
        selection_sha256="sha256:" + "b" * 64,
        policy_sha256=VisibilityPolicy().policy_hash,
        source_count=24,
        presented_pairs=120,
        coverage={"sparse_evidence": 2},
        limitations=["Synthetic test metadata; does not prove real coverage."],
        ontologies=[],
    )

    def app(value):
        package = publish_bundle(tmp_path, policy=VisibilityPolicy(), provenance=value)
        return create_prepared_app(package=package)

    with TestClient(app({"review_bundle": about.model_dump()})) as client:
        response = client.get("/api/v1/review-bundle")
        assert response.status_code == 200
        assert response.json() == about.model_dump()
    unsafe = {**about.model_dump(), "source_path": "PRIVATE_PREPARATION_PATH"}
    with TestClient(app({"review_bundle": unsafe})) as client:
        response = client.get("/api/v1/review-bundle")
        assert response.status_code == 409
        assert "PRIVATE_PREPARATION_PATH" not in response.text
    with TestClient(app({})) as client:
        assert client.get("/api/v1/review-bundle").status_code == 404
