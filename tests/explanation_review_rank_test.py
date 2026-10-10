"""Review array positions must not acquire a false production-ranking meaning."""

import copy
import json
import os
from types import SimpleNamespace

from fastapi.testclient import TestClient

from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import VisibilityPolicy
from exact_inspect.study.api import create_study_app
from tests.explanation_review_publication_test import review_fixture
from tools import prepare_review_bundle as preparation


def test_prepared_unresolved_review_keys_admit_empty_ranks_without_weakening_legacy(tmp_path):
    frozen = review_fixture(tmp_path)
    original_keys = copy.deepcopy(frozen["case_keys"])
    frozen["case_keys"] = [
        preparation.unresolved_review_key(case) for case in frozen["definition"]["cases"]
    ]
    app = create_study_app(
        database_url=os.environ.get(
            "EXACT_STUDY_TEST_DATABASE_URL", "sqlite:///" + str(tmp_path / "ranks.sqlite")
        ),
        signing_secret="s" * 48,
        researcher_token="r" * 48,
        origin="https://localhost",
        assets_dir=tmp_path,
        allow_test_sqlite=True,
    )
    with TestClient(app, base_url="https://localhost") as client:

        def publish(value):
            return client.post(
                "/api/v1/admin/studies", json=value, headers={"Authorization": "Bearer " + "r" * 48}
            )

        response = publish(frozen)
        assert response.status_code == 200, response.text
        assert publish(frozen).json() == response.json()
        with app.state.study_store.transaction() as db:
            raw = db.execute(
                "SELECT payload FROM researcher_case_keys WHERE study_revision = ?",
                (frozen["definition"]["study_revision"],),
            ).fetchone()["payload"]
        assert all(key["original_production_ranks"] == {} for key in json.loads(raw).values())

        # Even truthful ranks cannot be added to an already frozen key on retry.
        edited = copy.deepcopy(frozen)
        edited["case_keys"] = original_keys
        assert publish(edited).status_code == 409

        partial = copy.deepcopy(frozen)
        partial["definition"]["study_revision"] += "-partial"
        partial["case_keys"][0]["original_production_ranks"] = {
            frozen["definition"]["cases"][0]["candidates"][0]["candidate_id"]: 1
        }
        response = publish(partial)
        assert response.status_code == 422 and "all five" in response.text

        legacy = copy.deepcopy(frozen)
        legacy["definition"]["study_revision"] += "-legacy"
        legacy["definition"]["synthetic"] = True
        for field in ("purpose", "correctness_metrics", "review_provenance"):
            legacy["definition"].pop(field)
        response = publish(legacy)
        assert response.status_code == 422 and "development review purpose" in response.text
        legacy["case_keys"] = original_keys
        response = publish(legacy)
        assert response.status_code == 422 and "development review purpose" in response.text
        for unresolved_field in ("case_kind", "origin"):
            one_unresolved = copy.deepcopy(legacy)
            for key, case in zip(one_unresolved["case_keys"], legacy["definition"]["cases"]):
                key.update(
                    case_kind="answer_present",
                    origin="natural",
                    acceptable_candidate_ids=[case["candidates"][0]["candidate_id"]],
                )
                key[unresolved_field] = "unresolved"
            response = publish(one_unresolved)
            assert response.status_code == 422 and "development review purpose" in response.text
        with app.state.study_store.transaction() as db:
            assert (
                db.execute(
                    "SELECT revision FROM studies WHERE revision = ?",
                    (legacy["definition"]["study_revision"],),
                ).fetchone()
                is None
            )
        for key, case in zip(legacy["case_keys"], legacy["definition"]["cases"]):
            key.update(
                case_kind="answer_present",
                origin="natural",
                acceptable_candidate_ids=[case["candidates"][0]["candidate_id"]],
                original_production_ranks={},
            )
        response = publish(legacy)
        assert response.status_code == 422 and "all five" in response.text
        for key, case in zip(legacy["case_keys"], legacy["definition"]["cases"]):
            key["original_production_ranks"] = {
                candidate["candidate_id"]: index
                for index, candidate in enumerate(case["candidates"], 1)
            }
        response = publish(legacy)
        assert response.status_code == 200, response.text


def test_review_top_scores_retain_truthful_saved_array_positions(tmp_path, monkeypatch):
    atomic_json(
        tmp_path / "experiment_manifest.json", {"status": "complete", "split_role": "development"}
    )
    atomic_json(tmp_path / "run_manifest.json", {"fixture": "no reference labels"})
    atomic_json(
        tmp_path / "source_decisions.json",
        {
            "records": [
                {
                    "Src": "urn:S",
                    "candidates": [
                        {"target": "urn:" + label, "S_final": score}
                        for label, score in zip("ABCDEF", [0.3, 0.8, 0.1, 0.7, 0.9, 0.7])
                    ],
                }
            ]
        },
    )
    monkeypatch.setattr(
        preparation.RunReader, "open", lambda _: SimpleNamespace(iter_explanations=lambda: [])
    )
    monkeypatch.setattr(
        preparation,
        "packet_for",
        lambda *args: (
            SimpleNamespace(facts=[]),
            {"definitions": 0, "alternate_definitions": 0, "restrictions": 0},
        ),
    )
    context = SimpleNamespace(hierarchy=lambda *args, **kwargs: {"total_count": 0})
    plan, _ = preparation.plan_cases(
        tmp_path, {"source": context, "target": context}, "source", "target", VisibilityPolicy(), 1
    )
    selected = plan["cases"][0]
    assert [row["target"] for row in selected["recorded_candidates"]] == [
        "urn:E",
        "urn:B",
        "urn:D",
        "urn:F",
        "urn:A",
    ]
    assert [row["recorded_position"] for row in selected["recorded_candidates"]] == [5, 2, 4, 6, 1]
    assert "not a production rank" in plan["recorded_position_meaning"]
    assert preparation.unresolved_review_key(selected)["original_production_ranks"] == {}
