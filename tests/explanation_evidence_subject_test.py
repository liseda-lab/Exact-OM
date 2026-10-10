"""Retained real neighbour evidence never becomes a false focal-entity citation."""

import copy
import dataclasses
import json
import sqlite3
from pathlib import Path

import pyowl_core as core
import pytest
from fastapi.testclient import TestClient

from exact.runs import ExplanationStore
from exact_inspect.artifacts import publish_bundle
from exact_inspect.context import build_context_package
from exact_inspect.context_export import export_policy_context
from exact_inspect.context_resources import _native_admitted_axiom
from exact_inspect.contracts import DomainError, VisibilityPolicy, file_hash
from exact_inspect.decisions import DecisionStore, import_run
from exact_inspect.decisions_export import export_policy_run
from exact_inspect.evidence_metadata import filter_evidence_items, pair_evidence_metadata
from exact_inspect.models import SelectedEvidence
from exact_inspect.service import create_prepared_app
from exact_inspect.study.workspace import build_workspace_scope, validate_publication_workspaces
from tests.explanation_workspace_evidence_test import EvidenceStore, client_for


@pytest.fixture
def retained_neighbour(tmp_path):
    saved = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("explanation_evidence_subject_local300.json")
        .read_text()
    )
    record = saved["record"]
    focal = record["evidence"]["entity"]["iri"]
    neighbour = record["actual_refs"][0][0]
    target = record["target"]
    # Preserve the exact admitted real expression. Only the surrounding test
    # scope/declarations and explicit positive/punning controls are synthetic.
    template = core.load_snapshot(
        (
            f"Ontology(Declaration(Class(<{focal}>)) Declaration(NamedIndividual(<{focal}>)) "
            f"Declaration(Class(<{target}>)) SubClassOf(<{focal}> <{neighbour}>) "
            f"ClassAssertion(<{neighbour}> <{focal}>))"
        ).encode()
    ).root
    real = _native_admitted_axiom(saved["axiom"]["ast"], core)
    assert "sha256:" + core.structural_digest(real).hex() == saved["axiom"]["original_axiom_digest"]
    document = dataclasses.replace(template, axioms=core.CanonicalSet([*template.axioms, real]))
    context = build_context_package(
        core.load_snapshot(core.render_document(document, format=core.DocumentFormat.FUNCTIONAL)),
        tmp_path / "ontology",
    )
    source = {**record["evidence"]["entity"], "ontology_version_id": context.ontology_version_id}
    target = {**source, "iri": target}
    neighbour_id = context.resolve_feature(
        {"source_axiom_refs": [saved["axiom"]["original_axiom_digest"]]}, source
    )["fact_ids"][0]
    assert context.axiom(neighbour_id)["ast"] == saved["axiom"]["ast"]
    with pytest.raises(KeyError):
        context.fact(neighbour_id, source)
    direct_id = context.hierarchy(source)["items"][0]["axiom_id"]
    individual_id = context.facts({**source, "kind": "individual"}, category="types")["items"][0][
        "fact_id"
    ]
    rows = []
    for index, ids in enumerate(
        [[neighbour_id], [direct_id], [individual_id], [direct_id, neighbour_id]]
    ):
        row = copy.deepcopy(record["evidence"])
        row.update(entity=source, fact_ids=ids)
        if index:
            row["evidence_id"] += f"-control-{index}"
            row["feature_id"] += f"-control-{index}"
        SelectedEvidence.model_validate(row)
        rows.append(row)
    return context, source, target, rows


def assert_safe(rows, original):
    assert [r["evidence_id"] for r in rows] == [r["evidence_id"] for r in original]
    assert [r["origin_resolution"] for r in rows] == [
        "unresolved",
        "resolved",
        "unresolved",
        "unresolved",
    ]
    for index, row in enumerate(rows):
        SelectedEvidence.model_validate(row)
        assert row["values"] == original[index]["values"]
        if index != 1:
            assert row["fact_ids"] == [] and row["feature_id"] is None
            assert row["status"] == "not_exported"
            assert (
                row["reason"]
                == "Original facts do not bind to the selected feature's typed entity."
            )
    assert rows[1]["fact_ids"] == original[1]["fact_ids"]


def neighbour_metadata(source, target, rows):
    # A controlled recorded channel comparison exercises the historical metadata
    # contract around the exact retained real neighbour feature above.
    bridge = {
        "level": "channel_comparison",
        "channel": "hierarchy",
        "source_family": "is_a",
        "source": source,
        "target": target,
        "source_evidence_ids": [rows[0]["evidence_id"]],
        "source_fact_ids": rows[0]["fact_ids"],
    }
    return {
        "source": source,
        "target": target,
        "evidence_metadata": pair_evidence_metadata(
            {"evidence_bridges": [bridge]}, rows, source, target, ["controlled-bridge-fixture"]
        ),
    }


def test_real_retained_neighbour_main_read_and_new_policy_export_agree(
    retained_neighbour, tmp_path
):
    context, source, target, rows = retained_neighbour
    policy = VisibilityPolicy(ontology_ids=(context.ontology_version_id,))
    original_bytes = file_hash(context.database)
    raw_copy = copy.deepcopy(rows)
    safe = filter_evidence_items(rows, {context.ontology_version_id: context}, policy)
    assert_safe(safe, rows)
    assert rows == raw_copy
    assert filter_evidence_items(safe, {context.ontology_version_id: context}, policy) == safe
    portable = export_policy_context(context, tmp_path / "portable-context", policy)
    assert (
        filter_evidence_items(
            rows, {context.ontology_version_id: portable}, policy, physically_filtered=True
        )
        == safe
    )

    run = tmp_path / "raw-run"
    ExplanationStore.create(run).append([{"src_iri": source["iri"], "tgt_iri": target["iri"]}])
    import_run(
        run,
        tmp_path / "decisions",
        source_ontology_version_id=context.ontology_version_id,
        target_ontology_version_id=context.ontology_version_id,
        run_id="retained",
    )
    decisions = DecisionStore(tmp_path / "decisions")
    pair_id = decisions.candidates(source["iri"])["items"][0]["pair_id"]
    # Reconstruct the historical saved selected payload before freezing this
    # test package. Serving and exporting may not rewrite those retained bytes.
    with sqlite3.connect(tmp_path / "decisions/decisions.sqlite") as db:
        trace = json.loads(db.execute("SELECT trace FROM pairs").fetchone()[0])
        trace["evidence_metadata"] = pair_evidence_metadata(
            {"attributes": {}}, rows, source, target, ["retained-real-fixture"]
        )
        db.execute("UPDATE pairs SET evidence=?,trace=?", (json.dumps(rows), json.dumps(trace)))
    path = tmp_path / "decisions/decisions.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["database_hash"] = file_hash(tmp_path / "decisions/decisions.sqlite")
    path.write_text(json.dumps(manifest))
    frozen_hash = file_hash(tmp_path / "decisions/decisions.sqlite")
    package = publish_bundle(
        tmp_path,
        audience="local",
        policy=policy,
        ontologies={context.ontology_version_id: "ontology"},
        runs={"retained": "decisions"},
    )
    with TestClient(create_prepared_app(package)) as client:
        params = {"pair_id": pair_id, "limit": 1}
        actual = []
        while True:
            response = client.get("/api/v1/runs/retained/pair-evidence", params=params)
            assert response.status_code == 200, response.text
            page = response.json()
            assert page["total_count"] == len(rows) and page["returned_count"] == 1
            actual.extend(page["items"])
            if not page["next_cursor"]:
                break
            params["cursor"] = page["next_cursor"]
        expected = sorted(safe, key=lambda r: r["evidence_id"])
        assert actual == [SelectedEvidence.model_validate(row).model_dump() for row in expected]
        metadata = client.get("/api/v1/runs/retained/pair", params={"pair_id": pair_id}).json()[
            "evidence_metadata"
        ]
        assert metadata["counts"]["selected"] == 4
        assert metadata["counts"]["unresolved"] == 3 and metadata["counts"]["resolved"] == 1
    exported = export_policy_run(
        DecisionStore(tmp_path / "decisions"),
        tmp_path / "exported-decisions",
        contexts={context.ontology_version_id: portable},
        policy=policy,
    )
    assert sorted(exported.evidence(pair_id), key=lambda row: row["evidence_id"]) == expected
    assert file_hash(tmp_path / "decisions/decisions.sqlite") == frozen_hash
    assert file_hash(context.database) == original_bytes


@pytest.mark.parametrize("historical_raw_payload", [False, True])
def test_real_retained_neighbour_scoped_admission_and_legacy_http_keep_all_records(
    retained_neighbour, tmp_path, historical_raw_payload
):
    context, source, target, rows = retained_neighbour
    policy = VisibilityPolicy(ontology_ids=(context.ontology_version_id,))
    case = {
        "case_id": "case-a",
        "source": source,
        "candidates": [{"candidate_id": "candidate-a", "entity": target}],
        "explanation_refs": [],
    }
    metadata = neighbour_metadata(source, target, rows)
    scope = build_workspace_scope(
        tmp_path,
        scope_id="real-neighbour-scope",
        kind="case",
        case=case,
        contexts={context.ontology_version_id: context},
        policy=policy,
        evidence={"candidate-a": rows},
        pair_metadata={"candidate-a": metadata},
        components=["context", "evidence"],
    )
    database = tmp_path / scope.index_path
    if historical_raw_payload:
        with sqlite3.connect(database) as db:
            db.executemany(
                "UPDATE evidence SET payload=? WHERE id=?",
                [(json.dumps(row), row["evidence_id"]) for row in rows],
            )
            db.execute("UPDATE pair_metadata SET payload=?", (json.dumps(metadata),))
        scope = scope.model_copy(
            update={"index_sha256": file_hash(database).removeprefix("sha256:")}
        )
    study = {
        "contract_version": "exact-study/1.0",
        "study_revision": "retained-neighbour-revision",
        "cases": [case],
        "policy_hash": policy.policy_hash.removeprefix("sha256:"),
        "visibility_policy": policy.model_dump(mode="json"),
        "workspace_scopes": [scope.model_dump(mode="json")],
    }
    store = EvidenceStore(tmp_path, study)
    frozen_hash = file_hash(database)
    validate_publication_workspaces(store, study, {})
    with client_for(store) as client:
        url = "/api/v1/study/workspace/real-neighbour-scope/evidence"
        response = client.get(url, params={"candidate_id": "candidate-a"})
        assert response.status_code == 200, response.text
        assert response.json()["total_count"] == len(rows)
        assert_safe(response.json()["items"], rows)
        response = client.get(url + "-metadata", params={"candidate_id": "candidate-a"})
        assert response.status_code == 200, response.text
        served = response.json()["evidence_metadata"]
        assert served["counts"]["selected"] == 4
        assert served["counts"]["resolved"] == 1 and served["counts"]["unresolved"] == 3
        original_bridge = metadata["evidence_metadata"]["bridges"][0]
        bridge = served["bridges"][0]
        assert bridge["bridge_id"] == original_bridge["bridge_id"]
        assert bridge["source_evidence_ids"] == original_bridge["source_evidence_ids"]
        assert bridge["source_family"] == "is_a"
        assert bridge["status"] == served["bridge_status"] == "withheld"
        assert bridge["source_fact_ids"] == bridge["target_fact_ids"] == []
        # The same real original is available in its own scope but cannot be
        # presented as a fact of the focal typed entity.
        params = {
            "ontology_version_id": context.ontology_version_id,
            "iri": source["iri"],
            "kind": "class",
        }
        path = "/api/v1/study/workspace/real-neighbour-scope/axioms/" + rows[0]["fact_ids"][0]
        assert client.get(path, params=params).status_code == 404
        params.pop("iri")
        assert client.get(path, params=params).status_code == 200
        store.current["condition"] = "baseline"
        assert client.get(url, params={"candidate_id": "candidate-a"}).status_code == 403
    assert file_hash(database) == frozen_hash


@pytest.mark.parametrize(
    "invalid",
    ["foreign_evidence", "typed_pair", "bridge_pair", "channel", "family", "visible_fact"],
)
def test_historical_neighbour_metadata_still_rejects_unadmitted_identities_over_http(
    retained_neighbour, tmp_path, invalid
):
    context, source, target, rows = retained_neighbour
    policy = VisibilityPolicy(ontology_ids=(context.ontology_version_id,))
    case = {
        "case_id": "case-a",
        "source": source,
        "candidates": [{"candidate_id": "candidate-a", "entity": target}],
        "explanation_refs": [],
    }
    metadata = neighbour_metadata(source, target, rows)
    scope = build_workspace_scope(
        tmp_path,
        scope_id="metadata-negative",
        kind="case",
        case=case,
        contexts={context.ontology_version_id: context},
        policy=policy,
        evidence={"candidate-a": rows},
        pair_metadata={"candidate-a": metadata},
        components=["context", "evidence"],
    )
    bridge = metadata["evidence_metadata"]["bridges"][0]
    if invalid == "foreign_evidence":
        bridge["source_evidence_ids"] = ["foreign-selected-record"]
    elif invalid == "typed_pair":
        metadata["target"] = {**target, "kind": "individual"}
    elif invalid == "bridge_pair":
        bridge["source"] = {**source, "kind": "individual"}
    elif invalid == "channel":
        bridge["channel"] = "attributes"
    elif invalid == "family":
        bridge["source_family"] = "part_of"
    else:
        # A valid available focal record cannot acquire an unrelated citation.
        bridge["source_evidence_ids"] = [rows[1]["evidence_id"]]
        bridge["source_fact_ids"] = [*rows[1]["fact_ids"], "foreign-fact"]
    database = tmp_path / scope.index_path
    with sqlite3.connect(database) as db:
        db.executemany(
            "UPDATE evidence SET payload=? WHERE id=?",
            [(json.dumps(row), row["evidence_id"]) for row in rows],
        )
        db.execute("UPDATE pair_metadata SET payload=?", (json.dumps(metadata),))
    scope = scope.model_copy(update={"index_sha256": file_hash(database).removeprefix("sha256:")})
    study = {
        "contract_version": "exact-study/1.0",
        "study_revision": "metadata-negative-revision",
        "cases": [case],
        "policy_hash": policy.policy_hash.removeprefix("sha256:"),
        "visibility_policy": policy.model_dump(mode="json"),
        "workspace_scopes": [scope.model_dump(mode="json")],
    }
    store = EvidenceStore(tmp_path, study)
    frozen = file_hash(database)
    with pytest.raises((ValueError, DomainError)):
        validate_publication_workspaces(store, study, {})
    with client_for(store) as client:
        response = client.get(
            "/api/v1/study/workspace/metadata-negative/evidence-metadata",
            params={"candidate_id": "candidate-a"},
        )
        assert response.status_code == (503 if invalid == "typed_pair" else 422), response.text
        assert "foreign" not in response.text
    assert file_hash(database) == frozen
