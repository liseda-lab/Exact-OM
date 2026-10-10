"""Complete saved candidate pools survive absent explanation and mapping exports."""

import copy
import json
from pathlib import Path

import pyowl_core as core
import pytest
from fastapi.testclient import TestClient

from exact.runs import ExplanationStore
from exact.runs.reader import RunReader
from exact_inspect.artifacts import publish_bundle
from exact_inspect.context import build_context_package
from exact_inspect.contracts import DomainError, canonical_hash, file_hash
from exact_inspect.decisions import (
    DecisionStore,
    _global_mapping_path,
    _source_candidates,
    import_run,
)
from exact_inspect.service import create_prepared_app


@pytest.fixture
def actual():
    saved = json.loads(
        (
            Path(__file__).parent / "fixtures/explanation_source_candidates_local300.json"
        ).read_bytes()
    )
    assert canonical_hash(saved["source_record"]) == saved["provenance"]["retained_source_sha256"]
    assert (
        canonical_hash(saved["explanation"]) == saved["provenance"]["retained_explanation_sha256"]
    )
    return saved


def prepare(tmp_path, actual, *, mapping=True):
    run = tmp_path / "run"
    ExplanationStore.create(run).append([actual["explanation"]])
    source = copy.deepcopy(actual["source_record"])
    # Adversarial fields must be excluded even when they mimic score families.
    for row in source["candidates"]:
        row.update(
            P_gold=1.0, S_private_gold=0.9, cand_gold="PRIVATE-REFERENCE", label="PRIVATE-REFERENCE"
        )
    (run / "source_decisions.json").write_text(
        json.dumps({"schema_version": 1, "records": [source]})
    )
    if mapping:
        path = run / "alignment" / "paper.maps_global.tsv"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "SrcEntity\tTgtEntity\tScore\tRelation\tSrcKind\tTgtKind\n"
            + f"{source['Src']}\t{actual['omitted_target']}\t1.0\t=\tclass\tclass\n"
        )
        manifest_path = run / "run_manifest.json"
        manifest = RunReader.open(run).manifest()
        manifest.update(layout_version=2, synthesized=False)
        manifest.setdefault("artifacts", []).append(
            {
                "kind": "alignment",
                "path": "alignment/paper.maps_global.tsv",
                "sha256": file_hash(path)[7:],
                "bytes": path.stat().st_size,
            }
        )
        manifest_path.write_text(json.dumps(manifest))
    return run, source


def test_actual_full_source_candidate_array_is_paged_with_exact_saved_values_and_missingness(
    actual, tmp_path
):
    actual = copy.deepcopy(actual)
    # Distinct saved stages can disagree: retain the prior main score and the
    # exact source-decision score separately rather than silently overwrite.
    actual["explanation"]["confidences"]["S_final"] = -0.125
    run, source = prepare(tmp_path, actual)
    root = tmp_path / "package"
    contexts = [
        build_context_package(
            core.load_snapshot(f"Ontology(Declaration(Class(<{source['Src']}>)))".encode()),
            root / "source",
        ),
        build_context_package(
            core.load_snapshot(
                (
                    "Ontology("
                    + " ".join(f"Declaration(Class(<{c['target']}>))" for c in source["candidates"])
                    + ")"
                ).encode()
            ),
            root / "target",
        ),
    ]
    manifest = import_run(
        run,
        root / "decisions",
        source_ontology_version_id=contexts[0].ontology_version_id,
        target_ontology_version_id=contexts[1].ontology_version_id,
        run_id="actual-saved-candidates",
    )
    assert manifest["counts"]["pairs"] == source["candidate_count"] == 100
    assert manifest["counts"]["source_decision_candidates"] == 100
    assert manifest["counts"]["source_candidates_without_explanations"] == 99
    assert "alignment/paper.maps_global.tsv" in manifest["dependencies"]
    package = publish_bundle(
        root,
        ontologies={c.ontology_version_id: c.path.name for c in contexts},
        runs={manifest["run_id"]: "decisions"},
    )
    frozen = file_hash(root / "decisions" / "decisions.sqlite")
    source_hash = file_hash(run / "source_decisions.json")
    expected = {c["target"]: (i, c) for i, c in enumerate(source["candidates"])}
    rows, cursor, cursors = [], None, set()
    with TestClient(create_prepared_app(package)) as client:
        while True:
            page = client.get(
                "/api/v1/runs/actual-saved-candidates/candidates",
                params={
                    "source": source["Src"],
                    "limit": 7,
                    **({"cursor": cursor} if cursor else {}),
                },
            )
            assert page.status_code == 200, page.text
            assert not any(
                x in page.text
                for x in (
                    "PRIVATE-REFERENCE",
                    "P_gold",
                    "S_private_gold",
                    "cand_gold",
                    "reconstruction",
                )
            )
            body = page.json()
            assert body["total_count"] == 100 and body["returned_count"] == len(body["items"])
            rows.extend(body["items"])
            cursor = body["next_cursor"]
            if cursor is None:
                break
            assert cursor not in cursors
            cursors.add(cursor)
        assert len(rows) == len({r["pair_id"] for r in rows}) == 100
        assert {r["target"]["iri"] for r in rows} == set(expected)
        explained = next(r for r in rows if r["target"]["iri"] == actual["explanation"]["tgt_iri"])
        assert next(s["value"] for s in explained["scores"] if s["name"] == "S_final") == -0.125
        for row in rows:
            index, raw = expected[row["target"]["iri"]]
            provenance = row["source_decision_record"]
            assert provenance["artifact_ref"] == source_hash
            assert provenance["source_record_index"] == 0 and provenance["candidate_index"] == index
            assert provenance["ordering"] == "source_decisions_candidates_array"
            assert (
                next(s["value"] for s in provenance["scores"] if s["name"] == "S_final")
                == raw["S_final"]
            )
            assert row["ordinal_ranks"]["retrieval_rank"] is None
            assert row["ordinal_ranks"]["candidate_joint_rank"] is None
            assert all(
                s["name"] not in {"candidate_index", "source_record_index"} for s in row["scores"]
            )
        omitted = next(r for r in rows if r["target"]["iri"] == actual["omitted_target"])
        assert (
            omitted["decision_status"] == "not_exported"
            and omitted["saved_alignment_member"] is True
        )
        assert (
            omitted["membership_provenance"]["artifact_hash"]
            == manifest["dependencies"]["alignment/paper.maps_global.tsv"]
        )
        pair = client.get(
            "/api/v1/runs/actual-saved-candidates/pair", params={"pair_id": omitted["pair_id"]}
        )
        assert pair.status_code == 200, pair.text
        trace = pair.json()
        assert trace["evidence_metadata"]["status"] == "not_exported"
        assert trace["evidence_metadata"]["comparison_resolution"] is None
        assert trace["relation"] == "equivalent" and trace["relation_basis"] == "unavailable"
        assert all(e["status"] == "not_recorded" and not e["scores"] for e in trace["events"])
        evidence = client.get(
            "/api/v1/runs/actual-saved-candidates/pair-evidence",
            params={"pair_id": omitted["pair_id"]},
        )
        assert evidence.status_code == 200 and evidence.json()["total_count"] == 0
    assert file_hash(root / "decisions" / "decisions.sqlite") == frozen


@pytest.mark.parametrize("change", ["same", "conflicting_score", "conflicting_source_kind"])
def test_duplicate_or_conflicting_saved_candidate_identity_rejected(actual, change):
    source = copy.deepcopy(actual["source_record"])
    if change == "conflicting_source_kind":
        source["SrcKind"] = "object_property"
    else:
        source["candidates"].append(copy.deepcopy(source["candidates"][0]))
        if change == "conflicting_score":
            source["candidates"][-1]["S_final"] = 0.01
    with pytest.raises(DomainError, match="[Ii]dentity"):
        _source_candidates({"records": [source]}, "sha256:artifact")


def test_punned_kinds_are_distinct_and_array_positions_never_become_ranks():
    rows = [
        {"target": "urn:t", "SrcKind": kind, "TgtKind": kind, "S_final": 0.7}
        for kind in ("class", "object_property")
    ]
    saved = _source_candidates(
        {"records": [{"Src": "urn:s", "candidates": rows}]}, "sha256:artifact"
    )
    assert len(saved) == 2
    assert {r["record"]["src_kind"] for r in saved.values()} == {"class", "object_property"}
    assert [r["reference"]["candidate_index"] for r in saved.values()] == [0, 1]
    assert all(set(r["values"]) == {"S_final"} for r in saved.values())


@pytest.mark.parametrize("implementation", ["decision-adapter/6", "decision-adapter/7"])
def test_inventory_required_for_both_compact_revisions_and_legacy_remains_readable(
    tmp_path, actual, implementation
):
    run, _ = prepare(tmp_path, actual, mapping=False)
    output = tmp_path / "index"
    manifest = import_run(
        run, output, source_ontology_version_id="source", target_ontology_version_id="target"
    )
    path = output / "decisions.manifest.json"
    manifest["implementation"] = implementation
    path.write_text(json.dumps(manifest))
    DecisionStore(output)
    del manifest["dependency_inventory_hash"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(DomainError, match="inventory changed"):
        DecisionStore(output)
    manifest["implementation"] = "decision-adapter/5"
    path.write_text(json.dumps(manifest))
    DecisionStore(output)


@pytest.mark.parametrize("change", ["hash", "size", "escape", "ambiguous", "canonical_conflict"])
def test_declared_global_mapping_requires_unique_contained_exact_bytes(tmp_path, actual, change):
    run, _ = prepare(tmp_path, actual)
    reader = RunReader.open(run)
    manifest = reader.manifest()
    mapping = next(a for a in manifest["artifacts"] if a["kind"] == "alignment")
    if change == "hash":
        mapping["sha256"] = "0" * 64
    elif change == "size":
        mapping["bytes"] += 1
    elif change == "escape":
        mapping["path"] = "../outside.maps_global.tsv"
    elif change == "ambiguous":
        manifest["artifacts"].append(copy.deepcopy(mapping))
    else:
        (run / "alignment" / "maps_global.tsv").write_text("different")
    with pytest.raises(DomainError):
        _global_mapping_path(reader, manifest)


def test_synthesized_legacy_canonical_mapping_needs_no_preexisting_manifest_hash(tmp_path):
    run = tmp_path / "legacy"
    path = run / "model" / "alignment" / "src2tgt.maps_global.tsv"
    path.parent.mkdir(parents=True)
    path.write_text("SrcEntity\tTgtEntity\tScore\tRelation\nurn:s\turn:t\t0.7\t=\n")
    reader = RunReader.open(run)
    manifest = reader.manifest()
    assert manifest["synthesized"] and "sha256" not in manifest["artifacts"][0]
    assert _global_mapping_path(reader, manifest) == path
    indexed = import_run(
        run,
        tmp_path / "index",
        source_ontology_version_id="source",
        target_ontology_version_id="target",
    )
    assert indexed["counts"]["pairs"] == 1
    assert indexed["dependencies"]["model/alignment/src2tgt.maps_global.tsv"] == file_hash(path)
    candidate = DecisionStore(tmp_path / "index").candidates("urn:s")["items"][0]
    assert (
        candidate["saved_alignment_member"] is True and candidate["source_decision_record"] is None
    )
