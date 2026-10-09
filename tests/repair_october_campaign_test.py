"""Frozen October preparation targets, lineage and annotation scheduling."""

from collections import Counter
import dataclasses
import json
from pathlib import Path

import pytest
import pyowl_core as owl

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from tools.repair import corrective_campaign as campaign
from tools.repair.corpus import _case
from tools.repair.expanded_profile import MECHANISMS_SELECTED, parent_case
from tools.repair.expanded_corpus import binding, bound
from tools.repair.prepare import case_to_dict, case_from_dict, load_preparation


@pytest.mark.parametrize("family", MECHANISMS_SELECTED)
@pytest.mark.parametrize("sibling", (0, 1))
def test_targets_cover_all_families_mirrors_without_changing_deployment_inputs(family, sibling):
    original = _case("pure:" + family, family, "train", 2, sibling, 20261009)
    amended = campaign.semantic_queries(original)
    assert amended.problem is original.problem
    assert amended.intended_theory == original.intended_theory
    assert amended.intended_assignment == original.intended_assignment
    assert any(p.desired for p in amended.probes) and any(not p.desired for p in amended.probes)
    assert all(p.conditions() for p in amended.probes)
    assert len({p.probe_id for p in amended.probes}) == len(amended.probes)
    assert case_from_dict(case_to_dict(amended)) == amended
    assert case_to_dict(amended)["hash"] != case_to_dict(original)["hash"]
    assert campaign.semantic_queries(amended) == amended
    assert campaign.observable_queries(amended.problem) == campaign.observable_queries(
        original.problem
    )
    assert all(
        q["probe_id"].startswith("observed-") for q in campaign.observable_queries(amended.problem)
    )
    assert not any(
        q["probe_id"] == "corrective-unwanted/v1"
        for q in campaign.observable_queries(amended.problem)
    )


def inherited_fixture(path):
    release = path / "work/xr21-expanded-corpus/corpus/releases"
    ids = {}
    for split, depth, count in (("train", 2, 128), ("development", 6, 32), ("test", 10, 64)):
        rows = []
        for family in MECHANISMS_SELECTED:
            case = parent_case(family, depth, 20261009, split)
            payload = case_to_dict(case)
            destination = path / "cases" / f"{split}-{family}.json"
            write_artifact(destination, payload)
            rows.append(
                dict(
                    status="materialized",
                    case_id=case.case_id,
                    evaluator=binding(destination),
                    case_hash=payload["hash"],
                    input_hash=case.problem.content_hash,
                    split=split,
                )
            )
        rows.extend(
            dict(status="unavailable", case_id=f"unavailable-{split}-{n}")
            for n in range(count - len(rows))
        )
        ids[split] = [row["case_id"] for row in rows]
        write_artifact(release / (split + ".json"), dict(scheduled=count, rows=rows))
    write_artifact(path / "work/xr21-expanded-profile/profile/inventory.json", dict(selected=[]))
    return ids


def test_inherited_split_denominators_hashes_and_unavailable_slots_are_preserved(tmp_path):
    previous = tmp_path / "old"
    expected = inherited_fixture(previous)
    output = tmp_path / "new"
    cases, _ = campaign.inputs(previous, output)
    assert Counter(c.split for c in cases) == {"train": 8, "development": 8, "test": 8}
    for split, count in (("train", 128), ("development", 32), ("test", 64)):
        release = json.loads((output / (split + ".json")).read_text())
        assert release["scheduled"] == len(release["rows"]) == count
        assert [row["case_id"] for row in release["rows"]] == expected[split]
        for row in release["rows"]:
            if row["status"] == "materialized":
                payload = bound(row["evaluator"])
                assert row["case_hash"] == payload["hash"]
                assert row["case_hash"] != row["inherited_case_hash"]
                assert case_from_dict(payload).split == split
            else:
                assert row["inherited_unavailability"]
        prepared, caches, _ = load_preparation(output / (split + "-preparation.json"))
        assert len(prepared) == 8 and not caches
    # Preparation cannot quietly accept a different predeclared denominator.
    source = previous / "work/xr21-expanded-corpus/corpus/releases/test.json"
    invalid = json.loads(source.read_text())
    invalid["rows"].pop()
    write_artifact(source, invalid)
    with pytest.raises(ValueError, match="denominator"):
        campaign.inputs(previous, tmp_path / "invalid")


def test_actual_source_and_graph_contract_are_bound_instead_of_placeholders(tmp_path):
    source = campaign.source_identity()
    assert len(source["revision"]) == 40
    assert len(source["code_hash"]) == len(source["dirty_hash"]) == 64
    assert "exact/repair/pipeline.py" in source["source_files"]
    assert "tools/repair/corrective_campaign.py" in source["source_files"]
    assert source["code_hash"] == canonical_hash(source["source_files"])
    from exact.repair.graph_schema import generic_graph_schema

    assert source["graph_schema_hash"] == canonical_hash(generic_graph_schema())
    case = parent_case("overlap", 2, 20261009, "development")
    protocol = campaign.protocol(tmp_path, [case])
    assert protocol["identity"]["code_hash"] == source["code_hash"]
    assert protocol["identity"]["dirty_hash"] == source["dirty_hash"]
    assert protocol["model"]["graph_schema"] == generic_graph_schema()


def test_annotation_calibration_is_exactly32_native_qualified_train_slots(tmp_path):
    from tools.repair import corrective_calibration as calibration

    manifest_path = calibration.prepare(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["slots"]) == 32
    assert len({slot["id"] for slot in manifest["slots"]}) == 32
    assert Counter(slot["profile"] for slot in manifest["slots"]) == dict.fromkeys(
        calibration.SHORTLIST, 8
    )
    assert Counter(slot["swapped"] for slot in manifest["slots"]) == {False: 16, True: 16}
    assert set(manifest["parent_splits"].values()) == {"train"}
    assert "independent_test" not in {slot["profile"] for slot in manifest["slots"]}
    grouped = {}
    for slot in manifest["slots"]:
        packet = read_record(bound(slot["packet"]))
        assert packet.eligible and packet.split == "train"
        assert packet.plan_a.plan_id != packet.plan_b.plan_id
        assert set(packet.required_query_ids) == {"forward", "reverse"}
        grouped.setdefault((slot["profile"], packet.case_id), set()).add(slot["swapped"])
    assert len(grouped) == 16 and all(orders == {False, True} for orders in grouped.values())
    assert not (
        tmp_path / "annotations/ledger"
    ).exists()  # No paid transmissions during preparation.


def test_native_manifest_resume_keeps_completed_results_and_all_denominators(tmp_path, monkeypatch):
    from tools.repair import corrective_study as runner
    from exact.repair.candidates import mapping_candidates
    from exact.repair.records import RevisionObjectV2, PolicyV2, RepairInputV2
    from exact.repair.learning import TeacherProbe
    from tools.repair.corpus import GeneratedCase
    import time

    s, t = (owl.Class(owl.IRI("urn:runner:" + n)) for n in ("S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    problem = RepairInputV2(
        (owl.DisjointClasses(owl.CanonicalSet((s, t))),), (obj,), PolicyV2((s, t))
    )
    case = GeneratedCase(
        "corrupt",
        "parent",
        "overlap",
        "development",
        problem,
        (TeacherProbe("q", owl.SubClassOf(s, t), "desired", False),),
        (0,),
        13,
        False,
    )
    protocol = {
        "resources": {"case_wall_seconds": 20, "verification_seconds": 5, "case_rss_mb": 4096},
        "objective": {"edit_weights": {"mapping_deletion": 0.1}, "integer_scale": 1000},
    }
    settings = dict(
        arm="native_deletion", protocol=protocol, seconds=20, cpu_seconds=40, memory_mb=4096
    )
    payload = dict(
        schema=runner.SCHEMA,
        source_revision="fixture",
        rows=[
            dict(id=name, case=case_to_dict(case), settings=settings)
            for name in ("first", "second")
        ],
    )
    path = tmp_path / "manifest.json"
    write_artifact(path, payload)
    output = tmp_path / "results"
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 60))
    report = runner.run(path, output, stop=1)
    assert report["expected_rows"] == 2 and report["completed_rows"] == 1
    assert report["rows"][0]["status"] == "complete", report
    assert report["rows"][0]["result"]["logical_status"] == "VERIFIED_FEASIBLE"
    assert report["rows"][1]["status"] == "not_attempted"
    committed = (output / "first/completion.json").read_bytes()
    assert runner.run(path, output, stop=1) == report
    assert (output / "first/completion.json").read_bytes() == committed
    # The environment deadline is authoritative and reserves cleanup/receipts.
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 0.5))
    report = runner.run(path, output)
    assert report["completed_rows"] == 1 and not (output / "second/started.json").exists()
    payload["source_revision"] = "different-revision"
    write_artifact(path, payload)
    with pytest.raises(ValueError, match="incompatible corrective row resume"):
        runner.run(path, output)


def test_public_real_input_does_not_require_invented_generated_truth(tmp_path, monkeypatch):
    from tools.repair import corrective_study as runner
    from exact.repair.candidates import mapping_candidates
    from exact.repair.records import RevisionObjectV2, PolicyV2, RepairInputV2

    s, t = (owl.Class(owl.IRI("urn:public:" + n)) for n in ("S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    problem = RepairInputV2((), (obj,), PolicyV2((s, t)))
    record = dict(
        schema=runner.PUBLIC_INPUT_SCHEMA,
        input=problem.to_dict(),
        metadata=dict(
            case_id="captured-pair",
            family="conference",
            structural_parent="source-pair",
            split="development",
            control="observed_matcher_input",
        ),
    )
    protocol = {
        "resources": {"case_wall_seconds": 20, "verification_seconds": 5, "case_rss_mb": 4096},
        "objective": {"edit_weights": {"mapping_deletion": 0.1}, "integer_scale": 1000},
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("real inputs must not be converted to fabricated generated truth")

    monkeypatch.setattr(runner, "case_from_dict", forbidden)
    report = runner.evaluate_row(record, dict(arm="native_deletion", protocol=protocol), tmp_path)
    assert report["logical_status"] == "VERIFIED_FEASIBLE"
    assert report["semantic_benefit"] is None
    assert report["semantic_status"] == "independent_evaluation_not_provided"
    assert report["control"] == "observed_matcher_input"


def test_calibration_drafts_can_be_reprepared_with_fresh_native_reports_without_spending(tmp_path):
    from tools.repair import corrective_calibration as calibration

    path = calibration.prepare(tmp_path)
    first = json.loads(path.read_text())
    refreshed_path = calibration.prepare(tmp_path)
    assert refreshed_path == path
    second = json.loads(path.read_text())
    assert len(second["slots"]) == 32 and second["gold"] == first["gold"]
    assert second["implementation_source"] == first["implementation_source"]
    for row in second["slots"]:
        packet = read_record(bound(row["packet"]))
        assert packet.eligible
    assert not (tmp_path / "annotations/ledger").exists()
    assert not (tmp_path / "batches").exists()


@pytest.mark.parametrize("frozen_by", ("batches", "attempts", "hosted", "used", "results"))
def test_calibration_reprepare_rejects_before_any_write_or_native_work(
    tmp_path, monkeypatch, frozen_by
):
    from tools.repair import corrective_calibration as calibration

    directory = tmp_path / "annotations/calibration"
    write_artifact(
        directory / "manifest.json", dict(ledger_directory=str(tmp_path / "annotations/ledger"))
    )
    write_artifact(directory / "packets/0.json", dict(evidence="already frozen"))
    if frozen_by == "batches":
        (tmp_path / "batches").mkdir()
    elif frozen_by == "attempts":
        write_artifact(tmp_path / "ledger.json", dict(attempts={"job": {"state": "submitted"}}))
    elif frozen_by == "hosted":
        write_artifact(
            tmp_path / "annotations/ledger/phase-reservations.json",
            dict(reservations={"one": {"state": "reserved", "reserved_cost_usd": 0.05}}),
        )
    elif frozen_by == "used":
        write_artifact(directory / "used.json", dict(manifest_sha256="recorded"))
    else:
        write_artifact(directory / "results/report.json", dict(status="unavailable"))

    def contents():
        return {
            str(p.relative_to(tmp_path)): p.read_bytes() if p.is_file() else None
            for p in tmp_path.rglob("*")
        }

    before = contents()

    def forbidden(*args, **kwargs):
        raise AssertionError("freeze guard must precede native verification")

    monkeypatch.setattr(calibration, "_verified_plan", forbidden)
    with pytest.raises(ValueError, match="Calibration|Hosted"):
        calibration.prepare(tmp_path)
    assert contents() == before


def test_initialized_empty_hosted_ledger_allows_draft_refresh_without_changing_ledger(tmp_path):
    import sqlite3
    from tools.repair import corrective_calibration as calibration

    directory = tmp_path / "annotations/ledger"
    directory.mkdir(parents=True)
    database = directory / "requests.sqlite3"
    with sqlite3.connect(database) as db:
        for name in calibration._HOSTED_TABLES:
            db.execute(f'CREATE TABLE "{name}" (identity TEXT)')
    (directory / "requests.transaction.lock").touch()
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    calibration.prepare(tmp_path)
    calibration.prepare(tmp_path)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert not calibration._hosted_has_usage(directory)
    # Every recognized table is evidence of use, including pending reservations
    # and approvals; do not require an actual successful API response.
    for name in calibration._HOSTED_TABLES:
        with sqlite3.connect(database) as db:
            db.execute(f'INSERT INTO "{name}" VALUES (?)', ("consumed",))
        before_reject = {p.name: p.read_bytes() for p in directory.iterdir()}
        with pytest.raises(ValueError, match="Hosted annotation ledger has usage"):
            calibration.prepare(tmp_path)
        assert {p.name: p.read_bytes() for p in directory.iterdir()} == before_reject
        with sqlite3.connect(database) as db:
            db.execute(f'DELETE FROM "{name}"')
