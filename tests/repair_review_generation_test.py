"""Corrective REV-07/08/13/14 regressions; small fixtures, no experiments."""

import dataclasses
from pathlib import Path

import pyowl_core as owl
import pytest

from exact.repair.candidates import mapping_candidates
from exact.repair.grammar import mapping_grammar, with_immutable_context
from exact.repair.records import PolicyV2, RevisionObjectV2


def contextual_encoding(all_forbidden=False):
    a, b, s, t = (owl.Class(owl.IRI("urn:review:" + name)) for name in ("A", "B", "S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    encoding = mapping_grammar(obj, (a, b), source_classes=(a, b), max_depth=1, max_constructors=1)
    fixed = (owl.DisjointClasses((a, s)),)
    if all_forbidden:
        fixed += (owl.DisjointClasses((b, s)),)
    return with_immutable_context(encoding, fixed, PolicyV2((a, b, s, t))), b


def test_review07_first_forbidden_expression_does_not_close_family():
    encoding, b = contextual_encoding()
    representatives = encoding.representatives()
    special = [c for c in representatives if "specialise_subclass" in c.action_tags]
    assert special and all(encoding.candidate_assignments(c) for c in special)
    assert any(b in owl.signature(ax) for c in special for ax in c.axioms)


@pytest.mark.parametrize(
    "later",
    [
        {"enabled_actions": ("keep", "delete")},
        {"omitted_generation_symbols": ("urn:review:C",)},
        {"contextual_filtering": False},
        {"max_context_checks": 1},
    ],
)
def test_review08_reject_nonnested_schedule_before_first_stage(monkeypatch, later):
    from exact.repair import pipeline
    from tests.repair_generation_v3_test import neural_fixture

    problem, model = neural_fixture(False)

    def unexpected(*args, **kwargs):
        raise AssertionError("schedule was not validated before starting generation")

    monkeypatch.setattr(pipeline, "freeze_neural_round", unexpected)
    with pytest.raises(ValueError, match="nested|context"):
        pipeline.freeze_progressive_rounds(problem, model, ({}, later))


def test_review13_entry_disappears_between_enumeration_and_metadata(tmp_path, monkeypatch):
    from exact.repair.circuit import ProposalEncoding
    from exact.repair.compilation import _compile_persistent_worker

    victim = tmp_path / "prior.json"
    victim.write_text("old cache bytes")
    original = Path.glob
    deleted = []

    def disappearing(path, pattern):
        found = tuple(original(path, pattern))
        if path == tmp_path and pattern == "*.json" and victim in found:
            victim.unlink(missing_ok=True)
            deleted.append(True)
        return iter(found)

    monkeypatch.setattr(Path, "glob", disappearing)
    encoding = ProposalEncoding((("x", ("a", "b")),), ((True, False), (False, True)), ("a", "b"))
    result = _compile_persistent_worker(encoding, cache_directory=str(tmp_path))
    assert deleted and result.root.model_count() == 2


def test_review14_cold_and_warm_receipts_are_complete(tmp_path):
    from exact.repair.circuit import ProposalEncoding
    from exact.repair.compilation import _compile_persistent_worker

    encoding = ProposalEncoding((("x", ("a", "b")),), ((True, False), (False, True)), ("a", "b"))
    cold = _compile_persistent_worker(encoding, cache_directory=str(tmp_path), seconds=5)
    warm = _compile_persistent_worker(encoding, cache_directory=str(tmp_path), seconds=7)
    first, second = (dict(c.telemetry)["measurement_receipt"] for c in (cold, warm))
    assert first["cache_mode"] == "cold_compile" and second["cache_mode"] == "warm_load"
    assert first["structural_identity"] == second["structural_identity"]
    assert second["cold_receipt"] == first["cold_receipt"]
    assert second["cold_receipt"]["limits"]["seconds"] == 5
    assert second["admission_limits"]["seconds"] == 7


def test_review07_empty_resource_limited_and_editable_are_distinct(tmp_path):
    from exact.repair.grammar import compile_families, protected_representatives

    encoding, _ = contextual_encoding(all_forbidden=True)
    circuit = compile_families(encoding, seconds=10, cache_directory=str(tmp_path))
    _, rows = protected_representatives(encoding, circuit)
    assert {
        row["status"] for row in rows if row["action"] in {"specialise_subclass", "composite"}
    } == {"empty_language"}
    limited = compile_families(encoding, seconds=0.000001, cache_directory=str(tmp_path))
    controls, rows = protected_representatives(encoding, limited, max_checks=0)
    assert {
        row["status"] for row in rows if row["action"] in {"specialise_subclass", "composite"}
    } == {"compile_timeout"}
    assert {"keep", "delete"} <= {tag for c in controls for tag in c.action_tags}
    # Removing the fixed premise models an editable disjointness support.
    editable = dataclasses.replace(encoding, forbidden_assignments=(), context_proofs=())
    _, rows = protected_representatives(editable)
    assert all(row["status"] == "retained" for row in rows)
    _, exhausted = protected_representatives(editable, max_checks=0)
    assert any(row["status"] == "search_exhausted" for row in exhausted)


def test_review07_family_aliases_share_one_candidate_and_keep_origins():
    from exact.repair.grammar import protected_representatives

    encoding, _ = contextual_encoding()
    template = next(t for t in encoding.templates if t.action == "specialise_subclass")
    encoding = dataclasses.replace(
        encoding, templates=(template, dataclasses.replace(template, name="alias"))
    )
    # Re-encode contextual bans against the two-template reference before lookup.
    encoding = dataclasses.replace(encoding, forbidden_assignments=())
    candidates, rows = protected_representatives(encoding)
    assert len(candidates) == 1 and len(rows) == 2
    assert len({row["candidate_id"] for row in rows}) == 1
    assert {value for key, value in candidates[0].provenance if key == "grammar_template"} == {
        template.name,
        "alias",
    }


def _publish_concurrently(directory, tag, capacity, barrier, queue):
    from exact.repair import compilation
    from exact.repair.circuit import ProposalEncoding

    original = compilation._serialize_compile

    def synchronize(*args, **kwargs):
        result = original(*args, **kwargs)
        barrier.wait(timeout=15)
        return result

    compilation._serialize_compile = synchronize
    try:
        encoding = ProposalEncoding(
            (("x", ("a", "b")),), ((True, False), (False, True)), ("a", "b"), tag
        )
        result = compilation._compile_persistent_worker(
            encoding, cache_directory=directory, cache_bytes=capacity
        )
        queue.put(
            (
                "complete",
                result.root.model_count(),
                dict(result.telemetry)["measurement_receipt"]["cache_publication"],
            )
        )
    except Exception as error:
        queue.put(("error", str(error)))


def test_review13_independent_writers_publish_and_evict_concurrently(tmp_path):
    import multiprocessing

    from exact.repair.circuit import ProposalEncoding
    from exact.repair.compilation import _compile_persistent_worker

    encoding = ProposalEncoding(
        (("x", ("a", "b")),), ((True, False), (False, True)), ("a", "b"), "sizing"
    )
    _compile_persistent_worker(encoding, cache_directory=str(tmp_path))
    capacity = max(p.stat().st_size for p in tmp_path.glob("*.json")) + 1024
    for path in tmp_path.glob("*.json"):
        path.unlink()
    context = multiprocessing.get_context("spawn")
    barrier, queue = context.Barrier(2), context.Queue()
    children = [
        context.Process(
            target=_publish_concurrently, args=(str(tmp_path), tag, capacity, barrier, queue)
        )
        for tag in ("one", "two")
    ]
    try:
        for child in children:
            child.start()
        results = [queue.get(timeout=25) for _ in children]
        assert all(
            row[0] == "complete" and row[1] == 2 and row[2]["status"] == "complete"
            for row in results
        ), results
        assert sum(row[2]["evicted_entries"] for row in results) >= 1
        assert sum(p.stat().st_size for p in tmp_path.glob("*.json")) <= capacity
    finally:
        for child in children:
            child.join(5)
            if child.is_alive():
                child.kill()
                child.join()
        queue.close()


def test_review14_separate_worker_receipts_export_and_cache_policy(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    from exact.repair import study
    from tests.repair_study_test import frozen_case, synchronous, verifier

    code = """
import json, sys
from exact.repair.circuit import ProposalEncoding
from exact.repair.compilation import _compile_persistent_worker
encoding=ProposalEncoding((("x",("a","b")),),((True,False),(False,True)),("a","b"))
circuit=_compile_persistent_worker(encoding,cache_directory=sys.argv[1],seconds=float(sys.argv[2]))
print(json.dumps(dict(circuit.telemetry)))
"""
    cold, warm = [
        json.loads(
            subprocess.check_output(
                [sys.executable, "-c", code, str(tmp_path / "cache"), str(seconds)], text=True
            )
        )
        for seconds in (5, 8)
    ]
    first, second = cold["measurement_receipt"], warm["measurement_receipt"]
    assert (first["cache_mode"], second["cache_mode"]) == ("cold_compile", "warm_load")
    assert second["cold_receipt"] == first["cold_receipt"]
    assert second["cold_receipt"]["limits"]["seconds"] == 5
    assert second["admission_limits"]["seconds"] == 8
    assert first["cold_receipt"]["measurements"]["minimization_seconds"] is None
    assert first["cold_receipt"]["measurements"]["peak_manager_allocated_nodes"] is None
    assert {row["phase"] for row in first["phases"]} >= {
        "native_compile",
        "dag_serialization",
        "native_save",
        "evaluation_restore",
        "cache_publication_maintenance",
    }
    assert all(
        row["cpu_seconds"] >= 0 and row["peak_process_rss_bytes"] > 0 for row in first["phases"]
    )
    case = frozen_case()
    case = dataclasses.replace(
        case,
        problem=dataclasses.replace(
            case.problem,
            proposal_provenance=(
                {"object_id": "mapping", "compiler_telemetry": tuple(warm.items())},
            ),
        ),
    )
    exported = study.compiler_accounting(case.problem, policy="warm_required")
    assert exported["receipts"][0]["receipt"]["cold_receipt"] == second["cold_receipt"]
    assert exported["charged_wall_seconds"] == second["total"]["wall_seconds"]
    with pytest.raises(ValueError, match="cold/warm"):
        study.compiler_accounting(case.problem, policy="cold_required")
    monkeypatch.setattr(study, "bounded_call", synchronous)
    monkeypatch.setattr(study, "runtime_manifest", lambda: {"fixture": "review14"})
    import exact.repair.kernel as kernel

    monkeypatch.setattr(kernel, "bounded_call", synchronous)
    destination = tmp_path / "study"
    study.run_study(
        (case,),
        destination,
        arms=(study.StudyArmV2("none", "no_repair"),),
        verifier=verifier,
        compiler_cache_policy="warm_required",
    )
    saved = json.loads((destination / "results.json").read_text())["rows"][0][
        "compiler_preparation"
    ]
    assert saved["charged_wall_seconds"] == exported["charged_wall_seconds"]
    assert saved["receipts"][0]["receipt"]["cold_receipt"] == second["cold_receipt"]


def test_review07_zero_draw_pool_protection_caps_and_final_intervention(tmp_path, monkeypatch):
    from exact.repair import retrieval
    from exact.repair.graph import EffectivePreparation
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import freeze_neural_round
    from exact.repair.records import FrozenMapping, RepairInputV2, promote_input_v3

    encoding, b = contextual_encoding()
    a = encoding.classes[0]
    s, t = encoding.revision.source_entity, encoding.revision.target_entity
    fixed = (owl.DisjointClasses((a, s)),)
    problem = promote_input_v3(RepairInputV2(fixed, (encoding.revision,), PolicyV2((a, b, s, t))))
    retrieved = retrieval.RetrievalResult(
        (retrieval.ObjectMenus("m", source_classes=(a, b)),),
        (a, b, s, t),
        (),
        (),
        FrozenMapping({}),
    )
    monkeypatch.setattr(retrieval, "retrieve_vocabulary", lambda *args, **kwargs: retrieved)
    graph = EffectivePreparation(
        4096, 32768, 64, 128, pair_max_pairs=None, pair_max_factors=None
    ).graph(problem, retrieved)
    model = RepairModel(
        graph.metadata, hidden_dim=8, heads=2, layers=0, dropout=0, revision="v3", plan_risk=False
    )
    options = dict(
        graph=graph,
        draws_per_object=0,
        proposal_arm="grammar_uniform",
        max_depth=1,
        max_constructors=1,
        compiler_cache_directory=str(tmp_path),
        candidate_cap=16,
    )
    frozen = freeze_neural_round(problem, model, **options)
    report = frozen.proposal_reports[0]
    protected = [
        row
        for row in report["protected_family_reports"]
        if row["action"] in {"specialise_subclass", "composite"}
    ]
    assert protected and all(row["status"] == "retained" for row in protected)
    pool = {candidate.candidate_id: candidate for candidate in frozen.problem.objects[0].candidates}
    assert all(
        any(b in owl.signature(ax) for ax in pool[row["candidate_id"]].axioms) for row in protected
    )
    with pytest.raises(ValueError, match="candidate budget"):
        freeze_neural_round(problem, model, **{**options, "candidate_cap": 1})
    removed = protected[0]["candidate_id"]
    intervened = freeze_neural_round(
        problem,
        model,
        preserved_candidates={"m": tuple(pool.values())},
        final_candidate_removals={"m": (removed,)},
        **options,
    )
    assert removed not in {c.candidate_id for c in intervened.problem.objects[0].candidates}
    assert any(
        row["status"] == "removed_by_intervention"
        for row in intervened.proposal_reports[0]["protected_family_reports"]
    )
    with pytest.raises(ValueError, match="nested generation language"):
        freeze_neural_round(
            problem,
            model,
            preserved_candidates={"m": tuple(pool.values())},
            omitted_generation_symbols=(b.iri.value,),
            **options,
        )
    exhausted = freeze_neural_round(problem, model, representative_max_checks=0, **options)
    assert exhausted.proposal_reports[0]["generation_status"] == "PARTIAL_RESOURCE_LIMIT"


def test_review13_metadata_scan_deadline_returns_explicit_resource_status(tmp_path, monkeypatch):
    from exact.repair import compilation

    for index in range(3):
        (tmp_path / f"prior-{index}.json").write_text("cache-entry")
    observed = []
    original = Path.glob

    def tracked(path, pattern):
        for entry in original(path, pattern):
            observed.append(entry.name)
            yield entry

    monkeypatch.setattr(Path, "glob", tracked)
    monkeypatch.setattr(compilation, "monotonic", lambda: 2.0 if observed else 0.0)
    receipt = compilation._publish_cache(tmp_path, "new", b"valid", 100000, 1.0)
    assert receipt["status"] == "maintenance_timeout"
    assert receipt["detail"] == "capacity metadata scan incomplete"
    assert receipt["capacity_bytes"] == 100000
    assert len(observed) == 1
    assert (tmp_path / "new.json").read_bytes() == b"valid"
