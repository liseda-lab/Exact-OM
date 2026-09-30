"""Second review: exhaustive family receipts and finite SDD resource limits."""

import pyowl_core as owl
import pytest


def exhaustive_fixture(monkeypatch, all_forbidden):
    from exact.repair import retrieval
    from exact.repair.candidates import mapping_candidates
    from exact.repair.graph import EffectivePreparation
    from exact.repair.model import RepairModel
    from exact.repair.records import (
        FrozenMapping,
        PolicyV3,
        RepairInputV3,
        RevisionObjectV3,
    )

    a, b, s, t = (owl.Class(owl.IRI("urn:review2:" + name)) for name in ("A", "B", "S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV3("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    fixed = (owl.DisjointClasses((a, s)),)
    if all_forbidden:
        fixed += (owl.DisjointClasses((b, s)),)
    problem = RepairInputV3(fixed, (obj,), PolicyV3((a, b, s, t)))
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
    return problem, model, graph, b


@pytest.mark.parametrize("all_forbidden", [False, True])
def test_completed_enumeration_is_authoritative_for_family_coverage(monkeypatch, all_forbidden):
    from exact.repair.pipeline import freeze_neural_round

    problem, model, graph, b = exhaustive_fixture(monkeypatch, all_forbidden)
    frozen = freeze_neural_round(
        problem,
        model,
        graph=graph,
        proposal_arm="bounded_enumeration",
        draws_per_object=0,
        max_depth=1,
        max_constructors=1,
        representative_max_checks=0,
        candidate_cap=32,
    )
    report = frozen.proposal_reports[0]
    assert report["generation_status"] == "COMPLETE_DECLARED_ENUMERATION"
    assert (
        report["retained"]
        == report["enumerated_candidates"]
        == len(frozen.problem.objects[0].candidates)
    )
    protected = [
        row
        for row in report["protected_family_reports"]
        if row["action"] in {"specialise_subclass", "composite"}
    ]
    assert protected
    assert {row["status"] for row in protected} == {
        "empty_language" if all_forbidden else "retained"
    }
    if not all_forbidden:
        selected = {c.candidate_id: c for c in frozen.problem.objects[0].candidates}
        assert all(
            any(b in owl.signature(ax) for ax in selected[row["candidate_id"]].axioms)
            for row in protected
        )


@pytest.mark.parametrize(
    "options,error",
    [
        ({"max_enumerated_expressions": 1}, ValueError),
        ({"compile_seconds": 0.000001}, TimeoutError),
    ],
)
def test_incomplete_enumeration_is_not_reported_as_complete(monkeypatch, options, error):
    from exact.repair.pipeline import freeze_neural_round

    problem, model, graph, _ = exhaustive_fixture(monkeypatch, True)
    with pytest.raises(error, match="enumeration"):
        freeze_neural_round(
            problem,
            model,
            graph=graph,
            proposal_arm="bounded_enumeration",
            draws_per_object=0,
            max_depth=1,
            max_constructors=1,
            **options,
        )


def finite_three_choice():
    from exact.repair.circuit import ProposalEncoding

    return ProposalEncoding(
        (("choice", ("a", "b", "c")),),
        ((True, False, False), (False, True, False), (False, False, True)),
        ("a", "b", "c"),
        "review2-live-nodes",
    )


@pytest.mark.parametrize("warm", [False, True])
def test_finite_live_node_limit_is_enforced_on_cold_and_warm(tmp_path, warm):
    from exact.repair.compilation import _compile_persistent_worker

    encoding = finite_three_choice()
    if warm:
        first = _compile_persistent_worker(
            encoding, cache_directory=str(tmp_path), max_live_nodes=100
        )
        assert dict(first.manager.telemetry)["backend_measurements"]["manager_live_nodes"] > 1
    saved = {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
    with pytest.raises((ValueError, RuntimeError), match="live node"):
        _compile_persistent_worker(encoding, cache_directory=str(tmp_path), max_live_nodes=1)
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.json")} == saved


@pytest.mark.parametrize("layout", ["unsupported", "grouped"])
def test_finite_compiler_rejects_unsupported_vtree_instead_of_recording_it(tmp_path, layout):
    from exact.repair.compilation import _compile_persistent_worker

    with pytest.raises(ValueError, match="vtree"):
        _compile_persistent_worker(
            finite_three_choice(), cache_directory=str(tmp_path), vtree_type=layout
        )


@pytest.mark.parametrize(
    "limit", ["max_nodes", "max_live_nodes", "max_reachable_nodes", "max_elements"]
)
@pytest.mark.parametrize("warm", [False, True])
def test_finite_structural_limits_cannot_be_bypassed_by_cache(tmp_path, limit, warm):
    from exact.repair.compilation import _compile_persistent_worker

    encoding = finite_three_choice()
    if warm:
        _compile_persistent_worker(encoding, cache_directory=str(tmp_path))
    with pytest.raises((ValueError, RuntimeError), match="node|element"):
        _compile_persistent_worker(encoding, cache_directory=str(tmp_path), **{limit: 1})


@pytest.mark.parametrize("vtree", ["balanced", "right"])
@pytest.mark.parametrize("collect", [False, True])
def test_supported_finite_controls_have_matching_cold_warm_receipts(tmp_path, vtree, collect):
    from exact.repair.compilation import _compile_persistent_worker

    encoding = finite_three_choice()
    options = dict(
        cache_directory=str(tmp_path),
        vtree_type=vtree,
        collect=collect,
        max_live_nodes=100,
        max_reachable_nodes=100,
        max_elements=100,
    )
    cold = _compile_persistent_worker(encoding, **options)
    warm = _compile_persistent_worker(encoding, **options)
    assert cold.root.model_count() == warm.root.model_count() == 3
    for compiled, mode in ((cold, "cold_compile"), (warm, "warm_load")):
        native = dict(compiled.manager.telemetry)
        assert native["vtree"] == vtree and native["collect"] is collect
        receipt = dict(compiled.telemetry)["measurement_receipt"]
        assert receipt["cache_mode"] == mode
        assert receipt["resource_contract"] == "compiler-resource-contract/review-2"
        assert receipt["cold_receipt"]["measurements"]["manager_live_nodes"] <= 100


def test_finite_supervised_limit_failure_has_no_usable_circuit(tmp_path):
    from exact.repair.compilation import compile_bounded

    with pytest.raises(RuntimeError, match="live node") as failure:
        compile_bounded(
            finite_three_choice(), seconds=5, max_live_nodes=1, cache_directory=str(tmp_path)
        )
    receipt = failure.value.measurement_receipt
    assert receipt["cache_mode"] == "failed_attempt"
    assert receipt["supervised_call"]["status"] == "error"
    assert receipt["artifact_bytes"] is None
    assert not tuple(tmp_path.glob("*.json"))
