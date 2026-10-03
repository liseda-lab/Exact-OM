"""XR-2.1 finite distribution, lifetime, immutable cache and omission regressions."""

import json
import subprocess
import sys

import pyowl_core as owl
import pytest
import torch

from exact.repair.candidates import finite_expression_menu, mapping_candidates
from exact.repair.circuit import ConditionedMixture, FactoredConditionedMixture
from exact.repair.grammar import compile_families, compile_grammar, mapping_grammar
from exact.repair.records import RevisionObjectV2


def tiny():
    a, b, c = (owl.Class(owl.IRI("urn:v3:" + v)) for v in "ABC")
    pool = mapping_candidates("m", a, b, "=")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=a, target_entity=b)
    return mapping_grammar(
        obj,
        (a, b, c),
        max_depth=1,
        max_constructors=1,
        source_classes=(a, c),
        target_classes=(b, c),
    )


def assignments(encoding):
    expressions = finite_expression_menu(
        encoding.classes,
        encoding.properties,
        max_depth=encoding.max_depth,
        max_constructors=encoding.max_constructors,
    )
    result = {}
    for template in encoding.templates:
        for expression in (None,) if template.fixed else expressions:
            bits = encoding.assignment(template, expression)
            if encoding.accepts(bits):
                result[bits] = encoding.decode(bits)
    return result


def test_family_reduction_preserves_mass_aliases_and_gradients(tmp_path):
    encoding = tiny()
    compiled = compile_families(encoding, seconds=30, cache_directory=str(tmp_path))
    assert compiled.complete, compiled.telemetry
    assert all(
        f.circuit.encoding.variable_count < encoding.variable_count
        for f in compiled.families
        if f.circuit
    )
    torch.manual_seed(391)
    logits = torch.randn(2, encoding.variable_count, dtype=torch.float64, requires_grad=True)
    mixtures = torch.tensor([0.9, -0.4], dtype=torch.float64, requires_grad=True)
    factored = FactoredConditionedMixture(compiled, logits, mixtures)
    reference = ConditionedMixture(compile_grammar(encoding), logits, mixtures)
    assert torch.allclose(factored.component_log_normalizers, reference.component_log_normalizers)
    assert torch.allclose(factored.log_normalizer, reference.log_normalizer)
    pool = assignments(encoding)
    for bits, candidate in pool.items():
        assert factored.accepts(bits)
        assert torch.allclose(
            factored.candidate_log_probability(candidate),
            reference.candidate_log_probability(candidate),
        )
    candidate = list(pool.values())[-1]
    grads = torch.autograd.grad(
        factored.candidate_log_probability(candidate), (logits, mixtures), retain_graph=True
    )
    expected = torch.autograd.grad(
        reference.candidate_log_probability(candidate), (logits, mixtures)
    )
    assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(grads, expected))
    assert all(s.assignment in pool for s in factored.sample(50, seed=83))


@pytest.mark.parametrize("layout", ["right", "balanced", "grouped"])
def test_owned_collection_and_vtrees_keep_identical_support(layout):
    encoding = tiny()
    reference = ConditionedMixture(
        compile_grammar(encoding, vtree_type="right", collect=False),
        torch.zeros(1, encoding.variable_count),
    )
    compiled = compile_grammar(encoding, vtree_type=layout, collect=True)
    candidate = ConditionedMixture(compiled, torch.zeros(1, encoding.variable_count))
    assert torch.allclose(candidate.log_normalizer, reference.log_normalizer)
    assert all(candidate.accepts(bits) for bits in assignments(encoding))
    assert dict(compiled.telemetry)["phases"]
    assert compiled.root.model_count() == reference.circuit.root.model_count()


def test_persistent_artifact_cross_process_and_corruption(tmp_path):
    code = """
import json, sys
from exact.repair.circuit import ProposalEncoding
from exact.repair.compilation import compile_bounded,compilation_cache_info
value=ProposalEncoding((("x",("a","b")),),((True,False),(False,True)),("a","b"),sys.argv[2])
compiled=compile_bounded(value,seconds=10,cache_directory=sys.argv[1])
print(json.dumps(dict(compiled.telemetry)))
"""

    def run(identity="one"):
        output = subprocess.check_output(
            [sys.executable, "-c", code, str(tmp_path), identity], text=True
        )
        return json.loads(output)

    assert not run()["persistent_cache_hit"]
    assert run()["persistent_cache_hit"]
    assert not run("different-semantic-proof")["persistent_cache_hit"]
    for path in tmp_path.glob("*.json"):
        path.write_text("{truncated")
    assert not run()["persistent_cache_hit"]
    assert run()["persistent_cache_hit"]


def test_failed_families_retain_elementary_states(tmp_path):
    encoding = tiny()
    circuit = compile_families(encoding, seconds=0.0001, cache_directory=str(tmp_path))
    assert not circuit.complete
    dist = FactoredConditionedMixture(circuit, torch.zeros(1, encoding.variable_count))
    assert all(dist.accepts(encoding.assignment(t)) for t in encoding.templates if t.fixed)
    assert all(
        "keep" in c.action_tags or c.action_tags
        for c in (dist.candidate(s.assignment) for s in dist.sample(5))
    )


def test_immutable_range_disjointness_filters_active_expressions_only(tmp_path):
    from exact.repair.grammar import with_immutable_context
    from exact.repair.records import PolicyV2

    a, b, accepted, rejected = (
        owl.Class(owl.IRI("urn:context:" + v)) for v in ("A", "B", "Accepted", "Rejected")
    )
    role = owl.ObjectProperty(owl.IRI("urn:context:r"))
    fixed = (
        owl.ObjectPropertyRange(role, accepted),
        owl.DisjointClasses(owl.CanonicalSet((accepted, rejected))),
    )
    pool = mapping_candidates("m", a, b, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=a, target_entity=b)
    encoding = mapping_grammar(obj, (a, rejected), (role,), max_depth=2, max_constructors=2)
    policy = PolicyV2(monitored_classes=(a, b, accepted, rejected))
    argument = owl.ObjectSomeValuesFrom(role, rejected)
    template = next(t for t in encoding.templates if t.action == "specialise_subclass")
    bits = encoding.assignment(template, argument)
    assert encoding.accepts(bits)
    constrained = with_immutable_context(encoding, fixed, policy)
    assert not constrained.accepts(bits)
    assert constrained.context_proofs and all(
        set(p.asserted_support) <= set(fixed) for p in constrained.context_proofs
    )
    # When disjointness is editable it must not enter immutable B.
    editable = with_immutable_context(encoding, fixed[:1], policy)
    assert editable.accepts(bits)
    logits = torch.zeros(1, encoding.variable_count)
    dist = FactoredConditionedMixture(
        compile_families(constrained, seconds=30, cache_directory=str(tmp_path)), logits
    )
    assert not dist.accepts(bits)


def neural_fixture(retrieve, revision="v2"):
    from exact.repair.api import prepare_repair
    from exact.repair.graph import build_observable_graph
    from exact.repair.model import RepairModel
    from exact.repair.owl import snapshot_from_axioms
    from exact.repair.retrieval import retrieve_vocabulary

    a, c, b = (owl.Class(owl.IRI("urn:direct:" + v)) for v in ("A", "C", "B"))
    problem = prepare_repair(
        snapshot_from_axioms((owl.SubClassOf(a, c),)),
        snapshot_from_axioms((owl.Declaration(b),)),
        [{"Src": a.iri.value, "Tgt": b.iri.value, "Score": 0.8, "object_id": "m"}],
        retrieve=retrieve,
    )
    retrieval = retrieve_vocabulary(problem)
    graph = build_observable_graph(
        problem.objects,
        fixed_axioms=problem.fixed_axioms,
        source_axioms=problem.source_axioms,
        target_axioms=problem.target_axioms,
        evidence=retrieval.graph_evidence(problem.evidence),
        retrieved_symbols=retrieval.symbols,
        explanations=retrieval.explanations,
        feature_schema=f"exact-repair/observable-features/{revision}",
    )
    torch.manual_seed(14)
    model = RepairModel(
        graph.metadata, hidden_dim=8, heads=2, layers=0, dropout=0, revision=revision
    )
    return problem, model


def test_direct_and_prepared_retrieved_endpoints_materialize_same_ids(tmp_path):
    from exact.repair.pipeline import freeze_neural_round

    prepared, model = neural_fixture(True)
    direct, _ = neural_fixture(False)
    options = dict(
        proposal_arm="grammar_uniform",
        draws_per_object=0,
        candidate_cap=32,
        max_depth=1,
        max_constructors=1,
        compiler_cache_directory=str(tmp_path),
    )
    left = freeze_neural_round(prepared, model, **options)
    right = freeze_neural_round(direct, model, **options)

    def endpoints(round):
        return {
            c.candidate_id
            for c in round.problem.objects[0].candidates
            if "replace_endpoint" in c.action_tags
        }

    assert endpoints(left) and endpoints(left) == endpoints(right)
    assert left.objective.unary == right.objective.unary


@pytest.mark.parametrize("draws", [0, 30])
def test_final_bundle_removal_survives_every_producer(draws, tmp_path):
    from exact.repair.pipeline import freeze_neural_round

    problem, model = neural_fixture(False)
    options = dict(
        proposal_arm="grammar_uniform",
        draws_per_object=draws,
        candidate_cap=64,
        max_depth=1,
        max_constructors=1,
        compiler_cache_directory=str(tmp_path),
    )
    baseline = freeze_neural_round(problem, model, **options)
    target = next(
        c
        for c in baseline.problem.objects[0].candidates
        if "specialise_subclass" in c.action_tags and "keep" not in c.action_tags
    )
    result = freeze_neural_round(
        problem, model, final_candidate_removals={"m": (target.candidate_id,)}, **options
    )
    assert target.candidate_id not in {c.candidate_id for c in result.problem.objects[0].candidates}
    assert result.proposal_reports[0]["final_candidate_removals"] == (target.candidate_id,)
    # The designated answer remains outside the learned graph identity.
    assert baseline.graph_hash == result.graph_hash


def test_vocabulary_grammar_controls_and_progressive_epochs(tmp_path):
    from exact.repair.pipeline import freeze_neural_round, freeze_progressive_rounds

    problem, model = neural_fixture(False)
    options = dict(
        proposal_arm="grammar_uniform",
        candidate_cap=64,
        max_depth=1,
        max_constructors=1,
        compiler_cache_directory=str(tmp_path),
    )
    omitted = "urn:direct:C"
    frozen = freeze_neural_round(
        problem,
        model,
        draws_per_object=20,
        omitted_generation_symbols=(omitted,),
        enabled_actions=("keep", "delete", "replace_endpoint"),
        **options,
    )
    for obj in frozen.problem.objects:
        original = {str(e.iri.value) for a in obj.original_axioms for e in owl.signature(a)}
        assert all(
            not ({str(e.iri.value) for a in c.axioms for e in owl.signature(a)} - original)
            & {omitted}
            for c in obj.candidates
        )
        assert all(
            set(c.action_tags) <= {"keep", "delete", "replace_endpoint"} for c in obj.candidates
        )
    rounds = freeze_progressive_rounds(
        problem, model, ({"draws_per_object": 0}, {"draws_per_object": 8}), **options
    )
    old = {c.candidate_id for c in rounds[0].problem.objects[0].candidates}
    new = {c.candidate_id for c in rounds[1].problem.objects[0].candidates}
    assert old <= new
    assert rounds[0].problem.content_hash != rounds[1].problem.content_hash
    assert rounds[1].proposal_reports[0]["previous_pool_hash"] == rounds[0].problem.content_hash


def test_typed_capture_preserves_kinds_and_legacy_class_capture():
    from exact.repair.records import FrozenMapping
    from exact.repair.retrieval import ObjectMenus, RetrievalResult, _restore_capture

    role = owl.ObjectProperty(owl.IRI("urn:typed:r"))
    data = RetrievalResult(
        (ObjectMenus("m", endpoint_alternatives=(("source", role),)),),
        (),
        (),
        (),
        FrozenMapping({}),
    )
    restored = _restore_capture(data.capture())
    assert restored.menus[0].endpoint_alternatives == (("source", role),)
    legacy = dict(data.capture())
    row = dict(legacy["menus"][0])
    row["endpoint_alternatives"] = (("source", "urn:legacy:C"),)
    legacy["menus"] = (row,)
    assert isinstance(_restore_capture(legacy).menus[0].endpoint_alternatives[0][1], owl.Class)


def test_v3_frozen_risk_snapshot_and_objective_survive_supervised_transport(tmp_path):
    from exact.repair.pipeline import bounded_freeze_neural_round
    from exact.repair.records import ObjectiveV3, canonical_hash

    problem, model = neural_fixture(False, revision="v3")
    result = bounded_freeze_neural_round(
        problem,
        model,
        # This checks immutable transport, not generation speed. Native compiler
        # startup under concurrent Slurm work needs a separate finite fixture cap.
        seconds=120,
        draws_per_object=0,
        max_depth=0,
        max_constructors=0,
        candidate_cap=32,
        compiler_cache_directory=str(tmp_path),
        selected_other_assignment=(None,),
        pair_max_pairs=2,
        pair_max_factors=16,
    )
    assert result.status == "complete", result.detail
    frozen = result.value
    assert isinstance(frozen.objective, ObjectiveV3)
    assert frozen.objective.pool_hash == canonical_hash(frozen.problem.objects)
    assert frozen.objective.model_hash == frozen.model_hash
    assert frozen.objective.pair_selection_hash
    assert frozen.risk_scorer is not None
    before = frozen.risk_scorer((0,))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    assert before == frozen.risk_scorer((0,))


@pytest.mark.parametrize("limit", ["max_live_nodes", "max_reachable_nodes", "max_elements"])
def test_explicit_structural_resource_caps_are_enforced(limit):
    from exact.repair.grammar import CircuitBudgetExceeded

    with pytest.raises(CircuitBudgetExceeded):
        compile_grammar(tiny(), **{limit: 1})


def test_unknown_declared_circuit_limit_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported declared"):
        compile_families(tiny(), cache_directory=str(tmp_path), circuit_limits={"ignored_limit": 1})


def test_live_accelerator_models_require_checkpoint_transport(monkeypatch):
    from types import SimpleNamespace

    from exact.repair.pipeline import bounded_freeze_neural_round

    problem, model = neural_fixture(False)
    monkeypatch.setattr(
        model, "parameters", lambda: iter((SimpleNamespace(device=torch.device("cuda")),))
    )
    result = bounded_freeze_neural_round(problem, model)
    assert result.status == "unsupported" and "checkpoint artifact" in result.detail


def test_checkpoint_artifact_route_verifies_hash_inside_worker(tmp_path):
    import hashlib

    from exact.repair.pipeline import bounded_freeze_checkpoint

    problem, model = neural_fixture(False, revision="v3")
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_schema": "exact-repair/model/v3",
            "metadata": model.metadata,
            "config": model.config,
            "state_dict": model.state_dict(),
        },
        path,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    options = dict(
        seconds=30,
        draws_per_object=0,
        max_depth=0,
        max_constructors=0,
        compiler_cache_directory=str(tmp_path / "cache"),
    )
    result = bounded_freeze_checkpoint(problem, str(path), checkpoint_sha256=digest, **options)
    assert result.status == "complete", result.detail
    bad = bounded_freeze_checkpoint(problem, str(path), checkpoint_sha256="0" * 64, **options)
    assert bad.status == "error" and "integrity mismatch" in bad.detail


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA artifact source route requires an available GPU"
)
def test_cuda_owned_weights_use_cpu_artifact_for_supervised_inference(tmp_path):
    from exact.repair.pipeline import (
        bounded_freeze_checkpoint,
        bounded_freeze_neural_round,
    )

    problem, model = neural_fixture(False, revision="v3")
    model = model.cuda()
    assert bounded_freeze_neural_round(problem, model).status == "unsupported"
    # This small fixture stands in for the owning, already-supervised trainer.
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_schema": "exact-repair/model/v3",
            "metadata": model.metadata,
            "config": model.config,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        path,
    )
    result = bounded_freeze_checkpoint(
        problem,
        str(path),
        seconds=30,
        draws_per_object=0,
        max_depth=0,
        max_constructors=0,
        compiler_cache_directory=str(tmp_path / "cache"),
    )
    assert result.status == "complete", result.detail


def test_typed_generation_records_are_actual_pipeline_output_and_roundtrip(tmp_path):
    from exact.repair.pipeline import bounded_freeze_neural_round
    from exact.repair.records import GenerationReportV3, ProposalRecordV3, read_record

    problem, model = neural_fixture(False)
    result = bounded_freeze_neural_round(
        problem,
        model,
        seconds=30,
        draws_per_object=3,
        proposal_arm="grammar_uniform",
        max_depth=0,
        max_constructors=0,
        compiler_cache_directory=str(tmp_path),
    )
    assert result.status == "complete", result.detail
    frozen = result.value
    report = frozen.proposal_reports[0]
    assert isinstance(report, GenerationReportV3)
    assert frozen.problem.proposal_provenance == frozen.proposal_reports
    assert read_record(report.to_dict()) == report
    restored = read_record(frozen.problem.to_dict())
    assert isinstance(restored.proposal_provenance[0], GenerationReportV3)
    assert restored.content_hash == frozen.problem.content_hash
    assert dict(report)["valid_draws"] == 3
    sample = report["samples"][0]
    assert isinstance(sample, ProposalRecordV3)
    assert sample["candidate"].candidate_id == sample["candidate_id"]
    assert read_record(sample.to_dict()) == sample
    with pytest.raises(ValueError, match="status"):
        GenerationReportV3({**dict(report), "generation_status": "OPTIMAL"})
    with pytest.raises(ValueError, match="count accounting"):
        GenerationReportV3({**dict(report), "valid_draws": 2})
    with pytest.raises(ValueError, match="SHA256"):
        GenerationReportV3({**dict(report), "grammar_hash": "unbound"})
    with pytest.raises(ValueError, match="unknown"):
        GenerationReportV3({**dict(report), "invented_limit": 2})
    with pytest.raises(ValueError, match="log probability"):
        ProposalRecordV3({**dict(sample), "log_probability": 0.5})
    with pytest.raises(ValueError, match="Boolean"):
        ProposalRecordV3({**dict(sample), "assignment": (1, 0)})
