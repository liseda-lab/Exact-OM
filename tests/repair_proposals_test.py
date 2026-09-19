"""Matched enumeration, compiler-free rejection, and coverage-label separation."""

from itertools import product
from types import SimpleNamespace

import pyowl_core as owl
import pytest
import torch

from exact.repair.api import prepare_repair
from exact.repair.candidates import mapping_candidates
from exact.repair.circuit import ConditionedMixture
from exact.repair.grammar import compile_grammar, mapping_grammar
from exact.repair.owl import snapshot_from_axioms
from exact.repair.proposals import (
    UnconditionedMixture,
    compare_proposals,
    enumerate_grammar_candidates,
)
from exact.repair.records import FrozenMapping, RevisionObjectV2, make_objective
from exact.repair.workers import CallResult


def cls(name):
    return owl.Class(owl.IRI(f"urn:proposal:{name}"))


def encoding(depth=0, constructors=0):
    candidates = mapping_candidates("m", cls("S"), cls("T"), "<")
    obj = RevisionObjectV2(
        "m",
        "mapping",
        candidates[0].axioms,
        candidates,
        source_entity=cls("S"),
        target_entity=cls("T"),
    )
    return mapping_grammar(obj, (cls("S"),), max_depth=depth, max_constructors=constructors)


def test_enumeration_matches_exact_canonical_language_without_compiler(monkeypatch):
    import exact.repair.grammar as grammar

    language = encoding()
    monkeypatch.setattr(
        grammar,
        "compile_grammar",
        lambda *_args, **_kwargs: pytest.fail("enumeration must not compile"),
    )
    expected = {
        language.decode(bits).candidate_id
        for bits in product((False, True), repeat=language.variable_count)
        if language.accepts(bits)
    }
    actual = enumerate_grammar_candidates(language)
    assert {candidate.candidate_id for candidate in actual} == expected
    assert len(actual) == len(expected)


def test_enumeration_respects_side_menus_activation_bounds_and_caps():
    language = encoding(1, 1)
    language = mapping_grammar(
        language.revision,
        (cls("S"), cls("T")),
        max_depth=1,
        max_constructors=1,
        source_classes=(cls("S"),),
        target_classes=(cls("T"),),
    )
    dist = ConditionedMixture(
        compile_grammar(language), torch.zeros((1, language.variable_count), dtype=torch.float64)
    )
    pool = enumerate_grammar_candidates(language)
    assert sum(
        float(dist.candidate_log_probability(candidate).exp()) for candidate in pool
    ) == pytest.approx(1, abs=1e-6)
    assert all(language.candidate_assignments(candidate) for candidate in pool)
    with pytest.raises(ValueError, match="max_expressions"):
        enumerate_grammar_candidates(language, max_expressions=1)
    with pytest.raises(TimeoutError):
        enumerate_grammar_candidates(language, deadline=0.0)


def test_unconditioned_mass_matches_brute_force_and_sums_aliases():
    language = encoding()
    torch.manual_seed(1)
    logits = torch.randn((2, language.variable_count), dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([-0.4, 0.8], dtype=torch.float64, requires_grad=True)
    raw = UnconditionedMixture(language, logits, weights)
    accepted = []
    for bits in product((False, True), repeat=language.variable_count):
        if not language.accepts(bits):
            assert torch.isneginf(raw.log_probability(bits))
            continue
        selected = torch.tensor(bits)
        expected = (
            torch.where(selected[None, :], logits.sigmoid(), (-logits).sigmoid()).prod(dim=1)
            * weights.softmax(0)
        ).sum()
        assert torch.allclose(raw.log_probability(bits).exp(), expected)
        accepted.append(raw.log_probability(bits).exp())
    mass = torch.stack(accepted).sum()
    bundle_mass = torch.stack(
        [raw.candidate_log_probability(c).exp() for c in enumerate_grammar_candidates(language)]
    ).sum()
    assert torch.allclose(bundle_mass, mass)
    assert 0 < float(mass.detach()) < 1
    bundle_mass.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(weights.grad).all()
    with pytest.raises(ValueError, match="outside"):
        raw.candidate((False,) * language.variable_count)


def test_comparison_preserves_failed_denominator_and_never_exposes_coverage_labels(monkeypatch):
    import exact.repair.pipeline as pipeline
    import exact.repair.proposals as proposals
    from exact.repair.retrieval import retrieve_vocabulary

    empty = snapshot_from_axioms(())
    problem = prepare_repair(
        empty, empty, [{"Src": "urn:proposal:S", "Tgt": "urn:proposal:T", "object_id": "m"}]
    )
    retrieval = retrieve_vocabulary(problem)
    monkeypatch.setattr(
        proposals, "bounded_call", lambda *_args, **_kwargs: CallResult("complete", retrieval)
    )
    monkeypatch.setattr(pipeline, "model_digest", lambda _model: "frozen-model")
    seen = []

    def freeze(observed, _model, **options):
        seen.append((observed, options))
        if options["proposal_arm"] == "rejection":
            return CallResult("timeout", detail="bounded test")
        return CallResult(
            "complete",
            SimpleNamespace(
                problem=observed,
                objective=make_objective(observed.objects),
                graph_hash="graph",
                model_hash="frozen-model",
                proposal_reports=(FrozenMapping({"samples": ({"candidate_id": "x"},)}),),
            ),
        )

    monkeypatch.setattr(pipeline, "bounded_freeze_neural_round", freeze)
    keep = problem.objects[0].candidates[0].candidate_id
    result = compare_proposals(
        problem,
        object(),
        arms=("grammar_product", "rejection"),
        useful_candidate_ids={"m": (keep, "private-target")},
        required_symbols={"m": (cls("S"), cls("Hidden"))},
    )
    assert result["scheduled"] == result["recorded"] == 2
    assert result["vocabulary_coverage"]["recall"] == 0.5
    assert result["rows"][0]["useful_candidate_coverage"]["recall"] == 0.5
    assert result["rows"][1]["status"] == "timeout"
    assert "private-target" not in repr(seen) and "Hidden" not in repr(seen)
    assert seen[0][0] == seen[1][0]
    import json

    json.dumps(result, allow_nan=False)


def test_teacher_utility_is_unknown_outside_its_complete_finite_inventory():
    from dataclasses import replace

    from exact.repair.candidates import make_candidate
    from exact.repair.learning import RepairLabel, TeacherCache
    from exact.repair.proposals import _teacher_evaluation

    language = encoding()
    obj = replace(language.revision, candidates=(language.revision.candidates[0],))
    from exact.repair.records import PolicyV2, RepairInputV2

    problem = RepairInputV2((), (obj,), PolicyV2())
    cache = TeacherCache((1,), (RepairLabel((0,), True, 9.0, 1.0),), True, "complete", (), 0.0)
    known = SimpleNamespace(assignment=(0,), selected=(obj.candidates[0],))
    report = _teacher_evaluation(problem, known, cache)
    assert report["labelled_selected_utility"] == 8.0 and report["exact_regret"] == 0.0
    unknown = SimpleNamespace(assignment=(0,), selected=(make_candidate("m", (), ("delete",)),))
    report = _teacher_evaluation(problem, unknown, cache)
    assert report["labelled_selected_utility"] is None and report["exact_regret"] is None
    assert report["exact_teacher_optimum"] == 8.0
