"""E3b matches the exact grammar distribution on declared tractable full supports."""

from dataclasses import replace
from collections import defaultdict
import time

import pyowl_core as owl
import pytest
import torch

from exact.repair.candidates import mapping_candidates
from exact.repair.records import RevisionObjectV2, PolicyV2, RepairInputV2
from exact.repair.grammar import mapping_grammar, with_immutable_context, compile_families
from exact.repair.circuit import FactoredConditionedMixture
from exact.repair.proposals import (
    enumerate_grammar,
    UnconditionedMixture,
    EnumeratedConditionedMixture,
)


def cls(name):
    return owl.Class(owl.IRI("urn:decoder:" + name))


def tiny_grammar():
    s, t, a = map(cls, ("S", "T", "A"))
    pool = mapping_candidates("m", s, t, "=")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    grammar = mapping_grammar(obj, (s, t, a), max_depth=1, max_constructors=1)
    fixed = (owl.DisjointClasses(owl.CanonicalSet((s, a))),)
    return with_immutable_context(grammar, fixed, PolicyV2((s, t, a)))


def test_weighted_decoder_matches_native_circuit_alias_activation_exclusion_mass(tmp_path):
    grammar = tiny_grammar()
    assert grammar.forbidden_assignments
    enumeration = enumerate_grammar(grammar)
    random = torch.Generator().manual_seed(97)
    logits = torch.randn((3, grammar.variable_count), generator=random, dtype=torch.float64)
    components = torch.tensor((-0.2, 0.3, 1.1), dtype=torch.float64)
    circuit = compile_families(grammar, seconds=15, cache_directory=str(tmp_path / "cache"))
    assert circuit.complete
    reference = FactoredConditionedMixture(circuit, logits, components)
    decoder = EnumeratedConditionedMixture(
        UnconditionedMixture(grammar, logits, components),
        enumeration,
        deadline=time.monotonic() + 5,
    )
    assert decoder.log_normalizer.item() == pytest.approx(
        reference.log_normalizer.item(), abs=1e-10
    )
    assert decoder.component_log_normalizers.tolist() == pytest.approx(
        reference.component_log_normalizers.tolist()
    )
    assert sum(
        float(decoder.log_probability(bits).exp()) for bits in decoder.assignments
    ) == pytest.approx(1)
    aliases = defaultdict(list)
    for candidate in enumeration.candidates:
        assert decoder.candidate_log_probability(candidate).item() == pytest.approx(
            reference.candidate_log_probability(candidate).item(), abs=1e-10
        )
        aliases[candidate.axioms].append(candidate.active_expressions)
    assert any(
        len(values) > 1 for values in aliases.values()
    )  # Same assertions, different activations.
    assert any(len(grammar.candidate_assignments(c)) > 1 for c in enumeration.candidates)
    for bits in decoder.assignments:
        assert reference.accepts(bits)
        assert decoder.log_probability(bits).item() == pytest.approx(
            reference.log_probability(bits).item(), abs=1e-10
        )
    assert all(not decoder.accepts(bits) for bits in grammar.forbidden_assignments)
    draws = decoder.sample(37, seed=17)
    assert draws == decoder.sample(37, seed=17)
    assert len(draws) == 37
    assert all(
        draw.log_probability
        == pytest.approx(
            float(decoder.candidate_log_probability(decoder.candidate(draw.assignment)))
        )
        for draw in draws
    )
    with pytest.raises(TimeoutError):
        EnumeratedConditionedMixture(
            UnconditionedMixture(grammar, logits, components),
            enumeration,
            deadline=time.monotonic() - 1,
        )


def test_actual_pipeline_decoder_reports_sampling_and_fails_on_incomplete_enumeration(tmp_path):
    from exact.repair.pipeline import freeze_neural_round
    from exact.repair.model import RepairModel
    from exact.repair.graph_schema import generic_graph_schema, declared_metadata

    s, t = map(cls, ("S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    problem = RepairInputV2((), (obj,), PolicyV2((s, t)))
    model = RepairModel(
        declared_metadata(generic_graph_schema()),
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0,
        pairwise=False,
    )
    common = dict(
        draws_per_object=19,
        candidate_cap=32,
        max_depth=1,
        max_constructors=1,
        compile_seconds=10,
        compiler_cache_directory=str(tmp_path / "cache"),
    )
    frozen = freeze_neural_round(problem, model, proposal_arm="matched_grammar_decoder", **common)
    report = frozen.proposal_reports[0]
    assert report["arm"] == "matched_grammar_decoder"
    assert report["generation_status"] == "SAMPLED"
    assert report["attempted_draws"] == report["valid_draws"] == 19
    assert report["decoder_scope"] == "tractable_completed_exhaustive_support"
    assert report["decoder_support_identity"] and report["circuit_hash"] is None
    assert report["circuit_nodes"] == 0 and report["proposal_setup_seconds"] > 0
    assert all(
        row["probability_semantics"] == "conditioned_bundle_probability"
        for row in report["samples"]
    )
    with pytest.raises(ValueError, match="expression"):
        freeze_neural_round(
            problem,
            model,
            proposal_arm="matched_grammar_decoder",
            max_enumerated_expressions=1,
            **common
        )


def test_both_decoders_stop_at_same_unique_target_and_historical_receipts_remain_valid(tmp_path):
    from exact.repair.pipeline import freeze_neural_round
    from exact.repair.model import RepairModel
    from exact.repair.graph_schema import generic_graph_schema, declared_metadata
    from exact.repair.records import GenerationReportV3

    s, t = map(cls, ("S", "T"))
    pool = mapping_candidates("m", s, t, "<")
    obj = RevisionObjectV2("m", "mapping", pool[0].axioms, pool, source_entity=s, target_entity=t)
    problem = RepairInputV2((), (obj,), PolicyV2((s, t)))
    model = RepairModel(
        declared_metadata(generic_graph_schema()),
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0,
        pairwise=False,
    )
    reports = []
    for arm in ("grammar_mixture", "matched_grammar_decoder"):
        frozen = freeze_neural_round(
            problem,
            model,
            proposal_arm=arm,
            draws_per_object=50,
            unique_candidate_target=2,
            candidate_cap=32,
            max_depth=1,
            max_constructors=1,
            compile_seconds=10,
            compiler_cache_directory=str(tmp_path / "cache"),
        )
        report = frozen.proposal_reports[0]
        assert report["unique_draws"] == report["unique_candidate_target"] == 2
        assert report["unique_target_reached"] and report["attempted_draws"] <= 50
        assert report["distribution_scope"] == "declared_language"
        reports.append(report)
    assert reports[0]["grammar_hash"] == reports[1]["grammar_hash"]
    assert reports[0]["language_hash"] == reports[1]["language_hash"]
    old = dict(reports[0])
    for key in (
        "contextual_expensive_calls",
        "contextual_cache_hits",
        "decoder_scope",
        "decoder_support_identity",
        "unique_candidate_target",
        "unique_target_reached",
    ):
        old.pop(key)
    GenerationReportV3(old)
