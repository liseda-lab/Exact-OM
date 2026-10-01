"""Protected coverage must be witnessed in the current bounded grammar."""

from dataclasses import replace

import pyowl_core as owl
import pytest

from exact.repair.candidates import finite_expression_menu, mapping_candidates
from exact.repair.grammar import mapping_grammar
from exact.repair.proposals import enumerate_grammar_candidates
from exact.repair.records import (
    FrozenMapping,
    PolicyV3,
    RepairInputV3,
    RevisionObjectV3,
)


def current_family_members(encoding):
    """Independent finite oracle: never infer current membership from provenance."""
    expressions = finite_expression_menu(
        encoding.classes,
        encoding.properties,
        max_depth=encoding.max_depth,
        max_constructors=encoding.max_constructors,
    )
    result = {}
    for template in encoding.templates:
        members = set()
        for expression in (None,) if template.fixed is not None else expressions:
            assignment = encoding.assignment(template, expression)
            if encoding.accepts(assignment):
                members.add(encoding.decode(assignment).candidate_id)
        result[template.name] = members
    return result


def fixture(monkeypatch, *, reuse=True, compatible=False, poisoned=False):
    from exact.repair import retrieval
    from exact.repair.graph import EffectivePreparation
    from exact.repair.model import RepairModel

    s, t, a = (owl.Class(owl.IRI("urn:test:" + name)) for name in ("S", "T", "A"))
    elementary = mapping_candidates("m", s, t, "=")
    obj = RevisionObjectV3(
        "m",
        "mapping",
        next(c.axioms for c in elementary if "keep" in c.action_tags),
        elementary,
        source_entity=s,
        target_entity=t,
    )
    historical = mapping_grammar(obj, (s, t), max_depth=0, max_constructors=0)
    if reuse:
        obj = replace(obj, candidates=enumerate_grammar_candidates(historical))
    if poisoned:
        tags = tuple(
            ("grammar_template", template.name)
            for template in historical.templates
            if template.fixed is None
        )
        obj = replace(
            obj,
            candidates=tuple(
                (
                    replace(c, provenance=tuple(sorted(set(c.provenance + tags))))
                    if "keep" in c.action_tags
                    else c
                )
                for c in obj.candidates
            ),
        )
    menu_classes = (s, t) if compatible else (a,)
    menu = retrieval.ObjectMenus("m", source_classes=menu_classes, target_classes=menu_classes)
    retrieved = retrieval.RetrievalResult((menu,), (s, t, a), (), (), FrozenMapping({}))
    problem = RepairInputV3((), (obj,), PolicyV3((s, t, a)))
    monkeypatch.setattr(retrieval, "retrieve_vocabulary", lambda *args, **kwargs: retrieved)
    graph = EffectivePreparation(
        4096, 32768, 64, 128, pair_max_pairs=None, pair_max_factors=None
    ).graph(problem, retrieved)
    model = RepairModel(
        graph.metadata,
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0,
        revision="v3",
        plan_risk=False,
    )
    encoding = mapping_grammar(
        obj,
        menu.classes,
        max_depth=0,
        max_constructors=0,
        source_classes=menu.source_classes,
        target_classes=menu.target_classes,
    )
    return problem, model, graph, encoding


def freeze(problem, model, graph, cap):
    from exact.repair.pipeline import freeze_neural_round

    return freeze_neural_round(
        problem,
        model,
        graph=graph,
        proposal_arm="bounded_enumeration",
        draws_per_object=0,
        max_depth=0,
        max_constructors=0,
        candidate_cap=cap,
        representative_max_checks=0,
    )


def assert_current_coverage(frozen, encoding):
    members = current_family_members(encoding)
    final = {candidate.candidate_id for candidate in frozen.problem.objects[0].candidates}
    report = frozen.proposal_reports[0]
    assert report["generation_status"] == "COMPLETE_DECLARED_ENUMERATION"
    for row in report["protected_family_reports"]:
        expected = members[row["template"]]
        if expected:
            assert row["status"] == "retained"
            assert row["candidate_id"] in expected, (
                "Historical tags cannot witness this current family",
                row,
                expected,
            )
            assert row["candidate_id"] in final
        else:
            assert row["status"] == "empty_language"
            assert row["candidate_id"] is None
    return report


@pytest.mark.parametrize("reuse", [False, True])
def test_six_candidate_budget_cannot_claim_current_family_coverage(monkeypatch, reuse):
    problem, model, graph, encoding = fixture(monkeypatch, reuse=reuse)
    members = current_family_members(encoding)
    required = set().union(*members.values())
    # Every nonempty family is a singleton here: four elementary controls and
    # four current A-based necessary-condition bundles require eight slots.
    assert all(len(values) <= 1 for values in members.values())
    assert len(required) == 8
    try:
        frozen = freeze(problem, model, graph, 6)
    except ValueError as error:
        assert "candidate budget" in str(error) and "cap 6" in str(error)
        return
    final = {c.candidate_id for c in frozen.problem.objects[0].candidates}
    missing = [name for name, values in members.items() if values and not values & final]
    pytest.fail(
        f"Six-slot freeze accepted {len(final)} candidates with "
        f"status {frozen.proposal_reports[0]['generation_status']}; "
        f"current families without retained members: {missing}"
    )


def test_reused_inventory_receipts_name_actual_current_family_members(monkeypatch):
    problem, model, graph, encoding = fixture(monkeypatch)
    frozen = freeze(problem, model, graph, 8)
    assert len(frozen.problem.objects[0].candidates) == 8
    assert_current_coverage(frozen, encoding)
    before = next(c for c in problem.objects[0].candidates if "keep" in c.action_tags)
    after = next(
        c for c in frozen.problem.objects[0].candidates if c.candidate_id == before.candidate_id
    )
    assert any(key == "grammar_template" for key, _ in before.provenance)
    assert set(before.provenance) <= set(after.provenance)


def test_compatible_reuse_keeps_valid_family_aliases_and_history(monkeypatch):
    problem, model, graph, encoding = fixture(monkeypatch, compatible=True)
    frozen = freeze(problem, model, graph, 16)
    assert_current_coverage(frozen, encoding)
    keep = next(c for c in problem.objects[0].candidates if "keep" in c.action_tags)
    aliases = [
        name for name, ids in current_family_members(encoding).items() if keep.candidate_id in ids
    ]
    assert len(aliases) >= 3  # Fixed keep and two genuine current grammar aliases.
    final_keep = next(
        c for c in frozen.problem.objects[0].candidates if c.candidate_id == keep.candidate_id
    )
    assert set(keep.provenance) <= set(final_keep.provenance)
    ids = [c.candidate_id for c in frozen.problem.objects[0].candidates]
    assert len(ids) == len(set(ids))


def test_poisoned_historical_tags_cannot_supply_current_family_witnesses(monkeypatch):
    problem, model, graph, encoding = fixture(monkeypatch, reuse=False, poisoned=True)
    frozen = freeze(problem, model, graph, 8)
    assert_current_coverage(frozen, encoding)
    keep = next(c for c in problem.objects[0].candidates if "keep" in c.action_tags)
    final_keep = next(
        c for c in frozen.problem.objects[0].candidates if c.candidate_id == keep.candidate_id
    )
    assert set(keep.provenance) <= set(final_keep.provenance)


def receipt_encoding():
    s, t = (owl.Class(owl.IRI("urn:receipt:" + name)) for name in ("S", "T"))
    candidates = mapping_candidates("m", s, t, "=")
    obj = RevisionObjectV3(
        "m",
        "mapping",
        next(c.axioms for c in candidates if "keep" in c.action_tags),
        candidates,
        source_entity=s,
        target_entity=t,
    )
    return mapping_grammar(obj, (s, t), max_depth=0, max_constructors=0)


def test_enumeration_receipt_rejects_another_current_vocabulary():
    from exact.repair.grammar import protected_representatives
    from exact.repair.proposals import enumerate_grammar

    encoding = receipt_encoding()
    receipt = enumerate_grammar(encoding)
    changed = replace(encoding, classes=(owl.Class(owl.IRI("urn:receipt:A")),))
    assert receipt.language_hash != changed.content_hash
    with pytest.raises(ValueError, match="current grammar"):
        protected_representatives(changed, enumeration=receipt)


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_enumeration_receipt_requires_exact_template_coverage(mutation):
    from exact.repair.grammar import protected_representatives
    from exact.repair.proposals import enumerate_grammar

    encoding = receipt_encoding()
    receipt = enumerate_grammar(encoding)
    rows = receipt.family_witnesses
    malformed = replace(
        receipt, family_witnesses=rows + rows[:1] if mutation == "duplicate" else rows[1:]
    )
    with pytest.raises(ValueError, match="current grammar"):
        protected_representatives(encoding, enumeration=malformed)


@pytest.mark.parametrize("mutation", ["wrong_template", "unaccepted_assignment"])
def test_enumeration_receipt_replays_positive_current_template_witnesses(mutation):
    from exact.repair.grammar import protected_representatives
    from exact.repair.proposals import enumerate_grammar

    encoding = receipt_encoding()
    receipt = enumerate_grammar(encoding)
    positive = [(name, bits) for name, bits in receipt.family_witnesses if bits is not None]
    name, original = positive[0]
    replacement = positive[1][1] if mutation == "wrong_template" else tuple(False for _ in original)
    assert (encoding.accepts(replacement)) is (mutation == "wrong_template")
    malformed = replace(
        receipt,
        family_witnesses=tuple(
            (key, replacement if key == name else bits) for key, bits in receipt.family_witnesses
        ),
    )
    with pytest.raises(ValueError, match="current template"):
        protected_representatives(encoding, enumeration=malformed)


def test_current_witnesses_survive_candidate_dedup_and_tuple_wrapper():
    from exact.repair.grammar import protected_representatives
    from exact.repair.proposals import enumerate_grammar

    encoding = receipt_encoding()
    receipt = enumerate_grammar(encoding)
    candidates = enumerate_grammar_candidates(encoding)
    assert isinstance(candidates, tuple)
    assert candidates == receipt.candidates
    assert receipt.language_hash == encoding.content_hash
    expected = current_family_members(encoding)
    ids = {candidate.candidate_id for candidate in receipt.candidates}
    positive_ids = []
    for name, bits in receipt.family_witnesses:
        if not expected[name]:
            assert bits is None
            continue
        assert bits is not None and encoding.accepts(bits)
        assert encoding.choices(bits)["template"] == name
        identifier = encoding.decode(bits).candidate_id
        assert identifier in expected[name] & ids
        positive_ids.append(identifier)
    assert len(positive_ids) > len(set(positive_ids))
    protected, reports = protected_representatives(encoding, enumeration=receipt, max_checks=0)
    protected_ids = {candidate.candidate_id for candidate in protected}
    for row in reports:
        if expected[row["template"]]:
            assert row["status"] == "retained"
            assert row["candidate_id"] in expected[row["template"]] & protected_ids
        else:
            assert row["status"] == "empty_language"


def test_exhausted_enumeration_does_not_return_completed_witness_receipt():
    from exact.repair.proposals import enumerate_grammar

    with pytest.raises(ValueError, match="max_expressions"):
        enumerate_grammar(receipt_encoding(), max_expressions=1)
