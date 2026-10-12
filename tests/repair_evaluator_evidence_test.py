"""Exact DEV checks, reference-only judgments, resume and durable provenance."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from exact.repair.annotation_controls import response_format
from exact.repair.semantic_fidelity import (
    REFERENCE_PARSER_VERSION,
    REFERENCE_PROMPT_VERSION,
    TEACHER,
    aggregate_comparisons,
    validate_comparison,
)
from tests.repair_semantic_fidelity_test import (
    _offline_lookup_fixture,
    judgment,
    packet,
    adapter,
    response,
)
from tools.repair import evaluator_evidence as evidence
from tools.repair.dev_semantic_interface import native_evidence
from tools.repair.grounded_supervision import render
from tools.repair.historical_regression import binding
from tools.repair.shared_release import immutable


def reference_judgment(p, **kwargs):
    value = judgment(p, **kwargs)
    key = next(k for k, v in p.evidence.items() if v["symbolic_value"] in {"true", "false"})
    for criterion in value["criteria"]:
        criterion.update(evidence_ids=[], quotes=[], symbolic_claims=[dict(evidence_id=key)])
    return value


@pytest.mark.parametrize("swapped", [False, True])
def test_reference_is_its_own_citation_and_derives_only_the_supplied_value(swapped):
    p = packet()
    raw = reference_judgment(p, swapped=swapped)
    before = deepcopy(raw)
    parsed = validate_comparison(
        json.dumps(raw), p, swapped=swapped, annotator={"parser_version": REFERENCE_PARSER_VERSION}
    )
    assert parsed.global_target_eligible and raw == before
    for c in parsed.criteria:
        for claim in c["symbolic_claims"]:
            assert claim["evidence_id"] in c["evidence_ids"]
            assert claim["value"] == p.evidence[claim["evidence_id"]]["symbolic_value"]
    with pytest.raises(ValueError):
        validate_comparison(json.dumps(raw), p, swapped=swapped)


@pytest.mark.parametrize("bad", ["invented", "unchecked", "copied_value"])
def test_invalid_reference_cannot_become_a_checked_fact(bad):
    p = packet()
    raw = reference_judgment(p)
    if bad == "invented":
        raw["criteria"][0]["symbolic_claims"] = [{"evidence_id": "missing"}]
    elif bad == "unchecked":
        p = replace(
            p,
            evidence={
                **p.evidence,
                "definition": dict(
                    source_id="d",
                    release="v1",
                    kind="definition",
                    text="Definition",
                    symbolic_value=None,
                ),
            },
        )
        raw = reference_judgment(p)
        raw["criteria"][0]["symbolic_claims"] = [{"evidence_id": "definition"}]
    else:
        raw["criteria"][0]["symbolic_claims"][0]["value"] = "false"
    with pytest.raises(ValueError):
        validate_comparison(
            json.dumps(raw), p, annotator={"parser_version": REFERENCE_PARSER_VERSION}
        )


def test_reference_schema_prevents_value_copy_and_keeps_unknowns():
    p = packet()
    schema = response_format(
        p, fact_references=True, complete_criteria=True, verbatim_quotes=False
    )["json_schema"]["schema"]
    criterion = schema["properties"]["criteria"]["items"]["properties"]
    assert set(criterion["symbolic_claims"]["items"]["properties"]) == {"evidence_id"}
    assert "unknown" in criterion["status"]["enum"]
    assert "null" in criterion["a_score"]["type"]
    assert criterion["quotes"]["maxItems"] == 0


def test_reference_wire_replay_and_swap_aggregation_keep_unique_provenance(tmp_path, monkeypatch):
    p = packet()

    def handler(**kwargs):
        wire = json.loads(kwargs["content"])
        context = json.loads(wire["messages"][1]["content"])
        swap = context["swapped"]
        return response(
            reference_judgment(
                p,
                swapped=swap,
                decision="B" if swap else "A",
                a=0.25 if swap else 0.75,
                b=0.75 if swap else 0.25,
            )
        )

    annotator, calls = adapter(tmp_path, monkeypatch, handler=handler)
    annotator.run = replace(annotator.run, prompt_version=REFERENCE_PROMPT_VERSION)
    schedule = annotator.schedule(
        p, role=TEACHER, quorum=2, repetitions=(0,), presentation_orders=(False, True)
    )
    assert schedule.parser_versions == (REFERENCE_PARSER_VERSION,)
    a = annotator.annotate(p, role=TEACHER)
    b = annotator.annotate(p, role=TEACHER, swapped=True)
    result = aggregate_comparisons([a, b], schedule=schedule)
    assert result["eligible"] == 2 and result["decision"] == "A"
    assert annotator.annotate(p, role=TEACHER) == a and len(calls) == 2
    assert aggregate_comparisons([a, a], schedule=schedule)["decision"] == "abstain"


def setup_case(monkeypatch):
    case, p, _ = _offline_lookup_fixture("development")
    case.origin = "generated"
    case.intended_theory = (case.probes[0].axiom,)
    record = {"id": case.case_id}
    monkeypatch.setattr("tools.repair.prepare.case_from_dict", lambda _: case)
    d2 = dict(
        source_id="authored",
        release="v1",
        kind="authored_construction_commitments",
        symbolic_value=None,
        text="Intended editable commitments, together with the unchanged full fixed context:\n"
        + render(case.intended_theory),
    )
    p = replace(
        p,
        evidence=native_evidence(case.probes, (p.plan_a, p.plan_b), {"D2": d2}, compare_text=True),
    )
    return case, p, record


def test_native_intention_checks_distinguish_false_entailment_from_satisfiability(monkeypatch):
    _, p, record = setup_case(monkeypatch)
    a = evidence._check(record, (0,), p.plan_a.to_dict())
    b = evidence._check(record, (1,), p.plan_b.to_dict())
    assert a["rows"][0]["entailed"] is True
    assert b["rows"][0]["entailed"] is False
    assert a["rows"][0]["nonvacuous"] is b["rows"][0]["nonvacuous"] is True
    with pytest.raises(ValueError, match="complete theory"):
        evidence._check(record, (1,), p.plan_a.to_dict())


def test_native_domain_and_range_queries_are_direct_not_witness_substitutes(monkeypatch):
    import pyowl_core as owl

    case, p, record = setup_case(monkeypatch)
    prop = owl.ObjectProperty(owl.IRI("urn:offline:writes"))
    agent = owl.Class(owl.IRI("urn:offline:A"))
    case.intended_theory = (
        owl.ObjectPropertyDomain(prop, agent),
        owl.ObjectPropertyRange(prop, agent),
    )
    result = evidence._check(record, (0,), p.plan_a.to_dict())
    assert len(result["rows"]) == 2
    assert all(r["entailed"] is False and r["nonvacuous"] is True for r in result["rows"])
    assert any("ObjectPropertyDomain" in r["query"] for r in result["rows"])
    assert any("ObjectPropertyRange" in r["query"] for r in result["rows"])


def test_auxiliary_resume_keeps_exact_case_theory_and_intention_identity(tmp_path, monkeypatch):
    _, p, record = setup_case(monkeypatch)
    calls = []

    def local(fn, *args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            status="complete",
            cleanup_complete=True,
            detail=None,
            value=fn(*args),
            resource_usage=(),
        )

    monkeypatch.setattr("exact.repair.workers.bounded_call", local)
    revised = evidence.augment(p, record, ((0,), (1,)), tmp_path, seconds=30)
    assert (
        evidence.augment(p, record, ((0,), (1,)), tmp_path, seconds=30) == revised
        and len(calls) == 2
    )
    assert revised.plan_a == p.plan_a and revised.plan_b == p.plan_b
    assert revised.required_query_ids == p.required_query_ids
    assert any(k.startswith("INTENDED_ENTAILS_") for k in revised.evidence)
    assert all(not k.startswith("Q") for k in revised.evidence)
    assert "not missing evidence" in revised.task
    with pytest.raises(ValueError, match="frozen intended"):
        evidence.augment(
            replace(p, evidence={**p.evidence, "D2": {**p.evidence["D2"], "text": "changed"}}),
            record,
            ((0,), (1,)),
            tmp_path,
            seconds=30,
        )
    saved = next(tmp_path.glob("*.json"))
    data = json.loads(saved.read_text())
    data["context"] = "wrong"
    saved.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checkpoint identity"):
        evidence.augment(p, record, ((0,), (1,)), tmp_path, seconds=30)


def test_reallocation_preserves_spent_attempts_and_test_reserve(tmp_path):
    old = dict(calibration=91, train=325, development=96, test=192)
    new = dict(calibration=91, train=277, development=144, test=192)
    prior = dict(
        request_limits=old,
        reservations={str(i): dict(phase="train") for i in range(251)},
        lineage="campaign",
        cost_ceiling_usd=35,
    )
    previous = dict(
        request_limits=old,
        lineage_id="campaign",
        cost_ceiling_usd=35,
        ledger_directory=str(tmp_path),
    )
    immutable(tmp_path / "phase.json", prior)
    immutable(tmp_path / "previous.json", previous)
    amendment = dict(
        schema="exact-repair/dev-evaluator-request-reallocation/v1",
        authorized=True,
        user_instruction="Proceed with the fix",
        costs_reset=False,
        old_limits=old,
        new_limits=new,
        phase_before=binding(tmp_path / "phase.json"),
        previous_manifest=binding(tmp_path / "previous.json"),
    )
    immutable(tmp_path / "amendment.json", amendment)
    manifest = {
        **previous,
        "phase": "development",
        "request_limits": new,
        "evaluator_request_amendment": binding(tmp_path / "amendment.json"),
    }
    assert evidence.request_reallocation(manifest)[0] == amendment
    for bad in (
        {**new, "test": 191, "development": 145},
        {**new, "train": 250, "development": 171},
        {**new, "development": 145},
    ):
        with pytest.raises(ValueError):
            evidence.request_reallocation({**manifest, "request_limits": bad})


def test_phase_migration_preserves_history_and_resume_does_not_reserve_twice(tmp_path, monkeypatch):
    from tools.repair import corrective_semantics as runner
    from exact.repair.records import canonical_hash

    old = dict(calibration=91, train=325, development=96, test=192)
    new = {**old, "train": 277, "development": 144}
    historical = dict(
        phase="development",
        slot="old",
        comparison_id="same-pair",
        reserved_seconds=90,
        reserved_cost_usd=0.02,
        state="completed",
        packet_hash="old-packet",
    )
    prior = dict(
        lineage="campaign",
        cost_ceiling_usd=35,
        request_limits=old,
        request_budget_amendment="original-amendment",
        reservations={"old": historical},
    )
    manifest = dict(
        lineage_id="campaign",
        cost_ceiling_usd=35,
        request_limits=new,
        phase="development",
        ledger_directory=str(tmp_path),
        evaluator_request_amendment={"path": "approval", "sha256": "bound"},
    )
    monkeypatch.setattr(
        runner,
        "_request_budget",
        lambda _: (
            new,
            dict(
                binding="original-amendment",
                prior=dict(reservations={}),
                evaluator_quota=dict(old_limits=old),
                evaluator_prior=prior,
            ),
        ),
    )
    runner.write_artifact(tmp_path / "phase-reservations.json", prior)
    runner._reserve_phase(manifest, "new", "packet", 0.03, comparison_id="same-pair")
    first = runner.read(tmp_path / "phase-reservations.json")
    assert first["request_limits"] == new and first["reservations"]["old"] == historical
    assert len(first["reservations"]) == 2
    runner._reserve_phase(manifest, "new", "packet", 0.03, comparison_id="same-pair")
    assert runner.read(tmp_path / "phase-reservations.json") == first
    assert (
        first["reservations"][canonical_hash(("development", "new"))]["reserved_cost_usd"] == 0.03
    )
    first["reservations"].pop("old")
    runner.write_artifact(tmp_path / "phase-reservations.json", first)
    with pytest.raises(ValueError, match="historical reservations"):
        runner._reserve_phase(manifest, "another", "packet", 0.03, comparison_id="same-pair")


def test_incomplete_native_checkpoint_never_becomes_an_annotation_fact(tmp_path, monkeypatch):
    _, p, record = setup_case(monkeypatch)
    monkeypatch.setattr(
        "exact.repair.workers.bounded_call",
        lambda *a, **kw: SimpleNamespace(
            status="timeout", cleanup_complete=True, detail="timeout", value=None, resource_usage=()
        ),
    )
    with pytest.raises(ValueError, match="native evidence unavailable"):
        evidence.augment(p, record, ((0,), (1,)), tmp_path, seconds=30)
    monkeypatch.setattr(
        "exact.repair.workers.bounded_call",
        lambda *a, **kw: pytest.fail("silently retried terminal native evidence"),
    )
    with pytest.raises(ValueError, match="native evidence unavailable"):
        evidence.augment(p, record, ((0,), (1,)), tmp_path, seconds=30)
