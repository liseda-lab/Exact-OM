"""Citable native facts and a cohort-bound, non-destructive annotation revision."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from exact.repair.records import canonical_hash
from exact.repair.semantic_fidelity import (
    NATIVE_EVIDENCE_PROMPT_VERSION,
    NATIVE_EVIDENCE_PROMPT,
    validate_comparison,
)
from tests.repair_semantic_fidelity_test import _offline_lookup_fixture, judgment
from tools.repair import dev_semantic_interface as interface
from tools.repair.batch import read, sha
from tools.repair.historical_regression import binding
from tools.repair.shared_release import immutable


def test_typed_native_outcomes_are_citable_and_swap_safe():
    case, packet, _ = _offline_lookup_fixture("development")
    evidence = interface.native_evidence(case.probes, (packet.plan_a, packet.plan_b), {})
    revised = replace(packet, evidence=evidence)
    assert len(evidence) == 4
    for swapped in (False, True):
        for key, item in evidence.items():
            fact = json.loads(item["text"])
            plan = next(p for p in (packet.plan_a, packet.plan_b) if p.plan_id == fact["plan_id"])
            assert fact["query_basis_hash"] == plan.query_basis_hash
            assert fact["consequence_report_id"] == item["source_id"]
            assert "urn:offline:A" in fact["query"]
            if item["kind"] == "native_nonvacuity":
                assert fact["nonvacuity_expressions"]
            if item["symbolic_value"] == "false":
                assert "not entailment of negation" in fact["interpretation"]
            value = judgment(revised, swapped=swapped)
            for criterion in value["criteria"]:
                criterion.update(
                    evidence_ids=[key],
                    quotes=[],
                    symbolic_claims=[dict(evidence_id=key, value=item["symbolic_value"])],
                )
            assert validate_comparison(
                json.dumps(value), revised, swapped=swapped
            ).global_target_eligible
            value["criteria"][0]["symbolic_claims"][0]["value"] = (
                "false" if item["symbolic_value"] == "true" else "true"
            )
            with pytest.raises(ValueError, match="symbolic claim"):
                validate_comparison(json.dumps(value), revised, swapped=swapped)
    assert interface.native_evidence(case.probes, (packet.plan_a, packet.plan_a), {}) == {
        k: v
        for k, v in evidence.items()
        if json.loads(v["text"])["plan_id"] == packet.plan_a.plan_id
    }


@pytest.mark.parametrize("change", ["basis", "missing", "unknown"])
def test_unqualified_query_data_is_rejected(change):
    case, packet, _ = _offline_lookup_fixture("development")
    plan = packet.plan_a
    if change == "basis":
        plan = replace(plan, query_basis_hash="wrong")
    elif change == "missing":
        plan = replace(plan, query_outcomes={})
    else:
        plan = replace(plan, query_outcomes={"q1": "unknown"})
    with pytest.raises(ValueError):
        interface.native_evidence(case.probes, (plan,), {})


def test_identical_plan_requires_grounded_equal_scores_not_automatic_perfection():
    from exact.repair.annotation_controls import response_format

    _, packet, _ = _offline_lookup_fixture("development")
    packet = replace(packet, plan_b=packet.plan_a)
    schema = response_format(packet)["json_schema"]["schema"]
    assert schema["properties"]["decision"]["enum"] == ["tie", "abstain"]
    value = judgment(packet, decision="tie", a=0.25, b=0.25)
    assert validate_comparison(json.dumps(value), packet).overall_score_a == 0.25
    value["criteria"][0]["b_score"] = 1
    assert not validate_comparison(json.dumps(value), packet).global_target_eligible
    value = judgment(packet, decision="abstain", a=None, b=None)
    value["abstention_reason"] = "Missing meaning evidence"
    for row in value["criteria"]:
        row.update(status="unknown", preference="abstain")
    assert validate_comparison(json.dumps(value), packet).overall_score_a is None


@pytest.mark.parametrize("identical", [False, True])
def test_decoded_entrypoint_uses_full_content_native_citations_and_identical_plans(
    tmp_path, monkeypatch, identical
):
    from exact.repair import kernel
    from tools.repair import corrective_semantics as runner, prepare
    import exact.repair.workers

    case, packet, _ = _offline_lookup_fixture("development")
    slot = dict(seed=13, supervision_condition="symbolic", epoch=6, case_id=case.case_id)
    declaration = dict(
        phase="development",
        dev_interface_amendment=dict(path="bound", sha256="revision"),
        frozen_annotation_slots={canonical_hash(slot): dict(selection_slot=slot)},
        cases={
            case.case_id: dict(
                parent=case.structural_parent,
                query_basis_hash=packet.plan_a.query_basis_hash,
                counterpart_candidate_ids=[
                    (
                        packet.plan_a.complete_assignment[0][1]
                        if identical
                        else packet.plan_b.complete_assignment[0][1]
                    )
                ],
                task=packet.task,
                original_observation=packet.original_observation,
                evidence={k: v for k, v in packet.evidence.items() if k.startswith("D")},
                local_context=packet.local_context,
                coverage=packet.coverage,
            )
        },
        rubric_version=packet.rubric_version,
        criterion_weights=packet.criterion_weights,
    )
    manifest = tmp_path / "manifest.json"
    immutable(manifest, declaration)
    request = tmp_path / "request.json"
    immutable(
        request,
        dict(
            schema="exact-repair/post-decode-annotation-request/v1",
            manifest_hash=sha(manifest),
            selection_slot=slot,
            case={},
            assignment=[0],
            verification=kernel.verify_assignment(case.problem, (0,)).to_dict(),
        ),
    )
    monkeypatch.setattr(runner, "validate_manifest", lambda x: x)
    monkeypatch.setattr(prepare, "case_from_dict", lambda _: case)
    monkeypatch.setattr(
        exact.repair.workers,
        "bounded_call",
        lambda fn, record, assignment, **kw: SimpleNamespace(
            status="complete",
            cleanup_complete=True,
            value=packet.plan_a if assignment == (0,) else packet.plan_b,
        ),
    )
    calls = []
    monkeypatch.setattr(runner, "annotate_comparison", lambda p, *args, **kw: calls.append((p, kw)))
    runner.annotate_decoded(request, manifest, seconds=300)
    p, options = calls[0]
    assert p.plan_a.content == packet.plan_a.content
    assert p.plan_b.content == (packet.plan_a if identical else packet.plan_b).content
    assert "~B" not in str(p.to_dict()) and p.eligible
    assert any(v["kind"] == "native_entailment" for v in p.evidence.values())
    assert options["comparison_id"] != canonical_hash(slot)


def recovery_fixture(tmp_path):
    from tests.repair_october_learning_annotation_runner_test import manifest

    def save(name, value):
        p = tmp_path / (name + ".json")
        immutable(p, value)
        return binding(p)

    slot = dict(seed=13, supervision_condition="symbolic", epoch=6, case_id="c")
    old = dict(
        manifest(tmp_path),
        phase="development",
        cases={"c": dict(coverage=dict(byte_budget=100))},
        frozen_annotation_slots={canonical_hash(slot): dict(selection_slot=slot)},
        rubric_version="r",
        criterion_weights={"meaning_retention": 1},
    )
    previous = save("previous", old)
    request = save(
        "source",
        dict(
            request_identity="saved",
            manifest_hash=previous["sha256"],
            selection_slot=slot,
            assignment=[1],
            model_hash="weights",
        ),
    )
    plan = save("plan", dict(context_hash="original-provenance", plan={}))
    amendment = dict(
        schema=interface.SCHEMA,
        authorized=True,
        user_instruction="Please fix.",
        costs_reset=False,
        selection_rule_unchanged=True,
        test_opened=False,
        previous_manifest=previous,
        deadline_epoch=1000,
        requests={
            canonical_hash(slot): dict(
                source_request=request, historical_annotation_request=request, verified_plans=[plan]
            )
        },
        qualification_slots=[canonical_hash(slot)],
    )
    declaration = dict(
        old,
        prompt_version=NATIVE_EVIDENCE_PROMPT_VERSION,
        dev_interface_amendment=save("amendment", amendment),
        cases={"c": dict(coverage=dict(byte_budget=1000))},
        request_profiles={"teacher": dict(max_input_bytes=1000)},
        rate_limit_policy=dict(resume_deadline_epoch=1000),
    )
    return declaration, save("revised", declaration), request, plan


def test_revision_resume_keeps_native_provenance_and_does_not_overwrite_old_request(tmp_path):
    declaration, manifest, source, plan = recovery_fixture(tmp_path)
    revised = interface.rebind_request(source["path"], manifest["path"], tmp_path / "state")
    assert interface.rebind_request(source["path"], manifest["path"], tmp_path / "state") == revised
    assert sha(source["path"]) == source["sha256"]
    assert read(revised)["model_hash"] == "weights"
    assert read(revised)["assignment"] == [1]
    assert read(revised)["request_identity"] != read(source["path"])["request_identity"]
    assert read(revised.parent / "verified-plans/plan.json") == read(plan["path"])
    Path(plan["path"]).write_text("{}")
    with pytest.raises(ValueError):
        interface.rebind_request(source["path"], manifest["path"], tmp_path / "state")


@pytest.mark.parametrize(
    "field", ["frozen_annotation_slots", "cases", "profiles", "request_limits", "parent_splits"]
)
def test_interface_amendment_rejects_changed_design_before_request(tmp_path, field):
    declaration, _, _, _ = recovery_fixture(tmp_path)
    declaration = deepcopy(declaration)
    declaration[field] = {}
    with pytest.raises(ValueError):
        interface.validate_amendment(declaration)


def test_new_prompt_does_not_change_historical_request_bytes():
    from exact.repair.semantic_fidelity import INTERFACE_PROMPT, PROMPT

    assert (
        canonical_hash(INTERFACE_PROMPT)
        == "c506979b341abfaa460c4363464e698cacb38942654c819e0305615e38b14c15"
    )
    assert NATIVE_EVIDENCE_PROMPT.startswith(INTERFACE_PROMPT)
    assert NATIVE_EVIDENCE_PROMPT != PROMPT


def test_qualified_profile_allows_only_bound_interface_and_input_revision(tmp_path, monkeypatch):
    from tools.repair.teacher_selection import compatible_input_profile
    from tools.repair import annotation_profile

    declaration, _, _, _ = recovery_fixture(tmp_path)
    baseline = dict(max_output_tokens=2000, tokenizer={"sha256": "fixed"})
    proof = tmp_path / "input.json"
    immutable(proof, dict(previous_profile_hash=canonical_hash(baseline)))
    controls = dict(
        baseline,
        max_output_tokens=3000,
        max_input_tokens=100,
        max_input_bytes=1000,
        input_amendment=binding(proof),
    )
    declaration["request_profiles"] = {"teacher": controls}
    monkeypatch.setattr(
        annotation_profile, "request_profile", lambda m, p: m["request_profiles"][p]
    )
    source = dict(request_profiles={"teacher": baseline})
    assert compatible_input_profile(declaration, source, "teacher")
    declaration["request_profiles"]["teacher"]["tokenizer"] = {"sha256": "different"}
    assert not compatible_input_profile(declaration, source, "teacher")
    declaration["request_profiles"]["teacher"]["tokenizer"] = baseline["tokenizer"]
    declaration.pop("dev_interface_amendment")
    assert not compatible_input_profile(declaration, source, "teacher")


def test_readiness_review_requires_all_models_and_preserves_missing_semantics(
    tmp_path, monkeypatch
):
    def save(name, value):
        path = tmp_path / name
        immutable(path, value)
        return binding(path)

    declaration, _, _, _ = recovery_fixture(tmp_path)
    historical = save(
        "historical-review.json",
        dict(training_integrity_passed=True, full_dev_lifecycle_qualified=True),
    )
    monkeypatch.setattr(
        interface, "validate_amendment", lambda _: dict(historical_review=historical)
    )
    slots, models, reports = {}, [], []
    for i in range(6):
        seed, condition = (13, 37, 73)[i // 2], ("symbolic", "symbolic_plus_llm")[i % 2]
        slot = dict(seed=seed, supervision_condition=condition, epoch=6, case_id="c0")
        slots[canonical_hash(slot)] = dict(selection_slot=slot)
        source_rows = [save(f"source-{i}-{j}.json", dict(case_id=f"c{j}")) for j in range(32)]
        source = save(f"source-{i}.json", dict(rows=source_rows))
        artifact = save(f"weight-{i}.json", dict(tensor=i))
        weights = save(f"weights-{i}.json", dict(epoch=6, artifact=artifact))
        model = dict(seed=seed, condition=condition, source_report=source, weights=weights)
        models.append(model)
        output = tmp_path / f"model-{i}"
        for j, ref in enumerate(source_rows):
            immutable(
                output / "rows" / f"{j}.json",
                dict(
                    source=ref,
                    result=dict(
                        case_id=f"c{j}",
                        decoded=dict(
                            logical_status="VERIFIED_FEASIBLE", weak_benefit=None if i == 0 else 0.5
                        ),
                    ),
                ),
            )
        reports.append((output, source, artifact, i))
    declaration["frozen_annotation_slots"] = slots
    manifest = save("review-annotation.json", declaration)
    plan = save("review-plan.json", dict(annotation_manifest=manifest, models=models))
    paths = []
    for output, source, artifact, i in reports:
        path = output / "report.json"
        immutable(
            path,
            dict(
                source=source,
                status="complete",
                rows=32,
                epoch=6,
                annotation_manifest=manifest,
                interface_amendment=declaration["dev_interface_amendment"],
                selection_rule_unchanged=True,
                test_opened=False,
                costs_reset=False,
                selection_criterion=None if i == 0 else 0.5,
                selected_checkpoint=None if i == 0 else artifact,
            ),
        )
        paths.append(path)
    interface.review(plan["path"], paths, tmp_path / "review")
    result = read(tmp_path / "review/report.json")
    assert result["status"] == "review_required"
    assert result["selected_models_available"] == result["semantic_available"] == 5
    assert result["semantic_slots"] == 6 and result["verified_dev_rows"] == 192
    with pytest.raises(ValueError, match="all six"):
        interface.review(plan["path"], paths[:-1], tmp_path / "partial")
    with pytest.raises(ValueError, match="Duplicated"):
        interface.review(plan["path"], [paths[0]] * 6, tmp_path / "duplicate")


def test_quote_free_wire_preserves_citations_and_rejects_fabricated_quotes():
    from exact.repair.annotation_controls import response_format
    from exact.repair.semantic_fidelity import (
        CITATION_PROMPT,
        CITATION_PROMPT_VERSION,
        annotation_prompt,
    )

    _, packet, _ = _offline_lookup_fixture("development")
    old = response_format(packet)
    fixed = response_format(packet, verbatim_quotes=False)
    before = old["json_schema"]["schema"]["properties"]["criteria"]["items"]["properties"]
    after = fixed["json_schema"]["schema"]["properties"]["criteria"]["items"]["properties"]
    assert after["quotes"]["maxItems"] == 0 and "maxItems" not in before["quotes"]
    assert {k: v for k, v in before.items() if k != "quotes"} == {
        k: v for k, v in after.items() if k != "quotes"
    }
    assert annotation_prompt(CITATION_PROMPT_VERSION) == CITATION_PROMPT
    value = judgment(packet)
    for criterion in value["criteria"]:
        criterion["quotes"] = []
    assert validate_comparison(json.dumps(value), packet).global_target_eligible
    value["criteria"][0]["quotes"] = [dict(evidence_id="D1", span="invented quote")]
    with pytest.raises(ValueError, match="quote"):
        validate_comparison(json.dumps(value), packet)


def test_same_comparison_new_presentation_keeps_all_shared_request_costs(tmp_path):
    from tests.repair_october_learning_annotation_runner_test import manifest
    from tools.repair.corrective_semantics import _reserve_phase

    settings = dict(manifest(tmp_path), phase="development")
    _reserve_phase(
        settings, "first-interface:original", "same-packet", 0.02, comparison_id="fixed-pair"
    )
    before = read(tmp_path / "ledger/phase-reservations.json")
    _reserve_phase(
        settings, "quotation-interface:original", "same-packet", 0.03, comparison_id="fixed-pair"
    )
    after = read(tmp_path / "ledger/phase-reservations.json")
    assert all(after["reservations"][k] == v for k, v in before["reservations"].items())
    assert len(after["reservations"]) == 2
    assert len({r["comparison_id"] for r in after["reservations"].values()}) == 1
    assert sum(r["reserved_cost_usd"] for r in after["reservations"].values()) == 0.05
