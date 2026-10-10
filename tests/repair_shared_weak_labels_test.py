"""Required audits must survive offline aggregation, with no duplicate terms."""

import pytest

from exact.repair.semantic_fidelity import TEACHER, validate_fidelity_training_records
from tests.repair_semantic_fidelity_test import adapter, judgment, packet, response
from tools.repair.shared_weak_labels import combine, group_rows


def observed(tmp_path, monkeypatch, *, disagree=False):
    p = packet()
    def reply(**kwargs):
        # The order is determined by call sequence; these are independent receipts.
        swapped = bool(len(calls) % 2 == 0)
        return response(judgment(p, swapped=swapped, decision="A" if not swapped or disagree else "B",
                                 a=0.75 if not swapped or disagree else 0.25,
                                 b=0.25 if not swapped or disagree else 0.75))
    client, calls = adapter(tmp_path, monkeypatch, handler=reply)
    schedules = [client.schedule(p, role=TEACHER, quorum=1, repetitions=(0,), presentation_orders=(order,)) for order in (False, True)]
    observations = [client.annotate(p, role=TEACHER, swapped=order) for order in (False, True)]
    return p, schedules, observations


def test_joint_audit_is_one_term_and_preserves_both_observations(tmp_path, monkeypatch):
    p, schedules, observations = observed(tmp_path, monkeypatch)
    aggregate = combine(schedules, observations)
    assert aggregate.global_target_eligible
    assert aggregate.result["observed"] == aggregate.result["quorum"] == 2
    single = combine(schedules[:1], observations[:1])
    with pytest.raises(ValueError, match="reuse one durable observation"):
        validate_fidelity_training_records({p.case_id: [(p, single), (p, aggregate)]}, "train")


def test_missing_required_swap_masks_original(tmp_path, monkeypatch):
    _, schedules, observations = observed(tmp_path, monkeypatch)
    aggregate = combine(schedules, observations[:1])
    assert not aggregate.global_target_eligible
    assert aggregate.result["missing"] == 1
    assert aggregate.result["abstention_reason"] == "insufficient_quorum"


def test_discordant_orders_never_become_two_training_terms(tmp_path, monkeypatch):
    _, schedules, observations = observed(tmp_path, monkeypatch, disagree=True)
    assert not combine(schedules, observations).global_target_eligible
    with pytest.raises(ValueError, match="presentation"):
        combine(schedules[::-1], observations)


def test_closed_denominator_rejects_missing_original_and_duplicate_rows():
    row = dict(id="a", comparison_id="a", case_id="c", parent="p", family="f", control="clean", swapped=False)
    assert len(group_rows([row])) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        group_rows([row, row])
    with pytest.raises(ValueError, match="Missing original"):
        group_rows([dict(row, swapped=True)])


def test_lossless_context_restores_native_evidence_but_rejects_changed_proof():
    import dataclasses
    from tools.repair.semantic_packet_encoding import encode
    from tools.repair.semantic_rating_context import rating_context

    p = packet()
    shared = "A shared complete-theory line " * 8 + "\n"
    p = dataclasses.replace(p, plan_a=dataclasses.replace(p.plan_a, content=(shared + "A",)),
                            plan_b=dataclasses.replace(p.plan_b, content=(shared + "B",)))
    encoded, proof = encode(p)
    assert encoded.local_context != p.local_context
    ref = dict(original_packet=p.to_dict(), proof=proof)
    assert rating_context(encoded, ref) == p
    changed = dataclasses.replace(p, task=p.task + " changed")
    with pytest.raises(ValueError, match="round trip"):
        rating_context(encoded, dict(ref, original_packet=changed.to_dict()))
    bad = dict(proof, blocks={"~B0~": "invented evidence"})
    with pytest.raises(ValueError):
        rating_context(encoded, dict(ref, proof=bad))


def test_exact_context_correction_preserves_original_strict_anchor_policy(tmp_path, monkeypatch):
    import dataclasses
    from exact.repair.semantic_fidelity import offline_plan_label
    from tests.repair_october_learning_semantics_test import native_case, aggregate, POLICY
    from tools.repair.semantic_packet_encoding import encode

    case, p, _ = native_case()
    shared = "Complete shared text " * 10 + "\n"
    p = dataclasses.replace(p, plan_a=dataclasses.replace(p.plan_a, content=(shared + "A",)),
                            plan_b=dataclasses.replace(p.plan_b, content=(shared + "B",)))
    other = dataclasses.replace(p, plan_b=dataclasses.replace(p.plan_b, content=("other B",)))
    packets = [encode(original) for original in (p, other)]
    pairs = [(encoded, aggregate(tmp_path / str(i), monkeypatch, encoded))
             for i, (encoded, _) in enumerate(packets)]
    with pytest.raises(ValueError, match="incompatible"):
        offline_plan_label(case, (0,), (), pairs, rating_aggregation=POLICY)
    contexts = {encoded.content_hash: dict(original_packet=original.to_dict(), proof=proof)
                for original, (encoded, proof) in zip((p, other), packets)}
    label = offline_plan_label(case, (0,), (), pairs, rating_aggregation=POLICY,
                               rating_context_packets=contexts)
    assert label is not None and label.benefit == 0.75
