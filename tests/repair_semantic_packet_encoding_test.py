"""A smaller wire representation must retain every original packet byte."""

from dataclasses import replace
import copy

import pytest

from tests.repair_semantic_fidelity_test import packet
from tools.repair.semantic_packet_encoding import encode, decode


def source():
    value = packet()
    text = "<urn:exact:generated:0123456789abcdef:Paper_s>"
    return replace(
        value,
        original_observation=(text,),
        local_context=(text + " context",),
        evidence={k: {**v, "text": text + v["text"]} for k, v in value.evidence.items()},
        plan_a=replace(value.plan_a, content=(text * 40,)),
        plan_b=replace(value.plan_b, content=(text * 41,)),
    )


def test_exact_roundtrip_of_all_text_without_identity_or_query_changes():
    original = source()
    compact, proof = encode(original)
    assert decode(compact, proof).to_dict() == original.to_dict()
    assert compact.plan_a.plan_id == original.plan_a.plan_id
    assert compact.plan_a.query_outcomes == original.plan_a.query_outcomes
    assert compact.plan_b.theory_hash == original.plan_b.theory_hash
    assert sum(map(len, compact.plan_a.content)) < sum(map(len, original.plan_a.content))
    assert proof["axiom_omissions"] == 0
    assert encode(original) == (compact, proof)


@pytest.mark.parametrize("change", ["packet", "dictionary", "original_hash"])
def test_reject_changed_content_or_proof(change):
    compact, proof = encode(source())
    proof = copy.deepcopy(proof)
    if change == "packet":
        compact = replace(compact, plan_a=replace(compact.plan_a, content=("lost axiom",)))
    elif change == "dictionary":
        proof["dictionary"]["~N0~"] = "urn:exact:generated:ffffffffffffffff:"
    else:
        proof["original_packet_hash"] = "0" * 64
    with pytest.raises(ValueError):
        decode(compact, proof)


def test_no_recursive_or_ambiguous_encoding():
    compact, _ = encode(source())
    with pytest.raises(ValueError, match="already contains"):
        encode(compact)


def test_shared_full_context_is_present_once_and_reconstructs_both_theories():
    original = source()
    shared = "".join(f"SubClassOf(Class{i}, Class{i+1})\n" for i in range(200))
    original = replace(
        original,
        plan_a=replace(original.plan_a, content=(shared + "PlanA\n",)),
        plan_b=replace(original.plan_b, content=(shared + "PlanB\n",)),
    )
    compact, proof = encode(original)
    assert proof["blocks"] and any(shared in text for text in compact.local_context)
    assert decode(compact, proof).to_dict() == original.to_dict()
    proof = copy.deepcopy(proof)
    key = next(iter(proof["blocks"]))
    proof["blocks"][key] = "missing axioms"
    with pytest.raises(ValueError):
        decode(compact, proof)
