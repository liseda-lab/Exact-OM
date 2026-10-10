from dataclasses import replace

import pytest

from tests.repair_grounded_supervision_test import case
from tools.repair.grounded_supervision import controlled_evidence
from tools.repair.prepare_development_contract import case_entry, fixed_counterpart


def test_dev_evidence_requires_explicit_split_and_never_accepts_test():
    item = replace(case(), split="development")
    source = dict(path="source.json", sha256="a" * 64)
    with pytest.raises(ValueError):
        controlled_evidence(item, source)
    evidence = controlled_evidence(item, source, allowed_split="development")
    assert evidence["split"] == "development"
    assert evidence["input_hash"] == item.problem.content_hash
    for split in ("test", "train"):
        with pytest.raises(ValueError):
            controlled_evidence(replace(item, split=split), source, allowed_split="development")
    with pytest.raises(ValueError):
        controlled_evidence(replace(item, split="test"), source, allowed_split="test")


def test_dev_counterpart_and_evidence_do_not_depend_on_query_outcomes():
    item = replace(case(), split="development")
    source = dict(path="source.json", sha256="a" * 64)
    entry = case_entry(item, source, 65536)
    changed = case_entry(replace(item, probes=()), source, 65536)
    assert entry["evidence"] == changed["evidence"]
    assert entry["counterpart_candidate_ids"] == fixed_counterpart(item)
    assert entry["counterpart_candidate_ids"] == changed["counterpart_candidate_ids"]
    assert entry["query_basis_hash"] != changed["query_basis_hash"]
    assert entry["coverage"]["byte_budget"] == 65536


def test_post_decode_transport_keeps_full_native_packet_and_proof(tmp_path):
    import json
    from exact.repair.records import read_record
    from tests.repair_semantic_fidelity_test import packet
    from tools.repair.corrective_semantics import decoded_transport_packet
    from tools.repair.semantic_packet_encoding import REVISION, decode

    original = packet(split="development")
    encoded = decoded_transport_packet(original, {"packet_encoding": REVISION}, tmp_path)
    proof = json.loads((tmp_path / "transport-proof.json").read_text())
    assert decode(encoded, proof) == original
    assert read_record(json.loads((tmp_path / "original-packet.json").read_text())) == original
    assert encoded.plan_a.theory_hash == original.plan_a.theory_hash
    assert encoded.plan_b.theory_hash == original.plan_b.theory_hash
    assert decoded_transport_packet(original, {"packet_encoding": REVISION}, tmp_path) == encoded
    with pytest.raises(ValueError, match="encoding"):
        decoded_transport_packet(original, {"packet_encoding": "unknown"}, tmp_path)
    with pytest.raises(ValueError, match="split"):
        decoded_transport_packet(packet(), {"packet_encoding": REVISION}, tmp_path)
