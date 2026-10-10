"""Shared scale uses actual strict records and never changes symbolic masks."""

import copy
import time

import pytest

from tests.repair_october_learning_semantics_test import native_case, aggregate, POLICY
from tools.repair.semantic_packet_encoding import encode
from tools.repair.shared_capacity import scale_audit, wait_for
from tools.repair.train import DEFAULT_PROFILE


def test_scale_reuses_exact_plan_anchors_and_retains_missing_case(tmp_path, monkeypatch):
    case, packet, cache = native_case()
    encoded, proof = encode(packet)
    comparison = aggregate(tmp_path / "annotation", monkeypatch, encoded)
    absent = copy.copy(case)
    object.__setattr__(absent, "case_id", case.case_id + ":absent")
    protocol = dict(preferences=dict(cost_weights=dict(DEFAULT_PROFILE)),
                    llm_labels=dict(plan_rating_aggregation=POLICY))
    result = scale_audit([case, absent], {case.case_id: cache},
                         {case.case_id: [(encoded, comparison)]},
                         {encoded.content_hash: dict(original_packet=packet.to_dict(), proof=proof)},
                         protocol)
    assert result["alpha"] == result["beta"] == 0.2
    assert result["weak_anchors"] == 2
    assert len(result["rows"]) == 2
    assert result["rows"][1]["anchors"] == []
    assert not result["rows"][1]["symbolic_cache_available"]


def test_peer_wait_reserves_cleanup_and_never_extends_deadline(tmp_path):
    with pytest.raises(TimeoutError, match="owned deadline"):
        wait_for(tmp_path / "missing", time.time() + 14)
