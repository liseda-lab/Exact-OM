"""The demo must preserve ranked-query semantics and actual recorded model usage."""

import hashlib

import pytest

from exact_inspect.models import PairResponse
from tools.recover_reviewer_decisions import Reader
from tools.reviewer_decision_trace import recovered_pair


def records():
    entity = lambda iri: {"ontology_version_id": "sha256:" + "a" * 64, "iri": iri, "kind": "class"}
    case = {"case_id": "case", "source": entity("urn:source")}
    candidate = {
        "candidate_id": "pair",
        "entity": entity("urn:target"),
        "score": 1.0,
        "display_position": 1,
    }
    recovery = {
        "mode": "local_ranking",
        "worker_sha256": "b" * 64,
        "submission_sha256": "c" * 64,
        "cases": {
            "case": {
                "source": "urn:source",
                "query_id": 3,
                "candidate_count": 100,
                "candidates": [
                    {
                        "candidate_id": "pair",
                        "target": "urn:target",
                        "rank": 1,
                        "final_score": 1.0,
                        "utility": 19.0,
                        "protected_exact": False,
                        "native_scores": {"S_base": 0.7, "S_struct": 0.6},
                        "applied": {
                            "p_llm": None,
                            "S_pair_final": 0.7,
                            "cached_response_reused": False,
                            "effective_pair_score_contribution": False,
                        },
                        "artifact_refs": ["sha256:" + "d" * 64],
                    }
                ],
            }
        },
    }
    return recovery, case, candidate


def test_ranking_score_one_does_not_imply_exact_or_accepted_mapping():
    recovery, case, candidate = records()
    scores, trace = recovered_pair(recovery, case, candidate, "run", None)
    assert scores[0]["value"] == 1.0
    assert trace["saved_alignment_member"] is None and trace["relation"] is None
    assert trace["ranking"]["candidate_count"] == 100
    stages = {e["stage"]: e for e in trace["events"]}
    assert stages["prefilter"]["reason_code"] == "prefilter_retained"
    assert stages["selection"]["outcome"] == "retained"
    assert stages["selection"]["reason_code"] == "local_query_ranked"
    for stage in ("threshold", "cardinality", "relation_typing", "repair"):
        assert stages[stage]["status"] == "not_run"
    assert stages["llm_signal"]["reason_code"] == "local_model_signal_not_applied"
    assert all(
        e["artifact_refs"] and e["provenance_status"] == "derived_from_saved_artifacts"
        for e in trace["events"]
    )


@pytest.mark.parametrize(
    "changed,reason",
    [(True, "local_model_signal_applied"), (False, "local_model_signal_no_score_change")],
)
def test_actual_model_application_is_distinct_from_placeholder_zero(changed, reason):
    recovery, case, candidate = records()
    saved = recovery["cases"]["case"]["candidates"][0]
    saved["applied"].update(
        p_llm=0.8, cached_response_reused=True, effective_pair_score_contribution=changed
    )
    scores, trace = recovered_pair(recovery, case, candidate, "run", None)
    assert next(e for e in trace["events"] if e["stage"] == "llm_signal")["reason_code"] == reason
    assert next(s for s in scores if s["name"] == "p_llm")["value"] == 0.8


def test_protected_anchor_does_not_invent_pair_scores():
    recovery, case, candidate = records()
    saved = recovery["cases"]["case"]["candidates"][0]
    saved.update(protected_exact=True, native_scores={}, applied=None, utility=None)
    scores, trace = recovered_pair(recovery, case, candidate, "run", None)
    assert [s["name"] for s in scores] == ["S_final"]
    assert trace["events"][1]["reason_code"] == "exact_match_protected"
    assert trace["events"][2]["status"] == "not_run"
    assert trace["saved_alignment_member"] is None


@pytest.mark.parametrize(
    "field,value", [("target", "urn:other"), ("rank", 2), ("final_score", 0.8)]
)
def test_mismatched_original_record_is_rejected(field, value):
    recovery, case, candidate = records()
    recovery["cases"]["case"]["candidates"][0][field] = value
    with pytest.raises(ValueError, match="does not match"):
        recovered_pair(recovery, case, candidate, "run", None)


def test_original_artifact_hashes_are_checked(tmp_path):
    path = tmp_path / "record.json"
    raw = b'{"mode":"local_ranking"}'
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    assert Reader().json(path, expected)["mode"] == "local_ranking"
    path.write_text('{"mode":"global_alignment"}')
    with pytest.raises(ValueError, match="Changed original artifact"):
        Reader().json(path, expected)


def test_ranked_output_contract_rejects_impossible_rank_and_alignment_claim():
    recovery, case, candidate = records()
    _, trace = recovered_pair(recovery, case, candidate, "run", None)
    with pytest.raises(ValueError, match="not an accepted"):
        PairResponse.model_validate({**trace, "saved_alignment_member": True})
    with pytest.raises(ValueError, match="Rank exceeds"):
        PairResponse.model_validate({**trace, "ranking": {**trace["ranking"], "rank": 101}})


def test_study_admits_only_matching_complete_decision_records():
    from copy import deepcopy

    from exact_inspect.models import Candidate
    from exact_inspect.study.workspace import _admit_pair_metadata

    recovery, case, candidate = records()
    metadata = dict(
        status="none_selected",
        bridge_status="not_recorded",
        counts=dict.fromkeys(
            [
                "selected",
                "resolved",
                "unresolved",
                "not_exported",
                "withheld",
                "graph_supported",
                "graph_unsupported",
                "graph_not_recorded",
            ],
            0,
        ),
    )
    scores, trace = recovered_pair(recovery, case, candidate, "run", metadata)
    details = Candidate.model_validate(
        dict(
            pair_id="pair",
            source=case["source"],
            target=candidate["entity"],
            scores=scores,
            retrieval_channels=[],
            ordinal_ranks={},
            evidence_counts=[],
            nil={},
            saved_alignment_member=None,
            membership_provenance={"status": "unavailable", "artifact_hash": None},
            status="available",
            decision_status="available",
        )
    ).model_dump(mode="json")
    original = dict(source=case["source"], target=candidate["entity"], evidence_metadata=metadata)
    assert "decision_trace" not in _admit_pair_metadata(original, candidate, case, [])
    value = dict(**original, decision_trace=trace, score_details=details)
    assert _admit_pair_metadata(value, candidate, case, [])["decision_trace"]["pair_id"] == "pair"
    for kind in ("missing", "wrong_pair", "wrong_score"):
        bad = deepcopy(value)
        if kind == "missing":
            del bad["score_details"]
        elif kind == "wrong_pair":
            bad["decision_trace"]["pair_id"] = "other"
        else:
            bad["score_details"]["scores"][0]["value"] = 0.5
        with pytest.raises(ValueError):
            _admit_pair_metadata(bad, candidate, case, [])
