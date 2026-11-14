"""Reference-aware study selection must not rewrite a matcher's candidate lists."""

from tools.final_study_copy import participant_information
from tools.prepare_final_study_catalogue import choose_pool, eligible_queries


def test_selection_rejects_ambiguous_and_missing_answers_without_reordering():
    candidates = [{"target": f"t{i}", "score": 1 - i / 10} for i in range(5)]
    queries = [{"qid": i, "source": f"s{i}", "candidates": candidates} for i in range(3)]
    refs = [
        {"SrcEntity": "s0", "TgtEntity": "t4"},
        {"SrcEntity": "s1", "TgtEntity": "missing"},
        {"SrcEntity": "s2", "TgtEntity": "t0"},
        {"SrcEntity": "s2", "TgtEntity": "t1"},
    ]
    cases, excluded = eligible_queries(queries, refs)
    assert len(cases) == 1 and cases[0]["reference_rank"] == 5
    assert [(c["target"], c["score"]) for c in cases[0]["candidates"]] == [
        (c["target"], c["score"]) for c in candidates
    ]
    assert excluded == {"no_reference_in_five": 1, "multiple_references_in_five": 1}


def test_multiple_queries_never_duplicate_a_source_or_merge_candidate_sets():
    cases = [
        {"source": s, "query_id": q, "reference_rank": r, "quality": [n]}
        for s, q, r, n in [("s1", 1, 1, 9), ("s1", 2, 5, 2), ("s2", 3, 1, 7), ("s3", 4, 1, 1)]
    ]
    chosen = choose_pool(cases, {1: 1, 5: 1})
    assert {c["query_id"] for c in chosen} == {2, 3}


def test_consent_draft_never_invents_missing_owner_details():
    draft = participant_information()
    assert len(draft["missing_owner_details"]) == 3
    assert not draft["ready_for_participant_publication"]
    assert "to be confirmed" in draft["information_text"]


def test_final_information_follows_owner_edits_without_claiming_ethics_approval():
    from tools.final_study_copy import OWNER_DETAILS

    copy = participant_information(OWNER_DETAILS)
    text = copy["information_text"]
    assert "National Cancer Institute Thesaurus (NCIT)" in text
    assert "Disease Ontology (DOID)" in text
    assert "one year" in text and "session code" in text
    assert "pgcotovio@ciencias.ulisboa.pt" in text
    for removed in (
        "Study period:",
        "Time to take part:",
        "Ethics review:",
        "clinical decisions",
        "professional abilities",
        "to be confirmed",
    ):
        assert removed not in text
    assert copy["ready_for_participant_publication"]


def test_research_promotion_requires_actual_owner_selection_and_authorization():
    from copy import deepcopy

    import pytest

    from tools.prepare_final_study_resources import promote_definition

    catalogue = {
        "owner_selection": {"ready_for_selection_freeze": True, "case_ids": ["case"]},
        "options": {"forty_unequal": ["case"]},
    }
    definition = {
        "purpose": "development_review",
        "correctness_metrics": "disabled",
        "review_bundle": {},
        "review_provenance": {},
    }
    copy = {"missing_owner_details": []}
    for approvals in ([], [""]):
        with pytest.raises(ValueError, match="launch approvals"):
            promote_definition(deepcopy(definition), catalogue, approvals, copy)
    pending = deepcopy(catalogue)
    pending["owner_selection"]["ready_for_selection_freeze"] = False
    with pytest.raises(ValueError, match="completed owner selection"):
        promote_definition(deepcopy(definition), pending, ["Owner instruction"], copy)
    changed = deepcopy(catalogue)
    changed["options"]["forty_unequal"] = ["different"]
    with pytest.raises(ValueError, match="differs"):
        promote_definition(deepcopy(definition), changed, ["Owner instruction"], copy)
    promote_definition(definition, catalogue, ["Owner instruction"], copy)
    assert not definition["synthetic"]
    assert definition["launch_approvals"] == ["Owner instruction"]
    assert "purpose" not in definition and "review_bundle" not in definition
