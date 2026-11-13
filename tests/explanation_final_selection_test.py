from copy import deepcopy

from tools.extract_final_study_evidence import bind_retained_record
from tools.final_study_selection import resolve_selection, verify_frozen_cases


def fixture():
    counts = {1: 14, 2: 8, 3: 8, 4: 2, 5: 8}
    cases = []
    for rank, count in counts.items():
        for n in range(count):
            cid = f"case-{rank}-{n}"
            cases.append(
                {
                    "case_id": cid,
                    "query_id": len(cases),
                    "source": cid,
                    "reference_rank": rank,
                    "reference_target": f"{cid}-t{rank}",
                    "candidates": [
                        {"target": f"{cid}-t{r}", "original_rank": r, "score": 1 / r}
                        for r in range(1, 6)
                    ],
                }
            )
    plan = {
        "selected_case_ids": [c["case_id"] for c in cases],
        "fixed_case_ids": [c["case_id"] for c in cases[:12]],
        "decision_slots": [
            {"case_id": "case-4-0", "choices": [{"case_id": "case-4-0"}, {"case_id": "reserve"}]}
        ],
    }
    reserve = deepcopy(cases[0])
    reserve.update(case_id="reserve", source="reserve", query_id=100)
    return cases + [reserve], plan


def test_pending_owner_decisions_are_never_reported_as_approved():
    cases, plan = fixture()
    pending = resolve_selection(cases, plan)
    assert len(pending["case_ids"]) == 40 and not pending["errors"]
    assert not pending["ready_for_selection_freeze"]
    approved = resolve_selection(cases, plan, {"case-4-0": "case-4-0"})
    assert approved["ready_for_selection_freeze"]
    assert not approved["ready_for_publication"]


def test_owner_replacement_cannot_break_two_cases_per_rank():
    cases, plan = fixture()
    result = resolve_selection(cases, plan, {"case-4-0": "reserve"})
    assert len(set(result["case_ids"])) == 40
    assert result["counts_by_rank"][4] == 1
    assert any("at least two" in e for e in result["errors"])


def test_an_exclusion_or_discussion_note_prevents_freezing_the_pool():
    cases, plan = fixture()
    choices = {"case-4-0": "case-4-0"}
    excluded = resolve_selection(cases, plan, choices, {"case-2-0": "exclude"})
    assert not excluded["ready_for_selection_freeze"]
    assert any("Exclude" in e for e in excluded["errors"])
    discussed = resolve_selection(cases, plan, choices, {"case-2-0": "consider"})
    assert discussed["pending_case_ids"] == ["case-2-0"]


def test_alternative_cannot_fill_two_slots_or_replace_fixed_approval():
    cases, plan = fixture()
    plan["decision_slots"].append(
        {"case_id": "case-5-0", "choices": [{"case_id": "case-5-0"}, {"case_id": "reserve"}]}
    )
    result = resolve_selection(cases, plan, {"case-4-0": "reserve", "case-5-0": "reserve"})
    assert any("40 different" in e for e in result["errors"])
    plan["fixed_case_ids"].append("case-4-0")
    result = resolve_selection(cases, plan, {"case-4-0": "reserve"})
    assert any("approved" in e for e in result["errors"])


def test_original_ranks_and_scores_cannot_be_rewritten():
    import pytest

    cases, _ = fixture()
    inventory = {"cases": deepcopy(cases)}
    verify_frozen_cases(cases, inventory)
    for mutation in ("score", "order", "source", "reference"):
        changed = deepcopy(cases)
        if mutation == "score":
            changed[0]["candidates"][0]["score"] += 0.01
        if mutation == "order":
            changed[0]["candidates"].reverse()
        if mutation == "source":
            changed[0]["source"] = "other"
        if mutation == "reference":
            changed[0]["reference_rank"] = 5
        with pytest.raises(ValueError, match="differs from the frozen"):
            verify_frozen_cases(changed, inventory)


def test_exact_anchor_preserves_scored_features_without_inventing_a_binding():
    row = {
        "Src": "s",
        "Tgt": "t",
        "S_final": 0.8,
        "selected_attributions": {"attributes": {"source": ["fact"]}},
    }
    anchor = {"Src": "s", "Tgt": "t", "S_final": 1.0, "Score": 1.0}
    pair = {
        "records": [row, anchor],
        "record_origins": [[{"role": "row"}], [{"role": "anchor"}]],
        "protected_exact": True,
        "score": 1.0,
    }
    before = deepcopy(pair)
    bound = bind_retained_record(pair)
    assert bound["status"] == "scored_row_plus_exact_anchor"
    assert bound["record"]["selected_attributions"] == row["selected_attributions"]
    assert bound["record"]["S_final"] == 1.0
    assert pair == before
    pair["score"] = 0.9
    assert bind_retained_record(pair)["record"] is None


def test_rerender_preserves_compact_reviewed_claims_and_citations(tmp_path):
    import json
    import re

    from tools.render_final_study_catalogue import render

    fact = {
        "fact_id": "f1",
        "category": "definitions",
        "value": {"lexical_form": "Original definition"},
    }
    profile = {
        "claims": [{"text": "Grounded description", "fact_ids": ["f1"]}],
        "support_status": "supported",
        "citations": [fact],
    }
    data = {
        "entities": {"s": {"prepared_profile": profile}},
        "cases": [{"candidates": [{"prepared_comparison": profile}]}],
        "selection_plan": {"version": "owner-pool-review/1"},
        "limitations": ["Current pool limits"],
        "labels": {"s": "Source"},
    }
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(json.dumps(data))
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(json.dumps({"results": {}, "status": "reviewed"}))
    html = render(catalogue, reviewed_results=reviewed)
    rendered = json.loads(
        re.search(r'<script type="application/json" id="review-data">(.*?)</script>', html, re.S)[1]
    )
    assert rendered["entities"]["s"]["prepared_profile"] == profile
    assert rendered["cases"][0]["candidates"][0]["prepared_comparison"] == profile
    assert rendered["labels"]["s"] == "Source"
    assert rendered["limitations"] == ["Current pool limits"]
