"""Map verified Exact 2 local-ranking records onto the exploration trace contract."""

from exact_inspect.models import PairResponse

MEANINGS = {
    "S_final": "Recorded final ranking score within the original 100-candidate query.",
    "S_base": "Native pair score before the language-model signal is applied.",
    "S_pair_final": "Pair score after applying any admitted language-model signal; before query ranking.",
    "S_struct": "Native structural evidence score.",
    "p_llm": "Recorded local language-model signal; not calibrated semantic equivalence.",
    "rank_utility": "Recorded query-ranker utility before conversion to the final ranking score.",
    "s_label": "Recorded label-channel similarity.",
    "s_hier": "Recorded hierarchy-channel support.",
    "s_attr": "Recorded auxiliary attribute-channel support.",
    "s_sim": "Recorded relational similarity-channel support.",
    "s_diff": "Recorded difference-channel compatibility.",
    "s_strsim": "Recorded structural similarity-channel score, which may have zero fusion weight.",
}


def score(name, value, stage):
    return dict(
        name=name,
        value=value,
        stage=stage,
        meaning=MEANINGS[name],
        calibration_status="not_established",
    )


def recovered_pair(recovery, case, candidate, run_id, metadata):
    """Fail closed on mismatched query, pair, score or rank; never infer accepted membership."""
    selected = recovery["cases"][case["case_id"]]
    saved = next(
        c for c in selected["candidates"] if c["candidate_id"] == candidate["candidate_id"]
    )
    if (
        recovery["mode"] != "local_ranking"
        or selected["source"] != case["source"]["iri"]
        or saved["target"] != candidate["entity"]["iri"]
        or saved["rank"] != candidate["display_position"]
        or saved["final_score"] != candidate["score"]
        or selected["candidate_count"] != 100
    ):
        raise ValueError("Recovered decision does not match the frozen study candidate")
    refs = saved["artifact_refs"]
    worker = "sha256:" + recovery["worker_sha256"]
    submission = "sha256:" + recovery["submission_sha256"]
    native, applied = saved["native_scores"], saved["applied"]
    final = score("S_final", saved["final_score"], "selection")
    scores = [final]
    native_scores = [
        score(k, native[k], "pair_scoring")
        for k in (
            "S_base",
            "S_struct",
            "s_label",
            "s_hier",
            "s_attr",
            "s_sim",
            "s_diff",
            "s_strsim",
        )
        if k in native
    ]
    scores.extend(native_scores)
    signal_scores = []
    if applied:
        if applied["p_llm"] is not None:
            signal_scores.append(score("p_llm", applied["p_llm"], "llm_signal"))
        signal_scores.append(score("S_pair_final", applied["S_pair_final"], "llm_signal"))
    scores.extend(signal_scores)
    ranking_scores = [final]
    if saved["utility"] is not None:
        utility = score("rank_utility", saved["utility"], "selection")
        scores.append(utility)
        ranking_scores.append(utility)

    def event(stage, status, outcome, reason, values=(), extra_refs=()):
        return dict(
            stage=stage,
            status=status,
            outcome=outcome,
            reason_code=reason,
            scores=list(values),
            competitor_pair_ids=[],
            artifact_refs=list(dict.fromkeys([*refs, worker, *extra_refs])),
            implementation_id="exact2-local-ranking-recovery/1",
            provenance_status="derived_from_saved_artifacts",
        )

    events = [
        event("retrieval", "completed", "retained", "provided_candidate_pool"),
        event(
            "prefilter",
            "completed",
            "retained",
            "exact_match_protected" if saved["protected_exact"] else "prefilter_retained",
        ),
    ]
    events.append(
        event(
            "pair_scoring",
            "completed" if native else "not_run",
            "retained" if native else "not_applicable",
            "native_pair_scores_recorded" if native else "local_exact_anchor_no_pair_score",
            native_scores[:2],
        )
    )
    if not applied:
        events.append(
            event("llm_signal", "not_run", "not_applicable", "local_exact_anchor_not_fused")
        )
    elif applied["cached_response_reused"]:
        events.append(
            event(
                "llm_signal",
                "completed",
                "retained",
                (
                    "local_model_signal_applied"
                    if applied["effective_pair_score_contribution"]
                    else "local_model_signal_no_score_change"
                ),
                signal_scores,
            )
        )
    else:
        events.append(
            event(
                "llm_signal",
                "not_run",
                "not_applicable",
                "local_model_signal_not_applied",
                signal_scores,
            )
        )
    events.extend(
        [
            event("selection", "completed", "retained", "local_query_ranked", ranking_scores),
            event("threshold", "not_run", "not_applicable", "local_ranking_no_threshold"),
            event("cardinality", "not_run", "not_applicable", "local_ranking_no_cardinality"),
            event(
                "extraction",
                "completed",
                "retained",
                "ranked_submission_written",
                extra_refs=[submission],
            ),
            event("relation_typing", "not_run", "not_applicable", "local_ranking_no_relation"),
            event("repair", "not_run", "not_applicable", "local_ranking_no_repair"),
        ]
    )
    ranking = dict(
        mode="local_ranking",
        query_id=selected["query_id"],
        candidate_count=100,
        rank=saved["rank"],
        saved_output_member=True,
        artifact_ref=submission,
    )
    trace = PairResponse.model_validate(
        dict(
            artifact_type="pair_decision_trace",
            contract_version="exact-explain/1.0",
            run_id=run_id,
            pair_id=candidate["candidate_id"],
            source=case["source"],
            target=candidate["entity"],
            events=events,
            saved_alignment_member=None,
            relation=None,
            relation_basis="unavailable",
            assumption_refs=[],
            evidence_metadata=metadata,
            fixture_provenance="derived_from_saved_artifacts",
            ranking=ranking,
        )
    ).model_dump(mode="json")
    return scores, trace
