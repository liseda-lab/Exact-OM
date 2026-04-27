from exact.utils.candidate_generation import (
    lexical_candidate_pair_scores,
    make_candidate_labels,
    rank_channel_scores,
)


def test_lexical_candidate_scores_use_all_labels_and_char_token_overlap():
    src_records = make_candidate_labels(["s1"], {"s1": ["renal carcinoma"]})
    tgt_records = make_candidate_labels(
        ["t_good", "t_bad"],
        {
            "t_good": ["renal cell carcinoma"],
            "t_bad": ["heart failure"],
        },
    )

    scores = lexical_candidate_pair_scores(src_records, tgt_records, per_source_limit=5)

    assert scores[("s1", "t_good")] > 0.7
    assert ("s1", "t_bad") not in scores


def test_rank_channel_scores_unions_channels_and_respects_top_k():
    rows = rank_channel_scores(
        sources=["s1"],
        semantic_scores={("s1", "t_semantic"): 0.65},
        lexical_scores={("s1", "t_lexical"): 0.82, ("s1", "t_semantic"): 0.40},
        top_k=1,
    )

    assert len(rows) == 1
    assert rows[0]["Tgt"] == "t_lexical"
    assert rows[0]["cand_sim"] == 0.82
    assert rows[0]["cand_channels"] == "lexical"
