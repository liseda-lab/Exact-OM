from exact.utils.candidate_generation import (
    candidate_annotation_priority,
    candidate_token_key,
    lexical_candidate_pair_scores,
    make_candidate_labels,
    rank_channel_scores,
    select_candidate_annotation_literals,
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


def test_candidate_token_key_matches_order_and_possessive_variants_only():
    assert candidate_token_key(
        "Clear Cell Renal Cell Carcinoma"
    ) == candidate_token_key("renal clear cell carcinoma")
    assert candidate_token_key("Crohn Disease") == candidate_token_key(
        "Crohn's disease"
    )
    assert candidate_token_key(
        "Congenital Mesoblastic Nephroma"
    ) != candidate_token_key("classic congenital mesoblastic nephroma")


def test_candidate_annotation_priority_keeps_synonyms_and_rejects_noise():
    assert (
        candidate_annotation_priority(
            "Adult Malignant Fibrous Histiocytoma", "http://x#P90"
        )
        is not None
    )
    assert (
        candidate_annotation_priority(
            "Adult Unclassified Pleomorphic Sarcoma", "http://x#P107"
        )
        is not None
    )
    assert (
        candidate_annotation_priority(
            "LPFS1", "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"
        )
        is not None
    )
    assert (
        candidate_annotation_priority(
            "alternate disease name", "http://example.org#altLabel"
        )
        is not None
    )

    assert candidate_annotation_priority("Disease or Syndrome", "http://x#P106") is None
    assert candidate_annotation_priority("C3272295", "http://x#P207") is None
    assert candidate_annotation_priority("Malignant", "http://x#P363") is None
    assert (
        candidate_annotation_priority(
            "disease_ontology",
            "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace",
        )
        is None
    )
    assert (
        candidate_annotation_priority(
            "DOID:1234", "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId"
        )
        is None
    )
    assert (
        candidate_annotation_priority(
            "This is a comment", "http://www.w3.org/2000/01/rdf-schema#comment"
        )
        is None
    )
    assert (
        candidate_annotation_priority(
            "http://example.org/noise", "http://example.org#hasExactSynonym"
        )
        is None
    )
    assert (
        candidate_annotation_priority(
            "A rare disorder with a long explanatory sentence that describes symptoms rather than naming the entity",
            "http://example.org#hasExactSynonym",
        )
        is None
    )


def test_select_candidate_annotation_literals_applies_caps_and_existing_labels():
    annotations = [
        ("http://x#P108", "Primary Label"),
        ("http://x#P97", "A definition should not be selected."),
        (
            "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace",
            "disease_ontology",
        ),
    ]
    annotations.extend(("http://x#P90", f"Alias {idx}") for idx in range(10))
    annotations.extend(
        (
            "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
            f"Exact Synonym {idx}",
        )
        for idx in range(3)
    )

    selected = select_candidate_annotation_literals(
        annotations,
        seen_normalized={"primarylabel"},
        overall_cap=20,
    )

    assert "Primary Label" not in selected
    assert "A definition should not be selected." not in selected
    assert "disease_ontology" not in selected
    assert len([value for value in selected if value.startswith("Alias ")]) == 8
    assert len([value for value in selected if value.startswith("Exact Synonym ")]) == 3


def test_candidate_fusion_and_alias_overrides_change_runtime_selection() -> None:
    source = make_candidate_labels(["s"], {"s": ["renal carcinoma"]})
    targets = make_candidate_labels(["t"], {"t": ["renal cell carcinoma"]})
    assert lexical_candidate_pair_scores(source, targets, per_source_limit=1)
    assert (
        lexical_candidate_pair_scores(
            source,
            targets,
            per_source_limit=1,
            fusion_config={
                "token_weight": 0.0,
                "gram_weight": 0.0,
                "blend_token_weight": 0.0,
                "blend_gram_weight": 0.0,
            },
        )
        == {}
    )

    rows = rank_channel_scores(
        sources=["s"],
        semantic_scores={("s", "semantic"): 0.6},
        lexical_scores={("s", "lexical"): 0.9},
        top_k=1,
        fusion_config={"semantic_channel_weight": 1.0, "lexical_channel_weight": 0.0},
    )
    assert rows[0]["Tgt"] == "semantic"

    selected = select_candidate_annotation_literals(
        [
            ("http://x#P90", "one"),
            ("http://x#P90", "two"),
            ("http://x#P90", "two words"),
        ],
        overall_cap=10,
        alias_config={"exact_property_cap": 1, "max_tokens": 1},
    )
    assert len(selected) == 1
    assert selected[0] in {"one", "two"}
