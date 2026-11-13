from copy import deepcopy

from tools.audit_final_study_candidates import label_key, lexical_screen


def case(source, targets, rank=1):
    return {
        "source_summary": {"label": source, "counts": {"definitions": 1}},
        "reference_rank": rank,
        "candidates": [
            {"score": 1 - i / 10, "context_summary": {"label": label, "counts": {"definitions": 1}}}
            for i, label in enumerate(targets)
        ],
    }


def test_disease_disorder_is_only_a_private_screen_not_a_rewrite():
    value = case(
        "Lung Disorder",
        ["lung disease", "lung cancer", "airway disease", "pneumonia", "pulmonary artery disease"],
    )
    original = deepcopy(value)
    assert lexical_screen(value)["obvious_reference_label_twin"]
    assert value == original


def test_label_matching_preserves_distinguishing_qualifiers():
    for a, b in [
        ("type 1", "type 2"),
        ("BCR positive", "BCR negative"),
        ("with anosmia", "without anosmia"),
        ("autosomal dominant", "autosomal recessive"),
        ("acute leukemia", "chronic leukemia"),
    ]:
        assert label_key(a) != label_key(b)


def test_word_order_does_not_disguise_a_label_only_answer():
    value = case(
        "Ovarian Small Cell Carcinoma Pulmonary Type",
        [
            "pulmonary type ovarian small cell carcinoma",
            "ovarian small cell carcinoma",
            "hypercalcemic ovarian carcinoma",
            "ovarian cancer",
            "lung carcinoma",
        ],
    )
    assert lexical_screen(value)["obvious_reference_label_twin"]


def test_same_named_distractor_is_not_mistaken_for_an_easy_reference_answer():
    value = case(
        "Example syndrome",
        [
            "Example syndrome",
            "recessive Example syndrome",
            "dominant Example syndrome",
            "other syndrome",
            "distant condition",
        ],
        rank=2,
    )
    screen = lexical_screen(value)
    assert screen["label_twin_candidate_ranks"] == [1]
    assert not screen["obvious_reference_label_twin"]
    assert screen["manual_semantic_review_required"]
