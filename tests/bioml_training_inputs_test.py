import pandas as pd
import pytest

from exact.experiments.inputs import prepare_confirmed_bioml_training
from exact.impl.models.selector.fitting import safe_training_labels


def test_official_train_distractors_union_positives_and_preserve_unknown_scope(tmp_path):
    source = tmp_path / "local.train.cands.tsv"
    pd.DataFrame(
        [
            ("s", "a", repr(["a", "b", "negative"])),
            ("s", "b", repr(["b", "negative2"])),
            ("other", "c", repr(["c", "negative"])),
        ],
        columns=["SrcEntity", "TgtEntity", "TgtCandidates"],
    ).to_csv(source, sep="\t", index=False)
    record = prepare_confirmed_bioml_training(
        source, tmp_path / "out", dataset_revision="c" * 40, reporting_sources=["heldout"]
    )
    assert record["negative_policy"] == "confirmed_only"
    assert record["reference_completeness"] == "known_incomplete"
    assert record["negative_scope"] == "provided_training_candidate_pairs_only"
    assert record["positive_pairs"] == 3
    assert record["confirmed_negative_pairs"] == 3
    frame = pd.read_csv(record["outputs"]["candidates"]["path"], sep="\t")
    labeled, reference = safe_training_labels(
        frame, {("s", "a")}, {"negative_label_policy": "confirmed_negatives"}
    )
    assert {("s", "b"), ("other", "c")} <= reference
    assert labeled.set_index(["Src", "Tgt"]).loc[("s", "b"), "confirmed_label"] == 1
    assert not ((frame.Src == "s") & (frame.Tgt == "unobserved")).any()
    reference_frame = pd.read_csv(record["outputs"]["reference"]["path"], sep="\t")
    assert set(reference_frame.Relation) == {"="}
    assert len(reference_frame) == 3
    assert (
        prepare_confirmed_bioml_training(
            source, tmp_path / "out", dataset_revision="c" * 40, reporting_sources=["heldout"]
        )
        == record
    )
    with pytest.raises(ValueError, match="overlap"):
        prepare_confirmed_bioml_training(
            source, tmp_path / "bad", dataset_revision="c" * 40, reporting_sources=["s"]
        )
    with pytest.raises(ValueError, match="only local.train"):
        prepare_confirmed_bioml_training(
            tmp_path / "local.test.cands.tsv", tmp_path / "bad", dataset_revision="c" * 40
        )


def test_converter_requires_gold_in_pool_and_frozen_revision(tmp_path):
    source = tmp_path / "local.train.cands.tsv"
    pd.DataFrame(
        [("s", "gold", repr(["other"]))], columns=["SrcEntity", "TgtEntity", "TgtCandidates"]
    ).to_csv(source, sep="\t", index=False)
    with pytest.raises(ValueError, match="immutable official dataset revision"):
        prepare_confirmed_bioml_training(source, tmp_path / "out", dataset_revision="main")
    with pytest.raises(ValueError, match="inside its candidate pool"):
        prepare_confirmed_bioml_training(source, tmp_path / "out", dataset_revision="c" * 40)
