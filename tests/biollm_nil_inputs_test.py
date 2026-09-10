"""BioLLM labels stay within immutable benchmark pools and legacy ontology versions."""

import pandas as pd
import pytest

from exact.experiments.biollm_inputs import INPUTS, prepare_biollm_nil
from exact.utils.provenance import sha256_file


def fixture_inputs(tmp_path, monkeypatch):
    rows = []
    for index in range(100):
        candidates = [f"urn:t:{index}:{item}" for item in range(70 if index == 0 else 100)]
        rows.append(
            (f"urn:s:{index}", candidates[0] if index < 50 else "UnMatched", repr(candidates))
        )
    pool, source, target = [tmp_path / name for name in ["pool.tsv", "source.owl", "target.owl"]]
    pd.DataFrame(rows, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(
        pool, sep="\t", index=False
    )
    source.write_text("source ontology fixture")
    target.write_text("target ontology fixture")
    monkeypatch.setitem(
        INPUTS,
        "ncit-doid",
        {
            **INPUTS["ncit-doid"],
            **{
                f"{name}_sha256": sha256_file(path)
                for name, path in [("pool", pool), ("source", source), ("target", target)]
            },
        },
    )
    monkeypatch.setitem(INPUTS, "snomed-fma.body", dict(INPUTS["ncit-doid"]))
    return pool, source, target


def prepare(paths, destination, role="development"):
    return prepare_biollm_nil(
        *paths,
        destination,
        task="ncit-doid" if role == "development" else "snomed-fma.body",
        role=role,
    )


def read(record, name):
    return pd.read_csv(record["outputs"][name]["path"], sep="\t")


def test_stratified_partitions_preserve_short_pool_and_separate_labels(tmp_path, monkeypatch):
    paths = fixture_inputs(tmp_path, monkeypatch)
    record = prepare(paths, tmp_path / "prepared")
    assert prepare(paths, tmp_path / "prepared") == record
    splits = record["splits"]
    assert {name: len(value["sources"]) for name, value in splits.items()} == {
        "train": 60,
        "valid": 20,
        "internal_check": 20,
    }
    populations = [set(value["sources"]) for value in splits.values()]
    assert len(set.union(*populations)) == sum(map(len, populations)) == 100
    assert sum(value["pairs"] for value in splits.values()) == 9970
    assert record["natural_ontology_nil_claim"] is False
    assert record["unknown_outside_supplied_pool"] is True
    assert record["reference_completeness"] == "known_incomplete"
    for name, split in splits.items():
        per_class = 30 if name == "train" else 10
        assert split["source_status_counts"] == {"in_pool": per_class, "benchmark_nil": per_class}
        candidates, confirmed, labels = [
            read(split, key) for key in ["candidates", "confirmed_candidates", "source_labels"]
        ]
        assert list(candidates) == ["Src", "Tgt"]
        assert confirmed.confirmed_label.sum() == per_class
        nil_sources = set(labels.loc[labels.Status == "benchmark_nil", "Src"])
        assert not confirmed.loc[confirmed.Src.isin(nil_sources), "confirmed_label"].any()
        assert set(zip(candidates.Src, candidates.Tgt)) == set(zip(confirmed.Src, confirmed.Tgt))
        assert set(labels.Status) == {"benchmark_nil", "in_pool"}


def test_reporting_has_no_training_partition_or_ontology_nil_claim(tmp_path, monkeypatch):
    record = prepare(fixture_inputs(tmp_path, monkeypatch), tmp_path / "report", "reporting")
    assert list(record["splits"]) == ["test"]
    split = record["splits"]["test"]
    assert split["exposure"] == "public_heldout_reporting_only"
    assert len(read(split, "reference")) == 50
    assert set(read(split, "source_labels").Status) == {"benchmark_nil", "in_pool"}


def test_legacy_ontology_pin_blocks_label_transplant(tmp_path, monkeypatch):
    paths = fixture_inputs(tmp_path, monkeypatch)
    paths[2].write_text("expanded 2026 ontology")
    with pytest.raises(ValueError, match="target differs.*no label transplant"):
        prepare(paths, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_changed_output_is_not_silently_repaired(tmp_path, monkeypatch):
    paths = fixture_inputs(tmp_path, monkeypatch)
    record = prepare(paths, tmp_path / "prepared")
    from pathlib import Path

    labels = Path(record["splits"]["valid"]["outputs"]["source_labels"]["path"])
    labels.write_text("Src\tStatus\nchanged\tontology_nil\n")
    with pytest.raises(ValueError, match="Immutable BioLLM preparation conflict"):
        prepare(paths, tmp_path / "prepared")
    assert "ontology_nil" in labels.read_text()


def test_source_split_does_not_depend_on_original_row_order(tmp_path, monkeypatch):
    paths = fixture_inputs(tmp_path, monkeypatch)
    first = prepare(paths, tmp_path / "first")
    pd.read_csv(paths[0], sep="\t").iloc[::-1].to_csv(paths[0], sep="\t", index=False)
    monkeypatch.setitem(INPUTS["ncit-doid"], "pool_sha256", sha256_file(paths[0]))
    second = prepare(paths, tmp_path / "second")
    assert {name: value["sources"] for name, value in first["splits"].items()} == {
        name: value["sources"] for name, value in second["splits"].items()
    }


def test_reporting_task_cannot_be_prepared_for_training(tmp_path, monkeypatch):
    paths = fixture_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="reporting data cannot become training"):
        prepare_biollm_nil(*paths, tmp_path / "invalid", task="snomed-fma.body", role="development")
    assert not (tmp_path / "invalid").exists()
