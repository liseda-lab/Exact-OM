"""OpenEA preparation preserves official split boundaries and benchmark-negative scope."""

import csv
import json
import zipfile

import pandas as pd
import pytest

from exact.experiments import openea
from exact.experiments.campaign import CaseBinding
from exact.utils.provenance import sha256_file


@pytest.fixture
def package(tmp_path, monkeypatch):
    path = tmp_path / "fixture.zip"
    label = "http://www.w3.org/2004/02/skos/core#altLabel"
    with zipfile.ZipFile(path, "w") as archive:
        for side, prefix in [(1, "s"), (2, "t")]:
            archive.writestr(
                openea.PREFIX + f"rel_triples_{side}",
                f"{prefix}1\tr\t{prefix}2\n{prefix}2\tr\t{prefix}3\n{prefix}3\tr\t{prefix}4\n",
            )
            archive.writestr(
                openea.PREFIX + f"attr_triples_{side}",
                f"{prefix}1\t{label}\tName one\n{prefix}2\tdate\t2000\n{prefix}4\tempty\t\n",
            )
        archive.writestr(openea.PREFIX + f"{openea.SPLIT}/train_links", "s1\tt1\ns2\tt2\n")
        archive.writestr(openea.PREFIX + f"{openea.SPLIT}/valid_links", "s3\tt3\n")
        archive.writestr(openea.PREFIX + f"{openea.SPLIT}/test_links", "DO NOT READ")
        archive.writestr(openea.PREFIX + "ent_links", "DO NOT READ")
    monkeypatch.setattr(openea, "ARCHIVE_SHA256", sha256_file(path))
    monkeypatch.setattr(openea, "OFFICIAL_COUNTS", (4, 2, 1))
    return path


def test_only_approved_members_opened_and_no_alignment_graph_edges(package, tmp_path, monkeypatch):
    opened = []
    original = zipfile.ZipFile.open

    def guarded(self, name, *args, **kwargs):
        assert name in [openea.PREFIX + item for item in openea.PUBLIC_FILES]
        opened.append(name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded)
    destination = tmp_path / "prepared"
    result = openea.prepare_openea(package, destination, training_cap=2, source_cap=1)
    assert opened == result["opened_archive_members"]
    assert result["heldout_alignment_contents_read"] is False
    assert not list(destination.rglob("test_links"))
    assert not list(destination.rglob("ent_links"))
    fragment = json.loads((destination / "bindings-fragment.json").read_text())
    case = CaseBinding.model_validate(fragment["cases"][openea.CASE_ID])
    assert case.kind == "individual" and case.negative_policy == "confirmed_only"
    assert set(case.references) == {"train", "valid"}
    assert not case.candidates and not case.frozen_global_candidates
    assert case.source.verify(destination) == (destination / "views/source").resolve()
    for side, prefix in [("source", "s"), ("target", "t")]:
        with (destination / "views" / side / "relations.csv").open() as stream:
            edges = list(csv.DictReader(stream))
        assert [(row["src"], row["dst"]) for row in edges] == [
            (prefix + "1", prefix + "2"),
            (prefix + "2", prefix + "3"),
            (prefix + "3", prefix + "4"),
        ]
        descriptor = json.loads((destination / "views" / side / "kg.yaml").read_text())
        assert not descriptor["hierarchy_relations"]
    assert openea.prepare_openea(package, destination, training_cap=2, source_cap=1) == result


def test_csv_source_retains_kinds_attributes_and_observed_labels(package, tmp_path):
    from exact.io.sources.csv_kg import CsvKgSource
    from exact.core.entities.kinds import EntityKind

    destination = tmp_path / "prepared"
    openea.prepare_openea(package, destination, training_cap=2, source_cap=1)
    source = CsvKgSource.from_path(destination / "views/source")
    assert set(source.entities(EntityKind.INDIVIDUAL)) == {"s1", "s2", "s3", "s4"}
    assert not source.entities(EntityKind.CLASS)
    assert "Name one" not in source.entities(EntityKind.INDIVIDUAL)
    assert source.labels("s1") == ["Name one"]
    assert openea.verify_prepared(destination)["graphs"][0]["empty_attributes_omitted"] == 1


def test_training_negatives_are_scoped_without_changing_retrieval(tmp_path):
    links = tmp_path / "train_links"
    links.write_text("s1\tt1\ns2\tt2\n")
    frame = pd.DataFrame(
        {
            "Src": ["s1", "s1", "s1", "validation", "s2"],
            "Tgt": ["t1", "t2", "unknown", "t1", "t1"],
            "cand_sim": [0.1, 0.9, 0.8, 0.7, 0.6],
        },
        index=[8, 7, 6, 5, 4],
    )
    result = openea.label_training_candidates(frame, links, expected_sha256=sha256_file(links))
    pd.testing.assert_frame_equal(result.drop(columns="confirmed_label"), frame)
    assert result.confirmed_label.iloc[[0, 1, 4]].tolist() == [1, 0, 0]
    assert result.confirmed_label.iloc[[2, 3]].isna().all()
    assert "confirmed_label" not in frame
    from exact.impl.models.selector.fitting import safe_training_labels

    fitting, reference = safe_training_labels(
        result,
        {("s1", "t1"), ("s2", "t2")},
        {"negative_label_policy": "confirmed_negatives"},
    )
    assert list(zip(fitting.Src, fitting.Tgt)) == [("s1", "t1"), ("s1", "t2"), ("s2", "t1")]
    assert fitting.confirmed_label.tolist() == [1, 0, 0]
    assert reference == {("s1", "t1"), ("s2", "t2")}


@pytest.mark.parametrize("rows", ["s1\tt1\ns1\tt2\n", "s1\tt1\ns2\tt1\n", "s1\tt1\ns1\tt1\n"])
def test_nonbijective_training_links_cannot_generate_negatives(tmp_path, rows):
    path = tmp_path / "train_links"
    path.write_text(rows)
    with pytest.raises(ValueError, match="bijective"):
        openea.label_training_candidates(
            pd.DataFrame({"Src": ["s1"], "Tgt": ["t2"]}), path, expected_sha256=sha256_file(path)
        )


def test_label_scope_hash_and_published_input_changes_fail_closed(package, tmp_path):
    destination = tmp_path / "prepared"
    openea.prepare_openea(package, destination, training_cap=2, source_cap=1)
    path = destination / "official" / openea.SPLIT / "train_links"
    expected = sha256_file(path)
    path.write_text("s1\tt2\ns2\tt1\n")
    with pytest.raises(ValueError, match="identity changed"):
        openea.label_training_candidates(
            pd.DataFrame({"Src": ["s1"], "Tgt": ["t2"]}), path, expected_sha256=expected
        )
    with pytest.raises(ValueError, match="Prepared OpenEA input changed"):
        openea.verify_prepared(destination)


def test_split_overlap_rejects_publication(package, tmp_path, monkeypatch):
    with zipfile.ZipFile(package) as original:
        members = {
            name: original.read(name)
            for name in original.namelist()
            if name in [openea.PREFIX + item for item in openea.PUBLIC_FILES]
        }
    members[openea.PREFIX + f"{openea.SPLIT}/valid_links"] = b"s3\tt1\n"
    conflicting = tmp_path / "conflicting.zip"
    with zipfile.ZipFile(conflicting, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)
    monkeypatch.setattr(openea, "ARCHIVE_SHA256", sha256_file(conflicting))
    with pytest.raises(ValueError, match="endpoints overlap"):
        openea.prepare_openea(conflicting, tmp_path / "prepared", training_cap=2, source_cap=1)
    assert not (tmp_path / "prepared").exists()


def test_archive_identity_and_recipe_are_immutable(package, tmp_path):
    destination = tmp_path / "prepared"
    openea.prepare_openea(package, destination, training_cap=2, source_cap=1)
    with pytest.raises(ValueError, match="recipe changed"):
        openea.prepare_openea(package, destination, training_cap=2, source_cap=2)
    package.write_bytes(b"different archive")
    with pytest.raises(ValueError, match="approved v2.0"):
        openea.prepare_openea(package, tmp_path / "other", training_cap=2, source_cap=1)
