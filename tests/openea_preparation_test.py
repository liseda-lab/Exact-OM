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


def _pools(package, tmp_path, monkeypatch):
    from pathlib import Path
    from exact.experiments import openea_pools

    prepared = tmp_path / "prepared"
    openea.prepare_openea(package, prepared, training_cap=2, source_cap=1)
    calls = []

    def generate(config, case, sources, output, *, cap, seed, device):
        assert set(case) == {"source", "target"}  # No reference reaches retrieval.
        assert seed == 17 and device == "cpu"
        calls.append((list(sources), cap, config.candidates.model_dump(mode="json")))
        if "s1" in sources:
            frame = pd.DataFrame(
                {"Src": ["s1", "s1", "s1"], "Tgt": ["t1", "t2", "t3"], "cand_sim": [0.9, 0.8, 0.7]}
            )
        else:
            # Deliberately absent gold t3: preparation must never insert it.
            frame = pd.DataFrame({"Src": ["s3"], "Tgt": ["t4"], "cand_sim": [0.6]})
        return frame, sorted(sources), {"origin": "generated", "fixture": True}

    monkeypatch.setattr(openea_pools, "_generate", generate)
    config = Path(__file__).resolve().parents[1] / "exact/default_config.yaml"
    destination = tmp_path / "pools"
    fragment = openea_pools.prepare_pools(config, prepared, destination, device="cpu")
    return config, prepared, destination, fragment, calls


def test_own_pools_keep_retrieval_scores_split_boundaries_and_resume(
    package, tmp_path, monkeypatch
):
    from exact.experiments import openea_pools

    config, prepared, destination, fragment, calls = _pools(package, tmp_path, monkeypatch)
    assert len(calls) == 2
    train = pd.read_csv(destination / "train.candidates.tsv", sep="\t")
    valid = pd.read_csv(destination / "valid.candidates.tsv", sep="\t")
    assert train.cand_sim.tolist() == [0.9, 0.8, 0.7]
    assert train.confirmed_label.iloc[:2].tolist() == [1, 0]
    assert pd.isna(train.confirmed_label.iloc[2])
    assert valid.Tgt.tolist() == ["t4"] and "confirmed_label" not in valid
    proof = openea_pools.validate_pool_bindings(fragment)
    assert proof["roles"]["train"]["confirmed_negative"] == 1
    assert proof["roles"]["train"]["unknown"] == 1
    assert proof["roles"]["valid"]["gold_insertions"] == 0
    assert proof["generator"]["candidates"] == calls[0][2] == calls[1][2]
    assert openea_pools.prepare_pools(config, prepared, destination, device="cpu") == fragment
    assert len(calls) == 2
    (destination / "train.candidates.tsv").write_text("tampered")
    with pytest.raises(ValueError, match="pool input changed"):
        openea_pools.prepare_pools(config, prepared, destination, device="cpu")


def test_own_pool_provenance_rejects_transplanted_case_or_missing_proof(
    package, tmp_path, monkeypatch
):
    from copy import deepcopy
    from exact.experiments import openea_pools

    _, _, _, fragment, _ = _pools(package, tmp_path, monkeypatch)
    changed = deepcopy(fragment)
    changed["cases"][openea.CASE_ID]["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="KG identity changed"):
        openea_pools.validate_pool_bindings(changed)
    del fragment["e23_pool_provenance"]
    with pytest.raises(ValueError, match="lack retrieval provenance"):
        openea_pools.validate_pool_bindings(fragment)


def test_campaign_openea_is_separate_and_requires_own_pools(package, tmp_path):
    from tests.campaign_preparation_test import _cases
    from tests.graph_campaign_cases_test import prepare

    prepared = tmp_path / "inputs"
    openea.prepare_openea(package, prepared, training_cap=2, source_cap=1)
    fragment = json.loads((prepared / "bindings-fragment.json").read_text())
    cases = _cases()
    cases.update(fragment.pop("cases"))
    steps = prepare(tmp_path / "campaign", cases, **fragment)
    assert steps["E23"].case == openea.CASE_ID
    assert steps["E12"].case == "K0"
    assert steps["E23"].requires == ["E00"] and steps["E23"].inherits == []
    assert [arm.id for arm in steps["E23"].arms] == [
        "natural_graph_off",
        "natural_inductive",
        "graph_only",
        "graph_shuffled",
    ]
    assert all(
        stage.status == "blocked_input_resolution"
        for stages in steps["E23"].readiness.values()
        for stage in stages.values()
    )
    assert all(steps[f"E23-rich-{fraction}"].case == "D1" for fraction in (0, 50, 100))


def test_campaign_bound_pools_reject_stale_caps_and_inheritance(package, tmp_path, monkeypatch):
    from tests.campaign_preparation_test import _cases
    from tests.graph_campaign_cases_test import prepare

    _, _, _, fragment, _ = _pools(package, tmp_path, monkeypatch)
    cases = _cases()
    cases.update(fragment.pop("cases"))
    with pytest.raises(ValueError, match="caps/seed differ"):
        prepare(tmp_path / "wrong-cap", cases, **fragment)
    override = {"E23": {"source_cap": 1, "training_source_cap": 2}}
    steps = prepare(tmp_path / "matched", cases, steps=override, **fragment)
    assert steps["E23"].case == openea.CASE_ID
    assert all(
        stage.status == "implementing"
        for stages in steps["E23"].readiness.values()
        for stage in stages.values()
    )
    override["E23"]["inherits"] = ["instance_pool_freeze"]
    with pytest.raises(ValueError, match="another case's pool"):
        prepare(tmp_path / "wrong-inheritance", cases, steps=override, **fragment)


def test_source_sampling_preserves_legacy_default_and_individual_identity():
    import hashlib
    from exact.experiments.inputs import nested_sources

    sources = [f"entity{i}" for i in range(50)]
    expected = sorted(
        sources, key=lambda iri: (hashlib.sha256(f"17\x1f{iri}\x1fclass".encode()).digest(), iri)
    )[:10]
    assert nested_sources(sources, 10) == expected
    individual = sorted(
        sources,
        key=lambda iri: (hashlib.sha256(f"17\x1f{iri}\x1findividual".encode()).digest(), iri),
    )[:10]
    assert nested_sources(sources, 10, kind="individual") == individual
    assert individual != expected


def test_pool_generation_uses_normal_dataset_route_without_references(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from exact.core.entities.registry import ComponentRegistry
    from exact.experiments.openea_pools import _generate

    calls = []
    candidates = {"top_k": 20, "retrieval_strategy": "hybrid"}

    class Dataset:
        eligible_source_iris = ("s1",)
        candidate_pool_manifest = {"origin": "generated"}

        def __init__(self, **kwargs):
            assert kwargs["entity_kinds"] == ["individual"]
            assert kwargs["input_format"] == "csv-kg"
            assert kwargs["candidate_generation_params"] == candidates
            assert not any("reference" in key for key in kwargs)
            calls.append("init")

        def load_ontologies(self, source, target):
            assert source == tmp_path / "source" and target == tmp_path / "target"
            calls.append("ontologies")

        def freeze_source_universe(self, sources, *, cap, seed):
            assert sources == ["s1"] and cap == 1 and seed == 17
            calls.append("source_cap")

        def load_candidates(self, path, **kwargs):
            assert path is None
            assert kwargs["top_k"] == 20 and kwargs["retrieval_strategy"] == "hybrid"
            calls.append("retrieval")

        def candidate_recall_frames(self):
            return pd.DataFrame({"Src": ["s1"], "Tgt": ["t1"], "cand_sim": [0.8]}), None

    config = SimpleNamespace(
        resolve_dependencies=lambda: calls.append("resolve"),
        dataset_runtime=Dataset,
        dataset_params=SimpleNamespace(model_dump=lambda **_: {}),
        candidates=SimpleNamespace(model_dump=lambda **_: candidates),
        matching=SimpleNamespace(anchor_rescoring=SimpleNamespace(exact_policy=None)),
    )
    monkeypatch.setattr(ComponentRegistry, "get", lambda *_: lambda seed: calls.append(seed))
    frame, sources, manifest = _generate(
        config,
        {
            "source": {"path": str(tmp_path / "source")},
            "target": {"path": str(tmp_path / "target")},
        },
        ["s1"],
        tmp_path / "out",
        cap=1,
        seed=17,
        device="cpu",
    )
    assert calls == ["resolve", 17, "init", "ontologies", "source_cap", "retrieval"]
    assert sources == ["s1"] and manifest == {"origin": "generated"}
    assert frame.cand_sim.tolist() == [0.8]


def test_pool_cli_merges_only_case_fragment_before_campaign_preparation(tmp_path, monkeypatch):
    import sys
    from tools import prepare_openea_pools as cli

    base = tmp_path / "base.json"
    base.write_text(json.dumps({"cases": {"K0": {"task": "original"}}, "untouched": True}))
    fragment = {"cases": {openea.CASE_ID: {"task": "new"}}, "e23_natural_case": openea.CASE_ID}
    monkeypatch.setattr(cli, "prepare_pools", lambda *_, **__: fragment)
    seen = []
    monkeypatch.setattr(cli, "prepare_campaign", lambda *args: seen.append(args) or "campaign")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_openea_pools.py",
            "--config",
            str(tmp_path / "config.yaml"),
            "--prepared",
            str(tmp_path / "prepared"),
            "--output",
            str(tmp_path / "pools"),
            "--campaign-bindings",
            str(base),
            "--campaign-output",
            str(tmp_path / "campaign"),
        ],
    )
    cli.main()
    merged = json.loads((tmp_path / "pools/campaign-bindings.json").read_text())
    assert merged["cases"]["K0"] == {"task": "original"}
    assert merged["cases"][openea.CASE_ID] == {"task": "new"}
    assert merged["untouched"] is True and seen[0][2] == merged
