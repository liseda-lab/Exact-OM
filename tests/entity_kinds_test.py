from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from exact.core.entities.kinds import EntityKind
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.impl.datasets import base as base_module
from exact.impl.datasets.base import BaseAlignmentDataset
from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
from exact.impl.evaluators.builtin import BuiltinEvaluator
from exact.impl.models.selector.grouping import count_source_groups
from exact.impl.models.selector.selector import CandidateSetSelector
from exact.io.sources.csv_kg import CsvKgSource
from exact.ontology import load_ontology
from exact.runs.layout import RunLayout
from exact.utils.candidate_generation import (
    lexical_candidate_pair_scores,
    make_candidate_labels,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"
KG_FIXTURE = Path(__file__).parent / "fixtures" / "kg_csv"
SOURCE_PATH = FIXTURES / "mini_src.owl"
TARGET_PATH = FIXTURES / "mini_tgt.owl"
SRC = "http://example.org/mini/src#"
TGT = "http://example.org/mini/tgt#"
PART_OF = "http://purl.obolibrary.org/obo/BFO_0000050"


class _FakeSentenceTransformer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class KindDataset(BaseAlignmentDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.semantic_index_calls: list[
            tuple[set[EntityKind], int, set[EntityKind], int]
        ] = []
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        raise IndexError(idx)

    def __len__(self) -> int:
        return 0 if self.dataframe is None else len(self.dataframe)

    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def plot_feature_distributions(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_sanity_examples(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _semantic_label_pair_scores(self, src_records, tgt_records, **kwargs):
        self.semantic_index_calls.append(
            (
                {record.kind for record in src_records},
                len(src_records),
                {record.kind for record in tgt_records},
                len(tgt_records),
            )
        )
        return {
            (source.iri, target.iri): (
                1.0 if source.normalized == target.normalized else 0.1
            )
            for source in src_records
            for target in tgt_records
        }


def _dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    entity_kinds=None,
    name: str = "dataset",
) -> KindDataset:
    monkeypatch.setattr(base_module, "SentenceTransformer", _FakeSentenceTransformer)
    dataset = KindDataset(
        output_path=tmp_path / name,
        entity_kinds=entity_kinds,
        filter_exact_matches=False,
    )
    dataset.load_ontologies(SOURCE_PATH, TARGET_PATH)
    return dataset


def _generate(dataset: KindDataset) -> pd.DataFrame:
    dataset.generate_candidates(
        top_k=2,
        lexical_encoder_name="fixture-encoder",
        retrieval_strategy="hybrid",
        device=torch.device("cpu"),
        use_amp=False,
    )
    return dataset.candidates.copy()


def test_class_only_default_is_identical_to_explicit_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    implicit = _generate(_dataset(tmp_path, monkeypatch, name="implicit"))
    explicit = _generate(
        _dataset(
            tmp_path,
            monkeypatch,
            entity_kinds=[EntityKind.CLASS],
            name="explicit",
        )
    )

    assert implicit.to_csv(index=False) == explicit.to_csv(index=False)
    assert set(implicit["SrcKind"]) == {EntityKind.CLASS.value}
    assert set(implicit["TgtKind"]) == {EntityKind.CLASS.value}


def test_candidate_labels_and_dataset_indexes_are_partitioned_by_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class_records = make_candidate_labels(
        ["same-source"],
        {"same-source": ["same label"]},
        kind=EntityKind.CLASS,
    )
    property_records = make_candidate_labels(
        ["same-target"],
        {"same-target": ["same label"]},
        kind=EntityKind.OBJECT_PROPERTY,
    )
    assert (
        lexical_candidate_pair_scores(
            class_records, property_records, per_source_limit=5
        )
        == {}
    )

    kinds = [
        EntityKind.CLASS,
        EntityKind.OBJECT_PROPERTY,
        EntityKind.DATA_PROPERTY,
        EntityKind.INDIVIDUAL,
    ]
    dataset = _dataset(
        tmp_path,
        monkeypatch,
        entity_kinds=kinds,
        name="mixed",
    )
    candidates = _generate(dataset)

    assert (candidates["SrcKind"] == candidates["TgtKind"]).all()
    assert set(candidates["SrcKind"]) == {kind.value for kind in kinds}
    assert all(
        src_kinds == {kind} and tgt_kinds == {kind}
        for kind, (src_kinds, _, tgt_kinds, _) in zip(
            kinds, dataset.semantic_index_calls
        )
    )
    assert dataset.candidate_pool_sizes["object_property"]["source_entities"] == 4
    assert dataset.candidate_pool_sizes["data_property"]["source_entities"] == 1
    assert dataset.candidate_pool_sizes["individual"]["source_entities"] == 3


def test_property_features_use_subproperty_domain_and_range(tmp_path: Path) -> None:
    dataset = PairAdaptiveContextDataset(
        output_path=tmp_path,
        entity_kinds=[EntityKind.OBJECT_PROPERTY],
        only_taxonomy=True,
        verbaliser_name=None,
    )
    dataset.load_ontologies(SOURCE_PATH, TARGET_PATH)

    hierarchy = dataset.get_entity_features(
        SRC + "participatesIn", "src", EntityKind.OBJECT_PROPERTY
    )["hierarchy"]["is_a"]
    assert hierarchy[0]["object_iri"] == SRC + "relatedTo"
    assert hierarchy[0]["triple"][1] == "subPropertyOf"

    schema = dataset.get_entity_features(PART_OF, "src", EntityKind.OBJECT_PROPERTY)[
        "object_triples"
    ]
    schema_relations = {item["triple"][1] for item in schema}
    assert {"domain", "range"} <= schema_relations


def test_individual_features_use_type_closure_abox_and_data_values(
    tmp_path: Path,
) -> None:
    dataset = PairAdaptiveContextDataset(
        output_path=tmp_path,
        entity_kinds=[EntityKind.INDIVIDUAL],
        only_taxonomy=True,
        verbaliser_name=None,
    )
    dataset.load_ontologies(SOURCE_PATH, TARGET_PATH)

    features = dataset.get_entity_features(SRC + "alice", "src", EntityKind.INDIVIDUAL)
    hierarchy = features["hierarchy"]["is_a"]
    assert any(item["object_iri"] == SRC + "Patient" for item in hierarchy)
    assert any(
        item["object_iri"] == SRC + "Person" and item["type_closure"]
        for item in hierarchy
    )
    assert any(
        item["subject_iri"] == SRC + "alice" and item["object_iri"] == SRC + "trial1"
        for item in features["object_triples"]
    )
    assert any(item["value"] == "P-001" for item in features["attributes"])
    assert features["kind"] == EntityKind.INDIVIDUAL.value


def test_entity_kind_change_invalidates_dataset_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    classes = _dataset(
        tmp_path,
        monkeypatch,
        entity_kinds=[EntityKind.CLASS],
        name="class-cache",
    )
    properties = _dataset(
        tmp_path,
        monkeypatch,
        entity_kinds=[EntityKind.OBJECT_PROPERTY],
        name="property-cache",
    )

    assert classes.cache_fingerprint != properties.cache_fingerprint
    assert classes._cache_fingerprint_payload()["entity_kinds"] == ["class"]
    assert properties._cache_fingerprint_payload()["entity_kinds"] == [
        "object_property"
    ]


def test_base_dataset_uses_run_layout_and_source_registry(tmp_path: Path) -> None:
    run_dir = tmp_path / "csv-kg-run"
    dataset = KindDataset(
        output_path=run_dir,
        input_format="csv-kg",
        source_options={"hierarchy_relations": ["subclass_of"]},
        target_options={"hierarchy_relations": ["subclass_of"]},
        filter_exact_matches=False,
    )
    dataset.load_ontologies(KG_FIXTURE, KG_FIXTURE)

    assert isinstance(dataset.source, CsvKgSource)
    assert isinstance(dataset.target, CsvKgSource)
    assert dataset.output_path == RunLayout.open(run_dir).dataset_dir
    payload = dataset._cache_fingerprint_payload()
    assert payload["input_format"] == "csv-kg"
    assert payload["source_options"] == {"hierarchy_relations": ["subclass_of"]}
    assert payload["target_options"] == {"hierarchy_relations": ["subclass_of"]}


def test_selector_keeps_same_iri_pools_separate_across_kinds() -> None:
    rows = []
    for kind, prefix in (("class", "c"), ("object_property", "p")):
        for target, score in ((f"{prefix}-good", 0.9), (f"{prefix}-bad", 0.2)):
            rows.append(
                {
                    "Src": "shared-source",
                    "Tgt": target,
                    "SrcKind": kind,
                    "TgtKind": kind,
                    "S_final": score,
                    "s_label": score,
                    "S_struct": score,
                    "s_hier": score,
                    "s_sim": score,
                    "s_attr": score,
                    "s_diff": 0.9,
                    "cand_sim_prob": score,
                    "src_obj_ic_mean": 0.5,
                    "tgt_obj_ic_mean": 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    assert count_source_groups(frame) == 2

    selector = CandidateSetSelector(
        enabled=True,
        use_no_match=False,
        llm={"enabled": False},
    )
    selected = selector.forward(frame)["candidate_df"]
    winners = selected.groupby("SrcKind")["selection_winner"].sum().to_dict()
    assert winners == {"class": 1, "object_property": 1}


def test_builtin_evaluation_filters_and_reports_per_kind() -> None:
    source = load_ontology(SOURCE_PATH)
    target = load_ontology(TARGET_PATH)
    pairs = [
        (SRC + "Heart", TGT + "CardiacOrgan"),
        (SRC + "participatesIn", TGT + "enrolledIn"),
        (SRC + "alice", TGT + "bob"),
    ]
    predictions = [EntityMapping(src, tgt, "=", 0.9) for src, tgt in pairs]
    references = [ReferenceMapping(src, tgt, "=") for src, tgt in pairs]

    with pytest.warns(UserWarning) as caught:
        property_only = BuiltinEvaluator.global_eval(
            predictions,
            references,
            source_ontology=source,
            target_ontology=target,
            entity_kinds=[EntityKind.OBJECT_PROPERTY],
        )
    assert any("Filtered 2 reference" in str(item.message) for item in caught)
    assert property_only == {"P": 1.0, "R": 1.0, "F1": 1.0}

    mixed = BuiltinEvaluator.global_eval(
        predictions,
        references,
        source_ontology=source,
        target_ontology=target,
        entity_kinds=[
            EntityKind.CLASS,
            EntityKind.OBJECT_PROPERTY,
            EntityKind.INDIVIDUAL,
        ],
    )
    assert mixed["F1"] == 1.0
    assert mixed["class.F1"] == 1.0
    assert mixed["object_property.F1"] == 1.0
    assert mixed["individual.F1"] == 1.0

    local = []
    for kind, index in ((EntityKind.CLASS, 0), (EntityKind.OBJECT_PROPERTY, 1)):
        src, tgt = pairs[index]
        reference = ReferenceMapping(src, tgt, "=", src_kind=kind)
        candidates = [
            EntityMapping(src, "wrong", "=", 0.1, src_kind=kind),
            EntityMapping(src, tgt, "=", 0.9, src_kind=kind),
        ]
        local.append((reference, candidates))
    local_metrics = BuiltinEvaluator.local_eval(
        local,
        K=[1],
        entity_kinds=[EntityKind.CLASS, EntityKind.OBJECT_PROPERTY],
    )
    assert local_metrics["MRR"] == 1.0
    assert local_metrics["class.MRR"] == 1.0
    assert local_metrics["object_property.MRR"] == 1.0
