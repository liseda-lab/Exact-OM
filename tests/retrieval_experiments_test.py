import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from exact.core.entities.kinds import EntityKind
from exact.impl.datasets import base as base_module
from exact.impl.datasets.base import BaseAlignmentDataset
from exact.impl.retrieval import (
    resolve_local_retrieval_artifact,
    retrieval_artifact_requirement,
)
from exact.utils.candidate_generation import (
    adaptive_candidate_count,
    rank_channel_scores,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"
SOURCE_PATH = FIXTURES / "mini_src.owl"
TARGET_PATH = FIXTURES / "mini_tgt.owl"
SRC = "http://example.org/mini/src#"


class _RetrievalDataset(BaseAlignmentDataset):
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
        return {
            (source.iri, target.iri): (1.0 if source.normalized == target.normalized else 0.1)
            for source in src_records
            for target in tgt_records
        }


class _FakeSentenceTransformer:
    loaded_paths: list[str] = []

    def __init__(self, path: str, **kwargs: Any) -> None:
        self.loaded_paths.append(str(path))


class _FakeCrossEncoder:
    loaded_paths: list[str] = []

    def __init__(self, path: str, **kwargs: Any) -> None:
        self.loaded_paths.append(str(path))

    def predict(self, pairs, **kwargs):
        return np.asarray(
            [1.0 if str(left).lower() == str(right).lower() else 0.1 for left, right in pairs],
            dtype=np.float32,
        )


def _artifact(
    root: Path,
    *,
    kind: str,
    negative_policy: str = "complete_reference",
) -> Path:
    root.mkdir()
    model = root / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"synthetic model bytes")
    payload = {
        "schema_version": 1,
        "artifact_type": kind,
        "model_path": "model",
        "base_model": "fixture/base-model",
        "training_pairs": ["fixture/train"],
        "reference_completeness": "complete",
        "negative_policy": negative_policy,
        "dataset_lock": "a" * 64,
        "epochs": 1,
        "seed": 7,
    }
    if kind == "contrastive_encoder":
        payload["mining"] = {
            "candidate_pool_fingerprint": "b" * 64,
            "top_k": 50,
            "max_negatives_per_source": 5,
            "max_training_pairs": 100_000,
            "exclude_known_positives": True,
        }
    (root / "retrieval_artifact.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return root


def _dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    entity_kinds=(EntityKind.CLASS,),
    name: str = "dataset",
    candidate_generation_params=None,
) -> _RetrievalDataset:
    monkeypatch.setattr(base_module, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(base_module, "CrossEncoder", _FakeCrossEncoder)
    dataset = _RetrievalDataset(
        output_path=tmp_path / name,
        entity_kinds=entity_kinds,
        filter_exact_matches=False,
        candidate_generation_params=candidate_generation_params,
    )
    dataset.load_ontologies(SOURCE_PATH, TARGET_PATH)
    return dataset


def test_max_weighted_and_rrf_fusion_are_deterministic() -> None:
    semantic = {("s", "a"): 0.9, ("s", "b"): 0.8}
    lexical = {("s", "a"): 0.1, ("s", "b"): 0.95, ("s", "c"): 0.99}

    default_rows = rank_channel_scores(["s"], semantic, lexical, top_k=3)
    explicit_max = rank_channel_scores(
        ["s"], semantic, lexical, top_k=3, fusion_config={"mode": "max"}
    )
    weighted = rank_channel_scores(
        ["s"],
        semantic,
        lexical,
        top_k=3,
        fusion_config={
            "mode": "weighted",
            "weighted_semantic": 0.9,
            "weighted_lexical": 0.1,
        },
    )
    rrf = rank_channel_scores(
        ["s"],
        semantic,
        lexical,
        top_k=3,
        fusion_config={"mode": "rrf", "rrf_constant": 60.0},
    )

    assert default_rows == explicit_max
    assert [row["Tgt"] for row in default_rows] == ["c", "b", "a"]
    assert [row["Tgt"] for row in weighted] == ["a", "b", "c"]
    assert [row["Tgt"] for row in rrf] == ["a", "b", "c"]
    assert rrf[0]["cand_sim"] == pytest.approx((1.0 / 61.0) + (1.0 / 63.0))


def test_adaptive_gap_and_entropy_are_bounded_and_gold_free() -> None:
    assert (
        adaptive_candidate_count(
            [1.0, 0.91, 0.5, 0.49],
            top_k=20,
            adaptive_config={
                "enabled": True,
                "criterion": "gap",
                "gap": 0.2,
                "k_min": 2,
                "k_max": 4,
            },
        )
        == 2
    )
    assert (
        adaptive_candidate_count(
            [1.0, 0.91, 0.5, 0.49],
            top_k=20,
            adaptive_config={
                "enabled": True,
                "criterion": "gap",
                "gap": 0.5,
                "k_min": 2,
                "k_max": 4,
            },
        )
        == 4
    )
    assert (
        adaptive_candidate_count(
            [1.0, 0.01, 0.009, 0.008],
            top_k=20,
            adaptive_config={
                "enabled": True,
                "criterion": "entropy",
                "entropy": 0.75,
                "k_min": 2,
                "k_max": 4,
            },
        )
        == 2
    )
    assert (
        adaptive_candidate_count(
            [1.0, 0.99, 0.98, 0.97],
            top_k=20,
            adaptive_config={
                "enabled": True,
                "criterion": "entropy",
                "entropy": 0.75,
                "k_min": 2,
                "k_max": 4,
            },
        )
        == 4
    )


def test_fitted_retrieval_artifacts_are_local_self_describing_and_hashed(
    tmp_path: Path,
) -> None:
    root = _artifact(tmp_path / "encoder", kind="contrastive_encoder")
    resolved = resolve_local_retrieval_artifact(
        root,
        expected_kind="contrastive_encoder",
        negative_policy="complete_reference",
    )

    assert resolved.model_path == (root / "model").resolve()
    assert len(resolved.sha256) == 64
    before = resolved.sha256
    (root / "model" / "weights.bin").write_bytes(b"changed synthetic model bytes")
    after = resolve_local_retrieval_artifact(
        root,
        expected_kind="contrastive_encoder",
        negative_policy="complete_reference",
    ).sha256
    assert before != after

    with pytest.raises(ValueError, match="negative_policy"):
        resolve_local_retrieval_artifact(
            root,
            expected_kind="contrastive_encoder",
            negative_policy="positive_unlabelled",
        )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_local_retrieval_artifact(
            "sentence-transformers/remote-model",
            expected_kind="contrastive_encoder",
        )


def test_retrieval_artifact_preflight_distinguishes_absent_and_invalid_bindings(
    tmp_path: Path,
) -> None:
    missing = retrieval_artifact_requirement(
        None,
        expected_kind="contrastive_encoder",
        expected_dataset_identity="bioml/ncit-doid",
    )
    assert missing["status"] == "deferred_unavailable"
    assert missing["code"] == "missing_fitted_retrieval_artifact"

    root = _artifact(tmp_path / "encoder-bound", kind="contrastive_encoder")
    manifest = root / "retrieval_artifact.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["mining"]["candidate_pool_fingerprint"] = "b" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    ready = retrieval_artifact_requirement(
        root,
        expected_kind="contrastive_encoder",
        negative_policy="complete_reference",
        expected_dataset_lock="a" * 64,
        expected_candidate_pool_fingerprint="b" * 64,
        expected_dataset_identity="bioml/ncit-doid",
    )
    assert ready["status"] == "ready"
    assert ready["dataset_lock_sha256"] == "a" * 64
    assert ready["candidate_pool_fingerprint"] == "b" * 64

    mismatch = retrieval_artifact_requirement(
        root,
        expected_kind="contrastive_encoder",
        expected_candidate_pool_fingerprint="c" * 64,
    )
    assert mismatch["status"] == "invalid"
    assert mismatch["code"] == "candidate_pool_mismatch"


def test_retrieval_artifact_rejects_reporting_training_rows_and_mutation(
    tmp_path: Path,
) -> None:
    root = _artifact(tmp_path / "encoder-guarded", kind="contrastive_encoder")
    manifest = root / "retrieval_artifact.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["training_pairs"] = [{"task": "fixture", "split_role": "reporting"}]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden split role"):
        resolve_local_retrieval_artifact(root, expected_kind="contrastive_encoder")

    payload["training_pairs"] = [{"task": "fixture", "split_role": "development"}]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    artifact = resolve_local_retrieval_artifact(root, expected_kind="contrastive_encoder")
    (root / "model" / "weights.bin").write_bytes(b"mutation after validation")
    with pytest.raises(ValueError, match="changed after validation"):
        artifact.assert_unchanged()


def test_individual_multiview_uses_types_and_directional_relation_labels_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(
        tmp_path,
        monkeypatch,
        entity_kinds=(EntityKind.INDIVIDUAL,),
    )

    type_views = dataset._candidate_multiview_texts_by_iri(
        [SRC + "alice"],
        side="src",
        mode="labels_types",
    )[SRC + "alice"]
    relation_views = dataset._candidate_multiview_texts_by_iri(
        [SRC + "alice"],
        side="src",
        mode="labels_relations",
    )[SRC + "alice"]

    assert "type patient" in type_views
    assert "type person" in type_views
    assert any(text.startswith("out participates in ") for text in relation_views)
    assert all("http://" not in text for text in type_views + relation_views)


def test_non_individual_multiview_does_not_change_pool_and_manifest_is_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = _dataset(tmp_path, monkeypatch, name="labels")
    relations = _dataset(tmp_path, monkeypatch, name="relations")

    labels.generate_candidates(
        top_k=2,
        lexical_encoder_name="fixture-encoder",
        retrieval_strategy="hybrid",
        multi_view={"mode": "labels"},
        device=torch.device("cpu"),
        use_amp=False,
    )
    relations.generate_candidates(
        top_k=2,
        lexical_encoder_name="fixture-encoder",
        retrieval_strategy="hybrid",
        multi_view={"mode": "labels_relations"},
        device=torch.device("cpu"),
        use_amp=False,
    )

    assert labels.candidates.to_csv(index=False) == relations.candidates.to_csv(index=False)
    manifest = labels.candidate_pool_manifest
    assert labels.candidate_pool_fingerprint == manifest["fingerprint"]
    assert len(manifest["fingerprint"]) == 64
    assert len(manifest["per_kind"]["class"]["pool_sha256"]) == 64
    assert len(manifest["inputs"]["source"]["sha256"]) == 64
    inventory = manifest["ontology_inventory"]["per_kind"]["class"]
    assert inventory["source_entities"] == len(labels.source.entities(EntityKind.CLASS))
    assert inventory["target_entities"] == len(labels.target.entities(EntityKind.CLASS))
    assert manifest["gold_free_summary"]["candidate_pairs"] == len(labels.candidates)
    assert (labels.output_path / "candidate_pool_manifest.json").is_file()


def test_candidate_pool_manifest_binds_data_and_model_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path, monkeypatch, name="bound-pool")
    dataset.bind_candidate_provenance(
        data_lock="d" * 64,
        spec_lock="c" * 64,
        model_lock={
            "identifier": "fixture/encoder",
            "revision": "e" * 40,
        },
    )
    dataset.generate_candidates(
        top_k=2,
        lexical_encoder_name="fixture/encoder",
        retrieval_strategy="hybrid",
        device=torch.device("cpu"),
        use_amp=False,
    )

    manifest = dataset.candidate_pool_manifest
    assert manifest["inputs"]["data_lock"]["sha256"] == "d" * 64
    assert manifest["inputs"]["spec_lock"]["sha256"] == "c" * 64
    assert manifest["models"]["encoder"]["model_lock"] == {
        "identifier": "fixture/encoder",
        "revision": "e" * 40,
    }
    with pytest.raises(RuntimeError, match="before candidates are loaded"):
        dataset.bind_candidate_provenance(data_lock="f" * 64)


def test_runtime_loads_fitted_encoder_and_cross_encoder_only_from_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _artifact(tmp_path / "fitted-encoder", kind="contrastive_encoder")
    cross_encoder = _artifact(tmp_path / "cross-encoder", kind="cross_encoder")
    dataset = _dataset(tmp_path, monkeypatch, name="fitted")
    _FakeSentenceTransformer.loaded_paths.clear()
    _FakeCrossEncoder.loaded_paths.clear()

    dataset.generate_candidates(
        top_k=1,
        lexical_encoder_name="ignored/base-encoder",
        retrieval_strategy="primary_label",
        encoder_finetune={
            "mode": "contrastive",
            "artifact": encoder,
            "negative_policy": "complete_reference",
        },
        cross_encoder={"mode": "on", "artifact": cross_encoder, "top_k": 2},
        device=torch.device("cpu"),
        use_amp=False,
    )

    assert _FakeSentenceTransformer.loaded_paths == [str((encoder / "model").resolve())]
    assert _FakeCrossEncoder.loaded_paths == [str((cross_encoder / "model").resolve())]
    assert {"cand_sim_retrieval", "cand_sim_cross_encoder"} <= set(dataset.candidates.columns)
    assert len(dataset.candidate_pool_manifest["models"]["encoder"]["artifact"]["sha256"]) == 64
    assert len(dataset.candidate_pool_manifest["models"]["cross_encoder"]["sha256"]) == 64


def test_restrict_sources_filters_all_in_memory_frames_without_rewriting_pool_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path, monkeypatch, name="source-cap")
    rows = [
        {"Src": "same", "Tgt": "t1", "SrcKind": "class", "TgtKind": "class", "cand_sim": 0.9},
        {
            "Src": "same",
            "Tgt": "t2",
            "SrcKind": "individual",
            "TgtKind": "individual",
            "cand_sim": 0.8,
        },
        {"Src": "other", "Tgt": "t3", "SrcKind": "class", "TgtKind": "class", "cand_sim": 0.7},
        {"Src": "last", "Tgt": "t4", "SrcKind": "class", "TgtKind": "class", "cand_sim": 0.6},
    ]
    dataset._df = pd.DataFrame(rows)
    dataset._candidates = pd.DataFrame(rows)
    dataset._reference = pd.DataFrame(rows)
    dataset._exact_matches = pd.DataFrame(rows)
    dataset._candidate_pool_sizes = {
        "class": {"source_entities": 3},
        "individual": {"source_entities": 1},
    }
    dataset._active_candidate_config = {"top_k": 1}
    dataset._refresh_candidate_pool_manifest(origin="generated")
    persisted_before = dataset._candidate_pool_manifest_path.read_bytes()
    dataset._active_df_cache_key = (0, dataset.default_kind)
    dataset._active_df_cache = pd.DataFrame([{"stale": True}])

    selected = dataset.restrict_sources(cap=2, seed=11)
    expected_groups = sorted(
        {
            ("same", "class"),
            ("same", "individual"),
            ("other", "class"),
            ("last", "class"),
        },
        key=lambda item: (
            hashlib.sha256(f"11\x1f{item[0]}\x1f{item[1]}".encode()).digest(),
            item[0],
            item[1],
        ),
    )[:2]

    assert set(dataset.dataframe[["Src", "SrcKind"]].itertuples(index=False, name=None)) == set(
        expected_groups
    )
    assert selected == {src for src, _ in expected_groups}
    for frame in (dataset.candidates, dataset.reference, dataset.exact_matches):
        assert set(frame[["Src", "SrcKind"]].itertuples(index=False, name=None)) == set(
            expected_groups
        )
    assert dataset._active_df_cache is None
    assert dataset.candidate_pool_manifest["origin"] == "sampled"
    assert dataset.candidate_pool_manifest["retrieval_config"]["source_sample"]["cap"] == 2
    assert dataset._candidate_pool_manifest_path.read_bytes() == persisted_before
