"""E01 cardinality/anchor policy, including the real trainer output boundary."""

import json

import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from exact.core.contracts.model import IModel
from exact.core.entities.configs.dataset import DatasetMask
from exact.core.entities.configs.experimental import ExtractionConfig
from exact.core.entities.mappings import EntityMapping
from exact.impl.extraction import extract_global_alignment
from exact.impl.trainer import SemanticAlignmentRunner

MODES = [
    "greedy",
    "mutual_best",
    "stable_marriage",
    "assignment_accepted_utility",
    "assignment_legacy",
]


def _pairs(mappings):
    return {(item.head, item.tail) for item in mappings}


@pytest.mark.parametrize("mode", MODES)
def test_conflicting_anchors_compete_on_both_sides_without_changing_scores(mode):
    anchors = {("s1", "t1"), ("s1", "t2"), ("s2", "t2"), ("solo", "protected")}
    mappings = [
        EntityMapping(source, target, score=score)
        for source, target, score in [
            ("s1", "t1", 0.8),
            ("s1", "t2", 0.9),
            ("s2", "t2", 0.95),
            ("solo", "protected", 0.2),
            ("s1", "better", 0.99),
            ("other", "protected", 1.0),
            ("solo", "other", 1.0),
        ]
    ]
    result = extract_global_alignment(
        mappings,
        mode=mode,
        threshold=0.7,
        protected_pairs=anchors,
        anchor_conflict_policy="compete",
    )
    assert _pairs(result.mappings) == {("s1", "better"), ("s2", "t2"), ("solo", "protected")}
    assert all(item in mappings for item in result.mappings)
    assert result.diagnostics["protected_mappings"] == 1
    assert result.diagnostics["anchor_source_conflicts"] == [
        {"source": ["s1", "class"], "targets": [["t1", "class"], ["t2", "class"]]}
    ]
    assert result.diagnostics["anchor_target_conflicts"] == [
        {"target": ["t2", "class"], "sources": [["s1", "class"], ["s2", "class"]]}
    ]
    assert result.diagnostics["selected_conflicting_anchor_pairs"] == [["s2", "t2"]]
    assert result.diagnostics["suppressed_conflicting_anchor_pairs"] == [["s1", "t1"], ["s1", "t2"]]
    reverse = extract_global_alignment(
        list(reversed(mappings)),
        mode=mode,
        threshold=0.7,
        protected_pairs=anchors,
        anchor_conflict_policy="compete",
    )
    assert _pairs(reverse.mappings) == _pairs(result.mappings)
    assert reverse.diagnostics == result.diagnostics
    with pytest.raises(ValueError, match="Conflicting protected exact matches"):
        extract_global_alignment(mappings, mode=mode, protected_pairs=anchors)


@pytest.mark.parametrize("mode", [*MODES, "threshold"])
def test_demoted_anchor_below_threshold_does_not_survive(mode):
    result = extract_global_alignment(
        [EntityMapping("s", "t1", score=0.69), EntityMapping("s", "t2", score=0.9)],
        mode=mode,
        threshold=0.7,
        anchor_conflict_policy="compete",
        protected_pairs={("s", "t1"), ("s", "t2")},
        source_cardinality=None if mode == "threshold" else 1,
        target_cardinality=None if mode == "threshold" else 1,
    )
    assert _pairs(result.mappings) == {("s", "t2")}


@pytest.mark.parametrize("source,target", [(1, 1), (None, 1), (1, None)])
def test_threshold_rejects_a_hidden_cardinality(source, target):
    with pytest.raises(ValueError, match="unrestricted"):
        extract_global_alignment(
            [], mode="threshold", source_cardinality=source, target_cardinality=target
        )


def test_extraction_config_keeps_strict_default_and_validates_policy():
    assert ExtractionConfig().anchor_conflict_policy == "error"
    assert ExtractionConfig(mode="threshold", anchor_conflict_policy="compete").mode == "threshold"
    with pytest.raises(ValidationError):
        ExtractionConfig(anchor_conflict_policy="ignore")


class _Dataset:
    dataset_signature = "e01-cardinality-test"
    cache_fingerprint = "e01-cardinality-test-v1"
    scores = [("s1", "t3", 0.99), ("s2", "t2", 0.94), ("s3", "t5", 0.95), ("s4", "t4", 0.91)]

    def __init__(self, anchors=True):
        self.dataframe = pd.DataFrame(
            [("s1", "t1", 0.8), ("s1", "t2", 0.9), ("s3", "t4", 0.98)] if anchors else [],
            columns=["Src", "Tgt", "Scores"],
        )
        self.dataframe[DatasetMask.prefiltered] = True

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        source, target, _ = self.scores[index]
        return {
            "src_iri": source,
            "tgt_iri": target,
            "src_kind": "class",
            "tgt_kind": "class",
            "src_labels": [source],
            "tgt_labels": [target],
            "src_ctx_triples": [],
            "tgt_ctx_triples": [],
            "label": 0,
        }


class _E01ExtractionModel(IModel):
    def forward(self, *, src_iris, tgt_iris, **kwargs):
        lookup = {(source, target): score for source, target, score in _Dataset.scores}
        return {"S_final": torch.tensor([lookup[pair] for pair in zip(src_iris, tgt_iris)])}


@pytest.mark.parametrize("mode", [*MODES, "threshold"])
@pytest.mark.parametrize("anchors", [True, False])
def test_runner_preserves_amended_extraction_through_prefilter_and_audit(tmp_path, mode, anchors):
    runner = SemanticAlignmentRunner(
        dataset=_Dataset(anchors),
        model=_E01ExtractionModel,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        extraction_config={"mode": mode, "anchor_conflict_policy": "compete"},
    )
    cardinality = None if mode == "threshold" else 1
    predictions, _ = runner.predict(
        threshold=0.7,
        cardinality=cardinality,
        target_cardinality=cardinality,
        batch_size=4,
        num_workers=0,
        mixed_precision=False,
        enable_checkpoints=False,
        audit_shards_enabled=False,
        audit_shard_compression="none",
        cache_persist_policy="never",
        log_every=100,
    )
    assert runner._extraction_includes_prefilter
    # The final application boundary cannot reinsert discarded anchors or add
    # cardinality to already-extracted unrestricted mappings.
    final = runner.apply_prefilter(predictions, threshold=0.7, cardinality=1, target_cardinality=1)
    assert final is predictions
    if mode == "threshold":
        expected = {(s, t) for s, t, _ in _Dataset.scores}
        if anchors:
            expected |= {("s1", "t1"), ("s1", "t2"), ("s3", "t4")}
    elif anchors:
        expected = {("s1", "t3"), ("s2", "t2"), ("s3", "t4")}
    else:
        expected = {(s, t) for s, t, _ in _Dataset.scores}
    assert _pairs(final) == expected
    paths = runner.save_results(
        final, output_formats=["tsv-global"], save_json=False, save_stats_csv=False
    )
    trace = json.loads(paths["source_decisions_json"].read_text())
    candidates = {
        (row["Src"], candidate["target"]): candidate
        for row in trace["records"]
        for candidate in row["candidates"]
    }
    if anchors:
        assert not candidates[("s1", "t1")]["protected_exact"]
        assert not candidates[("s1", "t2")]["protected_exact"]
        assert candidates[("s3", "t4")]["protected_exact"]
        assert candidates[("s1", "t1")]["emitted"] == (mode == "threshold")


def test_default_checkpoint_identity_stays_compatible_but_compete_is_distinct(tmp_path):
    runner = SemanticAlignmentRunner(
        dataset=_Dataset(),
        model=_E01ExtractionModel,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    baseline = runner._build_checkpoint_fingerprint_payload()
    assert "trainer" not in baseline
    runner._extraction_config.pop("anchor_conflict_policy")
    assert runner._build_checkpoint_fingerprint_payload() == baseline
    runner._extraction_config["anchor_conflict_policy"] = "compete"
    assert (
        runner._build_checkpoint_fingerprint_payload()["trainer"]["extraction"][
            "anchor_conflict_policy"
        ]
        == "compete"
    )


def test_assignment_cap_fallback_competes_without_reinserting_anchors():
    result = extract_global_alignment(
        [
            EntityMapping("s", "first", score=0.8),
            EntityMapping("s", "second", score=0.9),
            EntityMapping("s", "better", score=0.99),
        ],
        mode="assignment_accepted_utility",
        threshold=0.7,
        assignment_component_cap=1,
        protected_pairs={("s", "first"), ("s", "second")},
        anchor_conflict_policy="compete",
    )
    assert _pairs(result.mappings) == {("s", "better")}
    assert result.diagnostics["assignment_fallback_components"] == 1
    assert result.diagnostics["suppressed_conflicting_anchor_pairs"] == [
        ["s", "first"],
        ["s", "second"],
    ]


@pytest.mark.parametrize("mode", [*MODES, "threshold"])
def test_difference_replay_keeps_ambiguous_exact_scores_fixed_but_not_mandatory(tmp_path, mode):
    from exact.experiments.difference_replay import write_difference_replay
    from tests.difference_replay_test import synthetic

    runner, frame, explanations, emitted, anchors, policy = synthetic(tmp_path)
    frame.loc[len(frame)] = {"Src": "anchor", "Tgt": "second", "S_final": 0.8}
    anchors.add(("anchor", "second"))
    policy["extraction"] = {"mode": mode, "anchor_conflict_policy": "compete"}
    if mode == "threshold":
        policy["source_cardinality"] = policy["target_cardinality"] = None
        emitted.add(("anchor", "second"))
    path = write_difference_replay(runner, frame, explanations, emitted, anchors, policy)
    payload = json.loads(path.read_text())
    assert payload["natural_parity"]
    assert {tuple(pair) for pair in payload["variants"]["natural"]["emitted"]} == emitted
    assert {
        tuple(pair) for pair in payload["variants"]["deletion_source"]["emitted"]
    } == emitted - {("s", "t")}
