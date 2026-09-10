import json
from pathlib import Path

import pandas as pd

from exact.impl.trainer.anchors import (
    AnchorPreparationMixin,
    corrupt_anchors,
    predicted_anchors,
)
from tests import kind_evidence_controls_test
from tests.pair_adaptive_experiments_test import _scorer

dataset = kind_evidence_controls_test.dataset
SRC = kind_evidence_controls_test.SRC
TGT = "http://example.org/mini/tgt#"


def test_predicted_anchor_rule_is_reciprocal_threshold_and_margin_only():
    rows = [
        {"Src": "a", "Tgt": "x", "score": 0.98},
        {"Src": "a", "Tgt": "y", "score": 0.97},
        {"Src": "b", "Tgt": "y", "score": 0.99},
        {"Src": "c", "Tgt": "z", "score": 0.99},
        {"Src": "c", "Tgt": "x", "score": 0.2},
    ]
    anchors = predicted_anchors(rows, threshold=0.95, margin=0.1)
    assert [(row["Src"], row["Tgt"]) for row in anchors] == [("c", "z")]
    assert predicted_anchors([], threshold=0.9, margin=0.1) == []


def test_anchor_noise_controls_are_nested_reproducible_and_reference_free():
    rows = [{"Src": f"s{i}", "Tgt": f"t{i}", "score": 1.0, "origin": "trusted"} for i in range(100)]
    kinds = {row["Tgt"]: "class" for row in rows}
    one, changed_one = corrupt_anchors(rows, fraction=0.01, seed=17, target_kinds=kinds)
    five, changed_five = corrupt_anchors(
        list(reversed(rows)), fraction=0.05, seed=17, target_kinds=kinds
    )
    assert len(changed_one) == 1 and len(changed_five) == 5
    assert changed_one == changed_five[:1]
    assert all(row["original_target"] != row["replacement_target"] for row in changed_five)
    assert one == corrupt_anchors(rows, fraction=0.01, seed=17, target_kinds=kinds)[0]


class Trainer(AnchorPreparationMixin):
    def _json_safe_value(self, value):
        return json.loads(json.dumps(value, default=str))

    def _build_checkpoint_fingerprint_payload(self):
        return {"anchor": self.model._anchor_manifest["inventory_sha256"]}

    def _hash_checkpoint_fingerprint_payload(self, payload):
        return str(payload)


def test_trusted_anchor_second_pass_binds_inventory_and_replays_without_model_calls(
    dataset, tmp_path
):
    dataset._df = pd.DataFrame({"Src": [SRC + "Heart"], "Tgt": [TGT + "CardiacOrgan"]})
    dataset._invalidate_active_dataframe_cache()
    anchors = tmp_path / "train.tsv"
    pd.DataFrame(
        {
            "Src": [SRC + "Organ", SRC + "Structure"],
            "Tgt": [TGT + "BodyOrgan", TGT + "BodyStructure"],
        }
    ).to_csv(anchors, sep="\t", index=False)
    scorer = _scorer(request_seed=17)
    scorer.attach_dataset(dataset)
    trainer = Trainer()
    trainer.dataset = dataset
    trainer.model = scorer
    trainer.output_dir = tmp_path
    trainer.anchor_config = {"mode": "one_pass", "source": "trusted", "trusted_file": str(anchors)}
    before = scorer.runtime_fingerprint_payload()
    trainer.prepare_anchors(batch_size=2)
    assert scorer.hier_enabled and scorer.hier_config["mode"] == "labels_overlap"
    assert scorer._hierarchy_anchor_terms(SRC + "Heart", TGT + "CardiacOrgan")[0] > 0
    assert scorer.runtime_fingerprint_payload() != before
    manifest = json.loads(Path(trainer.anchor_manifest["artifact"]).read_text())
    assert manifest["query_self_support"] == "excluded" and len(manifest["rows"]) == 2
    trainer.prepare_anchors(batch_size=2)
    assert json.loads(Path(trainer.anchor_manifest["artifact"]).read_text()) == manifest


def test_cycle_cannot_turn_query_anchor_into_its_own_ancestor_support(monkeypatch):
    from tests.pair_adaptive_experiments_test import _TinyDataset

    data = _TinyDataset()
    data.source.parents = {"s": ["sa"], "sa": ["s"]}
    data.target.parents = {"t": ["ta"], "ta": ["t"]}
    scorer = _scorer(hier={"enabled": True, "mode": "labels_overlap"})
    scorer.attach_dataset(data)
    scorer._exact_anchor_src_to_tgt = {"s": {"t"}}
    scorer._exact_anchor_tgt_to_src = {"t": {"s"}}
    assert scorer._hierarchy_anchor_terms("s", "t")[:2] == (0.0, 0.0)
