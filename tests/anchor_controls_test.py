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


def test_predicted_anchor_resume_skips_base_scoring_and_rebinds_changed_corruption(tmp_path):
    import torch

    from exact.core.entities.kinds import EntityKind
    from exact.impl.models.selector.fitting import fingerprint

    blocked, scored = [False], []

    class Source:
        def __init__(self, entities):
            self.iris = entities

        def entities(self, kind):
            assert not blocked[0], "resume must not reload ontology signatures"
            assert kind == EntityKind.CLASS
            return self.iris

    class Data:
        dataset_signature = "predicted-anchor-fixture"
        dataframe = pd.DataFrame({"Src": ["s1", "s1", "s2"], "Tgt": ["t1", "t2", "t2"]})
        source, target = Source(("s1", "s2")), Source(("t1", "t2"))

        def __getitem__(self, index):
            assert not blocked[0], "resume must not regenerate candidate features"
            row = self.dataframe.iloc[index]
            return {
                "src_iri": row.Src,
                "tgt_iri": row.Tgt,
                "src_labels": [row.Src],
                "tgt_labels": [row.Tgt],
            }

        def entity_kind_for(self, iri, side, *, warn_unknown):
            assert not blocked[0], "resume must reuse the verified anchor inventory"
            return EntityKind.CLASS

    class Model:
        use_llm, request_seed = True, 17

        def __init__(self):
            self.hier_config = {}

        def runtime_fingerprint_payload(self):
            return {"model": "fixed-base-scorer", "seed": self.request_seed}

        def forward(self, *, src_iris, tgt_iris, **kwargs):
            assert not blocked[0], "resume must not score a candidate again"
            assert self.use_llm is False, "anchor selection must never call the decision LLM"
            pairs = list(zip(src_iris, tgt_iris))
            scored.extend(pairs)
            scores = {("s1", "t1"): 0.99, ("s1", "t2"): 0.1, ("s2", "t2"): 0.98}
            return {"S_base": torch.tensor([scores[pair] for pair in pairs])}

    rule = tmp_path / "rule.json"
    rule.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "predicted_anchor_rule",
                "development_provenance": {"selection": "prespecified", "seed": 17},
                "threshold": 0.95,
                "margin": 0.1,
            }
        )
    )

    def trainer(corruption=0.0):
        value = Trainer()
        value.dataset, value.model, value.output_dir = Data(), Model(), tmp_path
        value.anchor_config = {
            "mode": "one_pass",
            "source": "predicted",
            "rule_artifact": str(rule),
            "corruption_fraction": corruption,
            "diagnostic": bool(corruption),
        }
        return value

    original = trainer()
    original.prepare_anchors(batch_size=2)
    assert scored == [("s1", "t1"), ("s1", "t2"), ("s2", "t2")]
    assert original.model.use_llm is True
    path = Path(original.anchor_manifest["artifact"])
    frozen = path.read_bytes()
    inventory = json.loads(frozen)
    assert inventory["inventory_sha256"] == fingerprint(inventory["rows"])
    assert inventory["corruptions"] == []
    assert original.model._exact_anchor_src_to_tgt == {"s1": {"t1"}, "s2": {"t2"}}

    blocked[0] = True
    resumed = trainer()
    resumed.prepare_anchors(batch_size=2)
    assert resumed.anchor_manifest == original.anchor_manifest
    assert resumed.model._exact_anchor_src_to_tgt == original.model._exact_anchor_src_to_tgt
    assert resumed.model.use_llm is True
    assert len(scored) == 3 and path.read_bytes() == frozen

    blocked[0] = False
    diagnostic = trainer(corruption=0.5)
    diagnostic.prepare_anchors(batch_size=2)
    changed = json.loads(Path(diagnostic.anchor_manifest["artifact"]).read_text())
    assert diagnostic.anchor_manifest["artifact"] != str(path)
    assert changed["diagnostic"] is True and len(changed["corruptions"]) == 1
    assert changed["realized_corruption_fraction"] == 0.5
    assert changed["input_identity"] != inventory["input_identity"]
    assert path.read_bytes() == frozen
