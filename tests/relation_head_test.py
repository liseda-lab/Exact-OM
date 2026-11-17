import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from exact.core.entities.kinds import EntityKind
from exact.impl.trainer.fitting import TrainingPoolMixin
from exact.io.relation_head import (
    fit_relation_artifact,
    predict_relation_head,
    typed_reference_frame,
)
from exact.io.relations import predict_relations


class Source:
    def __init__(self, prefix):
        self.prefix = prefix

    def entities(self, kind):
        return [self.prefix + str(i) for i in range(6)] if kind == EntityKind.CLASS else []

    def labels(self, iri):
        return ["concept " + iri[-1]]

    def direct_parents(self, iri, kind=EntityKind.CLASS):
        return [self.prefix + str(int(iri[-1]) - 1)] if int(iri[-1]) > 0 else []

    def direct_children(self, iri, kind=EntityKind.CLASS):
        return [self.prefix + str(int(iri[-1]) + 1)] if int(iri[-1]) < 5 else []


def typed_rows():
    return pd.DataFrame(
        [
            (f"s{i}", f"t{i+delta}", relation)
            for i in (1, 2, 3, 4)
            for delta, relation in ((0, "="), (-1, "<"), (1, ">"))
        ],
        columns=["Src", "Tgt", "Relation"],
    )


def test_grouped_three_way_head_checkpoint_repair_and_contributions(tmp_path, monkeypatch):
    source, target, rows = Source("s"), Source("t"), typed_rows()
    path = tmp_path / "typed.json"
    application = {"dataset_signature": "typed", "source_ids": ["s5"]}
    state = fit_relation_artifact(
        rows[["Src", "Tgt"]], rows, source, target, path, application=application, seed=17
    )
    assert state["relation_counts"] == {"=": 4, "<": 4, ">": 4}
    assert len(state["oof_predictions"]) == len(rows)
    assert all(not set(f["training_sources"]) & set(f["heldout_sources"]) for f in state["folds"])
    probs, contributions, logits, _ = predict_relation_head(rows, source, target, state)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(contributions.sum(axis=1) + state["bias"], logits)
    module = __import__("exact.io.relation_head", fromlist=["minimize"])
    original, calls = module.minimize, []

    def fit(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "minimize", fit)
    path.unlink()
    assert (
        fit_relation_artifact(
            rows[["Src", "Tgt"]], rows, source, target, path, application=application, seed=17
        )
        == state
    )
    assert len(calls) == 1
    candidates = rows[["Src", "Tgt"]].assign(Score=0.8)
    out = predict_relations(
        candidates,
        source,
        target,
        mode="learned_three_way",
        artifact=path,
        relation_confidence_threshold=0.0,
    )
    assert len(out) == len(rows) and set(out.Relation) <= {"=", "<", ">"}
    assert json.loads(out.iloc[0].relation_evidence)["artifact"] == state["fit_identity"]


def test_relation_fitting_hook_does_not_require_a_binary_negative_policy(tmp_path):
    rows = typed_rows()
    train = tmp_path / "typed.tsv"
    rows.to_csv(train, sep="\t", index=False)
    trainer = TrainingPoolMixin()
    trainer.output_dir = tmp_path
    trainer.model = SimpleNamespace(request_seed=17)
    trainer.dataset = SimpleNamespace(
        source=Source("s"),
        target=Source("t"),
        dataset_signature="typed",
        dataframe=pd.DataFrame({"Src": ["s5"]}),
    )
    trainer.relation_config = {
        "relation_prediction": "learned_three_way",
        "relation_training_file": str(train),
    }
    trainer.fit_training_pool()
    assert trainer.relation_fit_report["relation_counts"] == {"=": 4, "<": 4, ">": 4}
    assert trainer.relation_artifact.is_file()
    trainer.dataset.dataframe = pd.DataFrame({"Src": ["s1"]})
    trainer._relation_head_fitted = False
    with pytest.raises(ValueError, match="disjoint"):
        trainer.fit_training_pool()


def test_public_typed_list_adapter_preserves_direction(tmp_path):
    path = tmp_path / "typed.tsv"
    pd.DataFrame(
        {
            "SrcEntity": ["s1"],
            "TgtEntities": ["['t1','t2','t3']"],
            "Relations": ["['equivalent','subsumed_by','subsumes']"],
            "TgtCandidates": ["['t4']"],
        }
    ).to_csv(path, sep="\t", index=False)
    assert typed_reference_frame(path).Relation.tolist() == ["=", "<", ">"]
