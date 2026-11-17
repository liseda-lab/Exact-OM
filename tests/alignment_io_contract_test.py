from __future__ import annotations

import pandas as pd

from exact.core.contracts import alignment_io


def test_relation_typer_contract_forwards_semantic_options(monkeypatch) -> None:
    candidates = pd.DataFrame([{"Src": "s", "Tgt": "t", "Score": 0.8}])
    anchors = pd.DataFrame([{"Src": "a", "Tgt": "b", "Score": 1.0}])
    captured = {}

    def relation_typer(received, source, target, **kwargs):
        captured.update(
            {
                "candidates": received,
                "source": source,
                "target": target,
                **kwargs,
            }
        )
        return received

    monkeypatch.setattr(alignment_io, "_relation_typer", relation_typer)
    result = alignment_io.type_alignment_relations(
        candidates,
        "source-graph",
        "target-graph",
        mode="semantic_entailment",
        anchors=anchors,
        semantic_backend="bridge_reasoner",
        equivalence_anchor_threshold=0.92,
        equivalence_anchor_margin=0.18,
        relation_confidence_threshold=0.63,
        timeout_seconds=41.5,
    )

    assert result is candidates
    assert captured.pop("candidates") is candidates
    assert captured.pop("anchors") is anchors
    assert captured == {
        "source": "source-graph",
        "target": "target-graph",
        "mode": "semantic_entailment",
        "semantic_backend": "bridge_reasoner",
        "equivalence_anchor_threshold": 0.92,
        "equivalence_anchor_margin": 0.18,
        "relation_confidence_threshold": 0.63,
        "timeout_seconds": 41.5,
    }
