import json


from pathlib import Path


from types import SimpleNamespace


import pandas as pd


import pytest


from exact.utils.provenance import sha256_file


def evaluation_cell(tmp_path):
    labels = tmp_path / "evaluation-labels.tsv"
    pd.DataFrame(
        {
            "Src": ["mapped", "nil", "miss", "unknown"],
            "Status": ["mapped", "ontology_nil", "mapped", "unknown"],
        }
    ).to_csv(labels, sep="\t", index=False)
    references = tmp_path / "valid.tsv"
    pd.DataFrame({"SrcEntity": ["mapped", "miss"], "TgtEntity": ["t", "outside"]}).to_csv(
        references, sep="\t", index=False
    )
    records = [
        {
            "Src": "mapped",
            "candidates": [{"target": "t", "S_final": 0.9}, {"target": "other", "S_final": 0.1}],
            "emitted_targets": ["t"],
            "absence_semantics": "in_pool",
        },
        {
            "Src": "nil",
            "candidates": [],
            "emitted_targets": [],
            "absence_semantics": "ontology_nil",
        },
        {
            "Src": "miss",
            "candidates": [{"target": "wrong"}],
            "emitted_targets": [],
            "absence_semantics": "ontology_nil",
        },
        {"Src": "unknown", "candidates": [], "emitted_targets": []},
        {"Src": "unlabeled", "candidates": [], "emitted_targets": []},
    ]
    trace = {
        "schema_version": 2,
        "stage": "after_cardinality_and_relation_typing",
        "source_universe_status": "declared",
        "source_universe": [row["Src"] for row in records],
        "records": records,
    }
    (tmp_path / "source_decisions.json").write_text(json.dumps(trace))
    return SimpleNamespace(
        output_dir=tmp_path,
        split_role="development",
        reference_role="valid",
        reference_completeness="complete",
        diagnostics={
            "role": "development",
            "reference_role": "valid",
            "evaluation_source_labels": {"path": str(labels), "sha256": sha256_file(labels)},
        },
        resolved_config={"data": {"root": str(tmp_path), "refs": {"valid": str(references)}}},
    )
