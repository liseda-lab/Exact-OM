"""Historical metadata omissions never waive E21's matched-benefit prerequisite."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.experiments.judge_benefit import observed_judge_population
from exact.experiments.staged_selection import PrerequisiteUnavailable
from exact.utils.provenance import sha256_file


def write(root, path, payload):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True))


@pytest.fixture
def observed(tmp_path):
    items = []
    for side in ("old", "new"):
        root = tmp_path / side
        full = {
            "models": {"encoder": {"identifier": "same-encoder", "revision": "frozen"}},
            "retrieval_config": {"top_k": 20},
            "fingerprint": "parent",
        }
        write(root, "dataset/candidate_pool_manifest.json", full)
        sample = deepcopy(full)
        sample["fingerprint"] = side
        sample["retrieval_config"]["source_sample"] = {"sources": ["s"], "seed": 17}
        if side == "old":
            sample["models"]["encoder"] = {"identifier": None, "revision": None}
            sample["retrieval_config"].pop("top_k")
        write(root, "dataset/candidate_pool_sample_manifest.json", sample)
        write(
            root,
            "source_decisions.json",
            {
                "schema_version": 2,
                "reference_labels_used": False,
                "dataset_signature": "data",
                "source_universe": ["s"],
                "policy": {
                    "threshold": 0.7,
                    "source_cardinality": 1,
                    "target_cardinality": 1,
                    "llm": {"gate": side},
                },
                "records": [
                    {
                        "Src": "s",
                        "candidates": [
                            {"target": "t", "S_base": 0.8, "U": 0.4, "protected_exact": False}
                        ],
                    }
                ],
            },
        )
        refs = {}
        for role in ("full_reference", "train_reference"):
            path = root / (role + ".tsv")
            path.write_text("Src\tTgt\ns\tt\n")
            refs[role] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": 1,
                "bytes": path.stat().st_size,
            }
        write(root, "evaluation/evaluation_results.json", {"meta": {"refs": refs}})
        write(
            root,
            "stats/run_stats.json",
            {
                "source_sampling": {
                    "candidate_pool_fingerprint": side,
                    "cap": 200,
                    "seed": 17,
                    "selected_sources": 1,
                    "sample_sha256": "same-sample",
                }
            },
        )
        items.append(
            {
                "fingerprint_payload": {"output_dir": str(root)},
                "candidate_pool_fingerprint": side,
                "source_cap": 200,
                "seed": 17,
            }
        )
    config = SimpleNamespace(matching=SimpleNamespace(threshold=0.7))
    return items, config


def test_omitted_metadata_requires_fully_bound_identical_observed_population(observed):
    items, config = observed
    proof = observed_judge_population(*items, config, config)
    assert proof["source_count"] == proof["pair_count"] == 1
    assert proof["effective_acceptance"]["threshold"] == 0.7
    assert len(proof["inputs"]) == 2
    assert all(len(bindings) == 7 for bindings in proof["inputs"])


@pytest.mark.parametrize("fault", [None, "unprotected", "score", "partial", "emitted"])
@pytest.mark.parametrize("retained", [False, True])
def test_exact_anchors_keep_observed_missing_features_without_imputation(observed, fault, retained):
    items, config = observed
    for index, item in enumerate(items):
        root = Path(item["fingerprint_payload"]["output_dir"])
        relative = "source_decisions.json"
        data = json.loads((root / relative).read_text())
        row = data["records"][0]["candidates"][0]
        row.update(
            S_base=None,
            U=None,
            protected_exact=True,
            S_final=1.0,
            emitted=retained,
            pre_typing_selected=retained,
            reason="emitted" if retained else "cardinality_or_extraction",
        )
        if index == 0:
            if fault == "unprotected":
                row["protected_exact"] = False
            elif fault == "score":
                row["S_final"] = 0.9
            elif fault == "partial":
                row["U"] = 0.0
            elif fault == "emitted":
                row["emitted"] = not retained
        write(root, relative, data)
    if fault is None:
        assert observed_judge_population(*items, config, config)["pair_count"] == 1
    else:
        with pytest.raises(PrerequisiteUnavailable, match="inconsistent protected exact anchor"):
            observed_judge_population(*items, config, config)


def test_exact_anchor_extraction_state_must_match_both_populations(observed):
    items, config = observed
    for retained, item in zip((False, True), items):
        root = Path(item["fingerprint_payload"]["output_dir"])
        data = json.loads((root / "source_decisions.json").read_text())
        data["records"][0]["candidates"][0].update(
            S_base=None,
            U=None,
            protected_exact=True,
            S_final=1.0,
            emitted=retained,
            pre_typing_selected=retained,
            reason="emitted" if retained else "cardinality_or_extraction",
        )
        write(root, "source_decisions.json", data)
    with pytest.raises(
        PrerequisiteUnavailable, match="candidate scores or protected anchors differ"
    ):
        observed_judge_population(*items, config, config)


@pytest.mark.parametrize(
    "fault",
    [
        "parent",
        "encoder",
        "retrieval",
        "score",
        "anchor",
        "duplicate",
        "missing_source",
        "threshold",
        "sample",
        "reference",
        "missing",
    ],
)
def test_population_proof_rejects_real_mismatches(observed, fault):
    items, config = observed
    root = Path(items[0]["fingerprint_payload"]["output_dir"])
    if fault == "missing":
        (root / "source_decisions.json").unlink()
    elif fault == "reference":
        (root / "full_reference.tsv").write_text("changed reference")
    else:
        relative = (
            "dataset/candidate_pool_manifest.json"
            if fault == "parent"
            else (
                "dataset/candidate_pool_sample_manifest.json"
                if fault in {"encoder", "retrieval"}
                else "stats/run_stats.json" if fault == "sample" else "source_decisions.json"
            )
        )
        data = json.loads((root / relative).read_text())
        if fault == "parent":
            data["fingerprint"] = "different"
        elif fault == "encoder":
            data["models"]["encoder"]["identifier"] = "other"
        elif fault == "retrieval":
            data["retrieval_config"]["top_k"] = 30
        elif fault == "score":
            data["records"][0]["candidates"][0]["S_base"] = 0.9
        elif fault == "anchor":
            data["records"][0]["candidates"][0]["protected_exact"] = True
        elif fault == "duplicate":
            data["records"].append(data["records"][0])
        elif fault == "missing_source":
            data["source_universe"].append("absent")
        elif fault == "threshold":
            data["policy"]["threshold"] = 0.9
        elif fault == "sample":
            data["source_sampling"]["sample_sha256"] = "other"
        write(root, relative, data)
    with pytest.raises(PrerequisiteUnavailable, match="same population proof failed"):
        observed_judge_population(*items, config, config)
