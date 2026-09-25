"""Historical qualification reuse requires original provenance and actual populations."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from exact.experiments.population_audit import _population, audit_matched_measurements
from exact.experiments.recovery import ArtifactStore, stage_identity
from exact.utils.provenance import sha256_file


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _bind(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def _measurements(tmp_path, mutation=None):
    rows = []
    for arm in ("current", "unified_bank", "provenance_dedup"):
        root = tmp_path / arm
        output = root / "run"
        dataset = output / "dataset"
        dataset.mkdir(parents=True)
        frame = pd.DataFrame(
            [
                ("s1", "t1", "class", "class", 0.8, True, False),
                ("s2", "t2", "class", "class", 0.7, True, False),
                ("s1", "t3", "class", "class", None, False, True),
            ],
            columns=["Src", "Tgt", "SrcKind", "TgtKind", "cand_sim", "inference", "prefiltered"],
        )
        if arm == "provenance_dedup" and mutation:
            frame = mutation(frame)
        frame.to_csv(dataset / "dataset.csv", index=False)
        groups = [("s1", "class"), ("s2", "class")]
        sample = {
            "source_kind_groups": groups,
            "selected_groups": 2,
            "cap": 2,
            "seed": 17,
            "eligible_source_iris": ["s1", "s2"],
            "sha256": hashlib.sha256(b"s1\tclass\ns2\tclass").hexdigest(),
        }
        raw = {
            "gold_free_summary": {"candidate_pairs": 3},
            "per_kind": {"class": {"candidate_pairs": 3, "pool_sha256": "raw"}},
            "retrieval_config": {"top_k": 2},
        }
        effective = _population(frame.loc[frame.cand_sim.notna()], groups)
        sampled = {
            **(raw if arm == "provenance_dedup" else effective),
            "retrieval_config": {"source_sample": sample},
        }
        _write(dataset / "candidate_pool_manifest.json", raw)
        _write(dataset / "candidate_pool_sample_manifest.json", sampled)
        identity = stage_identity(
            "pair_scores",
            parameters={},
            inputs={},
            role="development",
            entity_kind="class",
            implementation={"code": "frozen"},
            dependencies={},
            seed=17,
        )
        ArtifactStore(root).publish(
            identity,
            {
                f"dataset/{name}": dataset / name
                for name in (
                    "dataset.csv",
                    "candidate_pool_manifest.json",
                    "candidate_pool_sample_manifest.json",
                )
            },
        )
        _write(output / "recovery-runtime.json", {"root": str(root), "identity": identity})
        _write(output / "experiment_manifest.json", {})
        _write(output / "validation-worker.json", {"return_code": 0})
        _write(
            output / "candidate_decisions.json",
            {
                "reference_labels_used": False,
                "candidate_pool": sampled,
                "records": [
                    {
                        "source": {"iri": r.Src, "kind": r.SrcKind},
                        "target": {"iri": r.Tgt, "kind": r.TgtKind},
                    }
                    for r in frame.itertuples(index=False)
                ],
            },
        )
        _write(
            output / "run_manifest.json",
            {
                "artifacts": [
                    {
                        "path": "candidate_decisions.json",
                        "sha256": sha256_file(output / "candidate_decisions.json"),
                    }
                ]
            },
        )
        row = {
            "name": arm,
            "output_dir": str(output),
            "status": "passed",
            "execution_status": "complete",
            "prefix": False,
            "new_usage": {"attempts": 0},
            "processed_pairs": 2,
            "dataset_rows": 2,
            "worker_calls": 1,
            "worker_measurement": {"return_code": 0},
            "generate_rationales": False,
            "no_private_test_references": True,
            "source_cap": 2,
            "seed": 17,
            "bindings": [
                _bind(output / name)
                for name in (
                    "recovery-runtime.json",
                    "experiment_manifest.json",
                    "validation-worker.json",
                    "dataset/candidate_pool_sample_manifest.json",
                )
            ],
        }
        _write(root / "measurement.json", row)
        rows.append(row)
    return rows


def test_saved_population_recovery_binds_original_evidence_without_writes(tmp_path):
    rows = _measurements(tmp_path)
    before = {p: sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    audit = audit_matched_measurements(rows)
    assert audit["candidate_population"]["gold_free_summary"]["candidate_pairs"] == 2
    assert [e["historical_manifest_scope"] for e in audit["evidence"]] == [
        "effective",
        "effective",
        "raw",
    ]
    assert all(e["exact_prefilter_rows"] == 1 for e in audit["evidence"])
    assert before == {p: sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert audit_matched_measurements(rows) == audit


def test_saved_population_recovery_rejects_changed_original_dataset(tmp_path):
    rows = _measurements(tmp_path)
    path = Path(rows[0]["output_dir"]) / "dataset/dataset.csv"
    path.write_text(path.read_text().replace("t1", "changed"))
    with pytest.raises(ValueError, match="evidence changed"):
        audit_matched_measurements(rows)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda frame: frame.assign(Tgt=["changed", "t2", "t3"]),
        lambda frame: frame.assign(cand_sim=[0.81, 0.7, None]),
        lambda frame: frame.iloc[::-1].reset_index(drop=True),
        lambda frame: frame.assign(Src=["outside", "s2", "s1"]),
        lambda frame: frame.assign(inference=[False, True, False]),
    ],
)
def test_saved_population_recovery_rejects_genuine_population_drift(tmp_path, mutation):
    rows = _measurements(tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match="population|identities|masks"):
        audit_matched_measurements(rows)


def test_saved_population_recovery_rejects_unbound_receipt(tmp_path):
    rows = _measurements(tmp_path)
    rows[0]["bindings"].pop()
    _write(Path(rows[0]["output_dir"]).parent / "measurement.json", rows[0])
    with pytest.raises(ValueError, match="lacks required"):
        audit_matched_measurements(rows)


def test_saved_population_recovery_rejects_resumed_measurement(tmp_path):
    rows = _measurements(tmp_path)
    rows[0]["worker_calls"] = 2
    _write(Path(rows[0]["output_dir"]).parent / "measurement.json", rows[0])
    with pytest.raises(ValueError, match="single-call"):
        audit_matched_measurements(rows)
