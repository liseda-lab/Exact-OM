from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from exact.experiments.harness import (
    ExperimentSource,
    LoadedSuite,
    _apply_bootstrap_multiplicity,
    _candidate_pool_guard,
    _finalize_stage_reports,
    _paired_bootstrap_rows,
    _require_successful_cells,
    _validate_dataset_lock,
    aggregate_stage,
)
from exact.experiments.schema import ExperimentConfig
from exact.impl.evaluators.builtin import BuiltinEvaluator
from exact.utils.provenance import file_provenance, sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "exact" / "default_config.yaml"


def _mapping(*, typed: bool = False, holm: bool = False) -> dict[str, Any]:
    decisions = [
        {
            "id": "primary",
            "baseline": "baseline",
            "candidates": ["candidate"],
            "metric": "F1",
        }
    ]
    if holm:
        decisions.append(
            {
                "id": "secondary",
                "baseline": "baseline",
                "candidates": ["candidate"],
                "metric": "F1",
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": "E99",
        "title": "Synthetic paper metric integration",
        "base_config": str(DEFAULT_CONFIG),
        "screen": {
            "tasks": [
                {
                    "id": "development",
                    "split_role": "development",
                    "track": "synthetic",
                    "task": "pair",
                    "reference_role": "valid",
                }
            ],
            "seeds": [7],
        },
        "confirm": {
            "tasks": [
                {
                    "id": "task-1",
                    "split_role": "reporting",
                    "track": "synthetic",
                    "task": "one",
                    "reference_role": "test",
                },
                {
                    "id": "task-2",
                    "split_role": "reporting",
                    "track": "synthetic",
                    "task": "two",
                    "reference_role": "test",
                },
            ],
            "seeds": [1, 2, 3],
        },
        "arms": [
            {"id": "baseline", "role": "baseline"},
            {"id": "candidate", "role": "candidate"},
        ],
        "selection": {"decisions": decisions},
        "design": {
            "primary_comparison": "candidate_vs_baseline",
            "primary_endpoint": "macro_F1",
            "independent_unit": "source_entity",
            "power_status": "descriptive",
            "assumptions": ["synthetic"],
            "power_slices": [
                {
                    "id": f"{task}-class-equivalence",
                    "task": task,
                    "entity_kind": "class",
                    "relation": "equivalence",
                    "status": "descriptive",
                    "hypothesized_effect": 0.0,
                    "assumptions": ["synthetic fixture"],
                }
                for task in ("task-1", "task-2")
            ],
            "required_slices": (["task", "entity_kind", "relation"] if typed else ["task"]),
            "multiplicity": "holm" if holm else "none",
        },
    }


def _suite(tmp_path: Path, *, typed: bool = False, holm: bool = False) -> LoadedSuite:
    config = ExperimentConfig.model_validate(_mapping(typed=typed, holm=holm))
    source_path = tmp_path / "E99.yaml"
    source_path.write_text("synthetic: true\n", encoding="utf-8")
    return LoadedSuite(
        suite_id="paper-metrics-suite",
        baseline_id=config.baseline_id,
        sources=(ExperimentSource(config=config, path=source_path),),
        suite_path=None,
        suite_hash="suite-sha",
        dataset_lock=None,
        dataset_lock_hash=None,
        specification={
            "path": str((tmp_path / "specs" / "experiments").resolve()),
            "sha256": "a" * 64,
            "files": 1,
            "algorithm": "sha256-length-prefixed-v1",
        },
    )


def _write_table(path: Path, header: str, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows, ""]), encoding="utf-8")
    return path


def _cell_artifacts(
    run: Path,
    reference: Path,
    *,
    correct: bool,
    typed: bool,
) -> dict[str, float]:
    if typed:
        alignment = _write_table(
            run / "alignment" / "maps_global.tsv",
            "SrcEntity\tTgtEntity\tScore\tRelation\tSrcKind\tTgtKind",
            [f"source\t{'target' if correct else 'wrong'}\t0.9\t=\tclass\tclass"],
        )
    else:
        alignment = _write_table(
            run / "alignment" / "maps_global.tsv",
            "SrcEntity\tTgtEntity\tScore",
            [f"source\t{'target' if correct else 'wrong'}\t0.9"],
        )
    builtin = BuiltinEvaluator.global_eval(alignment, reference)
    report = {
        "builtin": {key: builtin[key] for key in ("P", "R", "F1")},
        "meta": {
            "refs": {
                "alignment": file_provenance(alignment),
                "full_reference": file_provenance(reference),
            }
        },
    }
    report_path = run / "evaluation" / "evaluation_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return {key: float(builtin[key]) for key in ("P", "R", "F1")}


def _records(
    tmp_path: Path, *, typed: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    references: dict[str, Path] = {}
    for task in ("task-1", "task-2"):
        if typed:
            references[task] = _write_table(
                tmp_path / "references" / f"{task}.tsv",
                "SrcEntity\tTgtEntity\tRelation\tSrcKind\tTgtKind",
                ["source\ttarget\t=\tclass\tclass"],
            )
        else:
            references[task] = _write_table(
                tmp_path / "references" / f"{task}.tsv",
                "SrcEntity\tTgtEntity",
                ["source\ttarget"],
            )
    records: list[dict[str, Any]] = []
    for arm in ("baseline", "candidate"):
        for task in ("task-1", "task-2"):
            for seed in (1, 2, 3):
                run = tmp_path / "runs" / arm / task / f"seed-{seed}"
                metrics = _cell_artifacts(
                    run,
                    references[task],
                    correct=arm == "candidate",
                    typed=typed,
                )
                records.append(
                    {
                        "experiment_id": "E99",
                        "arm_id": arm,
                        "task_id": task,
                        "seed": seed,
                        "stage": "confirm",
                        "status": "complete",
                        "output_dir": str(run),
                        "candidate_pool_fingerprint": f"pool-{task}",
                        "metric_error": None,
                        "metrics": metrics,
                    }
                )
    return records, references


def test_harness_bootstrap_is_one_task_macro_row_with_raw_p_value(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    records, _references = _records(tmp_path)

    rows = _paired_bootstrap_rows(suite, records, stage="confirm", resamples=100, seed=11)

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "complete"
    assert row["metric"] == "task_macro_global_F1"
    assert row["delta"] == pytest.approx(1.0)
    assert row["n_tasks"] == 2
    assert row["n_cells"] == 6
    assert row["n_units"] == 2
    assert row["paired_seeds"] == [1, 2, 3]
    assert row["p_value"] == pytest.approx(1 / 101)
    assert row["p_value_adjustment"] == "none"


def test_declared_missing_cells_and_tampered_refs_are_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path)
    records, references = _records(tmp_path)
    missing = [
        record
        for record in records
        if not (
            record["arm_id"] == "candidate"
            and record["task_id"] == "task-2"
            and record["seed"] == 3
        )
    ]

    missing_row = _paired_bootstrap_rows(suite, missing, stage="confirm", resamples=10, seed=2)[0]
    assert missing_row["status"] == "unavailable"
    assert missing_row["reason_code"] == "unequal_or_missing_cells"
    assert "task-2/seed-3" not in missing_row["observed_cells"]["candidate"]

    references["task-1"].write_text(
        references["task-1"].read_text(encoding="utf-8") + "changed\trow\n",
        encoding="utf-8",
    )
    tampered_row = _paired_bootstrap_rows(suite, records, stage="confirm", resamples=10, seed=2)[0]
    assert tampered_row["status"] == "unavailable"
    assert tampered_row["reason_code"] == "artifact_verification_failed"
    assert tampered_row["fatal"] is True
    assert "SHA-256 mismatch" in tampered_row["reason"]


def test_fixed_retrieval_mismatch_writes_unavailable_row_before_failure(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path)
    records, _references = _records(tmp_path)
    records[-1]["candidate_pool_fingerprint"] = "different-pool"
    stage_root = tmp_path / "aggregates"

    with pytest.raises(ValueError, match="paired inference failed after writing"):
        _finalize_stage_reports(
            suite,
            stage="confirm",
            stage_root=stage_root,
            records=records,
        )

    payload = json.loads((stage_root / "paired_bootstrap.json").read_text(encoding="utf-8"))
    assert payload["rows"][0]["status"] == "unavailable"
    assert payload["rows"][0]["reason_code"] == "candidate_pool_mismatch"


def test_e05_pool_treatments_may_differ_but_each_arm_is_seed_stable() -> None:
    cells = {
        "baseline": {
            ("task", 1): {"candidate_pool_fingerprint": "base"},
            ("task", 2): {"candidate_pool_fingerprint": "base"},
        },
        "candidate": {
            ("task", 1): {"candidate_pool_fingerprint": "treatment"},
            ("task", 2): {"candidate_pool_fingerprint": "treatment"},
        },
    }

    status, _fingerprints, error = _candidate_pool_guard("E05", cells)
    assert status == "allowed_retrieval_treatment"
    assert error is None

    cells["candidate"][("task", 2)] = {"candidate_pool_fingerprint": "drift"}
    status, _fingerprints, error = _candidate_pool_guard("E05", cells)
    assert status == "retrieval_treatment_seed_drift"
    assert error is not None


def test_fixed_retrieval_compares_arms_within_each_task_seed() -> None:
    cells = {
        "baseline": {
            ("task", 1): {"candidate_pool_fingerprint": "seed-1"},
            ("task", 2): {"candidate_pool_fingerprint": "seed-2"},
        },
        "candidate": {
            ("task", 1): {"candidate_pool_fingerprint": "seed-1"},
            ("task", 2): {"candidate_pool_fingerprint": "seed-2"},
        },
    }

    status, _fingerprints, error = _candidate_pool_guard("E18", cells)
    assert status == "matched_fixed_retrieval"
    assert error is None

    cells["candidate"][("task", 2)] = {"candidate_pool_fingerprint": "wrong"}
    status, _fingerprints, error = _candidate_pool_guard("E18", cells)
    assert status == "candidate_pool_mismatch"
    assert "seed-2" in str(error)


def test_e20_allows_seed_specific_retrieval_treatment_pools() -> None:
    cells = {
        "zero_shot": {
            ("task", 1): {"candidate_pool_fingerprint": "zero-1"},
            ("task", 2): {"candidate_pool_fingerprint": "zero-2"},
        },
        "finetuned": {
            ("task", 1): {"candidate_pool_fingerprint": "trained-1"},
            ("task", 2): {"candidate_pool_fingerprint": "trained-2"},
        },
    }

    status, _fingerprints, error = _candidate_pool_guard("E20", cells)
    assert status == "allowed_retrieval_treatment"
    assert error is None


def test_e17_retrieval_treatment_requires_distinct_design_provenance() -> None:
    cells = {
        "rolling": {
            ("task", 1): {
                "candidate_pool_fingerprint": "rolling-pool",
                "candidate_pool_design_hash": "rolling-design",
            }
        },
        "stack_all": {
            ("task", 1): {
                "candidate_pool_fingerprint": "stack-pool",
                "candidate_pool_design_hash": "stack-design",
            }
        },
    }

    status, _fingerprints, error = _candidate_pool_guard("E17", cells)
    assert status == "allowed_retrieval_treatment"
    assert error is None

    cells["stack_all"][("task", 1)]["candidate_pool_design_hash"] = "rolling-design"
    status, _fingerprints, error = _candidate_pool_guard("E17", cells)
    assert status == "unbound_retrieval_treatment_change"
    assert error is not None


def test_explicit_enriched_artifacts_emit_typed_rows_and_legacy_records_unavailability(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path, typed=True)
    typed_records, _references = _records(tmp_path / "typed", typed=True)
    typed_rows = _paired_bootstrap_rows(suite, typed_records, stage="confirm", resamples=50, seed=4)

    assert len(typed_rows) == 3
    typed = [row for row in typed_rows if row["endpoint_scope"] == "typed_kind_relation"]
    assert {row["task_id"] for row in typed} == {"task-1", "task-2"}
    assert all(row["status"] == "complete" for row in typed)
    assert all(row["entity_kind"] == "class" for row in typed)
    assert all(row["relation"] == "equivalence" for row in typed)
    assert all(row["delta"] == pytest.approx(1.0) for row in typed)
    assert all(row["power_slice_id"].endswith("-class-equivalence") for row in typed)

    legacy_records, _references = _records(tmp_path / "legacy", typed=False)
    legacy_rows = _paired_bootstrap_rows(
        suite, legacy_records, stage="confirm", resamples=10, seed=4
    )
    typed_unavailable = [
        row for row in legacy_rows if row["endpoint_scope"] == "typed_kind_relation"
    ]
    assert len(typed_unavailable) == 2
    assert all(row["status"] == "unavailable" for row in typed_unavailable)
    assert all(row["reason_code"] == "typed_artifact_unavailable" for row in typed_unavailable)


def test_holm_is_applied_to_complete_confirmatory_rows_by_experiment(tmp_path: Path) -> None:
    suite = _suite(tmp_path, holm=True)
    records, _references = _records(tmp_path)
    rows = _paired_bootstrap_rows(suite, records, stage="confirm", resamples=100, seed=7)

    _apply_bootstrap_multiplicity(rows, suite, stage="confirm")

    assert len(rows) == 2
    assert all(row["p_value_adjustment"] == "holm" for row in rows)
    assert all(row["multiplicity_family_size"] == 2 for row in rows)
    assert all(row["p_value_adjusted"] == pytest.approx(2 / 101) for row in rows)


def test_aggregate_persists_metric_error_before_success_validation_fails(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path)
    run = tmp_path / "broken-run"
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "evaluation_results.json").write_text("not-json", encoding="utf-8")
    manifest = {
        "experiment_id": "E99",
        "arm_id": "baseline",
        "task_id": "task-1",
        "seed": 1,
        "stage": "confirm",
        "status": "complete",
        "fingerprint_payload": {"output_dir": str(run), "llm_required": False},
        "candidate_recall": 0.9,
        "candidate_recall_after_exact": 0.95,
        "candidate_recall_diagnostics": {"status": "available", "counts": {"hits": 9}},
        "mean_pool_size": 20.0,
        "gold_rank_p90": 3.0,
        "gold_rank_median": 2.0,
        "coverage": 0.8,
        "abstention_rate": 0.2,
        "metric_applicability": {"ranking": True},
        "explanation_reconstruction": {
            "status": "complete",
            "result_rows": 2,
            "reconstructed_rows": 2,
            "max_abs_error": 0.0,
        },
        "selection_evidence": {"schema_version": 1, "comparisons": []},
    }

    rows = aggregate_stage(
        suite,
        stage="confirm",
        output_root=tmp_path / "results",
        manifests=[manifest],
        finalize_reports=False,
    )

    assert rows[0]["metrics"] == {}
    assert rows[0]["metric_error"]["type"] == "ValueError"
    aggregate = tmp_path / "results" / suite.suite_id / "confirm" / "metrics.json"
    persisted = json.loads(aggregate.read_text(encoding="utf-8"))["rows"][0]
    assert persisted["metric_error"]["type"] == "ValueError"
    csv_path = tmp_path / "results" / suite.suite_id / "confirm" / "metrics.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        csv_row = next(csv.DictReader(stream))
    assert csv_row["llm_required"] == "False"
    assert csv_row["candidate_recall"] == "0.9"
    assert csv_row["candidate_recall_after_exact"] == "0.95"
    assert json.loads(csv_row["candidate_recall_diagnostics"])["status"] == "available"
    assert csv_row["mean_pool_size"] == "20.0"
    assert csv_row["gold_rank_p90"] == "3.0"
    assert csv_row["gold_rank_median"] == "2.0"
    assert csv_row["coverage"] == "0.8"
    assert csv_row["abstention_rate"] == "0.2"
    assert json.loads(csv_row["metric_applicability"]) == {"ranking": True}
    assert json.loads(csv_row["explanation_reconstruction"])["status"] == "complete"
    assert json.loads(csv_row["selection_evidence"])["schema_version"] == 1
    with pytest.raises(ValueError, match="metric extraction errors"):
        _require_successful_cells([manifest], stage="confirm")


def _write_lock(
    root: Path,
    *,
    descriptor_name: str,
    descriptor_revision: str,
    lock_revision: str,
) -> Path:
    descriptor = root / "descriptor.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": descriptor_name,
                "upstream": {"revision": descriptor_revision},
            }
        ),
        encoding="utf-8",
    )
    lock = root / "datasets.lock.yaml"
    lock.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "tracks": {
                    "track": {
                        "descriptor": descriptor.name,
                        "descriptor_sha256": sha256_file(descriptor),
                        "revision": lock_revision,
                        "roles": ["test"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return lock


def test_dataset_lock_binds_descriptor_name_and_declared_revision(tmp_path: Path) -> None:
    valid = _write_lock(
        tmp_path,
        descriptor_name="track",
        descriptor_revision="2026",
        lock_revision="2026",
    )
    _validate_dataset_lock(valid)

    wrong_name = _write_lock(
        tmp_path,
        descriptor_name="other",
        descriptor_revision="2026",
        lock_revision="2026",
    )
    with pytest.raises(ValueError, match="descriptor named"):
        _validate_dataset_lock(wrong_name)

    wrong_revision = _write_lock(
        tmp_path,
        descriptor_name="track",
        descriptor_revision="2025",
        lock_revision="2026",
    )
    with pytest.raises(ValueError, match="revision mismatch"):
        _validate_dataset_lock(wrong_revision)
