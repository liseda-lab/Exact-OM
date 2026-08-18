from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from exact.experiments import harness
from exact.experiments.harness import (
    ExperimentSource,
    LoadedSuite,
    RunCell,
    _component_arms,
    _metric_value,
    _model_identities,
    _path_provenance,
    _post_run_provenance,
    _prepare_cell,
    _require_successful_cells,
    _scores_by_arm,
    build_cells,
    cell_metrics,
    load_suite_or_experiment,
    run_stage,
    write_selection_record,
)
from exact.experiments.schema import BaselineManifest, ExperimentConfig, ResourceConfig
from exact.experiments.statistics import paired_bootstrap
from exact.utils.provenance import file_provenance

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "exact" / "default_config.yaml"


def _experiment_mapping(experiment_id: str = "E99") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "title": "Synthetic harness experiment",
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
                    "id": "reporting",
                    "split_role": "reporting",
                    "track": "synthetic",
                    "task": "pair",
                    "reference_role": "test",
                }
            ],
            "seeds": [7, 8, 9],
        },
        "arms": [
            {"id": "baseline", "role": "baseline"},
            {"id": "candidate", "role": "candidate"},
        ],
        "selection": {
            "decisions": [
                {
                    "id": "primary",
                    "baseline": "baseline",
                    "candidates": ["candidate"],
                    "metric": "f1",
                }
            ]
        },
        "design": {
            "primary_comparison": "candidate_vs_baseline",
            "primary_endpoint": "f1",
            "independent_unit": "source",
            "power_status": "descriptive",
            "assumptions": ["synthetic test"],
        },
    }


def _source_and_suite(
    tmp_path: Path,
    mapping: dict[str, Any] | None = None,
) -> tuple[ExperimentSource, LoadedSuite]:
    config = ExperimentConfig.model_validate(mapping or _experiment_mapping())
    source_path = tmp_path / f"{config.experiment_id}.yaml"
    source_path.write_text("synthetic: true\n", encoding="utf-8")
    source = ExperimentSource(config=config, path=source_path)
    suite = LoadedSuite(
        suite_id="synthetic-suite",
        baseline_id=config.baseline_id,
        sources=(source,),
        suite_path=None,
        suite_hash="suite-sha",
        dataset_lock=None,
        dataset_lock_hash=None,
    )
    return source, suite


def _run_cell(output_dir: Path) -> RunCell:
    return RunCell(
        suite_id="synthetic-suite",
        experiment_id="E99",
        stage="confirm",
        arm_id="baseline",
        arm_role="baseline",
        task_id="reporting",
        split_role="reporting",
        reference_role="test",
        seed=7,
        source_cap=None,
        resource=ResourceConfig(),
        output_dir=output_dir,
        resolved_config={"data": {}},
        config_hash="config-sha",
        experiment_config_hash="experiment-sha",
        design_hash="design-sha",
        selection_hash="selection-sha",
        supervision_label="target_label_free",
        resolved_supervision={},
        negative_label_policy="not_applicable",
    )


def _e17_mapping() -> dict[str, Any]:
    mapping = _experiment_mapping("E17")
    mapping["depends_on"] = ["E05", "E06"]
    mapping["arms"] = [
        {"id": "rolling", "role": "baseline"},
        {"id": "stack_all", "role": "candidate"},
    ]
    mapping["selection"] = {
        "decisions": [
            {
                "id": "primary",
                "baseline": "rolling",
                "candidates": ["stack_all"],
                "metric": "f1",
            }
        ]
    }
    mapping["composition"] = {
        "components": [
            {
                "id": "A",
                "overlay": {"candidates": {"top_k": 1}},
                "source_experiment": "E05",
                "claim": True,
            },
            {
                "id": "B",
                "overlay": {"candidates": {"top_k": 2}},
                "source_experiment": "E06",
                "claim": False,
            },
        ],
        "interactions": [
            {
                "id": "AB",
                "left": "A",
                "right": "B",
                "justification": "shared score boundary",
                "confirm": True,
            }
        ],
    }
    mapping["design"]["multiplicity"] = "holm"
    return mapping


def test_screen_resolved_config_strips_reporting_references(tmp_path: Path) -> None:
    mapping = _experiment_mapping()
    mapping["screen"]["tasks"][0] = {
        "id": "development",
        "split_role": "development",
        "reference_role": "valid",
        "overlay": {
            "data": {
                "root": str(tmp_path),
                "source": "source.owl",
                "target": "target.owl",
                "refs": {
                    "train": "train.tsv",
                    "valid": "valid.tsv",
                    "test": "test.tsv",
                    "full": "full.tsv",
                    "reporting": "reporting.tsv",
                },
                "candidate_source": "generated",
            }
        },
    }
    source, suite = _source_and_suite(tmp_path, mapping)

    cells = build_cells(
        suite,
        source,
        stage="screen",
        output_root=tmp_path / "results",
    )

    assert cells
    for cell in cells:
        data = cell.resolved_config["data"]
        assert data["reference_role"] == "valid"
        assert set(data["refs"]) == {"train", "valid"}


def test_screen_rejects_non_development_reference_alias() -> None:
    mapping = _experiment_mapping()
    mapping["screen"]["tasks"][0]["reference_role"] = "test-set"
    with pytest.raises(ValidationError, match="reporting references"):
        ExperimentConfig.model_validate(mapping)


def test_declared_missing_dataset_lock_fails(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "suite_id": "synthetic-suite",
                "baseline_id": "R_0",
                "baseline_manifest": "missing-baseline.yaml",
                "dataset_lock": "missing-lock.json",
                "experiments": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="declared dataset lock"):
        load_suite_or_experiment(suite_path=suite_path)


def _write_bound_single_config(tmp_path: Path) -> Path:
    collection = tmp_path / "experiments"
    experiment_dir = collection / "E99"
    baseline_dir = collection / "baselines"
    experiment_dir.mkdir(parents=True)
    baseline_dir.mkdir()
    descriptor = collection / "synthetic.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "descriptor_version": 1,
                "name": "synthetic",
                "provider": "local",
                "upstream": {"revision": "v1"},
                "tasks": {},
            }
        ),
        encoding="utf-8",
    )
    (collection / "datasets.lock.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "tracks": {
                    "synthetic": {
                        "descriptor": "synthetic.yaml",
                        "descriptor_sha256": file_provenance(descriptor)["sha256"],
                        "revision": "v1",
                        "roles": ["valid", "test"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "R_0.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "baseline_id": "R_0",
                "parent": "B0",
                "exact_om_version": "2.1.0",
                "pyowlcore_version": "0.2.0",
                "config": str(DEFAULT_CONFIG),
                "config_sha256": file_provenance(DEFAULT_CONFIG)["sha256"],
                "source_commit": "a" * 40,
                "source_tree_sha256": "b" * 64,
                "source_tree_files": 3,
                "status": "frozen_configuration",
                "experiment_flags": "disabled",
                "note": "synthetic frozen baseline",
            }
        ),
        encoding="utf-8",
    )
    mapping = _experiment_mapping()
    mapping["baseline_id"] = "R_0"
    mapping["base_config"] = str(DEFAULT_CONFIG)
    experiment_path = experiment_dir / "exp.yaml"
    experiment_path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return experiment_path


def test_single_config_auto_binds_and_validates_baseline_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_bound_single_config(tmp_path)
    monkeypatch.setattr(
        harness,
        "_package_versions",
        lambda: {"exact-om": "2.1.0", "pyowl-core": "0.2.0"},
    )
    monkeypatch.setattr(
        harness,
        "_git_provenance",
        lambda _root: {
            "commit": "a" * 40,
            "source_tree": {"sha256": "b" * 64, "files": 3},
        },
    )

    suite = load_suite_or_experiment(experiment_path=path)

    assert suite.baseline_manifest == path.parent.parent / "baselines" / "R_0.yaml"
    assert suite.baseline_manifest_hash
    assert suite.dataset_lock == path.parent.parent / "datasets.lock.yaml"
    assert suite.dataset_lock_hash
    assert suite.suite_hash


def test_single_config_rejects_task_role_outside_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_bound_single_config(tmp_path)
    lock_path = path.parent.parent / "datasets.lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    lock["tracks"]["synthetic"]["roles"] = ["valid"]
    lock_path.write_text(yaml.safe_dump(lock), encoding="utf-8")
    monkeypatch.setattr(
        harness,
        "_package_versions",
        lambda: {"exact-om": "2.1.0", "pyowl-core": "0.2.0"},
    )
    monkeypatch.setattr(
        harness,
        "_git_provenance",
        lambda _root: {
            "commit": "a" * 40,
            "source_tree": {"sha256": "b" * 64, "files": 3},
        },
    )

    with pytest.raises(ValueError, match="reference role 'test' is not locked"):
        load_suite_or_experiment(experiment_path=path)


def test_baseline_manifest_rejects_noncanonical_source_identity() -> None:
    mapping = {
        "schema_version": 1,
        "baseline_id": "R_0",
        "parent": "B0",
        "exact_om_version": "2.1.0",
        "pyowlcore_version": "0.2.0",
        "config": "default.yaml",
        "config_sha256": "not-a-sha",
        "source_commit": "short",
        "source_tree_sha256": "also-short",
        "source_tree_files": 0,
        "status": "frozen_configuration",
        "experiment_flags": "disabled",
        "note": "invalid identity",
    }

    with pytest.raises(ValidationError):
        BaselineManifest.model_validate(mapping)


def test_path_provenance_resolves_inputs_against_data_root(tmp_path: Path) -> None:
    for name, content in {
        "source.owl": "source",
        "target.owl": "target",
        "valid.tsv": "Src\tTgt\ns\tt\n",
    }.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    provenance = _path_provenance(
        {
            "data": {
                "root": str(tmp_path),
                "source": "source.owl",
                "target": "target.owl",
                "refs": {"valid": "valid.tsv"},
            }
        }
    )

    assert provenance["source"]["path"] == str((tmp_path / "source.owl").resolve())
    assert provenance["target"]["path"] == str((tmp_path / "target.owl").resolve())
    assert provenance["references"]["valid"]["sha256"]


def test_dirty_source_fingerprint_changes_with_source_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "exact" / "module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")

    def changed_paths(_workdir: Path, command: str, *_args: str) -> list[str]:
        return ["exact/module.py"] if command == "diff" else []

    monkeypatch.setattr(harness, "_git_null_paths", changed_paths)
    first = harness._dirty_source_fingerprint(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = harness._dirty_source_fingerprint(tmp_path)

    assert first is not None and second is not None
    assert first["sha256"] != second["sha256"]


def test_resume_rejects_nonempty_directory_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _run_cell(tmp_path / "orphan")
    cell.output_dir.mkdir()
    (cell.output_dir / "partial.txt").write_text("partial", encoding="utf-8")
    suite = LoadedSuite(
        suite_id=cell.suite_id,
        baseline_id="R_0",
        sources=(),
        suite_path=None,
        suite_hash="suite-sha",
        dataset_lock=None,
        dataset_lock_hash=None,
    )
    monkeypatch.setattr(
        harness,
        "_provenance_payload",
        lambda *_args, **_kwargs: {
            "fingerprint": "fingerprint",
            "fingerprint_payload": {},
            "git": {},
            "packages": {},
        },
    )

    with pytest.raises(FileExistsError, match="without a provenance manifest"):
        _prepare_cell(cell, suite, workdir=REPOSITORY_ROOT, resume=True)


def test_resume_validates_persisted_candidate_pool_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _run_cell(tmp_path / "complete")
    source_path = tmp_path / "source.owl"
    target_path = tmp_path / "target.owl"
    candidate_path = tmp_path / "candidates.tsv"
    source_path.write_text("source-v1", encoding="utf-8")
    target_path.write_text("target-v1", encoding="utf-8")
    candidate_path.write_text("candidates-v1", encoding="utf-8")
    pool_path = cell.output_dir / "dataset" / "candidate_pool_manifest.json"
    pool_path.parent.mkdir(parents=True)
    pool_payload = {
        "fingerprint": "pool-sha",
        "rows": 2,
        "inputs": {
            "source": file_provenance(source_path),
            "target": file_provenance(target_path),
            "candidate_file": file_provenance(candidate_path),
        },
    }
    pool_path.write_text(
        json.dumps(pool_payload),
        encoding="utf-8",
    )
    manifest = {
        "fingerprint": "fingerprint",
        "status": "complete",
        "candidate_pool_fingerprint": "pool-sha",
        "candidate_pool_manifest_provenance": file_provenance(pool_path),
    }
    cell.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    suite = LoadedSuite(
        suite_id=cell.suite_id,
        baseline_id="R_0",
        sources=(),
        suite_path=None,
        suite_hash="suite-sha",
        dataset_lock=None,
        dataset_lock_hash=None,
    )
    monkeypatch.setattr(
        harness,
        "_provenance_payload",
        lambda *_args, **_kwargs: {
            "fingerprint": "fingerprint",
            "fingerprint_payload": {},
            "git": {},
            "packages": {},
        },
    )

    reused_manifest, reused = _prepare_cell(
        cell,
        suite,
        workdir=REPOSITORY_ROOT,
        resume=True,
    )
    assert reused is True
    assert reused_manifest["candidate_pool_fingerprint"] == "pool-sha"

    pool_path.write_text(
        json.dumps({"fingerprint": "pool-sha", "rows": 3}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest sha256 changed"):
        _prepare_cell(cell, suite, workdir=REPOSITORY_ROOT, resume=True)

    pool_path.write_text(json.dumps(pool_payload), encoding="utf-8")
    source_path.write_text("source-v2", encoding="utf-8")
    with pytest.raises(ValueError, match="source input sha256 changed"):
        _prepare_cell(cell, suite, workdir=REPOSITORY_ROOT, resume=True)

    source_path.write_text("source-v1", encoding="utf-8")
    candidate_path.write_text("candidates-v2", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_file input sha256 changed"):
        _prepare_cell(cell, suite, workdir=REPOSITORY_ROOT, resume=True)


@pytest.mark.parametrize("missing_arm", ["baseline", "candidate", "control"])
def test_confirm_rejects_frozen_arm_without_confirm_stage(
    tmp_path: Path,
    missing_arm: str,
) -> None:
    mapping = _experiment_mapping()
    mapping["arms"].append(
        {
            "id": "control",
            "role": "control",
            "required_control": True,
        }
    )
    mapping["design"]["multiplicity"] = "holm"
    for arm in mapping["arms"]:
        if arm["id"] == missing_arm:
            arm["stages"] = ["screen"]
    source, suite = _source_and_suite(tmp_path, mapping)
    selection = {
        "selection_hash": "selection-sha",
        "experiments": {
            mapping["experiment_id"]: {
                "status": "selected",
                "decisions": [
                    {
                        "baseline": "baseline",
                        "selected_arm": "candidate",
                        "required_controls": [],
                    }
                ],
            }
        },
    }

    with pytest.raises(ValueError, match="not confirm-eligible"):
        build_cells(
            suite,
            source,
            stage="confirm",
            output_root=tmp_path / "results",
            selection_record=selection,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", "../escape"),
        ("arm", "../escape"),
        ("task", "../escape"),
        ("decision", "../escape"),
    ],
)
def test_path_unsafe_identifiers_are_rejected(field: str, value: str) -> None:
    mapping = _experiment_mapping()
    if field == "experiment_id":
        mapping["experiment_id"] = value
    elif field == "arm":
        mapping["arms"][0]["id"] = value
        mapping["selection"]["decisions"][0]["baseline"] = value
    elif field == "task":
        mapping["screen"]["tasks"][0]["id"] = value
    else:
        mapping["selection"]["decisions"][0]["id"] = value

    with pytest.raises(ValidationError, match="path-safe identifier"):
        ExperimentConfig.model_validate(mapping)


def test_duplicate_tasks_and_empty_matrices_are_rejected() -> None:
    duplicate = _experiment_mapping()
    duplicate["screen"]["tasks"].append(dict(duplicate["screen"]["tasks"][0]))
    with pytest.raises(ValidationError, match="task identifiers must be unique"):
        ExperimentConfig.model_validate(duplicate)

    no_arms = _experiment_mapping()
    no_arms["arms"] = []
    with pytest.raises(ValidationError, match="at least one arm"):
        ExperimentConfig.model_validate(no_arms)

    no_decisions = _experiment_mapping()
    no_decisions["selection"]["decisions"] = []
    with pytest.raises(ValidationError, match="at least one decision"):
        ExperimentConfig.model_validate(no_decisions)


def test_design_declarations_and_multiplicity_are_validated() -> None:
    empty_assumptions = _experiment_mapping()
    empty_assumptions["design"]["assumptions"] = []
    with pytest.raises(ValidationError, match="assumptions must be non-empty"):
        ExperimentConfig.model_validate(empty_assumptions)

    multiple = _experiment_mapping()
    multiple["selection"]["decisions"].append(
        {
            "id": "secondary",
            "baseline": "baseline",
            "candidates": ["candidate"],
            "metric": "mrr",
        }
    )
    with pytest.raises(ValidationError, match="require a multiplicity rule"):
        ExperimentConfig.model_validate(multiple)
    multiple["design"]["multiplicity"] = "holm"
    ExperimentConfig.model_validate(multiple)


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "duplicate_component",
        "missing_source",
        "duplicate_source",
        "self_pair",
        "unknown",
        "blank",
        "duplicate_interaction",
        "duplicate_unordered_pair",
    ],
)
def test_e17_rejects_malformed_composition(mutation: str) -> None:
    mapping = _e17_mapping()
    composition = mapping["composition"]
    if mutation == "empty":
        composition["components"] = []
        composition["interactions"] = []
    elif mutation == "duplicate_component":
        composition["components"].append(dict(composition["components"][0]))
    elif mutation == "missing_source":
        composition["components"][0]["source_experiment"] = None
    elif mutation == "duplicate_source":
        composition["components"][1]["claim"] = True
        composition["components"][1]["source_experiment"] = "E05"
    elif mutation == "self_pair":
        composition["interactions"][0]["right"] = "A"
    elif mutation == "unknown":
        composition["interactions"][0]["right"] = "missing"
    elif mutation == "blank":
        composition["interactions"][0]["justification"] = " "
    elif mutation == "duplicate_interaction":
        composition["interactions"].append(dict(composition["interactions"][0]))
    else:
        composition["interactions"].append(
            {
                "id": "BA",
                "left": "B",
                "right": "A",
                "justification": "same pair in reverse",
                "confirm": True,
            }
        )

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(mapping)


@pytest.mark.parametrize(
    "arm_id",
    ["stack_minus_A", "interaction_AB_00"],
)
def test_e17_mandatory_controls_cannot_be_downgraded(arm_id: str) -> None:
    mapping = _e17_mapping()
    mapping["arms"].append(
        {
            "id": arm_id,
            "role": "control",
            "stages": ["screen"],
            "required_control": False,
        }
    )

    with pytest.raises(ValidationError, match="must include screen and confirm"):
        ExperimentConfig.model_validate(mapping)


def test_e17_mandatory_primary_arms_require_screen_and_confirm() -> None:
    mapping = _e17_mapping()
    mapping["arms"][0]["stages"] = ["confirm"]

    with pytest.raises(ValidationError, match="must include screen and confirm"):
        ExperimentConfig.model_validate(mapping)


def test_e17_primary_decision_cannot_select_a_leaveout() -> None:
    mapping = _e17_mapping()
    mapping["selection"]["decisions"][0]["candidates"].append("stack_minus_A")

    with pytest.raises(ValidationError, match="compare only stack_all"):
        ExperimentConfig.model_validate(mapping)


@pytest.mark.parametrize("kind", ["leaveout", "interaction"])
def test_e17_exploratory_generated_arms_cannot_be_forced_into_confirm(
    kind: str,
) -> None:
    mapping = _e17_mapping()
    if kind == "leaveout":
        arm_id = "stack_minus_B"
    else:
        mapping["composition"]["interactions"][0]["confirm"] = False
        arm_id = "interaction_AB_00"
    mapping["arms"].append(
        {
            "id": arm_id,
            "role": "control",
            "stages": ["screen", "confirm"],
        }
    )

    with pytest.raises(ValidationError, match="cannot include confirm"):
        ExperimentConfig.model_validate(mapping)


def test_e17_missing_dependency_promotion_fails() -> None:
    config = ExperimentConfig.model_validate(_e17_mapping())
    with pytest.raises(ValueError, match="missing the selected promotion"):
        _component_arms(config, promoted_overlays={"E05": {}})


def test_e17_generated_matrix_respects_claim_and_confirm_flags() -> None:
    config = ExperimentConfig.model_validate(_e17_mapping())
    arms = {
        arm.id: arm
        for arm in _component_arms(
            config,
            promoted_overlays={"E05": {}, "E06": {}},
        )
    }

    assert arms["stack_minus_A"].required_control is True
    assert "confirm" in arms["stack_minus_A"].stages
    assert arms["stack_minus_B"].required_control is False
    assert arms["stack_minus_B"].stages == ["screen"]
    assert arms["interaction_AB_00"].required_control is True
    assert "confirm" in arms["interaction_AB_00"].stages


def test_e17_screen_dry_run_uses_placeholders_without_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, suite = _source_and_suite(tmp_path, _e17_mapping())
    output_root = tmp_path / "results"

    result = run_stage(
        suite,
        stage="screen",
        output_root=output_root,
        jobs=1,
        resume=False,
        workdir=REPOSITORY_ROOT,
        dry_run=True,
    )

    assert result is None
    assert "RUN\tE17/screen/stack_all" in capsys.readouterr().out
    assert not output_root.exists()


def test_metric_lookup_requires_exact_or_unique_suffix() -> None:
    assert _metric_value({"f1": 0.8, "global.f1": 0.9}, "f1") == 0.8
    assert _metric_value({"global.f1": 0.9}, "f1") == 0.9
    with pytest.raises(ValueError, match="ambiguous"):
        _metric_value({"global.f1": 0.9, "local.f1": 0.1}, "f1")


def test_metric_extraction_ignores_derived_csv_and_run_stats(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "evaluation_results.json").write_text(
        json.dumps(
            {
                "builtin": {"P": 0.9, "R": 0.9, "F1": 0.9},
                "meta": {"refs": {"full_reference": {"path": "unused"}}},
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "evaluation_results.csv").write_text("Metric,Value\nF1,0.2\n", encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"f1": 0.1}), encoding="utf-8")
    (tmp_path / "run_stats.json").write_text(json.dumps({"f1": 0.0}), encoding="utf-8")

    assert cell_metrics(tmp_path) == {"P": 0.9, "R": 0.9, "F1": 0.9}


def test_selection_rejects_complete_cell_missing_endpoint() -> None:
    records = [
        {
            "experiment_id": "E99",
            "arm_id": "baseline",
            "task_id": "one",
            "seed": 7,
            "status": "complete",
            "metrics": {"f1": 0.5},
        },
        {
            "experiment_id": "E99",
            "arm_id": "baseline",
            "task_id": "two",
            "seed": 7,
            "status": "complete",
            "metrics": {},
        },
    ]
    with pytest.raises(ValueError, match="complete cells are missing"):
        _scores_by_arm(records, experiment_id="E99", metric="f1")


def test_complete_cell_requires_at_least_one_finite_metric(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    manifest = {
        "experiment_id": "E99",
        "arm_id": "baseline",
        "task_id": "task",
        "seed": 7,
        "status": "complete",
        "fingerprint_payload": {"output_dir": str(output_dir)},
    }
    with pytest.raises(ValueError, match="no finite metrics"):
        _require_successful_cells([manifest], stage="confirm")

    evaluation = output_dir / "evaluation"
    evaluation.mkdir()
    (evaluation / "evaluation_results.json").write_text(
        json.dumps(
            {
                "builtin": {"F1": 0.5},
                "meta": {"refs": {"full_reference": {"path": "unused"}}},
            }
        ),
        encoding="utf-8",
    )
    _require_successful_cells([manifest], stage="confirm")


@pytest.mark.parametrize("stage", ["screen", "confirm"])
def test_stage_raises_when_any_cell_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    source, suite = _source_and_suite(tmp_path)
    selection = {
        "selection_hash": "selection-sha",
        "experiments": {
            source.config.experiment_id: {
                "status": "selected",
                "decisions": [],
            }
        },
    }
    monkeypatch.setattr(
        harness,
        "load_and_validate_selection",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(harness, "build_dataset_inventory", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(harness, "build_cells", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(
        harness,
        "run_cells",
        lambda *_args, **_kwargs: [
            {
                "experiment_id": "E99",
                "arm_id": "baseline",
                "task_id": "task",
                "seed": 7,
                "status": "failed",
            }
        ],
    )

    with pytest.raises(ValueError, match="failed or incomplete"):
        run_stage(
            suite,
            stage=stage,
            output_root=tmp_path / "results",
            jobs=1,
            resume=False,
            workdir=REPOSITORY_ROOT,
            selection_record_path=(tmp_path / "selection.json" if stage == "confirm" else None),
        )


def test_model_identity_manifest_redacts_secret_profile_fields() -> None:
    identities = _model_identities(
        {
            "pipeline": [],
            "llm": {
                "profiles": {
                    "hosted": {
                        "backend": "remote",
                        "model": "provider/model@revision",
                        "tokenizer": "provider/tokenizer@revision",
                        "api_base": (
                            "https://user:SECRET@example.invalid:8443/v1" "?api_key=SECRET#fragment"
                        ),
                        "api_key_env": "SECRET_ENV",
                        "api_key_path": "/secret/key",
                        "extra_headers": {"Authorization": "SECRET"},
                        "provider": {"token": "SECRET"},
                    }
                }
            },
        }
    )

    assert identities["llm_profiles"]["hosted"] == {
        "api_base": "https://example.invalid:8443/v1",
        "backend": "remote",
        "model": "provider/model@revision",
        "tokenizer": "provider/tokenizer@revision",
    }
    assert "SECRET" not in json.dumps(identities)


def test_runtime_llm_usage_retains_counts_but_redacts_secrets(tmp_path: Path) -> None:
    cell = _run_cell(tmp_path / "runtime")
    stats_dir = cell.output_dir / "stats"
    stats_dir.mkdir(parents=True)
    (stats_dir / "run_stats.json").write_text(
        json.dumps(
            {
                "llm_usage": {
                    "model": "provider/model@revision",
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "api_base": ("https://user:SECRET@example.invalid/v1?api_key=SECRET"),
                    "headers": {"Authorization": "SECRET"},
                    "request_id": "SECRET",
                    "non_finite": float("nan"),
                }
            }
        ),
        encoding="utf-8",
    )

    usage = _post_run_provenance(cell)["llm_usage"]
    assert usage == {
        "model": "provider/model@revision",
        "input_tokens": 12,
        "output_tokens": 3,
        "api_base": "https://example.invalid/v1",
    }
    assert "SECRET" not in json.dumps(usage)


def test_paired_bootstrap_identical_positive_and_deterministic() -> None:
    identical = paired_bootstrap([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], resamples=200)
    assert identical.delta == identical.ci_low == identical.ci_high == 0.0

    positive = paired_bootstrap(
        {"a": 0.0, "b": 1.0},
        {"a": 1.0, "b": 2.0},
        resamples=200,
        seed=11,
    )
    assert positive.delta == positive.ci_low == positive.ci_high == 1.0
    assert positive == paired_bootstrap(
        {"a": 0.0, "b": 1.0},
        {"a": 1.0, "b": 2.0},
        resamples=200,
        seed=11,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_paired_bootstrap_rejects_non_finite_inputs(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        paired_bootstrap([0.0, value], [1.0, 2.0], resamples=10)


def test_paired_bootstrap_rejects_overflowing_deltas() -> None:
    with pytest.raises(ValueError, match="paired deltas.*non-finite"):
        paired_bootstrap([-1e308], [1e308], resamples=10)


def test_frozen_selection_cannot_be_overwritten_after_confirm_artifact(
    tmp_path: Path,
) -> None:
    suite = LoadedSuite(
        suite_id="synthetic-suite",
        baseline_id="R_0",
        sources=(),
        suite_path=None,
        suite_hash="suite-sha",
        dataset_lock=None,
        dataset_lock_hash=None,
    )
    artifact = tmp_path / suite.suite_id / "confirm" / "runs" / "artifact.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="confirm artifacts"):
        write_selection_record(suite, {}, output_root=tmp_path)
