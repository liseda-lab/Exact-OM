from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from exact.core.entities.configs.config import ConfigModel
from exact.experiments import harness
from exact.experiments.harness import (
    ExperimentSource,
    LoadedSuite,
    RunCell,
    _component_arms,
    _llm_runtime_required,
    _metric_value,
    _model_identities,
    _path_provenance,
    _post_run_provenance,
    _prepare_cell,
    _require_successful_cells,
    _scores_by_arm,
    _validate_paired_llm_identities,
    build_cells,
    cell_metrics,
    experiment_design_hash,
    hash_payload,
    load_and_bind_confirmed_components,
    load_suite_or_experiment,
    run_stage,
    select_experiment,
    selected_experiment_overlays,
    write_selection_record,
)
from exact.experiments.schema import (
    BaselineManifest,
    ExperimentConfig,
    ResourceConfig,
    load_experiment,
)
from exact.experiments.statistics import paired_bootstrap
from exact.utils.provenance import file_provenance

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "exact" / "default_config.yaml"


def _complete_model_lock() -> dict[str, Any]:
    requested = (
        "sentence-transformers/all-MiniLM-L6-v2",
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "BAAI/bge-large-en-v1.5",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Xenova/gpt-4o",
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen/Qwen3-235B-A22B-2507",
        "meta-llama/Llama-3.3-70B-Instruct",
        "deepseek-ai/DeepSeek-V3.1",
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "models": {
            f"model_{index}": {
                "requested_id": model_id,
                "resolved_revision": f"{index + 1:x}" * 40,
            }
            for index, model_id in enumerate(requested)
        },
    }


def _specification(tmp_path: Path) -> dict[str, Any]:
    return {
        "path": str((tmp_path / "specs" / "experiments").resolve()),
        "sha256": "a" * 64,
        "files": 1,
        "algorithm": "sha256-length-prefixed-v1",
    }


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
            "power_slices": [
                {
                    "id": "reporting-class-equivalence",
                    "task": "reporting",
                    "entity_kind": "class",
                    "relation": "equivalence",
                    "status": "descriptive",
                    "hypothesized_effect": 0.0,
                    "assumptions": ["synthetic fixture"],
                }
            ],
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
        specification=_specification(tmp_path),
        model_lock=tmp_path / "models.lock.yaml",
        model_lock_hash="model-lock-sha",
        model_lock_payload=_complete_model_lock(),
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
        reference_completeness="complete",
        seed=7,
        source_cap=None,
        resource=ResourceConfig(),
        output_dir=output_dir,
        resolved_config={
            "config_version": 2,
            "data": {},
            "pipeline": [
                {
                    "name": "PairAdaptiveSemanticScorer",
                    "params": {"generate_llm_rationales": False},
                }
            ],
        },
        config_hash="config-sha",
        experiment_config_hash="experiment-sha",
        design_hash="design-sha",
        selection_hash="selection-sha",
        supervision_label="target_label_free",
        resolved_supervision={},
        negative_label_policy="not_applicable",
    )


def test_specification_tree_uses_exact_length_prefixed_wire_format(tmp_path: Path) -> None:
    specs = tmp_path / "specs" / "experiments"
    nested = specs / "nested"
    nested.mkdir(parents=True)
    (specs / "z.md").write_bytes(b"z\n")
    (nested / "a.md").write_bytes(b"alpha\x00beta")
    (specs / "ignored.txt").write_text("not part of the specification tree", encoding="utf-8")
    expected = hashlib.sha256()
    for relative, content in (
        ("specs/experiments/nested/a.md", b"alpha\x00beta"),
        ("specs/experiments/z.md", b"z\n"),
    ):
        encoded = relative.encode("utf-8")
        expected.update(len(encoded).to_bytes(8, "big"))
        expected.update(encoded)
        expected.update(len(content).to_bytes(8, "big"))
        expected.update(content)

    identity = harness.specification_tree_identity(specs, relative_to=tmp_path)

    assert identity == {
        "path": str(specs.resolve()),
        "sha256": expected.hexdigest(),
        "files": 2,
        "algorithm": "sha256-length-prefixed-v1",
    }


def test_frozen_constants_are_bound_into_experiment_design_hash(tmp_path: Path) -> None:
    mapping = _experiment_mapping()
    mapping["frozen_constants"] = {"tie_rule": ["score", "source_iri"]}
    source, suite = _source_and_suite(tmp_path, mapping)
    first = experiment_design_hash(
        source,
        baseline_manifest_hash=suite.baseline_manifest_hash,
    )
    changed = source.config.model_copy(
        update={"frozen_constants": {"tie_rule": ["score", "target_iri"]}}
    )
    changed_source = ExperimentSource(config=changed, path=source.path)
    second = experiment_design_hash(
        changed_source,
        baseline_manifest_hash=suite.baseline_manifest_hash,
    )
    assert first != second


def test_confirmation_rejects_changed_specification_identity(tmp_path: Path) -> None:
    _source, suite = _source_and_suite(tmp_path)
    record = {
        "stage": "screen",
        "suite_id": suite.suite_id,
        "suite_hash": suite.suite_hash,
        "baseline_id": suite.baseline_id,
        "baseline_manifest_hash": suite.baseline_manifest_hash,
        "dataset_lock_hash": suite.dataset_lock_hash,
        "model_lock_hash": suite.model_lock_hash,
        "specification": {**suite.specification, "sha256": "b" * 64},
    }
    record["selection_hash"] = harness.hash_payload(record)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="specification tree changed"):
        harness.load_and_validate_selection(path, suite)


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


def _confirmed_component_fixture(
    tmp_path: Path,
) -> tuple[LoadedSuite, Path, Path, dict[str, Any]]:
    dependency_mapping = _experiment_mapping("E05")
    dependency_mapping["arms"][1]["overlay"] = {"selector": {"emit_candidate_scores": True}}
    dependency_config = ExperimentConfig.model_validate(dependency_mapping)
    dependency_path = tmp_path / "E05.yaml"
    dependency_path.write_text("fixture: E05\n", encoding="utf-8")
    dependency = ExperimentSource(config=dependency_config, path=dependency_path)

    e17_mapping = _e17_mapping()
    e17_mapping["depends_on"] = ["E05"]
    e17_mapping["composition"]["components"] = [e17_mapping["composition"]["components"][0]]
    e17_mapping["composition"]["interactions"] = []
    e17_mapping["selection"]["decisions"][0]["required_controls"] = ["stack_minus_A"]
    e17_config = ExperimentConfig.model_validate(e17_mapping)
    e17_path = tmp_path / "E17.yaml"
    e17_path.write_text("fixture: E17\n", encoding="utf-8")
    e17 = ExperimentSource(config=e17_config, path=e17_path)

    pool_hash = "c" * 64
    suite = LoadedSuite(
        suite_id="synthetic-suite",
        baseline_id="R_0",
        sources=(dependency, e17),
        suite_path=None,
        suite_hash="suite-sha",
        dataset_lock=None,
        dataset_lock_hash="d" * 64,
        specification=_specification(tmp_path),
        model_lock=tmp_path / "models.lock.yaml",
        model_lock_hash="e" * 64,
        model_lock_payload=_complete_model_lock(),
        baseline_manifest_hash="f" * 64,
    )
    artifact_path = tmp_path / "selector-artifact.json"
    artifact_path.write_text('{"fixture":true}\n', encoding="utf-8")
    overlay = {
        "selector": {
            "emit_candidate_scores": True,
            "rerank": {"artifact": artifact_path.name},
        }
    }
    design_hash = experiment_design_hash(
        dependency,
        baseline_manifest_hash=suite.baseline_manifest_hash,
    )
    selection: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "suite_hash": suite.suite_hash,
        "stage": "screen",
        "baseline_id": suite.baseline_id,
        "baseline_manifest_hash": suite.baseline_manifest_hash,
        "dataset_lock_hash": suite.dataset_lock_hash,
        "model_lock_hash": suite.model_lock_hash,
        "specification": suite.specification,
        "experiments": {
            "E05": {
                "status": "selected",
                "experiment_config_hash": dependency.raw_hash(),
                "design_hash": design_hash,
                "combined_selected_overlay": overlay,
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
    selection["selection_hash"] = hash_payload(selection)
    selection_path = tmp_path / "screen-selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    confirmation_rows: list[dict[str, Any]] = []
    for arm_id in ("baseline", "candidate"):
        for seed in (7, 8, 9):
            output_dir = tmp_path / "confirm-runs" / arm_id / f"seed-{seed}"
            output_dir.mkdir(parents=True)
            candidate_manifest_path = output_dir / "candidate_pool_manifest.json"
            candidate_manifest_path.write_text(
                json.dumps({"fingerprint": "9" * 64, "cell": [arm_id, seed]}),
                encoding="utf-8",
            )
            fingerprint_payload = {
                "experiment_config_hash": dependency.raw_hash(),
                "design_hash": design_hash,
                "selection_hash": selection["selection_hash"],
                "baseline_manifest_hash": suite.baseline_manifest_hash,
                "dataset_lock_hash": suite.dataset_lock_hash,
                "model_lock_hash": suite.model_lock_hash,
                "specification_sha256": suite.specification["sha256"],
                "candidate_pool_design_hash": pool_hash,
            }
            manifest = {
                "experiment_id": "E05",
                "stage": "confirm",
                "arm_id": arm_id,
                "task_id": "reporting",
                "seed": seed,
                "status": "complete",
                "fingerprint_payload": fingerprint_payload,
                "fingerprint": hash_payload(fingerprint_payload),
                "candidate_pool_fingerprint": "9" * 64,
                "candidate_pool_manifest_provenance": file_provenance(candidate_manifest_path),
            }
            (output_dir / harness.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
            confirmation_rows.append(
                {
                    "output_dir": str(output_dir),
                    "experiment_id": "E05",
                    "arm_id": arm_id,
                    "task_id": "reporting",
                    "seed": seed,
                    "status": "complete",
                    "metric_error": None,
                }
            )
    confirmation_path = tmp_path / "confirm-metrics.json"
    confirmation_path.write_text(
        json.dumps({"schema_version": 1, "rows": confirmation_rows}),
        encoding="utf-8",
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "stage": "confirm",
        "baseline_id": suite.baseline_id,
        "dataset_lock_hash": suite.dataset_lock_hash,
        "model_lock_hash": suite.model_lock_hash,
        "specification": suite.specification,
        "composition_candidate_pool_design_hash": pool_hash,
        "components": {
            "E05": {
                "status": "confirmed",
                "experiment_config_hash": dependency.raw_hash(),
                "design_hash": experiment_design_hash(
                    dependency,
                    baseline_manifest_hash=suite.baseline_manifest_hash,
                ),
                "selection_record": selection_path.name,
                "selection_hash": selection["selection_hash"],
                "confirmation_manifest": confirmation_path.name,
                "confirmation_manifest_hash": harness.sha256_file(confirmation_path),
                "selected_arm": "candidate",
                "overlay": overlay,
                "overlay_hash": hash_payload(overlay),
                "pool_contract": {
                    "mode": "bound",
                    "candidate_pool_design_hash": pool_hash,
                },
                "artifacts": [
                    {
                        **file_provenance(artifact_path),
                        "path": artifact_path.name,
                        "candidate_pool_design_hash": pool_hash,
                    }
                ],
            }
        },
    }
    record["confirmed_components_hash"] = hash_payload(record)
    record_path = tmp_path / "confirmed-components.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return suite, record_path, artifact_path, record


def test_e17_confirmed_components_bind_valid_frozen_overlay(tmp_path: Path) -> None:
    suite, record_path, _artifact_path, record = _confirmed_component_fixture(tmp_path)

    bound = load_and_bind_confirmed_components(record_path, suite)

    assert bound.confirmed_components_hash == record["confirmed_components_hash"]
    assert bound.confirmed_component_overlays == {"E05": record["components"]["E05"]["overlay"]}


def test_e17_confirmed_components_reject_record_tampering(tmp_path: Path) -> None:
    suite, record_path, _artifact_path, record = _confirmed_component_fixture(tmp_path)
    record["components"]["E05"]["selected_arm"] = "baseline"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="confirmed_components_hash"):
        load_and_bind_confirmed_components(record_path, suite)


def test_e17_confirmed_components_reject_pool_mismatch(tmp_path: Path) -> None:
    suite, record_path, _artifact_path, record = _confirmed_component_fixture(tmp_path)
    record["components"]["E05"]["pool_contract"]["candidate_pool_design_hash"] = "9" * 64
    record.pop("confirmed_components_hash")
    record["confirmed_components_hash"] = hash_payload(record)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible candidate pool"):
        load_and_bind_confirmed_components(record_path, suite)


def test_e17_confirmed_components_reject_mutated_artifact(tmp_path: Path) -> None:
    suite, record_path, artifact_path, _record = _confirmed_component_fixture(tmp_path)
    artifact_path.write_text('{"fixture":false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_and_bind_confirmed_components(record_path, suite)


def test_e17_confirmed_components_reject_mutated_selection_artifact(
    tmp_path: Path,
) -> None:
    suite, record_path, _artifact_path, record = _confirmed_component_fixture(tmp_path)
    selection_path = tmp_path / record["components"]["E05"]["selection_record"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["experiments"]["E05"]["decisions"][0]["selected_arm"] = "baseline"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="selection record/hash is not authentic"):
        load_and_bind_confirmed_components(record_path, suite)


def test_e17_confirmed_components_reject_incomplete_confirmation_matrix(
    tmp_path: Path,
) -> None:
    suite, record_path, _artifact_path, record = _confirmed_component_fixture(tmp_path)
    component = record["components"]["E05"]
    confirmation_path = tmp_path / component["confirmation_manifest"]
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    confirmation["rows"].pop()
    confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")
    component["confirmation_manifest_hash"] = harness.sha256_file(confirmation_path)
    record.pop("confirmed_components_hash")
    record["confirmed_components_hash"] = hash_payload(record)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="confirmation cells differ from the frozen matrix"):
        load_and_bind_confirmed_components(record_path, suite)


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
        assert cell.resolved_config["candidates"]["encoder_revision"] == "1" * 40
        params = cell.resolved_config["pipeline"][0]["params"]
        assert params["lexical_model_revision"] == "2" * 40
        assert params["context_model_revision"] == "3" * 40
        assert params["llm_model_revision"] == "5" * 40
        profiles = cell.resolved_config["llm"]["profiles"]
        assert profiles["local_verbaliser_default"]["revision"] == "4" * 40
        assert profiles["local_llm_default"]["revision"] == "5" * 40


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
                "specification": {
                    "path": "missing-specs",
                    "sha256": "a" * 64,
                    "files": 1,
                },
                "dataset_lock": "missing-lock.json",
                "model_lock": "missing-model-lock.yaml",
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
    (collection / "models.lock.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "complete",
                "models": {
                    "candidate": {
                        "requested_id": "sentence-transformers/all-MiniLM-L6-v2",
                        "resolved_revision": "c" * 40,
                    },
                    "lexical": {
                        "requested_id": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                        "resolved_revision": "c" * 40,
                    },
                    "context": {
                        "requested_id": "BAAI/bge-large-en-v1.5",
                        "resolved_revision": "c" * 40,
                    },
                    "verbaliser": {
                        "requested_id": "Qwen/Qwen2.5-3B-Instruct",
                        "resolved_revision": "c" * 40,
                    },
                    "llm": {
                        "requested_id": "Qwen/Qwen2.5-7B-Instruct",
                        "resolved_revision": "c" * 40,
                    },
                    "gpt4o_tokenizer": {
                        "requested_id": "Xenova/gpt-4o",
                        "resolved_revision": "c" * 40,
                    },
                    "qwen35_tokenizer": {
                        "requested_id": "Qwen/Qwen3.5-122B-A10B",
                        "resolved_revision": "c" * 40,
                    },
                    "qwen3_tokenizer": {
                        "requested_id": "Qwen/Qwen3-235B-A22B-2507",
                        "resolved_revision": "c" * 40,
                    },
                    "llama33_tokenizer": {
                        "requested_id": "meta-llama/Llama-3.3-70B-Instruct",
                        "resolved_revision": "c" * 40,
                    },
                    "deepseek_tokenizer": {
                        "requested_id": "deepseek-ai/DeepSeek-V3.1",
                        "resolved_revision": "c" * 40,
                    },
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


def test_single_config_rejects_model_lock_missing_configured_identity(
    tmp_path: Path,
) -> None:
    path = _write_bound_single_config(tmp_path)
    lock_path = path.parent.parent / "models.lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    lock["models"].pop("context")
    lock_path.write_text(yaml.safe_dump(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="missing configured runtime model identities"):
        load_suite_or_experiment(experiment_path=path)


def test_model_lock_binding_covers_arm_only_model_identity(tmp_path: Path) -> None:
    mapping = _experiment_mapping()
    mapping["arms"][1]["overlay"] = {"candidates": {"encoder": "org/arm-only"}}
    source, _suite = _source_and_suite(tmp_path, mapping)

    with pytest.raises(ValueError, match="org/arm-only"):
        harness._validate_model_lock_bindings((source,), _complete_model_lock())


def test_complete_model_lock_binds_every_runtime_loader_and_identity() -> None:
    revisions = {
        "org/candidate": "1" * 40,
        "org/lexical": "2" * 40,
        "org/context": "3" * 40,
        "org/pipeline-llm": "4" * 40,
        "org/profile-llm": "5" * 40,
        "org/profile-tokenizer": "6" * 40,
    }
    lock = {
        "schema_version": 1,
        "status": "complete",
        "models": {
            f"model_{index}": {
                "requested_id": model_id,
                "resolved_revision": revision,
            }
            for index, (model_id, revision) in enumerate(revisions.items())
        },
    }
    mapping = {
        "candidates": {"encoder": "org/candidate"},
        "pipeline": [
            {
                "name": "Semantic",
                "params": {
                    "lexical_model_name": "org/lexical",
                    "context_model_name": "org/context",
                    "llm_model_name": "org/pipeline-llm",
                },
            }
        ],
        "llm": {
            "profiles": {
                "local": {"backend": "local_hf", "model": "org/profile-llm"},
                "hosted": {
                    "backend": "openrouter",
                    "model": "hosted/mutable",
                    "tokenizer": "org/profile-tokenizer",
                },
            }
        },
    }

    bound = harness._bind_model_lock_revisions(mapping, lock, require_complete=True)

    assert bound["candidates"]["encoder_revision"] == revisions["org/candidate"]
    params = bound["pipeline"][0]["params"]
    assert params["lexical_model_revision"] == revisions["org/lexical"]
    assert params["context_model_revision"] == revisions["org/context"]
    assert params["llm_model_revision"] == revisions["org/pipeline-llm"]
    assert bound["llm"]["profiles"]["local"]["revision"] == revisions["org/profile-llm"]
    assert "revision" not in bound["llm"]["profiles"]["hosted"]
    assert (
        bound["llm"]["profiles"]["hosted"]["tokenizer_revision"]
        == revisions["org/profile-tokenizer"]
    )

    identities = _model_identities(bound)
    assert identities["candidate_retrieval"]["revision"] == revisions["org/candidate"]
    assert identities["pipeline"][0]["context_revision"] == revisions["org/context"]
    assert identities["llm_profiles"]["local"]["revision"] == revisions["org/profile-llm"]
    assert (
        identities["llm_profiles"]["hosted"]["tokenizer_revision"]
        == revisions["org/profile-tokenizer"]
    )


def test_model_lock_binding_fails_closed_without_loadable_revision() -> None:
    lock = {
        "schema_version": 1,
        "status": "complete",
        "models": {
            "candidate": {
                "requested_id": "org/candidate",
                "artifact_sha256": "a" * 64,
            }
        },
    }

    with pytest.raises(ValueError, match="40-hex resolved_revision"):
        harness._bind_model_lock_revisions(
            {"candidates": {"encoder": "org/candidate"}},
            lock,
            require_complete=True,
        )


def test_incomplete_model_lock_is_screen_only() -> None:
    mapping = {"candidates": {"encoder": "org/candidate", "encoder_revision": None}}
    lock = {"schema_version": 1, "status": "incomplete", "models": {}}

    assert harness._bind_model_lock_revisions(mapping, lock, require_complete=False) == mapping
    with pytest.raises(ValueError, match="complete immutable model lock"):
        harness._bind_model_lock_revisions(mapping, lock, require_complete=True)


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
        specification=_specification(tmp_path),
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
        specification=_specification(tmp_path),
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


def test_selection_cannot_promote_non_deployable_diagnostic() -> None:
    mapping = _experiment_mapping()
    mapping["arms"][1].update({"role": "diagnostic", "deployable": False})

    with pytest.raises(ValidationError, match="cannot promote non-deployable"):
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


def _e17_component_removal_records() -> list[dict[str, Any]]:
    audit = {
        "status": "complete",
        "result_rows": 4,
        "reconstructed_rows": 4,
        "max_abs_error": 0.0,
    }
    return [
        {
            "experiment_id": "E17",
            "stage": "screen",
            "arm_id": arm,
            "task_id": "development",
            "seed": 7,
            "status": "complete",
            "metrics": {"F1": f1, "local.MRR": mrr},
            "wall_seconds": 100.0,
            "explanation_reconstruction": dict(audit),
        }
        for arm, f1, mrr in (
            ("rolling", 0.79, 0.79),
            ("stack_all", 0.80, 0.80),
            ("stack_minus_A", 0.80, 0.80),
            ("stack_minus_B", 0.80, 0.80),
        )
    ]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"metrics": {"F1": 0.806, "local.MRR": 0.80}}, "macro_f1_improvement"),
        ({"metrics": {"F1": 0.80, "local.MRR": 0.806}}, "local_mrr_improvement"),
        (
            {"metrics": {"F1": 0.796, "local.MRR": 0.80}, "wall_seconds": 80.0},
            "cost_reduction_with_quality_non_inferiority",
        ),
        (
            {
                "stack_audit_failure": True,
            },
            "repairs_explanation_reconstruction_hard_guard",
        ),
    ],
)
def test_e17_leaveout_removal_triggers_fail_before_freeze(
    tmp_path: Path,
    mutation: dict[str, Any],
    reason: str,
) -> None:
    mapping = _e17_mapping()
    mapping["frozen_constants"] = {
        "component_removal": {
            "macro_f1_improvement": 0.005,
            "local_mrr_improvement": 0.005,
            "cost_reduction_fraction": 0.20,
            "cost_non_inferiority_margin": -0.005,
            "cost_metric": "wall_seconds",
            "hard_guard_repairs_trigger_removal": True,
            "reconstruction_tolerance": 1e-6,
            "decision_data": "development_only",
        }
    }
    source, _suite = _source_and_suite(tmp_path, mapping)
    records = _e17_component_removal_records()
    leaveout = next(record for record in records if record["arm_id"] == "stack_minus_A")
    if mutation.get("stack_audit_failure"):
        stack = next(record for record in records if record["arm_id"] == "stack_all")
        stack["explanation_reconstruction"]["max_abs_error"] = 1e-4
    else:
        leaveout.update(mutation)

    with pytest.raises(ValueError, match=reason):
        select_experiment(
            source,
            records,
            promoted_component_overlays={"E05": {}, "E06": {}},
        )


def test_e17_component_removal_audit_is_bound_into_selection(tmp_path: Path) -> None:
    mapping = _e17_mapping()
    mapping["frozen_constants"] = {
        "component_removal": {
            "macro_f1_improvement": 0.005,
            "local_mrr_improvement": 0.005,
            "cost_reduction_fraction": 0.20,
            "cost_non_inferiority_margin": -0.005,
            "cost_metric": "wall_seconds",
            "hard_guard_repairs_trigger_removal": True,
            "reconstruction_tolerance": 1e-6,
            "decision_data": "development_only",
        }
    }
    source, _suite = _source_and_suite(tmp_path, mapping)

    selection = select_experiment(
        source,
        _e17_component_removal_records(),
        promoted_component_overlays={"E05": {}, "E06": {}},
    )

    assert selection["component_removal_audit"]["removal_triggers"] == {}
    assert selection["component_removal_audit"]["cost_metric"] == "wall_seconds"


def test_e17_screen_dry_run_defers_without_confirmed_components_record(
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
    assert "DEFERRED_RUNTIME\tE17\tconfirmed_components_record_required" in (
        capsys.readouterr().out
    )
    assert not output_root.exists()


def test_e17_dependency_subphase_records_runtime_deferral_without_blocking_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, suite = _source_and_suite(tmp_path, _e17_mapping())
    monkeypatch.setattr(harness, "build_dataset_inventory", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(harness, "aggregate_stage", lambda *_args, **_kwargs: [])
    output_root = tmp_path / "results"

    selection_path = run_stage(
        suite,
        stage="screen",
        output_root=output_root,
        jobs=1,
        resume=False,
        workdir=REPOSITORY_ROOT,
    )

    assert selection_path is not None
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["experiments"]["E17"]["status"] == "deferred_runtime"
    assert selection["experiments"]["E17"]["reason_code"] == "confirmed_components_record_required"

    assert (
        run_stage(
            suite,
            stage="confirm",
            output_root=output_root,
            jobs=1,
            resume=False,
            workdir=REPOSITORY_ROOT,
            selection_record_path=selection_path,
        )
        is None
    )


def test_e17_bound_screen_appends_selection_after_component_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, suite = _source_and_suite(tmp_path, _e17_mapping())
    monkeypatch.setattr(harness, "build_dataset_inventory", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(harness, "aggregate_stage", lambda *_args, **_kwargs: [])
    output_root = tmp_path / "results"

    parent_path = run_stage(
        suite,
        stage="screen",
        output_root=output_root,
        jobs=1,
        resume=False,
        workdir=REPOSITORY_ROOT,
    )
    assert parent_path is not None
    parent_bytes = parent_path.read_bytes()
    parent = json.loads(parent_bytes)

    confirmation_artifact = (
        output_root / suite.suite_id / "confirm" / "runs" / "component" / "manifest.json"
    )
    confirmation_artifact.parent.mkdir(parents=True)
    confirmation_artifact.write_text("{}\n", encoding="utf-8")

    promoted = {"E05": {}, "E06": {}}
    bound = replace(
        suite,
        confirmed_components_hash="c" * 64,
        confirmed_parent_selection_hash=parent["selection_hash"],
        confirmed_component_overlays=promoted,
    )
    screened_out = harness._runtime_deferred_selection(
        source,
        bound,
        reason_code="unused",
        reason="unused",
    )
    screened_out.pop("reason_code")
    screened_out.pop("reason")
    screened_out.update(
        {
            "status": "screened_out",
            "resolved_arms_hash": hash_payload(
                {
                    arm.id: arm.overlay
                    for arm in harness._component_arms(
                        source.config,
                        promoted_overlays=promoted,
                    )
                }
            ),
        }
    )
    monkeypatch.setattr(harness, "build_cells", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        harness,
        "select_experiment",
        lambda *_args, **_kwargs: dict(screened_out),
    )

    downstream_path = run_stage(
        bound,
        stage="screen",
        output_root=output_root,
        jobs=1,
        resume=True,
        workdir=REPOSITORY_ROOT,
    )

    assert downstream_path is not None
    assert downstream_path != parent_path
    assert downstream_path.name == f"selection.e17.{'c' * 64}.json"
    assert parent_path.read_bytes() == parent_bytes
    downstream = harness.load_and_validate_selection(downstream_path, bound)
    assert downstream["lifecycle"] == {
        "phase": "e17_post_component_confirmation",
        "parent_selection": "selection.json",
        "parent_selection_hash": parent["selection_hash"],
        "confirmed_components_hash": "c" * 64,
    }
    assert downstream["experiments"]["E17"]["status"] == "screened_out"

    assert (
        run_stage(
            bound,
            stage="screen",
            output_root=output_root,
            jobs=1,
            resume=True,
            workdir=REPOSITORY_ROOT,
        )
        == downstream_path
    )
    assert (
        run_stage(
            bound,
            stage="confirm",
            output_root=output_root,
            jobs=1,
            resume=True,
            workdir=REPOSITORY_ROOT,
            selection_record_path=downstream_path,
        )
        is None
    )


def test_e17_downstream_phases_do_not_rerun_frozen_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e17_source, e17_suite = _source_and_suite(tmp_path, _e17_mapping())
    dependency_config = ExperimentConfig.model_validate(_experiment_mapping("E05"))
    dependency_path = tmp_path / "E05.yaml"
    dependency_path.write_text("synthetic: E05\n", encoding="utf-8")
    dependency = ExperimentSource(config=dependency_config, path=dependency_path)
    suite = replace(e17_suite, sources=(dependency, e17_source))
    output_root = tmp_path / "results"
    built: list[tuple[str, str]] = []
    inventoried: list[tuple[str, tuple[str, ...]]] = []

    def inventory(source_suite: LoadedSuite, *, stage: str, **_kwargs: Any) -> list[Any]:
        inventoried.append(
            (stage, tuple(source.config.experiment_id for source in source_suite.sources))
        )
        return []

    monkeypatch.setattr(harness, "build_dataset_inventory", inventory)
    monkeypatch.setattr(harness, "aggregate_stage", lambda *_args, **_kwargs: [])

    def build_for(source_suite: LoadedSuite, source: ExperimentSource, **kwargs: Any) -> list[Any]:
        built.append((kwargs["stage"], source.config.experiment_id))
        return []

    def screened_out(
        source: ExperimentSource,
        _records: Any,
        *,
        promoted_component_overlays: dict[str, dict[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        frozen = harness._runtime_deferred_selection(
            source,
            suite,
            reason_code="fixture",
            reason="fixture",
        )
        frozen.pop("reason_code")
        frozen.pop("reason")
        frozen["status"] = "screened_out"
        arms = harness._component_arms(
            source.config,
            promoted_overlays=promoted_component_overlays,
        )
        frozen["resolved_arms_hash"] = hash_payload({arm.id: arm.overlay for arm in arms})
        return frozen

    monkeypatch.setattr(harness, "build_cells", build_for)
    monkeypatch.setattr(harness, "select_experiment", screened_out)

    parent_path = run_stage(
        suite,
        stage="screen",
        output_root=output_root,
        jobs=1,
        resume=False,
        workdir=REPOSITORY_ROOT,
    )
    assert parent_path is not None
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    assert built == [("screen", "E05")]
    assert inventoried == [("screen", ("E05", "E17"))]

    confirmation_artifact = output_root / suite.suite_id / "confirm" / "metrics.json"
    confirmation_artifact.parent.mkdir(parents=True)
    confirmation_artifact.write_text('{"component":"frozen"}\n', encoding="utf-8")
    bound = replace(
        suite,
        confirmed_components_hash="c" * 64,
        confirmed_parent_selection_hash=parent["selection_hash"],
        confirmed_component_overlays={"E05": {}, "E06": {}},
    )

    built.clear()
    inventoried.clear()
    downstream_path = run_stage(
        bound,
        stage="screen",
        output_root=output_root,
        jobs=1,
        resume=True,
        workdir=REPOSITORY_ROOT,
    )
    assert downstream_path is not None
    assert built == [("screen", "E17")]
    assert inventoried == [("screen", ("E17",))]

    built.clear()
    inventoried.clear()
    run_stage(
        bound,
        stage="confirm",
        output_root=output_root,
        jobs=1,
        resume=True,
        workdir=REPOSITORY_ROOT,
        selection_record_path=downstream_path,
    )
    assert built == [("confirm", "E17")]
    assert inventoried == [("confirm", ("E17",))]


def test_e17_bound_reports_do_not_overwrite_component_confirmation(
    tmp_path: Path,
) -> None:
    _source, suite = _source_and_suite(tmp_path, _e17_mapping())
    bound = replace(suite, confirmed_components_hash="c" * 64)
    output_root = tmp_path / "results"
    canonical = output_root / suite.suite_id / "confirm" / "metrics.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"component":"frozen"}\n', encoding="utf-8")
    frozen_bytes = canonical.read_bytes()

    harness.aggregate_stage(
        bound,
        stage="confirm",
        output_root=output_root,
        manifests=[],
        finalize_reports=False,
    )

    assert canonical.read_bytes() == frozen_bytes
    versioned = output_root / suite.suite_id / "confirm" / "e17" / ("c" * 64) / "metrics.json"
    assert json.loads(versioned.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "rows": [],
    }


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


def _selection_source(tmp_path: Path, mapping: dict[str, Any]) -> ExperimentSource:
    config = ExperimentConfig.model_validate(mapping)
    path = tmp_path / f"{config.experiment_id}.yaml"
    path.write_text("synthetic: true\n", encoding="utf-8")
    return ExperimentSource(config=config, path=path)


def _selection_row(
    arm: str,
    task: str,
    metrics: dict[str, float],
    *,
    seed: int = 7,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "experiment_id": "E99",
        "arm_id": arm,
        "task_id": task,
        "seed": seed,
        "status": "complete",
        "metrics": metrics,
        **extra,
    }


def _paired_selection_evidence(*, lower: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "comparisons": [
            {
                "decision_id": "primary",
                "baseline": "baseline",
                "candidate": "candidate",
                "metric": "f1",
                "direction": "max",
                "estimate": 0.01,
                "confidence_interval": {
                    "confidence": 0.95,
                    "lower": lower,
                    "upper": 0.02,
                    "paired": True,
                    "method": "paired_bootstrap",
                    "independent_unit": "source_entity",
                },
            }
        ],
    }


def test_selection_applies_each_task_guard_without_hiding_regressions(
    tmp_path: Path,
) -> None:
    mapping = _experiment_mapping()
    mapping["selection"]["decisions"][0].update(
        {
            "min_delta": 0.0,
            "guards": [
                {
                    "id": "per_task_f1",
                    "metric": "f1",
                    "direction": "max",
                    "scope": "each_task",
                    "min_delta": -0.005,
                }
            ],
        }
    )
    source = _selection_source(tmp_path, mapping)
    records = [
        _selection_row("baseline", "one", {"f1": 0.5}),
        _selection_row("baseline", "two", {"f1": 0.5}),
        _selection_row("candidate", "one", {"f1": 0.52}),
        _selection_row("candidate", "two", {"f1": 0.494}),
    ]

    selected = select_experiment(source, records)

    assert selected["status"] == "screened_out"
    evaluation = selected["decisions"][0]["candidate_evaluations"]["candidate"]
    assert evaluation["primary"]["passed"] is True
    assert evaluation["guards"][0]["passed"] is False
    assert evaluation["rejection_reasons"] == ["guard_failed:per_task_f1"]


def test_selection_requires_declared_guard_metric(tmp_path: Path) -> None:
    mapping = _experiment_mapping()
    mapping["selection"]["decisions"][0]["guards"] = [
        {
            "id": "ranking_guard",
            "metric": "MRR",
            "direction": "max",
            "min_delta": -0.005,
        }
    ]
    source = _selection_source(tmp_path, mapping)
    records = [
        _selection_row("baseline", "one", {"f1": 0.5}),
        _selection_row("candidate", "one", {"f1": 0.6}),
    ]

    with pytest.raises(ValueError, match="missing selection metric 'MRR'"):
        select_experiment(source, records)


def test_selection_requires_and_applies_paired_ci_lower_bound(tmp_path: Path) -> None:
    mapping = _experiment_mapping()
    mapping["selection"]["decisions"][0].update(
        {"min_delta": 0.0, "evidence": "paired_ci_lower", "strict": True}
    )
    source = _selection_source(tmp_path, mapping)
    bare_records = [
        _selection_row("baseline", "one", {"f1": 0.5}),
        _selection_row("candidate", "one", {"f1": 0.51}),
    ]

    with pytest.raises(ValueError, match="missing paired 95% CI evidence"):
        select_experiment(source, bare_records)

    excludes_improvement = [dict(record) for record in bare_records]
    excludes_improvement[0]["selection_evidence"] = _paired_selection_evidence(lower=0.0)
    assert select_experiment(source, excludes_improvement)["status"] == "screened_out"

    positive = [dict(record) for record in bare_records]
    positive[0]["selection_evidence"] = _paired_selection_evidence(lower=0.001)
    result = select_experiment(source, positive)
    assert result["status"] == "selected"
    assert result["selection_evidence_hash"] is not None


def test_selection_uses_tolerant_lexicographic_ties_deterministically(
    tmp_path: Path,
) -> None:
    mapping = _experiment_mapping()
    mapping["arms"] = [
        {"id": "baseline", "role": "baseline"},
        {
            "id": "candidate_a",
            "role": "candidate",
            "overlay": {"matching": {"extraction": {"mode": "mutual_best"}}},
        },
        {
            "id": "candidate_b",
            "role": "candidate",
            "overlay": {"matching": {"extraction": {"mode": "assignment"}}},
        },
    ]
    mapping["selection"]["decisions"] = [
        {
            "id": "primary",
            "baseline": "baseline",
            "candidates": ["candidate_a", "candidate_b"],
            "metric": "f1",
            "min_delta": 0.0,
            "tie_tolerance": 0.002,
            "tie_breaks": [{"kind": "metric", "metric": "MRR", "direction": "max"}],
        }
    ]
    source = _selection_source(tmp_path, mapping)
    records = [
        _selection_row("baseline", "one", {"f1": 0.7, "MRR": 0.7}),
        _selection_row("candidate_a", "one", {"f1": 0.801, "MRR": 0.7}),
        _selection_row("candidate_b", "one", {"f1": 0.800, "MRR": 0.8}),
    ]

    forward = select_experiment(source, records)
    reverse = select_experiment(source, list(reversed(records)))

    assert forward["decisions"][0]["selected_arm"] == "candidate_b"
    assert forward["decisions"][0]["ranking_evidence"]["lexicographic_order"] == [
        "candidate_b",
        "candidate_a",
    ]
    assert reverse["decisions"] == forward["decisions"]


def test_independent_decisions_compose_deltas_and_allow_partial_survival(
    tmp_path: Path,
) -> None:
    mapping = _experiment_mapping()
    mapping["arms"] = [
        {"id": "baseline", "role": "baseline"},
        {
            "id": "extractor",
            "role": "candidate",
            "overlay": {"matching": {"extraction": {"mode": "mutual_best"}}},
        },
        {
            "id": "anchors",
            "role": "candidate",
            "overlay": {"matching": {"anchor_rescoring": {"mode": "one_pass"}}},
        },
    ]
    mapping["selection"]["decisions"] = [
        {
            "id": "extractor_choice",
            "baseline": "baseline",
            "candidates": ["extractor"],
            "metric": "extractor_quality",
        },
        {
            "id": "anchor_choice",
            "baseline": "baseline",
            "candidates": ["anchors"],
            "metric": "anchor_quality",
        },
    ]
    mapping["design"]["multiplicity"] = "holm"
    source = _selection_source(tmp_path, mapping)
    records = [
        _selection_row("baseline", "one", {"extractor_quality": 0.5, "anchor_quality": 0.5}),
        _selection_row("extractor", "one", {"extractor_quality": 0.6}),
        _selection_row("anchors", "one", {"anchor_quality": 0.6}),
    ]

    result = select_experiment(source, records)
    assert result["combined_selected_overlay"]["matching"] == {
        "anchor_rescoring": {"mode": "one_pass"},
        "extraction": {"mode": "mutual_best"},
    }
    inherited = selected_experiment_overlays({"experiments": {"E99": result}}, ["E99"])
    assert inherited["E99"] == result["combined_selected_overlay"]

    records[-1]["metrics"]["anchor_quality"] = 0.4
    partial = select_experiment(source, records)
    assert partial["status"] == "selected"
    assert partial["all_decisions_selected"] is False
    assert partial["decisions"][1]["status"] == "screened_out"
    assert "anchor_rescoring" not in partial["combined_selected_overlay"]["matching"]


def test_independent_decisions_reject_conflicting_overlay_deltas(tmp_path: Path) -> None:
    mapping = _experiment_mapping()
    mapping["arms"] = [
        {"id": "baseline", "role": "baseline"},
        {
            "id": "candidate_a",
            "role": "candidate",
            "overlay": {"matching": {"extraction": {"mode": "mutual_best"}}},
        },
        {
            "id": "candidate_b",
            "role": "candidate",
            "overlay": {"matching": {"extraction": {"mode": "assignment"}}},
        },
    ]
    mapping["selection"]["decisions"] = [
        {
            "id": "one",
            "baseline": "baseline",
            "candidates": ["candidate_a"],
            "metric": "a",
        },
        {
            "id": "two",
            "baseline": "baseline",
            "candidates": ["candidate_b"],
            "metric": "b",
        },
    ]
    mapping["design"]["multiplicity"] = "holm"
    source = _selection_source(tmp_path, mapping)
    records = [
        _selection_row("baseline", "one", {"a": 0.5, "b": 0.5}),
        _selection_row("candidate_a", "one", {"a": 0.6}),
        _selection_row("candidate_b", "one", {"b": 0.6}),
    ]

    with pytest.raises(ValueError, match="conflicting overlays at matching.extraction.mode"):
        select_experiment(source, records)


def test_normalized_manifests_bind_structured_selection_contracts() -> None:
    configs = {
        experiment_id: load_experiment(
            REPOSITORY_ROOT / "exp" / "experiments" / experiment_id / "exp.yaml"
        )
        for experiment_id in ("E05", "E08", "E10", "E18", "E21", "E22", "E26")
    }

    assert configs["E05"].selection.decisions[0].guards[0].comparison == "matched"
    assert configs["E08"].selection.decisions[1].evidence == "paired_ci_lower"
    assert configs["E10"].selection.decisions[0].tie_tolerance == 0.002
    assert configs["E18"].selection.decisions[0].guards[0].min_delta == -0.005
    assert configs["E21"].selection.decisions[1].metric == "llm.calls"
    assert configs["E22"].selection.decisions[0].min_relative_delta == 0.25
    assert configs["E26"].selection.decisions[0].guards[0].comparison == ("absolute_threshold")


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


@pytest.mark.parametrize("mode", ["off", "oracle"])
def test_non_calling_llm_gate_is_not_marked_runtime_required(mode: str) -> None:
    default_mapping = ConfigModel.load_config(DEFAULT_CONFIG).model_dump(mode="json", by_alias=True)
    mapping = ConfigModel.from_mapping(
        harness.deep_merge(
            default_mapping,
            {"llm": {"experiment": {"enabled": True, "gate": {"mode": mode}}}},
        ),
        warn_v1=False,
    ).model_dump(mode="json", by_alias=True)
    assert _llm_runtime_required(mapping) is False


def test_post_run_provenance_forwards_observed_execution(tmp_path: Path) -> None:
    cell = _run_cell(tmp_path / "runtime-device")
    stats_dir = cell.output_dir / "stats"
    stats_dir.mkdir(parents=True)
    (stats_dir / "run_stats.json").write_text(
        json.dumps(
            {
                "explanation_reconstruction": {
                    "status": "complete",
                    "result_rows": 2,
                    "reconstructed_rows": 2,
                    "max_abs_error": 0.0,
                },
                "candidate_recall": 0.75,
                "candidate_recall_after_exact": 1.0,
                "candidate_recall_diagnostics": {"status": "available"},
                "mean_pool_size": 20.0,
                "gold_rank_p90": 3.0,
                "gold_rank_median": 2.0,
                "selection_evidence": {
                    "schema_version": 1,
                    "comparisons": [],
                },
                "observed_execution": {
                    "device_type": "cuda",
                    "device": "cuda:2",
                    "device_name": "fixture GPU",
                },
            }
        ),
        encoding="utf-8",
    )
    assert _post_run_provenance(cell)["observed_execution"] == {
        "device_type": "cuda",
        "device": "cuda:2",
        "device_name": "fixture GPU",
    }
    assert _post_run_provenance(cell)["explanation_reconstruction"] == {
        "status": "complete",
        "result_rows": 2,
        "reconstructed_rows": 2,
        "max_abs_error": 0.0,
    }
    assert _post_run_provenance(cell)["candidate_recall"] == 0.75
    assert _post_run_provenance(cell)["candidate_recall_after_exact"] == 1.0
    assert _post_run_provenance(cell)["candidate_recall_diagnostics"] == {"status": "available"}
    assert _post_run_provenance(cell)["mean_pool_size"] == 20.0
    assert _post_run_provenance(cell)["gold_rank_p90"] == 3.0
    assert _post_run_provenance(cell)["gold_rank_median"] == 2.0
    assert _post_run_provenance(cell)["selection_evidence"] == {
        "schema_version": 1,
        "comparisons": [],
    }


def test_frozen_default_rejects_enabled_experiment_switch(tmp_path: Path) -> None:
    harness._assert_experiment_flags_disabled(DEFAULT_CONFIG)
    mapping = ConfigModel.load_config(DEFAULT_CONFIG).model_dump(mode="json", by_alias=True)
    mapping["matching"]["fusion"]["enabled"] = True
    changed = tmp_path / "enabled.yaml"
    changed.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="matching.fusion.enabled"):
        harness._assert_experiment_flags_disabled(changed)


def test_e00_replay_failure_record_is_persisted_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exact.experiments import replay as replay_module

    _source, suite = _source_and_suite(tmp_path, _experiment_mapping("E00"))
    manifest = {
        "experiment_id": "E00",
        "arm_id": "baseline",
        "task_id": "development",
        "seed": 7,
        "status": "complete",
    }
    failure = {
        "schema_version": 1,
        "experiment_id": "E00",
        "status": "failed",
        "failure": {
            "type": "ReplayValidationError",
            "code": "fixture_failure",
            "message": "fixture replay failed",
            "details": {},
        },
    }
    monkeypatch.setattr(harness, "build_dataset_inventory", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(harness, "build_cells", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(harness, "run_cells", lambda *_args, **_kwargs: [manifest])
    monkeypatch.setattr(harness, "aggregate_stage", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(harness, "_require_successful_cells", lambda *_args, **_kwargs: None)

    def fail_replay(_manifests: Any) -> Any:
        raise replay_module.ReplayValidationError("fixture replay failed", failure)

    monkeypatch.setattr(replay_module, "validate_e00_replay", fail_replay)
    output_root = tmp_path / "results"

    with pytest.raises(replay_module.ReplayValidationError, match="fixture replay failed"):
        run_stage(
            suite,
            stage="screen",
            output_root=output_root,
            jobs=1,
            resume=False,
            workdir=REPOSITORY_ROOT,
        )

    replay_path = output_root / suite.suite_id / "screen" / "replay" / "E00.json"
    assert json.loads(replay_path.read_text(encoding="utf-8")) == failure


def _complete_llm_identity(**updates: Any) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "backend": "local_hf",
        "provider": "huggingface",
        "requested_model": "org/model",
        "requested_revision": "a" * 40,
        "effective_model": "org/model",
        "resolved_revision": "b" * 40,
        "endpoint_identity": "local:cuda-0",
        "tokenizer": "org/tokenizer",
        "tokenizer_revision": "c" * 40,
        "prompt_hashes": ["d" * 64],
        "decoding": {"temperature": 0.0, "do_sample": False},
        "request_seed": 17,
        "cache_hashes": ["e" * 64],
        "request_time": "2026-08-28T12:00:00+00:00",
    }
    identity.update(updates)
    return identity


def _llm_manifest(arm: str, identity: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "experiment_id": "E07",
        "arm_id": arm,
        "task_id": "task",
        "seed": 17,
        "status": "complete",
        "fingerprint_payload": {"llm_required": True},
        "llm_usage": (
            {"backend_identities": {"decision": [identity]}} if identity is not None else None
        ),
    }


def test_confirmatory_llm_identity_is_complete_immutable_and_paired() -> None:
    identity = _complete_llm_identity()
    _validate_paired_llm_identities(
        [_llm_manifest("baseline", identity), _llm_manifest("candidate", dict(identity))]
    )

    with pytest.raises(ValueError, match="no resolved backend identity"):
        _validate_paired_llm_identities([_llm_manifest("baseline", None)])
    incomplete = _complete_llm_identity(cache_hashes=[])
    with pytest.raises(ValueError, match="cache_hashes"):
        _validate_paired_llm_identities([_llm_manifest("baseline", incomplete)])
    mutable_tokenizer = _complete_llm_identity(tokenizer_revision="main")
    with pytest.raises(ValueError, match="tokenizer revision is not an immutable"):
        _validate_paired_llm_identities([_llm_manifest("baseline", mutable_tokenizer)])
    mutable = _complete_llm_identity(resolved_revision="latest")
    with pytest.raises(ValueError, match="mutable resolved revision"):
        _validate_paired_llm_identities([_llm_manifest("baseline", mutable)])
    drifted = _complete_llm_identity(effective_model="org/other-model")
    with pytest.raises(ValueError, match="identity changed within paired confirm arms"):
        _validate_paired_llm_identities(
            [_llm_manifest("baseline", identity), _llm_manifest("candidate", drifted)]
        )


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
        specification=_specification(tmp_path),
    )
    artifact = tmp_path / suite.suite_id / "confirm" / "runs" / "artifact.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="confirm artifacts"):
        write_selection_record(suite, {}, output_root=tmp_path)


@pytest.mark.parametrize("generate_rationales", [False, True])
def test_rationales_require_explicit_experiment_opt_in(tmp_path, generate_rationales):
    mapping = _experiment_mapping()
    if generate_rationales:
        mapping["generate_rationales"] = True
    source, suite = _source_and_suite(tmp_path, mapping)
    cells = build_cells(suite, source, stage="screen", output_root=tmp_path / "runs")
    assert cells
    for cell in cells:
        assert cell.generate_rationales is generate_rationales
        assert (
            cell.resolved_config["pipeline"][0]["params"]["generate_llm_rationales"]
            is generate_rationales
        )
        assert cell.resolved_config["pipeline"][0]["params"]["return_explanations"] is True
        assert cell.config_hash == ConfigModel.from_mapping(cell.resolved_config).fingerprint()
    changed = replace(
        source,
        config=source.config.model_copy(update={"generate_rationales": not generate_rationales}),
    )
    assert experiment_design_hash(source) != experiment_design_hash(changed)
    for task in source.config.confirm.tasks:
        resolved, *_ = harness._resolve_config(
            source,
            task=task,
            arm=source.config.arms[0],
            stage="confirm",
            seed=7,
            source_cap=None,
            inherited_overlay={},
        )
        assert resolved["pipeline"][0]["params"]["generate_llm_rationales"] is generate_rationales


def test_rationale_policy_resolves_missing_defaults_and_preserves_production():
    from exact.experiments.rationale_policy import apply_rationale_policy

    production = ConfigModel().model_dump(mode="json", by_alias=True)
    original = json.loads(json.dumps(production))
    assert production["pipeline"][0]["params"]["generate_llm_rationales"] is True
    disabled = apply_rationale_policy({"config_version": 2})
    assert disabled["pipeline"][0]["params"]["generate_llm_rationales"] is False
    assert apply_rationale_policy(production, generate_rationales=True) == production
    assert apply_rationale_policy(disabled, generate_rationales=True) == disabled
    assert production == original


def test_rationale_policy_overrides_inherited_and_arm_settings(tmp_path):
    mapping = _experiment_mapping()
    enabled = {
        "pipeline": [
            {
                "name": "PairAdaptiveSemanticScorer",
                "params": {
                    "generate_llm_rationales": True,
                },
            }
        ]
    }
    mapping["arms"][0]["overlay"] = enabled
    source, suite = _source_and_suite(tmp_path, mapping)
    cells = build_cells(
        suite, source, stage="screen", output_root=tmp_path / "runs", inherited_overlay=enabled
    )
    assert all(
        cell.resolved_config["pipeline"][0]["params"]["generate_llm_rationales"] is False
        for cell in cells
    )


def test_execute_rejects_prebuilt_rationales_without_opt_in(tmp_path, monkeypatch):
    _, suite = _source_and_suite(tmp_path)
    production = ConfigModel().model_dump(mode="json", by_alias=True)
    cell = replace(_run_cell(tmp_path / "run"), resolved_config=production)
    with pytest.raises(ValueError, match="requires explicit generate_rationales"):
        harness.execute_cell(cell, suite, workdir=tmp_path, resume=False)
    assert not cell.output_dir.exists()
    monkeypatch.setattr(harness, "_prepare_cell", lambda *a, **k: ({"status": "complete"}, True))
    result = harness.execute_cell(
        replace(cell, generate_rationales=True), suite, workdir=tmp_path, resume=True
    )
    assert result["status"] == "complete"
