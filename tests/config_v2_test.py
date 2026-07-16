from __future__ import annotations

import logging
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.configs.migration import V1_TO_V2, migrate_v1_mapping
from exact.core.entities.configs.yaml_io import default_config_is_in_sync
from exact.delivery.cli.config import main as config_main
from tools.hparam_tuner import resolve_base_config, translate_parameter_path
from tools.run_exact_job import build_exact_command


def _v1_defaults() -> dict:
    default = ConfigModel()
    sequence = default.get_model_sequence()
    return {
        "logging_level": default.run.logging_level,
        "seed": default.run.seed,
        "use_file_cache": default.run.use_file_cache,
        "data": None,
        "dataset_track": None,
        "k": default.evaluation.k,
        "evaluation": {
            "backends": default.evaluation.backends,
            "bioml": default.evaluation.bioml,
        },
        "alignment_params": default.alignment_params.model_dump(mode="python"),
        "dataset_params": default.dataset_params.model_dump(mode="python"),
        "candidates_params": default.candidates.model_dump(
            mode="python", exclude={"fusion", "aliases"}
        ),
        "sanity_check_params": default.sanity_check_params.model_dump(mode="python"),
        "plot_params": default.plot_params.model_dump(mode="python"),
        "inference_params": default.inference_params.model_dump(mode="python"),
        "llm_profiles": {
            name: profile.model_dump(mode="python")
            for name, profile in default.llm.profiles.items()
        },
        "llm_routing": default.llm.routing.model_dump(mode="python"),
        "model": {"name": sequence[0].name, "params": sequence[0].params},
        "second_model": {"name": sequence[1].name, "params": sequence[1].params},
    }


def test_exhaustive_v1_defaults_migrate_to_identical_v2_fingerprint() -> None:
    v1 = _v1_defaults()
    migrated, _ = migrate_v1_mapping(v1)

    from_v1 = ConfigModel.from_mapping(v1, warn_v1=False)
    from_v2 = ConfigModel.from_mapping(migrated, warn_v1=False)

    assert from_v1.model_dump(mode="json", by_alias=True) == ConfigModel().model_dump(
        mode="json", by_alias=True
    )
    assert from_v1.fingerprint() == from_v2.fingerprint() == ConfigModel().fingerprint()
    assert all(root in V1_TO_V2 for root in ("model", "second_model", "model_chain"))


def test_v1_warning_once_and_unknown_key_suggestions(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="exact.config")
    ConfigModel.from_mapping({"seed": 7})
    messages = [record.message for record in caplog.records if "deprecated v1" in record.message]
    assert len(messages) == 1

    with pytest.raises(ValueError, match="seed"):
        ConfigModel.from_mapping({"sead": 7}, warn_v1=False)
    with pytest.raises(ValueError, match="threshold"):
        ConfigModel.from_mapping(
            {"config_version": 2, "matching": {"threshhold": 0.4}}, warn_v1=False
        )


def test_too_new_version_is_clear() -> None:
    with pytest.raises(ValueError, match="too new"):
        ConfigModel.from_mapping({"config_version": 3}, warn_v1=False)


def test_second_pass_and_removed_reasoner_controls_are_reported() -> None:
    migrated, report = migrate_v1_mapping(
        {
            "second_pass_params": {"enabled": False},
            "dataset_params": {
                "reasoner_timeout_secs": 60,
                "reasoner_force_hermit": True,
            },
        }
    )

    assert migrated["pipeline"][1]["name"] == "SecondPassReranker"
    rendered = report.render()
    assert "second_pass_params" in rendered
    assert "reasoner_timeout_secs" in rendered
    assert "reasoner_force_hermit" in rendered


def test_generated_default_self_feeds_and_is_in_sync(tmp_path: Path, capsys) -> None:
    assert config_main(["default"]) == 0
    rendered = capsys.readouterr().out
    path = tmp_path / "default.yaml"
    path.write_text(rendered, encoding="utf-8")

    assert ConfigModel.load_config(path).fingerprint() == ConfigModel().fingerprint()
    assert default_config_is_in_sync()


def test_migrate_cli_preserves_comments_and_prints_report(tmp_path: Path, capsys) -> None:
    old = tmp_path / "old.yaml"
    new = tmp_path / "new.yaml"
    old.write_text(
        "# retained header\nseed: 11 # retained inline\nsecond_pass_params:\n"
        "  enabled: false\ndataset_params:\n  reasoner_timeout_secs: 5\n",
        encoding="utf-8",
    )

    assert config_main(["migrate", str(old), "-o", str(new)]) == 0
    text = new.read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "# retained header" in text
    assert "# retained inline" in text
    assert "second_pass_params" in output
    assert "reasoner_timeout_secs" in output
    assert ConfigModel.load_config(new).run.seed == 11


def test_pipeline_names_remain_serializable_after_dependency_resolution() -> None:
    config = ConfigModel()
    config.resolve_dependencies()

    assert isinstance(config.get_model_sequence()[0].name, type)
    assert config.model_dump()["pipeline"][0]["name"] == "PairAdaptiveSemanticScorer"


def test_tuner_translates_v1_paths_and_resolves_templates() -> None:
    assert translate_parameter_path("model.params.tau_LLM") == "pipeline.0.params.tau_LLM"
    assert translate_parameter_path("dataset_params.n_hops") == "dataset.n_hops"
    resolved = resolve_base_config({"model": {"params": {"tau_LLM": 0.25}}})
    assert resolved["config_version"] == 2
    assert resolved["pipeline"][0]["params"]["tau_LLM"] == 0.25


def test_job_runner_supports_v2_data_paths_and_tracks(tmp_path: Path) -> None:
    explicit = build_exact_command(
        {
            "data": {
                "root": str(tmp_path),
                "source": "source.owl",
                "target": "target.owl",
                "refs": {"train": "train.tsv"},
            },
            "job": {
                "output_dir": str(tmp_path / "run"),
                "config_file": str(tmp_path / "config.yaml"),
            },
        }
    )
    assert explicit[1:5] == [
        "-s",
        str((tmp_path / "source.owl").resolve()),
        "-t",
        str((tmp_path / "target.owl").resolve()),
    ]
    assert "-r" in explicit

    tracked = build_exact_command(
        {
            "data": {"track": "bioml", "task": "ncit-doid"},
            "job": {
                "output_dir": str(tmp_path / "track-run"),
                "config_file": str(tmp_path / "track.yaml"),
            },
        }
    )
    assert "-s" not in tracked and "-t" not in tracked


def test_default_yaml_is_valid_yaml_12() -> None:
    payload = YAML(typ="safe").load(Path("exact/default_config.yaml").read_text(encoding="utf-8"))
    assert payload["config_version"] == 2
