from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import exact.core.actions.alignment as alignment_actions
import exact.delivery.common as delivery_common
from exact.core.entities.configs.config import ConfigModel
from exact.delivery.api.align import AlignmentRunner
from exact.delivery.api.eval import EvaluationRunner
from exact.delivery.cli.align import main as align_main
from exact.delivery.cli.align import parse_arguments as parse_align_arguments
from exact.delivery.cli.eval import main as eval_main


def _file(root: Path, name: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name, encoding="utf-8")
    return path


def test_alignment_action_is_a_deprecated_function_alias(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(alignment_actions, "run_alignment", lambda **kwargs: sentinel)

    assert inspect.isfunction(alignment_actions.run_alignment)
    assert not getattr(alignment_actions.AlignmentAction, "_is_protocol", False)
    with pytest.deprecated_call(match="run_alignment"):
        result = alignment_actions.AlignmentAction.run(output_dir_path="unused")

    assert result is sentinel


def test_plain_alignment_function_uses_the_functional_session_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _file(tmp_path, "source.owl")
    target = _file(tmp_path, "target.owl")
    captured: list[dict] = []
    monkeypatch.setattr(ConfigModel, "resolve_dependencies", lambda self: None)

    def _session(**kwargs):
        captured.append(kwargs)
        return None, None

    monkeypatch.setattr(alignment_actions, "_run_alignment_session", _session)

    result, timings = alignment_actions.run_alignment(
        source_file_path=source,
        target_file_path=target,
        output_dir_path=tmp_path / "run",
        configs_file_path=ConfigModel.model_validate({}),
    )

    assert result is None
    assert timings["Total"] >= 0.0
    assert captured[0]["source_file_path"] == source.resolve()
    assert captured[0]["target_file_path"] == target.resolve()
    run_dir = tmp_path / "run"
    assert json.loads((run_dir / "run_manifest.json").read_text())["layout_version"] == 2
    assert "config_version: 2" in (run_dir / "config.yaml").read_text()


def test_cli_and_api_alignment_share_preparation_and_action_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _file(tmp_path, "source.owl")
    target = _file(tmp_path, "target.owl")
    training = _file(tmp_path, "train.tsv")
    captured: list[dict] = []
    monkeypatch.setattr(
        delivery_common,
        "run_alignment",
        lambda **kwargs: captured.append(kwargs),
    )

    cli_output = tmp_path / "cli-output"
    assert (
        align_main(
            [
                "-s",
                str(source),
                "-t",
                str(target),
                "-o",
                str(cli_output),
                "-r",
                str(training),
                "-l",
                "-e",
            ]
        )
        is None
    )

    api_output = tmp_path / "api-output"
    runner = AlignmentRunner(
        source_ontology_file=str(source),
        target_ontology_file=str(target),
        output_dir=str(api_output),
        training_reference_file=str(training),
        save_logs=True,
        run_eval=True,
    )
    runner.run()

    assert len(captured) == 2
    for request, output in zip(captured, (cli_output, api_output)):
        assert request["source_file_path"] == source.resolve()
        assert request["target_file_path"] == target.resolve()
        assert request["training_reference_file_path"] == training.resolve()
        assert request["output_dir_path"] == output.resolve()
        assert request["log_file_path"] == output.resolve() / "exact.log"
        assert request["run_eval"] is True
        assert isinstance(request["configs_file_path"], ConfigModel)


def test_alignment_cli_format_overrides_flow_through_shared_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    options = tmp_path / "target-options.yaml"
    options.write_text("hierarchy_relations: [broader]\n", encoding="utf-8")
    captured: list[dict] = []
    monkeypatch.setattr(
        delivery_common,
        "run_alignment",
        lambda **kwargs: captured.append(kwargs),
    )

    args = parse_align_arguments(
        [
            "-s",
            str(source),
            "-t",
            str(target),
            "-o",
            str(tmp_path / "run"),
            "--input-format",
            "csv-kg",
            "--source-options",
            "include_abox=true",
            'hierarchy_relations=["subclass_of"]',
            "--target-options",
            str(options),
            "--output-formats",
            "typed-tsv",
            "json",
            "--relation-prediction",
            "hierarchy_heuristic",
        ]
    )
    invocation = delivery_common.prepare_alignment_namespace(args)
    delivery_common.execute_alignment(invocation)

    config = captured[0]["configs_file_path"]
    assert config.io.input_format == "csv-kg"
    assert config.io.source_options == {
        "include_abox": True,
        "hierarchy_relations": ["subclass_of"],
    }
    assert config.io.target_options == {"hierarchy_relations": ["broader"]}
    assert config.io.output_formats == ["typed-tsv", "json"]
    assert config.matching.relation_prediction == "hierarchy_heuristic"


def test_cli_and_api_evaluation_share_preparation_but_keep_output_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alignment = _file(tmp_path, "alignment.tsv")
    sentinel = object()
    captured: list[dict] = []

    def _capture(**kwargs):
        captured.append(kwargs)
        return sentinel

    monkeypatch.setattr(delivery_common, "run_evaluation", _capture)
    cli_output = tmp_path / "cli-eval"
    assert eval_main(["-a", str(alignment), "-o", str(cli_output)]) is sentinel
    assert cli_output.is_dir()

    missing_api_output = tmp_path / "missing-api-eval"
    with pytest.raises(FileNotFoundError, match="Output directory"):
        EvaluationRunner(str(alignment), str(missing_api_output)).run()

    api_output = tmp_path / "api-eval"
    api_output.mkdir()
    assert EvaluationRunner(str(alignment), str(api_output)).run() is sentinel

    assert len(captured) == 2
    assert captured[0]["alignment"] == alignment.resolve()
    assert captured[1]["alignment"] == alignment.resolve()
    assert captured[0]["output_dir_path"] == cli_output.resolve()
    assert captured[1]["output_dir_path"] == api_output.resolve()
    assert captured[0]["backends"] == captured[1]["backends"] == ["builtin"]
