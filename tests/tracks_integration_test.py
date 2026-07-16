from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

import exact.core.actions.alignment as alignment_module
import exact.delivery.cli.data as data_cli
from exact.core.actions.alignment import AlignmentAction, _merge_run_stats
from exact.core.entities.configs.config import ConfigModel, DataConfig
from exact.delivery.cli.main import main as cli_main
from exact.tracks.provider import TaskLayout, TrackStatus, VerificationReport


class FakeProvider:
    name = "fake"

    def __init__(self, layout: TaskLayout):
        self.layout = layout
        self.calls: list[tuple[str, Path, Optional[str], bool]] = []

    def tasks(self) -> list[str]:
        return ["demo"]

    def materialize(
        self,
        task: str,
        data_root: Path,
        *,
        revision: str | None = None,
        update: bool = False,
    ) -> TaskLayout:
        self.calls.append((task, data_root, revision, update))
        return self.layout

    def verify(self, task: str, data_root: Path) -> VerificationReport:
        return VerificationReport(self.name, task, "ok", checked_files=2)

    def status(self, task: str, data_root: Path) -> TrackStatus:
        return "ok"


def _file(root: Path, name: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name, encoding="utf-8")
    return path


def _layout(root: Path) -> TaskLayout:
    return TaskLayout(
        source=_file(root, "layout/source.owl"),
        target=_file(root, "layout/target.owl"),
        refs={
            "train": _file(root, "layout/train.tsv"),
            "test": _file(root, "layout/test.tsv"),
        },
        candidates=_file(root, "layout/test.cands.tsv"),
        provenance={
            "provider": "fake",
            "upstream_id": "fixture/repository",
            "revision": "commit-a",
            "hashes": {"source.owl": "abc"},
        },
    )


def test_config_loader_accepts_data_and_dataset_track_shim(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  track: fake",
                "  task: demo",
                f"  root: {tmp_path / 'data'}",
                "  revision: stable",
                "  refs:",
                "    train: explicit-train.tsv",
                "dataset_track:",
                "  track: legacy",
                "  task: legacy-task",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = ConfigModel.load_config(config_path)

    assert config.data is not None
    assert config.data.track == "fake"
    assert config.data.refs == {"train": Path("explicit-train.tsv")}
    assert config.dataset_track is not None
    assert config.dataset_track.track == "legacy"
    assert config.effective_data_config() is config.data


def test_alignment_input_precedence_is_cli_then_config_then_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    provider = FakeProvider(layout)
    monkeypatch.setattr(alignment_module, "get_track", lambda name: provider)
    cli_source = _file(tmp_path, "cli/source.owl")
    cli_full = _file(tmp_path, "cli/full.tsv")
    config_target = _file(tmp_path, "config/target.owl")
    config_train = _file(tmp_path, "config/train.tsv")
    config_candidates = _file(tmp_path, "config/candidates.tsv")
    data_root = tmp_path / "materialized"
    config = ConfigModel(
        data=DataConfig(
            track="fake",
            task="demo",
            root=data_root,
            revision="stable",
            target=config_target,
            refs={"train": config_train},
            candidates=config_candidates,
        )
    )

    resolved = AlignmentAction.resolve_inputs(
        configs=config,
        source_file_path=cli_source,
        full_reference_file_path=cli_full,
    )

    assert resolved.source == cli_source.resolve()
    assert resolved.target == config_target.resolve()
    assert resolved.training_reference == config_train.resolve()
    assert resolved.full_reference == cli_full.resolve()
    assert resolved.candidates == config_candidates.resolve()
    assert resolved.task_name == "demo"
    assert resolved.track_provenance == {
        **layout.provenance,
        "track": "fake",
        "task": "demo",
    }
    assert provider.calls == [("demo", data_root.resolve(), "stable", False)]


def test_alignment_can_use_explicit_config_paths_without_a_track(tmp_path: Path) -> None:
    source = _file(tmp_path, "source.owl")
    target = _file(tmp_path, "target.owl")
    config = ConfigModel(data=DataConfig(source=source, target=target))

    resolved = AlignmentAction.resolve_inputs(configs=config)

    assert resolved.source == source.resolve()
    assert resolved.target == target.resolve()
    assert resolved.track_provenance is None


def test_alignment_requires_paths_or_a_track() -> None:
    with pytest.raises(ValueError, match="Source ontology is required"):
        AlignmentAction.resolve_inputs(configs=ConfigModel())


def test_run_stats_merge_preserves_existing_metadata_and_adds_dataset_provenance(
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "run_stats.json"
    stats_path.write_text(
        json.dumps({"n_mappings": 3, "provenance": {"calibration": {"sha256": "old"}}}),
        encoding="utf-8",
    )

    _merge_run_stats(
        stats_path,
        {"provenance": {"dataset": {"provider": "fake", "revision": "commit-a"}}},
    )
    _merge_run_stats(stats_path, {"timing": {"run_id": "run-a"}})

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["n_mappings"] == 3
    assert payload["provenance"]["calibration"]["sha256"] == "old"
    assert payload["provenance"]["dataset"]["revision"] == "commit-a"
    assert payload["timing"]["run_id"] == "run-a"


def test_track_run_creates_provenance_stats_without_trainer_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(_layout(tmp_path))
    monkeypatch.setattr(alignment_module, "get_track", lambda name: provider)
    monkeypatch.setattr(ConfigModel, "resolve_dependencies", lambda self: None)
    stats_path = tmp_path / "run" / "model" / "alignment" / "demo" / "run_stats.json"
    monkeypatch.setattr(
        AlignmentAction,
        "_run_session",
        staticmethod(lambda **kwargs: (None, stats_path)),
    )
    config = ConfigModel(data=DataConfig(track="fake", task="demo", root=tmp_path / "materialized"))

    AlignmentAction.run(
        output_dir_path=tmp_path / "run",
        configs_file_path=config,
    )

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["dataset"]["revision"] == "commit-a"
    assert payload["provenance"]["dataset"]["task"] == "demo"
    assert payload["timing"]["run_id"]


def test_flat_alignment_cli_remains_backward_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _file(tmp_path, "source.owl")
    target = _file(tmp_path, "target.owl")
    output = tmp_path / "output"
    captured = []

    monkeypatch.setattr(
        "exact.delivery.cli.align.run_alignment",
        lambda args: captured.append(args),
    )

    result = cli_main(["-s", str(source), "-t", str(target), "-o", str(output)])

    assert result is None
    assert captured[0].source_ontology_file == str(source)
    assert captured[0].target_ontology_file == str(target)
    assert output.is_dir()


def test_alignment_cli_accepts_config_only_track_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  track: fake\n  task: demo\n", encoding="utf-8")
    captured = []
    monkeypatch.setattr(
        "exact.delivery.cli.align.run_alignment",
        lambda args: captured.append(args),
    )

    cli_main(["-y", str(config_path), "-o", str(tmp_path / "output")])

    assert captured[0].source_ontology_file is None
    assert captured[0].target_ontology_file is None


def test_data_dispatcher_lists_and_pulls_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = FakeProvider(_layout(tmp_path))
    monkeypatch.setattr(data_cli, "list_tracks", lambda: ["fake"])
    monkeypatch.setattr(data_cli, "get_track", lambda name: provider)

    assert cli_main(["data", "list"]) == 0
    assert capsys.readouterr().out.strip() == "fake\tdemo"

    root = tmp_path / "data"
    assert cli_main(["data", "pull", "fake/demo", "--root", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["track"] == "fake"
    assert payload["task"] == "demo"
    assert provider.calls[-1] == ("demo", root.resolve(), None, False)


def test_data_verify_returns_nonzero_for_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(_layout(tmp_path))
    provider.verify = lambda task, root: VerificationReport(
        provider.name, task, "local-drift", issues=("checksum mismatch",)
    )
    monkeypatch.setattr(data_cli, "get_track", lambda name: provider)

    assert data_cli.main(["verify", "fake/demo", "--root", str(tmp_path)]) == 1
