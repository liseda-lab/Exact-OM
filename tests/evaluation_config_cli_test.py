from pathlib import Path

from exact.core.entities.configs.config import ConfigModel
from exact.delivery.cli.eval import parse_arguments


def test_evaluation_config_defaults_to_builtin() -> None:
    assert ConfigModel().evaluation.backends == ["builtin"]


def test_evaluation_config_preserves_backend_order_and_deduplicates(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "evaluation:\n  backends: [builtin, bioml, builtin]\n  bioml:\n    candidate_count: 50\n",
        encoding="utf-8",
    )

    config = ConfigModel.load_config(config_path)

    assert config.evaluation.backends == ["builtin", "bioml"]
    assert config.evaluation.bioml == {"candidate_count": 50}


def test_eval_cli_accepts_hyphenated_backend_flag() -> None:
    args = parse_arguments(
        [
            "-a",
            "alignment.tsv",
            "-o",
            "output",
            "--eval-backends",
            "builtin",
            "bioml",
            "-k",
            "1",
            "5",
        ]
    )

    assert args.eval_backends == ["builtin", "bioml"]
    assert args.K == [1, 5]
