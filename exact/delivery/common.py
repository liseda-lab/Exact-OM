"""Shared delivery preparation for the CLI and Python API surfaces."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from exact.core.actions.alignment import run_alignment
from exact.core.actions.evaluation import run_evaluation
from exact.core.entities.configs.config import ConfigModel
from exact.utils.logs import configure_exact_logger

PathLike = Union[str, Path]


def optional_path(value: Optional[PathLike]) -> Optional[Path]:
    """Resolve a supplied path while preserving ``None`` for optional inputs."""
    return Path(value).expanduser().resolve() if value is not None else None


def require_file(value: Optional[PathLike], label: str) -> Optional[Path]:
    """Resolve and validate one optional input file."""
    path = optional_path(value)
    if path is not None and not path.exists():
        raise FileNotFoundError(f"{label} {value} does not exist")
    return path


def prepare_output_dir(value: PathLike, *, create: bool) -> Path:
    """Resolve an output directory using the caller's frozen creation policy."""
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Output directory {value} is not a directory")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise FileNotFoundError(f"Output directory {value} does not exist")
    return path


def warn_ignored_jvm(value: Optional[str], option: str) -> None:
    """Keep the 2.0 accepted-but-ignored Java option behavior in one place."""
    if value is not None:
        warnings.warn(
            f"{option} is deprecated and ignored; Exact-OM no longer needs Java.",
            DeprecationWarning,
            stacklevel=3,
        )


def parse_adapter_options(
    values: Optional[Sequence[str]],
    *,
    label: str,
) -> Optional[dict[str, Any]]:
    """Parse adapter options from ``key=value`` tokens or one YAML file."""

    if values is None:
        return None
    tokens = [str(value) for value in values]
    if len(tokens) == 1 and "=" not in tokens[0]:
        path = require_file(tokens[0], f"{label} options file")
        assert path is not None
        from exact.core.entities.configs.yaml_io import load_yaml_mapping

        return dict(load_yaml_mapping(path))
    parsed: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(
                f"{label} options must be key=value pairs or one YAML file; got {token!r}"
            )
        key, raw_value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{label} option keys cannot be empty")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        parsed[key] = value
    return parsed


def _override_alignment_config(
    configs: ConfigModel,
    *,
    input_format: Optional[str],
    source_options: Optional[Mapping[str, Any]],
    target_options: Optional[Mapping[str, Any]],
    output_formats: Optional[Sequence[str]],
    relation_prediction: Optional[str],
) -> ConfigModel:
    """Apply delivery-only I/O overrides through normal schema validation."""

    if all(
        value is None
        for value in (
            input_format,
            source_options,
            target_options,
            output_formats,
            relation_prediction,
        )
    ):
        return configs
    payload = configs.model_dump(mode="python")
    io_config = dict(payload["io"])
    matching_config = dict(payload["matching"])
    if input_format is not None:
        io_config["input_format"] = input_format
    if source_options is not None:
        io_config["source_options"] = dict(source_options)
    if target_options is not None:
        io_config["target_options"] = dict(target_options)
    if output_formats is not None:
        io_config["output_formats"] = list(output_formats)
    if relation_prediction is not None:
        matching_config["relation_prediction"] = relation_prediction
    payload["io"] = io_config
    payload["matching"] = matching_config
    return ConfigModel.model_validate(payload)


@dataclass(frozen=True)
class AlignmentInvocation:
    source_file_path: Optional[Path]
    target_file_path: Optional[Path]
    output_dir_path: Path
    configs: ConfigModel
    configs_source: Optional[Path]
    training_reference_file_path: Optional[Path]
    full_reference_file_path: Optional[Path]
    candidates_file_path: Optional[Path]
    log_file_path: Optional[Path]
    run_eval: bool
    task_name: Optional[str]
    device: Optional[int]

    def action_kwargs(self) -> dict[str, Any]:
        return {
            "source_file_path": self.source_file_path,
            "target_file_path": self.target_file_path,
            "output_dir_path": self.output_dir_path,
            "configs_file_path": self.configs,
            "configs_source": self.configs_source,
            "training_reference_file_path": self.training_reference_file_path,
            "full_reference_file_path": self.full_reference_file_path,
            "candidates_file_path": self.candidates_file_path,
            "log_file_path": self.log_file_path,
            "run_eval": self.run_eval,
            "task_name": self.task_name,
            "device": self.device,
        }


def prepare_alignment(
    *,
    source_ontology_file: Optional[PathLike],
    target_ontology_file: Optional[PathLike],
    output_dir: PathLike,
    training_reference_file: Optional[PathLike] = None,
    full_reference_file: Optional[PathLike] = None,
    candidates_file: Optional[PathLike] = None,
    config_file: Optional[PathLike] = None,
    save_logs: bool = False,
    run_eval: bool = False,
    task_name: Optional[str] = None,
    device: Optional[int] = None,
    full_reference_label: str = "Full reference file",
    input_format: Optional[str] = None,
    source_options: Optional[Mapping[str, Any]] = None,
    target_options: Optional[Mapping[str, Any]] = None,
    output_formats: Optional[Sequence[str]] = None,
    relation_prediction: Optional[str] = None,
) -> AlignmentInvocation:
    """Validate and assemble one alignment invocation for any delivery surface."""
    source = require_file(source_ontology_file, "Source ontology file")
    target = require_file(target_ontology_file, "Target ontology file")
    training = require_file(training_reference_file, "Training reference file")
    full = require_file(full_reference_file, full_reference_label)
    candidates = require_file(candidates_file, "Candidates file")
    output = prepare_output_dir(output_dir, create=True)
    config_path = require_file(config_file, "Configuration file")
    configs = (
        ConfigModel.load_config(config_path)
        if config_path is not None
        else ConfigModel.model_validate({})
    )
    configs = _override_alignment_config(
        configs,
        input_format=input_format,
        source_options=source_options,
        target_options=target_options,
        output_formats=output_formats,
        relation_prediction=relation_prediction,
    )
    log_path = output / "exact.log" if save_logs else None
    configure_exact_logger(
        logging.getLogger("exact"),
        configs.logging_level,
        log_file_path=log_path,
    )
    return AlignmentInvocation(
        source_file_path=source,
        target_file_path=target,
        output_dir_path=output,
        configs=configs,
        configs_source=config_path,
        training_reference_file_path=training,
        full_reference_file_path=full,
        candidates_file_path=candidates,
        log_file_path=log_path,
        run_eval=bool(run_eval),
        task_name=task_name,
        device=device,
    )


def prepare_alignment_namespace(args: Any) -> AlignmentInvocation:
    """Translate the frozen alignment CLI namespace through the shared seam."""
    return prepare_alignment(
        source_ontology_file=args.source_ontology_file,
        target_ontology_file=args.target_ontology_file,
        output_dir=args.output_dir,
        training_reference_file=args.training_reference_file,
        full_reference_file=args.full_reference_file,
        candidates_file=args.candidates_file,
        config_file=args.config_file,
        save_logs=args.save_logs,
        run_eval=args.run_eval,
        device=args.device,
        input_format=args.input_format,
        source_options=parse_adapter_options(args.source_options, label="Source adapter"),
        target_options=parse_adapter_options(args.target_options, label="Target adapter"),
        output_formats=args.output_formats,
        relation_prediction=args.relation_prediction,
    )


def execute_alignment(invocation: AlignmentInvocation):
    """Invoke the functional core action from a prepared delivery request."""
    return run_alignment(**invocation.action_kwargs())


@dataclass(frozen=True)
class EvaluationInvocation:
    alignment: Path
    output_dir_path: Path
    error_on_fail: bool
    k: Optional[list[int]]
    source_file_path: Optional[Path]
    target_file_path: Optional[Path]
    train_reference_file_path: Optional[Path]
    full_reference_file_path: Optional[Path]
    reference_candidates: Optional[Path]
    log_file_path: Optional[Path]
    log_level: Optional[str]
    backends: list[str]
    backend_options: dict[str, Mapping[str, Any]]

    def action_kwargs(self) -> dict[str, Any]:
        return {
            "alignment": self.alignment,
            "output_dir_path": self.output_dir_path,
            "error_on_fail": self.error_on_fail,
            "K": self.k,
            "source_file_path": self.source_file_path,
            "target_file_path": self.target_file_path,
            "train_reference_file_path": self.train_reference_file_path,
            "full_reference_file_path": self.full_reference_file_path,
            "reference_candidates": self.reference_candidates,
            "log_file_path": self.log_file_path,
            "log_level": self.log_level,
            "backends": self.backends,
            "backend_options": self.backend_options,
        }


def prepare_evaluation(
    *,
    alignment_file: PathLike,
    output_dir: PathLike,
    error_on_fail: bool = False,
    k: Optional[Sequence[int]] = None,
    source_ontology_file: Optional[PathLike] = None,
    target_ontology_file: Optional[PathLike] = None,
    train_reference_file: Optional[PathLike] = None,
    full_reference_file: Optional[PathLike] = None,
    reference_candidates: Optional[PathLike] = None,
    log_level: Optional[str] = None,
    save_logs: bool = False,
    backends: Optional[Sequence[str]] = None,
    backend_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    create_output: bool,
    log_filename: str,
    train_reference_label: str = "Training reference file",
) -> EvaluationInvocation:
    """Validate and assemble one evaluator invocation for any delivery surface."""
    alignment = require_file(alignment_file, "Alignment file")
    assert alignment is not None
    output = prepare_output_dir(output_dir, create=create_output)
    source = require_file(source_ontology_file, "Source ontology file")
    target = require_file(target_ontology_file, "Target ontology file")
    training = require_file(train_reference_file, train_reference_label)
    full = require_file(full_reference_file, "Full reference file")
    candidates = require_file(reference_candidates, "Reference candidates file")
    return EvaluationInvocation(
        alignment=alignment,
        output_dir_path=output,
        error_on_fail=bool(error_on_fail),
        k=list(k) if k is not None else None,
        source_file_path=source,
        target_file_path=target,
        train_reference_file_path=training,
        full_reference_file_path=full,
        reference_candidates=candidates,
        log_file_path=output / log_filename if save_logs else None,
        log_level=log_level,
        backends=list(backends or ["builtin"]),
        backend_options={
            str(name): dict(options) for name, options in dict(backend_options or {}).items()
        },
    )


def prepare_evaluation_namespace(args: Any) -> EvaluationInvocation:
    """Translate the frozen evaluator CLI namespace through the shared seam."""
    typed_submission = require_file(args.bioml_typed_submission, "BioML typed submission")
    typed_answers = require_file(args.bioml_typed_answers, "BioML typed answers")
    bioml_options = {
        key: value
        for key, value in {
            "typed_submission_path": typed_submission,
            "typed_answers_path": typed_answers,
            "preferred_pairs_path": optional_path(args.bioml_preferred_pairs),
            "graded_relevance_path": optional_path(args.bioml_graded_relevance),
            "hierarchy_path": optional_path(args.bioml_hierarchy),
            "candidate_count": args.bioml_candidate_count,
            "coherence_reasoner": args.bioml_coherence_reasoner,
            "coherence_timeout_s": args.bioml_coherence_timeout,
            "coherence_skip_invalid": args.bioml_skip_invalid_iris,
        }.items()
        if value is not None
    }
    return prepare_evaluation(
        alignment_file=args.alignment_file,
        output_dir=args.output_dir,
        error_on_fail=args.error_on_fail,
        k=args.K,
        source_ontology_file=args.source_ontology_file,
        target_ontology_file=args.target_ontology_file,
        train_reference_file=args.train_reference_file,
        full_reference_file=args.full_reference_file,
        reference_candidates=args.reference_candidates,
        log_level=args.log_level,
        save_logs=args.save_logs,
        backends=args.eval_backends,
        backend_options={"bioml": bioml_options},
        create_output=True,
        log_filename="OAEI_bio_ml_eval.log",
    )


def execute_evaluation(invocation: EvaluationInvocation):
    """Invoke the functional evaluator action from a prepared delivery request."""
    return run_evaluation(**invocation.action_kwargs())


__all__ = [
    "AlignmentInvocation",
    "EvaluationInvocation",
    "execute_alignment",
    "execute_evaluation",
    "optional_path",
    "parse_adapter_options",
    "prepare_alignment",
    "prepare_alignment_namespace",
    "prepare_evaluation",
    "prepare_evaluation_namespace",
    "prepare_output_dir",
    "require_file",
    "warn_ignored_jvm",
]
