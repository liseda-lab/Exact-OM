"""Orchestration for registered evaluation backends."""

from __future__ import annotations

import functools
import importlib
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Union

from exact.core.entities.evaluation import BackendEvaluation, EvaluationRequest
from exact.core.entities.mappings import EntityMapping, ReferenceMapping
from exact.core.entities.registry import ComponentRegistry, ComponentType
from exact.utils.data import save_dict_to_csv
from exact.utils.logs import configure_exact_logger
from exact.utils.provenance import file_provenance
from exact.utils.run_context import current_run_session
from exact.utils.timing import TimingLedger, config_fingerprint

_BUILTIN_BACKEND_MODULES = {
    "builtin": "exact.impl.evaluators.builtin",
    "bioml": "exact.impl.evaluators.bioml",
}


def _load_evaluator(name: str):
    if name not in ComponentRegistry.list(ComponentType.EVALUATOR):
        module_name = _BUILTIN_BACKEND_MODULES.get(name)
        if module_name:
            importlib.import_module(module_name)
    try:
        return ComponentRegistry.get(ComponentType.EVALUATOR, name)
    except ValueError as exc:
        available = sorted(
            set(ComponentRegistry.list(ComponentType.EVALUATOR)) | set(_BUILTIN_BACKEND_MODULES)
        )
        raise ValueError(
            f"Unknown evaluation backend {name!r}; available backends: {', '.join(available)}"
        ) from exc


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _path_inputs(request: EvaluationRequest) -> dict[str, dict[str, Any]]:
    paths = {
        "alignment": request.alignment,
        "train_reference": request.train_reference,
        "full_reference": request.full_reference,
        "reference_candidates": request.reference_candidates,
    }
    return {
        name: file_provenance(path)
        for name, path in paths.items()
        if isinstance(path, Path) and path.is_file()
    }


def _find_training_reference_sha(payload: Any) -> Optional[str]:
    if isinstance(payload, Mapping):
        direct = payload.get("training_reference_sha256")
        if isinstance(direct, str):
            return direct
        provenance = payload.get("training_reference")
        if isinstance(provenance, Mapping) and isinstance(provenance.get("sha256"), str):
            return str(provenance["sha256"])
        for value in payload.values():
            found = _find_training_reference_sha(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_training_reference_sha(value)
            if found:
                return found
    return None


def _update_run_stats(
    path: Path,
    provenance: Mapping[str, Any],
    *,
    logger: logging.Logger,
) -> list[str]:
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read existing run statistics at %s: %s", path, exc)
    warnings: list[str] = []
    calibrated_sha = _find_training_reference_sha(payload)
    evaluated_train = provenance.get("train_reference")
    evaluated_sha = evaluated_train.get("sha256") if isinstance(evaluated_train, Mapping) else None
    if calibrated_sha and evaluated_sha and calibrated_sha != evaluated_sha:
        message = (
            "Evaluation training-reference hash differs from the selector calibration hash "
            f"({evaluated_sha} != {calibrated_sha})."
        )
        logger.warning(message)
        warnings.append(message)
    payload["evaluation_inputs"] = dict(provenance)
    _atomic_json(path, payload)
    return warnings


def _save_report(
    output_dir: Path,
    backend_names: list[str],
    results: Mapping[str, BackendEvaluation],
    request: EvaluationRequest,
    *,
    run_stats_path: Optional[Path],
    logger: logging.Logger,
) -> dict[str, Optional[float]]:
    provenance = _path_inputs(request)
    stats_path = run_stats_path or output_dir / "run_stats.json"
    warnings = _update_run_stats(stats_path, provenance, logger=logger)

    skipped: dict[str, str] = {}
    versions: dict[str, Optional[str]] = {}
    report: dict[str, Any] = {}
    flat: dict[str, Optional[float]] = {}
    single_builtin = backend_names == ["builtin"]
    for backend in backend_names:
        result = results[backend]
        report[backend] = dict(result.metrics)
        versions[backend] = result.version
        for metric, value in result.metrics.items():
            flat_name = metric if single_builtin else f"{backend}.{metric}"
            flat[flat_name] = value
        for metric, reason in result.skipped.items():
            skipped[f"{backend}.{metric}"] = reason
    report["meta"] = {
        "refs": provenance,
        "k": list(request.k),
        "versions": versions,
        "skipped": skipped,
        "warnings": warnings,
    }
    _atomic_json(output_dir / "evaluation_results.json", report)

    if single_builtin:
        save_dict_to_csv(
            data=dict(results["builtin"].metrics),
            file_path=output_dir / "evaluation_results.csv",
            columns=["Metric", "Value"],
        )
    else:
        save_dict_to_csv(
            data=flat,
            file_path=output_dir / "evaluation_results.csv",
            columns=["Metric", "Value"],
        )
    return flat


def run_evaluation(
    alignment: Union[List[Tuple[ReferenceMapping, List[EntityMapping]]], List[EntityMapping], Path],
    output_dir_path: Path,
    error_on_fail: bool = False,
    K: Optional[List[int]] = None,
    source_file_path: Any = None,
    target_file_path: Any = None,
    train_reference_file_path: Optional[Path] = None,
    full_reference_file_path: Optional[Path] = None,
    reference_candidates: Optional[Path] = None,
    log_file_path: Optional[Path] = None,
    log_level: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    backends: Optional[List[str]] = None,
    backend_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    run_stats_path: Optional[Path] = None,
    _timing_managed: bool = False,
) -> Optional[dict[str, Optional[float]]]:
    """Run all selected evaluation backends and persist their reports."""

    start_time = time.perf_counter()
    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if logger is None:
        resolved_log_level = (
            getattr(logging, str(log_level).upper()) if log_level is not None else logging.INFO
        )
        logger = configure_exact_logger(
            logging.getLogger("exact.eval"),
            resolved_log_level,
            log_file_path=log_file_path,
        )

    backend_names = list(dict.fromkeys(backends or ["builtin"]))
    if not backend_names:
        raise ValueError("At least one evaluation backend must be selected.")
    if current_run_session() is None and not _timing_managed:
        fingerprint = config_fingerprint(
            {
                "backends": backend_names,
                "backend_options": dict(backend_options or {}),
                "k": list(K or (1, 5, 10)),
                "mode": "global" if full_reference_file_path is not None else "local",
            },
            run_dir=output_dir_path,
        )
        ledger = TimingLedger.open(output_dir_path)
        with ledger.session(command="eval", config_fingerprint=fingerprint) as session:
            with session.stage("Postprocess.Evaluation"):
                return run_evaluation(
                    alignment=alignment,
                    output_dir_path=output_dir_path,
                    error_on_fail=error_on_fail,
                    K=K,
                    source_file_path=source_file_path,
                    target_file_path=target_file_path,
                    train_reference_file_path=train_reference_file_path,
                    full_reference_file_path=full_reference_file_path,
                    reference_candidates=reference_candidates,
                    log_file_path=log_file_path,
                    log_level=log_level,
                    logger=logger,
                    backends=backend_names,
                    backend_options=backend_options,
                    run_stats_path=run_stats_path,
                    _timing_managed=True,
                )
    request_base = {
        "alignment": alignment,
        "full_reference": full_reference_file_path,
        "train_reference": train_reference_file_path,
        "reference_candidates": reference_candidates,
        "source": source_file_path,
        "target": target_file_path,
        "k": tuple(K or (1, 5, 10)),
    }

    logger.info("Starting evaluation with backends: %s", ", ".join(backend_names))
    backend_results: dict[str, BackendEvaluation] = {}
    for backend in backend_names:
        evaluator = _load_evaluator(backend)
        request = EvaluationRequest(
            **request_base,
            options=dict((backend_options or {}).get(backend, {})),
        )
        try:
            backend_results[backend] = evaluator.run(request)
        except ValueError as exc:
            logger.error("Error during %s evaluation: %s", backend, exc, exc_info=True)
            if error_on_fail:
                raise
            backend_results[backend] = BackendEvaluation(metrics={}, skipped={"backend": str(exc)})

    request = EvaluationRequest(**request_base)
    flat = _save_report(
        output_dir_path,
        backend_names,
        backend_results,
        request,
        run_stats_path=run_stats_path,
        logger=logger,
    )
    elapsed = time.perf_counter() - start_time
    logger.info("Finished evaluation in %.3f seconds", elapsed)
    if backend_names == ["builtin"] and not backend_results["builtin"].metrics:
        return None
    return flat


class EvaluationAction:
    """Deprecated namespace alias for :func:`run_evaluation`."""

    @staticmethod
    @functools.wraps(run_evaluation)
    def run(*args, **kwargs):
        warnings.warn(
            "EvaluationAction.run is deprecated; use run_evaluation instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return run_evaluation(*args, **kwargs)


__all__ = ["EvaluationAction", "run_evaluation"]
