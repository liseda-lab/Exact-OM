"""Fail-closed application of immutable E03 score-calibration artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from exact.utils.provenance import file_provenance

from .calibration_helpers import ScoreCalibrator, score_calibrator_from_dict


class ScoreCalibrationMixin:
    """Apply a pre-fitted calibrator without reading labels at inference time."""

    def _load_matching_score_calibrator(self) -> Tuple[ScoreCalibrator, Dict[str, Any]]:
        mode = str(self.matching_calibration.get("mode", "none")).strip().lower()
        artifact_value = self.matching_calibration.get("artifact")
        if mode == "none":
            raise ValueError("internal error: no calibrator exists for matching.calibration=none")
        if not artifact_value:
            raise ValueError(
                f"matching.calibration.mode={mode!r} requires an immutable fitted artifact; "
                "runtime fitting or silent identity calibration is forbidden"
            )
        path = Path(str(artifact_value)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"score-calibration artifact does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid score-calibration artifact {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("score-calibration artifact must contain a JSON object")
        if payload.get("schema_version") != 1:
            raise ValueError("score-calibration artifact schema_version must be 1")
        if payload.get("kind") != "score_calibrator":
            raise ValueError("score-calibration artifact kind must be 'score_calibrator'")
        calibrator_payload = payload.get("calibrator")
        if not isinstance(calibrator_payload, Mapping):
            raise ValueError("score-calibration artifact requires a calibrator object")
        calibrator = score_calibrator_from_dict(calibrator_payload, expected_mode=mode)
        provenance = dict(file_provenance(path))
        metadata: Dict[str, Any] = {
            "mode": mode,
            "artifact": provenance,
            "parameters": dict(calibrator_payload),
            "fit_provenance": dict(payload.get("fit_provenance") or {}),
        }
        return calibrator, metadata

    def _apply_matching_score_calibration(self, df: pd.DataFrame) -> pd.DataFrame:
        mode = str(self.matching_calibration.get("mode", "none")).strip().lower()
        artifact_value = self.matching_calibration.get("artifact")
        if mode == "none":
            if artifact_value:
                raise ValueError(
                    "matching.calibration.artifact is set while mode='none'; refusing an unused artifact"
                )
            self._score_calibration_meta = {"mode": "none", "applied": False}
            return df

        calibrator, metadata = self._load_matching_score_calibrator()
        raw_scores = [float(value) for value in df["S_pair_final"].tolist()]
        if any(not math.isfinite(value) for value in raw_scores):
            raise ValueError("S_pair_final contains non-finite values before score calibration")
        calibrated = calibrator.predict(raw_scores)
        if len(calibrated) != len(raw_scores) or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0 for value in calibrated
        ):
            raise ValueError("score calibrator emitted invalid probabilities")

        df["S_pair_pre_calibration"] = raw_scores
        df["S_pair_final"] = calibrated
        # S_final is the public score when selector replacement is disabled.
        # Keeping both columns synchronized also prevents later feature code
        # from accidentally consuming the uncalibrated value.
        df["S_final"] = calibrated
        self._score_calibration_meta = {
            **metadata,
            "applied": True,
            "rows": len(raw_scores),
        }
        return df


__all__ = ["ScoreCalibrationMixin"]
