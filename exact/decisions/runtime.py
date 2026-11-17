"""Lazy adapter for the pinned, patched OneJev noncausal slot readout."""

from __future__ import annotations

import importlib.util
from importlib.metadata import version
import os
from pathlib import Path
from time import monotonic

from .pipeline import file_hash, read


class ElephantEngine:
    """Run independent pair requests locally with the qualified numerical settings.

    Models and runtime code must already exist locally. This adapter never falls
    back to a hosted API, a chat model, or an unpatched causal readout.
    """

    def __init__(self, model_dir: Path, *, device: str = "cuda:0"):
        model_dir = Path(model_dir).resolve()
        if not model_dir.is_dir():
            raise ValueError("Elephant requires an existing local model directory")
        spec = importlib.util.find_spec("qev")
        if spec is None or spec.origin is None:
            raise RuntimeError(
                "Install pinned OneJev with Elephant's serve.patch; see docs/decisions.md"
            )
        manifest = read(Path(__file__).with_name("runtime.json"))
        for distribution, expected in manifest["dependency_versions"].items():
            if version(distribution) != expected:
                raise ValueError(
                    f"Pinned Elephant dependency differs: {distribution} requires {expected}"
                )
        runtime_dir = Path(spec.origin).parent
        for root, bindings in (
            (runtime_dir, manifest["runtime_files"]),
            (model_dir, manifest["model_files"]),
        ):
            for relative, expected in bindings.items():
                path = root / relative
                if not path.is_file() or file_hash(path) != expected:
                    raise ValueError(f"Pinned Elephant content differs: {path}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from qev.engine import DecisionEngine
        from qev.schema import SystemOneRequest

        self.request_type = SystemOneRequest
        self.engine = DecisionEngine(
            str(model_dir),
            device=device,
            dtype="bfloat16",
            head_dtype="float32",
            fork_mode="sequential",
            max_branch_tokens=16384,
            max_request_tokens=16384,
        )
        if (
            self.engine.info.extra.get("attention") != "noncausal"
            or self.engine.info.extra.get("readout") != "slot_lm"
        ):
            raise RuntimeError("Elephant must use noncausal attention and slot_lm readout")

    def predict(self, payload: dict) -> dict:
        """Return a calibrated local probability without sampling text tokens."""
        start = monotonic()
        response, _ = self.engine.decide(self.request_type(**payload))
        return {
            "probability": float(response.answers["equivalent"].noul),
            "usage": response.usage.model_dump(),
            "elapsed_seconds": monotonic() - start,
            "local_model_call_attempted": True,
            "live_provider_call_attempted": False,
            "selector_stage_consultation": False,
        }
