"""Subprocess worker for the WP-L hard-interruption acceptance test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from exact.core.contracts.model import IModel
from exact.impl.trainer import SemanticAlignmentRunner
from exact.runs import finalize_artifacts

CRASH_EXIT_CODE = 86


class _FixtureDataset:
    dataset_signature = "wp-l-interrupted-run"
    cache_fingerprint = "wp-l-interrupted-candidates-v1"

    def __init__(self, size: int = 6):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "src_iri": f"https://example.org/source/{index}",
            "tgt_iri": f"https://example.org/target/{index}",
            "src_kind": "class",
            "tgt_kind": "class",
            "src_labels": [f"Source concept {index}"],
            "tgt_labels": [f"Target concept {index}"],
            "src_ctx_triples": [],
            "tgt_ctx_triples": [],
            "label": index % 2 == 0,
        }


class _FixtureModel(IModel):
    def __init__(self, crash_before_call: int | None = None, **kwargs: Any):
        super().__init__()
        self.crash_before_call = crash_before_call
        self.calls = 0

    def runtime_fingerprint_payload(self, **kwargs: Any) -> dict[str, str]:
        # The crash injection is a harness concern, not a scoring change, so a
        # healthy process can resume the checkpoint written by the crash worker.
        return {"model": "wp-l-interrupted-fixture-v1"}

    def runtime_fingerprint(self) -> str:
        payload = json.dumps(self.runtime_fingerprint_payload(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def forward(self, *, src_iris: list[str], tgt_iris: list[str], **kwargs: Any):
        self.calls += 1
        if self.crash_before_call == self.calls:
            os._exit(CRASH_EXIT_CODE)
        scores = [0.95 - 0.08 * int(str(src).rsplit("/", 1)[-1]) for src in src_iris]
        return {
            "S_final": torch.tensor(scores, dtype=torch.float32),
            "explanations": [
                {
                    "description": "deterministic CPU interruption fixture",
                    "prediction": {"model_decision": score >= 0.5},
                    "context": {
                        "source": [f"source context for {src}"],
                        "target": [f"target context for {tgt}"],
                    },
                }
                for src, tgt, score in zip(src_iris, tgt_iris, scores)
            ],
        }


def _run(output: Path, *, crash: bool, resume: bool) -> None:
    runner = SemanticAlignmentRunner(
        dataset=_FixtureDataset(),
        model=_FixtureModel,
        model_params={"crash_before_call": 3 if crash else None},
        device=torch.device("cpu"),
        output_dir=output,
    )
    predictions, _ = runner.predict(
        threshold=0.5,
        batch_size=1,
        num_workers=0,
        mixed_precision=False,
        checkpoint_file="inference_acceptance.json",
        checkpoint_every=1,
        resume_from_checkpoint=resume,
        enable_checkpoints=True,
        audit_shard_compression="zstd",
        cache_persist_policy="never",
    )
    runner.save_results(
        predictions,
        save_json=False,
        output_formats=["tsv-global"],
    )
    finalize_artifacts(
        output,
        run_id=runner._run_id,
        checkpoint_retention="latest",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("crash", "resume", "uninterrupted"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    _run(
        args.output,
        crash=args.mode == "crash",
        resume=args.mode != "uninterrupted",
    )


if __name__ == "__main__":
    main()
