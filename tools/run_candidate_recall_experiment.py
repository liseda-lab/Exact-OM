#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import yaml

from exact import init_jvm
from exact.analysis.candidate_recall import (
    analyze_candidate_recall,
    flatten_candidate_recall,
    pairs_from_table,
    write_absent_gold_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run candidate-only recall experiments for Exact.")
    parser.add_argument(
        "--run-config", type=Path, required=True, help="Run YAML with dataset/job paths."
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[20, 40, 60], help="Candidate top-k values."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for recall outputs."
    )
    parser.add_argument("--jvm-heap-size", default="32G", help="Heap size for exact.init_jvm.")
    parser.add_argument(
        "--device", default=None, help="CUDA device id; CPU is used when CUDA is unavailable."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.output_dir)

    init_jvm(_normalize_heap_size(args.jvm_heap_size))

    from exact.impl import bootstrap_components

    bootstrap_components()
    from exact.core.entities.configs.config import ConfigModel

    run_cfg = _load_yaml(args.run_config)
    dataset_cfg = run_cfg["dataset"]
    job_cfg = run_cfg.get("job", {})
    config_path = Path(job_cfg.get("config_file") or args.output_dir / "config.yaml").resolve()
    configs = ConfigModel.load_config(config_path) if config_path.exists() else ConfigModel()
    configs.resolve_dependencies()

    data_dir = Path(dataset_cfg["data_dir"]).resolve()
    source_path = _dataset_path(data_dir, dataset_cfg.get("source"))
    target_path = _dataset_path(data_dir, dataset_cfg.get("target"))
    reference_path = _dataset_path(data_dir, dataset_cfg.get("full_reference"))
    train_reference_path = _dataset_path(
        data_dir, dataset_cfg.get("train_reference"), required=False
    )
    if reference_path is None:
        raise ValueError(
            "Run config must provide dataset.full_reference for candidate recall analysis."
        )

    reference_pairs = pairs_from_table(reference_path)
    train_pairs = pairs_from_table(train_reference_path) if train_reference_path else set()
    device = _resolve_device(args.device)
    logger = logging.getLogger("exact.candidate_recall")

    dataset = configs.dataset(
        output_path=args.output_dir / "work",
        logger=logger,
        cache_ok=False,
        device=device,
        llm_profiles={name: value.model_dump() for name, value in configs.llm_profiles.items()},
        llm_routing=configs.llm_routing.model_dump(),
        request_seed=configs.seed,
        candidate_generation_params={
            **configs.candidates_params.model_dump(),
            "candidates_file_path": None,
        },
        **configs.dataset_params.model_dump(),
    )
    dataset.load_ontologies(source_path, target_path)

    exact_pairs = set()
    if configs.dataset_params.filter_exact_matches:
        dataset.get_exact_matches()
        exact_df = dataset.exact_matches if dataset.exact_matches is not None else pd.DataFrame()
        if not exact_df.empty:
            exact_pairs = set(
                (str(src), str(tgt))
                for src, tgt in exact_df[["Src", "Tgt"]].dropna().itertuples(index=False)
            )

    summary: Dict[str, Any] = {
        "run_config": str(args.run_config.resolve()),
        "config_file": str(config_path),
        "source": str(source_path),
        "target": str(target_path),
        "reference": str(reference_path),
        "train_reference": str(train_reference_path) if train_reference_path else None,
        "device": str(device),
        "top_k": [int(value) for value in args.top_k],
        "runs": {},
    }
    rows = []
    for top_k in args.top_k:
        candidate_params = {
            **configs.candidates_params.model_dump(),
            "top_k": int(top_k),
        }
        logger.info("Generating candidates for top_k=%s", top_k)
        dataset.generate_candidates(device=device, **candidate_params)
        candidates = dataset.candidates if dataset.candidates is not None else pd.DataFrame()
        analysis = analyze_candidate_recall(
            candidates,
            reference_pairs=reference_pairs,
            train_pairs=train_pairs,
            exact_pairs=exact_pairs,
        )
        summary["runs"][str(top_k)] = analysis
        rows.append(flatten_candidate_recall(int(top_k), analysis))
        write_absent_gold_tsv(args.output_dir / f"absent_gold_top_k_{int(top_k)}.tsv", analysis)
        write_absent_gold_tsv(
            args.output_dir / f"absent_gold_after_exact_top_k_{int(top_k)}.tsv",
            analysis,
            after_exact=True,
        )
        _write_outputs(args.output_dir, summary, rows)
        metrics = analysis["metrics"]
        counts = analysis["counts"]
        logger.info(
            "top_k=%s rows=%s generated_recall=%.4f oracle_recall=%.4f absent=%s absent_after_exact=%s",
            top_k,
            len(candidates),
            metrics["generated_candidate_recall"],
            metrics["exact_prefilter_oracle_recall"],
            counts["absent_gold_pairs"],
            counts["absent_gold_pairs_after_exact"],
        )

    _write_outputs(args.output_dir, summary, rows)
    print(pd.DataFrame(rows).to_string(index=False))


def _write_outputs(output_dir: Path, summary: Dict[str, Any], rows: list[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_recall_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(output_dir / "candidate_recall_by_top_k.csv", index=False)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dataset_path(data_dir: Path, value: Optional[str], required: bool = True) -> Optional[Path]:
    if not value:
        if required:
            raise ValueError("Required dataset path is missing.")
        return None
    path = Path(value)
    if not path.is_absolute():
        path = data_dir / path
    path = path.resolve()
    if required and not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    if not required and not path.exists():
        return None
    return path


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is None or str(device_arg).strip() == "":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(device_arg)}")
    return torch.device("cpu")


def _normalize_heap_size(value: str) -> str:
    text = str(value or "32G").strip()
    return f"{text}G" if text.isdigit() else text


def _setup_logging(output_dir: Path) -> None:
    logger = logging.getLogger("exact.candidate_recall")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output_dir / "candidate_recall.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


if __name__ == "__main__":
    main()
