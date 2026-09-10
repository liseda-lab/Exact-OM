#!/usr/bin/env python3
"""Freeze an R_v2 child and local model revisions using cached files only."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exact.core.entities.configs.yaml_io import dump_yaml_document, load_yaml_mapping  # noqa: E402
from exact.experiments.campaign import digest, openrouter_only  # noqa: E402
from exact.experiments.harness import (  # noqa: E402
    _assert_experiment_flags_disabled,
    _bind_model_lock_revisions,
    _configured_model_ids,
    _source_tree_fingerprint,
    _validate_model_lock,
)
from exact.experiments.schema import BaselineManifest, load_baseline  # noqa: E402
from exact.utils.provenance import sha256_file  # noqa: E402


def cached_model(
    model_id: str,
    cache: Path,
    *,
    revision: str | None = None,
    tokenizer_only: bool = False,
) -> dict[str, Any]:
    """Verify every snapshot file and required model/tokenizer assets without loading it."""
    folder = cache / ("models--" + model_id.replace("/", "--"))
    if revision is None:
        ref = folder / "refs/main"
        if ref.is_file():
            revision = ref.read_text().strip()
        else:
            snapshots = sorted((folder / "snapshots").glob("*"))
            if len(snapshots) != 1:
                raise ValueError(f"{model_id}: choose an explicit cached revision")
            revision = snapshots[0].name
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{model_id}: revision must be an immutable 40-hex commit")
    snapshot = folder / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"{model_id}: snapshot is not cached: {revision}")
    files = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            continue
        if not path.is_file():
            raise FileNotFoundError(f"{model_id}: broken snapshot asset: {path.name}")
        files[path.relative_to(snapshot).as_posix()] = sha256_file(path)
    tokenizers = {"tokenizer.json", "vocab.txt", "tokenizer.model", "sentencepiece.bpe.model"}
    if not tokenizers.intersection(files):
        raise ValueError(f"{model_id}: cached tokenizer assets are incomplete")
    if not tokenizer_only:
        if "config.json" not in files:
            raise ValueError(f"{model_id}: cached model config is missing")
        weights = [name for name in files if name.endswith(".safetensors") or name.endswith(".bin")]
        if not weights:
            raise ValueError(f"{model_id}: cached model weights are missing")
        for name in files:
            if name.endswith((".safetensors.index.json", ".bin.index.json")):
                index = json.loads((snapshot / name).read_text())
                missing = set(index["weight_map"].values()) - files.keys()
                if missing:
                    raise ValueError(
                        f"{model_id}: cached weight shards are missing: {sorted(missing)}"
                    )
    return {
        "requested_id": model_id,
        "resolved_revision": revision,
        "artifact_sha256": digest(files),
        "snapshot_path": str(snapshot.resolve()),
        "tokenizer_only": tokenizer_only,
        "files": files,
        "bytes": sum((snapshot / name).stat().st_size for name in files),
    }


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"immutable lock output differs: {path}; use a new output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def prepare(
    base_path: Path,
    destination: Path,
    *,
    cache: Path,
    parent_path: Path,
    profile: str = "openrouter_gpt4o_mini",
    models: tuple[str, ...] = (),
    hosted_model: str = "openai/gpt-4o-mini-2024-07-18",
    hosted_revision: str = "2024-07-18",
    provider: str = "OpenAI",
    note: str,
) -> dict[str, Any]:
    """Write immutable declarations only; no HTTP client or model loader is called."""
    parent = load_baseline(parent_path)
    if parent.baseline_id != "R_0":
        raise ValueError("R_v2 must retain the historical R_0 parent")
    _assert_experiment_flags_disabled(base_path)
    mapping = openrouter_only(load_yaml_mapping(base_path), profile)
    mapping["llm"]["profiles"][profile].update(
        model=hosted_model,
        revision=hosted_revision,
        provider={"only": [provider], "allow_fallbacks": False, "require_parameters": True},
    )
    required = _configured_model_ids(mapping, hosted_only=True)
    tokenizer_ids = {
        item["tokenizer"]
        for name, item in mapping["llm"]["profiles"].items()
        if name == profile and item.get("tokenizer")
    }
    revisions = {}
    for specification in models:
        model_id, separator, revision = specification.partition("@")
        if not model_id:
            raise ValueError("extra model identity cannot be empty")
        if separator:
            revisions[model_id] = revision
        required.add(model_id)
    entries = {
        model_id: cached_model(
            model_id,
            cache,
            revision=revisions.get(model_id),
            tokenizer_only=model_id in tokenizer_ids,
        )
        for model_id in sorted(required)
    }
    model_lock = {
        "schema_version": 1,
        "status": "complete",
        "models": entries,
        "hosted_profiles": {profile: mapping["llm"]["profiles"][profile]},
    }
    mapping = _bind_model_lock_revisions(
        mapping, model_lock, require_complete=True, hosted_only=True
    )
    model_lock["hosted_profiles"] = {profile: mapping["llm"]["profiles"][profile]}
    destination = destination.resolve()
    config_path = destination / "R_v2.config.yaml"
    lock_path = destination / "models.lock.yaml"
    baseline_path = destination / "R_v2.yaml"
    _write_once(config_path, dump_yaml_document(mapping))
    _assert_experiment_flags_disabled(config_path)
    _write_once(lock_path, dump_yaml_document(model_lock))
    _validate_model_lock(lock_path)
    source = _source_tree_fingerprint(ROOT)
    if source is None:
        raise ValueError("source tree provenance could not be determined")
    manifest = BaselineManifest(
        baseline_id="R_v2",
        parent="R_0",
        exact_om_version=importlib.metadata.version("exact-om"),
        pyowlcore_version=importlib.metadata.version("pyowl-core"),
        config=Path(config_path.name),
        config_sha256=sha256_file(config_path),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_tree_sha256=source["sha256"],
        source_tree_files=source["files"],
        status="frozen_configuration",
        experiment_flags="disabled",
        note=note,
    )
    _write_once(baseline_path, dump_yaml_document(manifest.model_dump(mode="json")))
    result = {
        "schema_version": 1,
        "base_config": str(config_path),
        "baseline_manifest": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
        "model_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "historical_parent": {
            "path": str(parent_path.resolve()),
            "sha256": sha256_file(parent_path),
        },
        "input_config_sha256": sha256_file(base_path),
        "source_tree": source,
        "preparation": "cached_files_only; no model execution or hosted requests",
    }
    _write_once(
        destination / "preparation.json", json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=ROOT / "exact/default_config.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--parent-manifest", type=Path, default=ROOT / "exp/experiments/baselines/R_0.yaml"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")),
    )
    parser.add_argument("--profile", default="openrouter_gpt4o_mini")
    parser.add_argument("--hosted-model", default="openai/gpt-4o-mini-2024-07-18")
    parser.add_argument("--hosted-revision", default="2024-07-18")
    parser.add_argument("--provider", default="OpenAI")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Additional cached model ID, optionally ID@40-hex-revision.",
    )
    parser.add_argument(
        "--note", required=True, help="Record the accepted changes from historical R_0."
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.base_config,
                args.output_root,
                cache=args.cache_dir,
                parent_path=args.parent_manifest,
                profile=args.profile,
                models=tuple(args.model),
                hosted_model=args.hosted_model,
                hosted_revision=args.hosted_revision,
                provider=args.provider,
                note=args.note,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
