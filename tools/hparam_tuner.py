#!/usr/bin/env python3
"""
Utility to generate Exact experiments for hyperparameter sweeps.

Supports traditional grid search as well as a lightweight low-discrepancy sampler
with optional local exploitation to cover the space with fewer trials.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml

from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.configs.migration import V1_TO_V2


@dataclass(frozen=True)
class ParamSpec:
    name: str
    dtype: str
    values: Sequence
    bounds: tuple[float, float] | None
    scale: str
    quantize: float | None

    @classmethod
    def from_mapping(cls, name: str, data: dict) -> "ParamSpec":
        dtype = data.get("type", "float")
        values = data.get("values")
        bounds = tuple(data["bounds"]) if "bounds" in data else None
        scale = data.get("scale", "linear")
        quantize = data.get("quantize")
        if bounds is not None and len(bounds) != 2:
            raise ValueError(f"Param '{name}' bounds must have size 2.")
        if dtype not in {"float", "int", "categorical"}:
            raise ValueError(f"Param '{name}' has unsupported dtype '{dtype}'.")
        if dtype == "categorical" and not values:
            raise ValueError(f"Param '{name}' of type categorical needs explicit values.")
        if values is not None and not isinstance(values, list):
            raise ValueError(f"Param '{name}' values must be a list.")
        return cls(
            name=name,
            dtype=dtype,
            values=values or [],
            bounds=bounds,
            scale=scale,
            quantize=quantize,
        )

    def convert(self, raw):
        if self.dtype == "int":
            return int(round(raw))
        if self.dtype == "float":
            return float(raw)
        return raw

    def value_from_unit(self, u: float) -> float:
        if self.bounds is None:
            raise ValueError(f"Param '{self.name}' does not define bounds for continuous sampling.")
        lo, hi = self.bounds
        if self.scale == "log":
            lo = math.log(lo)
            hi = math.log(hi)
            value = math.exp(lo + (hi - lo) * u)
        else:
            value = lo + (hi - lo) * u
        if self.quantize:
            value = round(value / self.quantize) * self.quantize
        return self.convert(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Exact hyperparameter sweep jobs.")
    parser.add_argument(
        "--tuner-config",
        required=True,
        type=Path,
        help="Path to tuner YAML description.",
    )
    parser.add_argument(
        "--strategy",
        choices=("grid", "smart", "per_param"),
        default="grid",
        help="Grid search, low-discrepancy smart sampler, or per-parameter sweeps.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of smart samples to draw (overrides YAML).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing files.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit sbatch commands after generation.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def set_nested(mapping: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    target = mapping
    for part in parts[:-1]:
        if isinstance(target, list):
            if not part.isdigit() or int(part) >= len(target):
                raise KeyError(f"Cannot set '{dotted_key}': invalid list index '{part}'.")
            target = target[int(part)]
        else:
            if part not in target:
                raise KeyError(f"Cannot set '{dotted_key}': missing intermediate key '{part}'.")
            target = target[part]
    leaf = parts[-1]
    if isinstance(target, list):
        if not leaf.isdigit() or int(leaf) >= len(target):
            raise KeyError(f"Cannot set '{dotted_key}': invalid list index '{leaf}'.")
        target[int(leaf)] = value
    else:
        target[leaf] = value


def get_nested(mapping: dict, dotted_key: str):
    parts = dotted_key.split(".")
    target = mapping
    for part in parts:
        target = target[int(part)] if isinstance(target, list) else target[part]
    return target


def translate_parameter_path(dotted_key: str) -> str:
    """Translate one legacy tuner path to its canonical v2 location."""

    dynamic_prefixes = {
        "model.params.": "pipeline.0.params.",
        "second_model.params.": "pipeline.1.params.",
        "llm_profiles.": "llm.profiles.",
        "llm_routing.": "llm.routing.",
        "evaluation.bioml.": "evaluation.bioml.",
        "dataset_params.hierarchical_relation_families.": (
            "dataset.hierarchical_relation_families."
        ),
    }
    for old_prefix, new_prefix in dynamic_prefixes.items():
        if dotted_key.startswith(old_prefix):
            return new_prefix + dotted_key[len(old_prefix) :]
    target = V1_TO_V2.get(dotted_key)
    return target if isinstance(target, str) else dotted_key


def resolve_base_config(raw: dict) -> dict:
    """Migrate v1 templates and expand them to a complete canonical v2 dump."""

    return ConfigModel.from_mapping(raw, warn_v1=False).model_dump(mode="json", by_alias=True)


def grid_trials(params: Sequence[ParamSpec]) -> List[Dict[str, object]]:
    for spec in params:
        if not spec.values:
            raise ValueError(f"Param '{spec.name}' requires explicit values for grid search.")
    all_values = [[(spec.name, spec.convert(value)) for value in spec.values] for spec in params]
    combinations = []
    for combo in itertools.product(*all_values):
        trial = {name: value for name, value in combo}
        combinations.append(trial)
    return combinations


def per_param_trials(params: Sequence[ParamSpec]) -> List[Dict[str, object]]:
    """Generate trials that change one parameter at a time from the baseline."""
    trials: List[Dict[str, object]] = []
    for spec in params:
        if not spec.values:
            raise ValueError(
                f"Param '{spec.name}' requires explicit values for per-parameter sweeps."
            )
        for value in spec.values:
            trials.append({spec.name: spec.convert(value)})
    return trials


def halton(dim: int, count: int, skip: int = 0) -> List[List[float]]:
    def _primes(n: int) -> List[int]:
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if candidate % p == 0:
                    is_prime = False
                    break
                if p * p > candidate:
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes

    def _van_der_corput(index: int, base: int) -> float:
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    if dim == 0:
        return [[] for _ in range(count)]
    bases = _primes(dim)
    seq = []
    for n in range(skip + 1, skip + count + 1):
        seq.append([_van_der_corput(n, base) for base in bases])
    return seq


def smart_trials(
    params: Sequence[ParamSpec],
    cfg: dict,
    base_config: dict,
    override_num_samples: int | None,
) -> List[Dict[str, object]]:
    num_samples = override_num_samples or cfg.get("num_samples", 10)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive for smart strategy.")
    exploit_fraction = cfg.get("exploit_fraction", 0.25)
    exploit_noise = cfg.get("exploit_noise", 0.1)
    rng = random.Random(cfg.get("random_seed", 42))

    num_exploit = int(round(num_samples * exploit_fraction))
    num_exploit = min(num_exploit, num_samples)
    num_explore = num_samples - num_exploit

    continuous = [p for p in params if p.bounds is not None]
    discrete = [p for p in params if p.bounds is None]

    trials: List[Dict[str, object]] = []
    halton_points = halton(len(continuous), num_explore, skip=rng.randint(0, 1000))

    for idx in range(num_explore):
        point = {}
        for c_idx, spec in enumerate(continuous):
            point[spec.name] = spec.value_from_unit(halton_points[idx][c_idx])
        for spec in discrete:
            if spec.values:
                value = spec.values[(idx + len(trials)) % len(spec.values)]
            else:
                value = get_nested(base_config, spec.name)
            point[spec.name] = value
        trials.append(point)

    anchors = cfg.get("anchor_configs") or []
    anchor_values = [anchor.get("values", {}) for anchor in anchors]
    if not anchor_values:
        anchor_values.append({spec.name: get_nested(base_config, spec.name) for spec in params})

    def _perturb(spec: ParamSpec, base_value):
        if spec.bounds is None:
            return base_value
        span = spec.bounds[1] - spec.bounds[0]
        noise = rng.uniform(-exploit_noise, exploit_noise)
        candidate = base_value + noise * span
        candidate = max(spec.bounds[0], min(spec.bounds[1], candidate))
        if spec.quantize:
            candidate = round(candidate / spec.quantize) * spec.quantize
        return spec.convert(candidate)

    for idx in range(num_exploit):
        anchor = anchor_values[idx % len(anchor_values)]
        point = {}
        for spec in params:
            base_value = anchor.get(spec.name, get_nested(base_config, spec.name))
            if spec.bounds is not None:
                base_value = spec.convert(base_value)
                point[spec.name] = _perturb(spec, base_value)
            else:
                if spec.values:
                    base_value = spec.values[idx % len(spec.values)]
                point[spec.name] = (
                    spec.convert(base_value) if spec.dtype != "categorical" else base_value
                )
        trials.append(point)

    return trials


def slugify(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("/", "_")


def build_trial_name(prefix: str, index: int, params: Dict[str, object]) -> str:
    fragments = [f"{key.split('.')[-1]}={slugify(value)}" for key, value in params.items()]
    suffix = "__".join(fragments)
    return f"{prefix}_{index:03d}_{suffix}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_extra_files(entries: Iterable, config_dir: Path) -> List[Tuple[Path, Path]]:
    if not entries:
        return []
    if isinstance(entries, (str, bytes)) or isinstance(entries, dict):
        entries = [entries]
    normalized: List[Tuple[Path, Path]] = []
    for idx, entry in enumerate(entries):
        if isinstance(entry, str):
            src_str = entry
            dst_str = Path(entry).name
        elif isinstance(entry, dict):
            src_str = entry.get("source") or entry.get("src")
            if not src_str:
                raise ValueError(f"extra_files[{idx}] is missing 'source'.")
            dst_str = (
                entry.get("destination")
                or entry.get("dest")
                or entry.get("target")
                or Path(src_str).name
            )
        else:
            raise ValueError("extra_files entries must be strings or mappings with 'source'.")
        src_path = Path(src_str).expanduser()
        candidates: List[Path] = []
        if src_path.is_absolute():
            candidates.append(src_path)
        else:
            candidates.append((Path.cwd() / src_path))
            candidates.append((config_dir / src_path))
        resolved_src = None
        for candidate in candidates:
            candidate_resolved = candidate.resolve()
            if candidate_resolved.exists():
                resolved_src = candidate_resolved
                break
        if resolved_src is None:
            raise FileNotFoundError(
                f"Extra file '{src_str}' was not found relative to the current working directory or {config_dir}."
            )
        dst_path = Path(dst_str)
        if dst_path.is_absolute():
            raise ValueError(
                f"Destination path '{dst_str}' for extra_files[{idx}] must be relative."
            )
        if any(part == ".." for part in dst_path.parts):
            raise ValueError(
                f"Destination path '{dst_str}' for extra_files[{idx}] cannot contain '..'."
            )
        normalized.append((resolved_src, dst_path))
    return normalized


def copy_extra_files(trial_dir: Path, extra_files: Sequence[Tuple[Path, Path]]) -> None:
    for src, rel_dst in extra_files:
        dest = trial_dir / rel_dst
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            ensure_dir(dest.parent)
            shutil.copy2(src, dest)


def create_trial_files(
    trial_dir: Path,
    config_template: dict,
    param_values: Dict[str, object],
) -> Path:
    config = json.loads(json.dumps(config_template))
    for key, value in param_values.items():
        set_nested(config, key, value)
    config = resolve_base_config(config)
    ensure_dir(trial_dir)
    config_path = trial_dir / "config.yaml"
    write_yaml(config_path, config)
    meta_path = trial_dir / "trial.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump({"params": param_values}, handle, indent=2)
    return config_path


def build_export_vars(
    dataset_cfg: dict, config_path: Path, trial_dir: Path, job_name: str
) -> Dict[str, str]:
    export = {
        "JOB_NAME": job_name,
        "EXP_DIR": str(trial_dir),
        "CONFIG_FILE": str(config_path),
        "DATA_DIR": dataset_cfg["data_dir"],
        "SOURCE": dataset_cfg["source"],
        "TARGET": dataset_cfg["target"],
        "FULL_REFERENCE": dataset_cfg["full_reference"],
        "CANDIDATES": dataset_cfg["candidates"],
    }
    if dataset_cfg.get("reference"):
        export["REFERENCE"] = dataset_cfg["reference"]
    if dataset_cfg.get("memory"):
        export["MEMORY"] = str(dataset_cfg["memory"])
    if dataset_cfg.get("device") is not None:
        export["DEVICE"] = str(dataset_cfg["device"])
    if dataset_cfg.get("run_eval") is not None:
        export["RUN_EVAL"] = "1" if dataset_cfg["run_eval"] else "0"
    if dataset_cfg.get("save_logs") is not None:
        export["SAVE_LOGS"] = "1" if dataset_cfg["save_logs"] else "0"
    return export


def export_string(env: Dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in env.items())


def write_submit_script(
    commands: List[str],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#!/bin/bash\nset -euo pipefail\n\n")
        for cmd in commands:
            handle.write(f"{cmd}\n")


def maybe_submit(commands: List[str]) -> None:
    import subprocess

    for cmd in commands:
        print(f"[SUBMIT] {cmd}")
        subprocess.run(cmd, check=True, shell=True)


def main() -> None:
    args = parse_args()
    tuner_config_path = args.tuner_config.resolve()
    tuner_cfg = load_yaml(tuner_config_path)
    config_dir = tuner_config_path.parent
    extra_files = resolve_extra_files(tuner_cfg.get("extra_files"), config_dir)
    base_config_path = Path(tuner_cfg["base_config"]).resolve()
    base_config = resolve_base_config(load_yaml(base_config_path))
    params = [
        ParamSpec.from_mapping(translate_parameter_path(name), spec)
        for name, spec in tuner_cfg["search_space"].items()
    ]
    experiment_root = Path(tuner_cfg["experiment_root"]).resolve()
    if args.dry_run:
        print(f"[DRY RUN] Would generate experiments under {experiment_root}")
    dataset_cfg = tuner_cfg["dataset"]
    job_prefix = tuner_cfg.get("job_name_prefix", "tune")
    slurm = tuner_cfg["slurm"]
    sbatch_args = slurm.get("sbatch_args", [])
    slurm_script = slurm["script"]

    if args.strategy == "grid":
        trial_params = grid_trials(params)
    elif args.strategy == "per_param":
        trial_params = per_param_trials(params)
    else:
        smart_cfg = json.loads(json.dumps(tuner_cfg.get("smart", {})))
        for anchor in smart_cfg.get("anchor_configs", []) or []:
            anchor["values"] = {
                translate_parameter_path(name): value
                for name, value in (anchor.get("values") or {}).items()
            }
        trial_params = smart_trials(params, smart_cfg, base_config, args.num_samples)

    if not trial_params:
        raise SystemExit("No trials generated.")

    commands: List[str] = []
    manifest = []

    for index, param_values in enumerate(trial_params):
        job_name = build_trial_name(job_prefix, index, param_values)
        trial_dir = experiment_root / job_name
        config_path = create_trial_files(trial_dir, base_config, param_values)
        if extra_files:
            copy_extra_files(trial_dir, extra_files)
        export_vars = build_export_vars(dataset_cfg, config_path, trial_dir, job_name)
        slurm_dir = trial_dir / "slurm"
        ensure_dir(slurm_dir)
        args_segment = " ".join(sbatch_args)
        if args_segment:
            args_segment = f"{args_segment} "
        cmd = (
            f"sbatch --job-name {job_name} "
            f"--output {slurm_dir}/slurm_%j.out --error {slurm_dir}/slurm_%j.err "
            f"{args_segment}"
            f"--export=ALL,{export_string(export_vars)} {slurm_script}"
        )
        commands.append(" ".join(cmd.split()))
        manifest.append(
            {"job_name": job_name, "params": param_values, "config_path": str(config_path)}
        )

    if args.dry_run:
        for item in manifest:
            print(f"[TRIAL] {item['job_name']}: {item['params']}")
        return

    ensure_dir(experiment_root)
    manifest_path = experiment_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    submit_script = experiment_root / "submit_all.sh"
    write_submit_script(commands, submit_script)
    submit_script.chmod(0o755)

    print(f"Wrote {len(commands)} trials to {experiment_root}")
    print(f"- Manifest: {manifest_path}")
    print(f"- Submit helper: {submit_script}")

    if args.submit:
        maybe_submit(commands)


if __name__ == "__main__":
    main()
