"""One pinned LogMap comparator, executed by the shared experiment scheduler."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from exact.io.writers.oaei_rdf import read_alignment
from exact.utils.data import read_table
from exact.utils.provenance import sha256_file, sha256_path
from exact.utils.timing import TimingLedger


def validate_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject mutable executables, unbounded jobs, and unsupported matcher options."""
    required = {"matcher", "bundle", "jar", "timeout_seconds", "java_heap_gb", "java_threads"}
    if set(value) != required or value["matcher"] != "logmap":
        raise ValueError("published matcher requires the explicit bounded LogMap binding")
    bundle = value["bundle"]
    if not isinstance(bundle, Mapping) or set(bundle) != {"path", "sha256"}:
        raise ValueError("LogMap bundle requires path and sha256")
    digest = str(bundle["sha256"])
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("LogMap bundle requires an immutable SHA256")
    jar = Path(str(value["jar"]))
    if jar.is_absolute() or ".." in jar.parts or jar.suffix != ".jar":
        raise ValueError("LogMap jar must be inside its pinned bundle")
    for key, cap in (("timeout_seconds", 14400), ("java_heap_gb", 32), ("java_threads", 8)):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= cap:
            raise ValueError(f"LogMap {key} must be an integer in [1, {cap}]")
    return dict(value)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _population(cell: Any) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    data = cell.resolved_config["data"]
    if data.get("execution_mode") != "global_alignment":
        raise ValueError(
            "LogMap comparator supports global alignment only; it has its own retrieval"
        )
    if cell.resolved_config["matching"]["entity_kinds"] != ["class"]:
        raise ValueError("This bounded LogMap comparison supports biomedical classes only")
    if not data.get("source_universe"):
        raise ValueError("LogMap requires the independently frozen Exact source universe")
    root = Path(data.get("root") or ".")
    sources = sorted(
        {
            line.strip()
            for line in (root / data["source_universe"]).read_text().splitlines()
            if line.strip()
        }
    )
    if not sources:
        raise ValueError("Frozen source universe is empty")
    groups = [(source, "class") for source in sources]
    ranked = sorted(
        groups,
        key=lambda item: (
            hashlib.sha256(f"{cell.seed}\x1f{item[0]}\x1f{item[1]}".encode()).digest(),
            item,
        ),
    )
    groups = sorted(ranked[: cell.source_cap] if cell.source_cap else ranked)
    sample = {
        "cap": cell.source_cap or len(groups),
        "seed": cell.seed,
        "selected_groups": len(groups),
        "eligible_source_iris": [source for source, _ in groups],
        "source_kind_groups": groups,
        "sha256": hashlib.sha256(
            "\n".join(f"{source}\t{kind}" for source, kind in groups).encode()
        ).hexdigest(),
    }
    return groups, sample


def _invoke(
    command: list[str], cwd: Path, output: Path, timeout: int, stop: Path | None
) -> tuple[int, float, int]:
    started, peak = time.monotonic(), 0
    with (output / "experiment.stdout.log").open("w") as stdout, (
        output / "experiment.stderr.log"
    ).open("w") as stderr:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout or (stop and stop.exists()):
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    if stop and stop.exists():
                        _json(
                            output / "interrupted.json",
                            {"status": "interrupted", "reason": "campaign STOP"},
                        )
                    return 124, time.monotonic() - started, peak
                try:
                    for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                        if line.startswith(("VmHWM:", "VmRSS:")):
                            peak = max(peak, int(line.split()[1]))
                except (OSError, ValueError):
                    pass
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return int(process.returncode), time.monotonic() - started, peak


def _evaluate(cell: Any, groups: list[tuple[str, str]]) -> None:
    from exact.core.actions.alignment import _materialize_evaluation_reference
    from exact.core.actions.evaluation import run_evaluation

    data = cell.resolved_config["data"]
    root, refs = Path(data.get("root") or "."), data.get("refs") or {}
    reporting = refs.get(cell.reference_role) or refs.get("full")
    if not reporting:
        raise ValueError("Comparator evaluation requires the authorized reporting reference")
    sources = {source for source, _ in groups}
    reference = root / reporting
    frame = read_table(reference).rename(columns={"Src": "SrcEntity", "Tgt": "TgtEntity"})
    frame = frame.loc[frame["SrcEntity"].astype(str).isin(sources)].copy()
    for name in ("SrcKind", "TgtKind"):
        if name in frame and not frame[name].astype(str).eq("class").all():
            raise ValueError("Comparator reference mixes unsupported entity kinds")
        frame[name] = "class"
    report = _materialize_evaluation_reference(
        frame,
        parent_path=reference,
        output_path=cell.output_dir / "evaluation_inputs/full_reference.tsv",
    )
    training = refs.get("train") or refs.get("training")
    train_path = None
    if training:
        train = read_table(root / training)
        column = "SrcEntity" if "SrcEntity" in train else "Src"
        train = train.loc[train[column].astype(str).isin(sources)]
        train_path = cell.output_dir / "dataset/sampled_inputs/training_reference.tsv"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(train_path, sep="\t", index=False)
    options = cell.resolved_config.get("evaluation") or {}
    run_evaluation(
        cell.output_dir / "alignment/maps_global.tsv",
        cell.output_dir / "evaluation",
        error_on_fail=True,
        full_reference_file_path=report,
        train_reference_file_path=train_path,
        backends=options.get("backends"),
        K=options.get("k"),
        backend_options={"bioml": options.get("bioml") or {}},
        run_stats_path=cell.output_dir / "stats/run_stats.json",
    )


def evaluate_cell(cell: Any) -> None:
    """Replay evaluation on the same frozen canonical population without running Java."""
    _evaluate(cell, _population(cell)[0])


def run_cell(cell: Any) -> tuple[int, float, int]:
    """Run no fitting/LLM code; preserve raw mappings, then use Exact's evaluator."""
    binding = validate_binding(cell.published_matcher)
    bundle = Path(binding["bundle"]["path"]).resolve()
    if sha256_path(bundle) != binding["bundle"]["sha256"]:
        raise ValueError("Pinned LogMap bundle changed")
    groups, sample = _population(cell)
    data = cell.resolved_config["data"]
    root, output = Path(data.get("root") or "."), cell.output_dir
    output.mkdir(parents=True, exist_ok=True)
    work = output / "published/logmap"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle / "parameters.txt", work / "parameters.txt")
    command = [
        "java",
        f"-Xmx{binding['java_heap_gb']}G",
        f"-XX:ActiveProcessorCount={binding['java_threads']}",
        "-DentityExpansionLimit=10000000",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "-jar",
        str(bundle / binding["jar"]),
        "MATCHER",
        (root / data["source"]).resolve().as_uri(),
        (root / data["target"]).resolve().as_uri(),
        str(work.resolve()) + os.sep,
        "false",
    ]
    provenance = {
        "binding": binding,
        "command": command,
        "mode": "global_alignment",
        "supervision": "target_label_free",
        "retrieval": "LogMap own opaque retrieval; candidate recall unavailable",
        "full_classification": False,
    }
    _json(work / "recipe.json", provenance)
    stop = Path(cell.recovery["root"]) / "STOP" if cell.recovery else None
    with TimingLedger.open(output).session(
        command="published_logmap", config_fingerprint=cell.config_hash
    ) as timing:
        code, elapsed, peak = _invoke(command, work, output, binding["timeout_seconds"], stop)
        timing.record("Inference", seconds=elapsed)
        if code:
            return code, elapsed, peak
        raw = work / "logmap2_mappings.rdf"
        predictions = read_alignment(raw)
        if not predictions["Score"].map(lambda value: math.isfinite(float(value))).all():
            raise ValueError("LogMap emitted non-finite confidence")
        selected = predictions.loc[
            predictions["SrcEntity"].isin(sample["eligible_source_iris"])
            & predictions["Relation"].eq("=")
        ].copy()
        selected["SrcKind"], selected["TgtKind"] = "class", "class"
        selected = selected.drop_duplicates(["SrcEntity", "TgtEntity"]).sort_values(
            ["SrcEntity", "TgtEntity"]
        )
        (output / "alignment").mkdir(exist_ok=True)
        selected.to_csv(output / "alignment/maps_global.tsv", sep="\t", index=False)
        manifest = {
            "origin": "published_matcher",
            "retrieval_config": {"source_sample": sample},
            "published_matcher": provenance,
            "candidate_recall_available": False,
            "raw_predictions_sha256": sha256_file(raw),
        }
        manifest["fingerprint"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()
        _json(output / "dataset/candidate_pool_manifest.json", manifest)
        _json(
            output / "stats/run_stats.json",
            {
                "published_matcher": {
                    **provenance,
                    "raw_rows": len(predictions),
                    "equivalence_rows_in_population": len(selected),
                },
                "source_sampling": sample,
                "metric_applicability": {"candidate_recall": False},
                "llm": {"attempts": 0},
            },
        )
        evaluation_start = time.monotonic()
        _evaluate(cell, groups)
        elapsed += time.monotonic() - evaluation_start
    return code, elapsed, peak
