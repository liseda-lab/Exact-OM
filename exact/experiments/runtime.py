"""Small adapters connecting campaign recovery to the existing scorer and runner."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from exact.experiments.recovery import (
    ArtifactStore,
    build_reuse_plan,
    create_attempt,
    finish_attempt,
    stage_identity,
)
from exact.utils.provenance import sha256_file, sha256_path


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _files(root: Path, directories: Sequence[str]) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for name in directories
        for path in ([root / name] if (root / name).is_file() else (root / name).rglob("*"))
        if path.is_file() and not path.name.endswith(".tmp") and not path.name.startswith(".")
    }


def _code_identity(root: Path, *, evaluation: bool) -> dict[str, Any]:
    """Conservative dynamic implementation scope; evaluator changes spare predictions.

    All otherwise unknown executable modules invalidate prediction scope. Shared
    utilities belong to both scopes, so normalisation changes cannot escape it.
    """
    files = {}
    for path in sorted((root / "exact").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        evaluator = "/evaluators/" in relative or path.stem in {
            "evaluation",
            "evaluator",
            "paper_metrics",
            "reporting",
            "statistics",
        }
        shared = "/utils/" in relative or "/entities/" in relative
        if (evaluation and (evaluator or shared)) or (not evaluation and not evaluator):
            files[relative] = sha256_file(path)
    return {"sha256": _hash(files), "files": files}


class CellRecovery:
    """Reuse verified prediction outputs independently of evaluation and provenance."""

    def __init__(self, cell: Any, provenance: Mapping[str, Any], workdir: Path):
        self.cell = cell
        self.metadata = dict(cell.recovery or {})
        self.store = ArtifactStore(Path(self.metadata["root"]))
        self.index_relative = (
            Path("recovery/cells") / f"{_hash([cell.suite_id, cell.cell_id])}.json"
        )
        self.index = self.store.root / self.index_relative
        old_root = Path(self.metadata.get("resume_from") or self.store.root)
        old_index = old_root / self.index_relative
        self.previous = json.loads(old_index.read_text()) if old_index.is_file() else {}
        self.old_root = old_root
        self.input_files: dict[str, Path | bytes] = {}
        fingerprint = provenance["fingerprint_payload"]
        input_metadata = fingerprint["inputs"]
        input_hashes: dict[str, str] = {}
        reporting_hashes: dict[str, str] = {}
        replacements: dict[str, Any] = {}
        for name, value in input_metadata.items():
            entries = value.items() if name == "references" else [(name, value)]
            for role, item in entries:
                if not isinstance(item, Mapping) or not item.get("sha256"):
                    continue
                label = f"references/{role}" if name == "references" else str(role)
                replacements[str(item["path"])] = {"sha256": item["sha256"]}
                data = cell.resolved_config.get("data") or {}
                configured = (
                    (data.get("refs") or {}).get(role) if name == "references" else data.get(role)
                )
                if configured:
                    replacements[str(configured)] = {"sha256": item["sha256"]}
                if name == "references" and role not in {"train", "training"}:
                    reporting_hashes[label] = item["sha256"]
                else:
                    input_hashes[label] = item["sha256"]
                    input_path = Path(item["path"])
                    if input_path.is_dir():
                        for child in sorted(input_path.rglob("*")):
                            if child.is_file():
                                relative = child.relative_to(input_path).as_posix()
                                self.input_files[f"_locked_inputs/{label}/{relative}"] = child
                    else:
                        self.input_files[f"_locked_inputs/{label}"] = input_path

        for label, item in (fingerprint.get("artifacts") or {}).items():
            if isinstance(item, Mapping) and item.get("sha256"):
                replacements[str(item["path"])] = {"sha256": item["sha256"]}
                input_hashes[f"artifact/{label}"] = item["sha256"]

        def normalise(value: Any, *, key: str = "") -> Any:
            if isinstance(value, Mapping):
                return {
                    str(k): normalise(v, key=str(k))
                    for k, v in value.items()
                    if k
                    not in {
                        "output",
                        "logging",
                        "root",
                        "cache_dir",
                        "api_key_path",
                        "extra_headers",
                    }
                    and (key != "refs" or k in {"train", "training"})
                    and not (k == "seed" and cell.published_matcher and cell.source_cap is None)
                }
            if isinstance(value, list):
                return [normalise(item) for item in value]
            if str(value) in replacements:
                return replacements[str(value)]
            if isinstance(value, str) and ("/" in value or Path(value).suffix):
                path = Path(value).expanduser()
                if path.is_file() or path.is_dir():
                    return {"sha256": sha256_path(path)}
            return value

        parameters = normalise(cell.resolved_config)
        if cell.published_matcher:
            parameters["published_matcher"] = normalise(cell.published_matcher)
        evaluation_options = parameters.pop("evaluation", {})
        parameters.update(
            source_cap=cell.source_cap,
            supervision=cell.resolved_supervision,
            negative_label_policy=cell.negative_label_policy,
        )
        packages = {name.lower(): version for name, version in provenance["packages"].items()}
        common = dict(
            role=cell.split_role,
            entity_kind=str(cell.resolved_config.get("entity_kind", "all")),
            seed=None if cell.published_matcher and cell.source_cap is None else cell.seed,
        )
        self.identities = {}
        self.identities["inputs"] = stage_identity(
            "inputs",
            parameters={"task": cell.task_id},
            inputs=input_hashes,
            implementation={"schema": "locked-bytes-v2"},
            dependencies={},
            **common,
        )
        self.identities["extraction"] = stage_identity(
            "extraction",
            parameters=parameters,
            inputs=input_hashes,
            parents=[self.identities["inputs"]["artifact_id"]],
            implementation=_code_identity(workdir, evaluation=False),
            dependencies={
                name: packages.get(name)
                for name in (
                    "torch",
                    "transformers",
                    "tokenizers",
                    "numpy",
                    "scipy",
                    "pyowl-core",
                    "pyowl2vec-star-projector",
                    "sentence-transformers",
                    "pandas",
                )
            },
            **common,
        )
        self.identities["evaluation"] = stage_identity(
            "evaluation",
            parameters=evaluation_options,
            inputs=reporting_hashes,
            parents=[self.identities["extraction"]["artifact_id"]],
            implementation=_code_identity(workdir, evaluation=True),
            dependencies={
                name: packages.get(name) for name in ("numpy", "pandas", "oaei-bioml-eval")
            },
            **common,
        )
        repair = self.metadata.get("repair_record") or {}
        if isinstance(repair, (str, Path)):
            repair = json.loads(Path(repair).read_text())
        self.repair = repair
        if repair.get("reporting_labels_exposed") and not repair.get(
            "scientific_choices_unchanged", False
        ):
            raise ValueError(
                "Post-reporting method amendments require a new exploratory design or fresh final evidence"
            )
        previous_ids = self.previous.get("artifacts", {})
        # Imports are verified against the original manifests, never fabricated from current config.
        if old_root.resolve() != self.store.root and not self.metadata.get("reuse_plan_only"):
            for relative in ("openrouter/requests.sqlite3", "embeddings/vectors.sqlite3"):
                source_db, target_db = old_root / relative, self.store.root / relative
                if source_db.is_file() and not target_db.exists():
                    target_db.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target_db.with_suffix(f".{os.getpid()}.pending")
                    with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as source:
                        with sqlite3.connect(temporary) as target:
                            source.backup(target)
                    try:
                        os.link(temporary, target_db)
                    except FileExistsError:
                        pass
                    finally:
                        temporary.unlink(missing_ok=True)
            for artifact_id in previous_ids.values():
                try:
                    self.store.import_artifact(old_root, artifact_id)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            self.store.import_checkpoint(old_root, self.identities["extraction"]["artifact_id"])
        self.plan = build_reuse_plan(
            ArtifactStore(old_root) if self.metadata.get("reuse_plan_only") else self.store,
            self.identities,
            previous_ids,
            changed_stages=repair.get("affected_stages", ()),
        )
        self.reuse = {row["stage"] for row in self.plan["stages"] if row["action"] == "reuse"}
        if self.metadata.get("reuse_plan_only"):
            self.attempt = {}
            _write(self.store.root / "recovery/plans" / f"{_hash(cell.cell_id)}.json", self.plan)
            return
        self.attempt = create_attempt(
            self.store.root,
            design_id=cell.design_hash,
            provenance=dict(provenance),
            parent_attempt=self.previous.get("attempt_id"),
            repair_record=repair or None,
        )
        _write(cell.output_dir / "reuse-plan.json", self.plan)
        _write(
            self.store.root / "attempts" / self.attempt["attempt_id"] / "reuse-plan.json", self.plan
        )

    def prepare(self) -> None:
        """Restore compatible outputs/checkpoints, then materialize this attempt's locks."""
        self.store.publish(
            self.identities["inputs"], self.input_files or {"_locked_inputs/manifest.json": b"{}"}
        )
        if "extraction" in self.reuse:
            self.store.restore(self.identities["extraction"]["artifact_id"], self.cell.output_dir)
            if "evaluation" in self.reuse:
                self.store.restore(
                    self.identities["evaluation"]["artifact_id"], self.cell.output_dir
                )
        else:
            checkpoint = self.store.latest_checkpoint(self.identities["extraction"]["artifact_id"])
            archive = self.store.root / "attempts" / self.attempt["attempt_id"] / "superseded"
            for name in (
                "alignment",
                "explanations",
                "checkpoints",
                "stats",
                "dataset",
                "cache",
                "evaluation",
                "timings.json",
                "source_decisions.json",
                "published",
                "evaluation_inputs",
                "fitting",
                "diagnostics",
            ):
                current = self.cell.output_dir / name
                if current.exists():
                    archive.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(current), archive / name)
            if checkpoint is not None:
                self.store.restore_checkpoint(checkpoint, self.cell.output_dir)
        (self.cell.output_dir / "interrupted.json").unlink(missing_ok=True)
        _write(
            self.cell.output_dir / "recovery-runtime.json",
            {
                "root": str(self.store.root),
                "identity": self.identities["extraction"],
                "stop_after_checkpoint": bool(self.metadata.get("stop_after_checkpoint")),
            },
        )
        # Restored reports retain numerical contents; runtime path provenance is repackaged.
        old_output = self.previous.get("output_dir")
        if old_output and str(self.cell.output_dir) != old_output:
            for path in (self.cell.output_dir / "evaluation").glob("*.json"):
                value = path.read_text().replace(old_output, str(self.cell.output_dir))
                path.write_text(value)

    def environment(self) -> dict[str, str]:
        limits = self.metadata.get("budget_limits", {})
        reserve_final = self.metadata.get("stage") != "confirm"
        budget = {
            f"EXACT_OPENROUTER_{unit.upper()}_CAP": str(
                limits[f"{unit}s_cap"]
                - (limits.get(f"final_{unit}s_reserved", 0) if reserve_final else 0)
            )
            for unit in ("request", "token")
            if f"{unit}s_cap" in limits
        }
        return {
            **budget,
            **{
                name: os.environ.get(name, "2")
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "EXACT_EXPERIMENT_MODE": "1",
            "EXACT_EXPERIMENT_RUNTIME": str(self.cell.output_dir / "recovery-runtime.json"),
            "EXACT_OPENROUTER_LEDGER_DIR": str(self.store.root / "openrouter"),
            "EXACT_EMBEDDING_CACHE_DIR": str(self.store.root / "embeddings"),
            "EXACT_EXPERIMENT_ROLE": self.cell.split_role,
            "EXACT_EXPERIMENT_STOP_FILE": str(self.store.root / "STOP"),
        }

    def evaluate(self) -> None:
        """Execute the existing evaluator directly, without loading any scoring model."""
        if self.cell.published_matcher:
            from exact.experiments.published_matcher import evaluate_cell

            evaluate_cell(self.cell)
            return
        from exact.core.actions.evaluation import run_evaluation

        data = self.cell.resolved_config.get("data") or {}
        root = Path(data.get("root") or ".")
        refs = data.get("refs") or {}
        options = self.cell.resolved_config.get("evaluation") or {}
        mode = data.get("execution_mode", "global_alignment")
        alignment = (
            self.cell.output_dir
            / "alignment"
            / ("maps_local.tsv" if mode == "local_ranking" else "maps_global.tsv")
        )
        if not alignment.is_file():
            raise FileNotFoundError(f"Reusable predictions lack canonical alignment: {alignment}")
        train = refs.get("train") or refs.get("training")
        report = refs.get(self.cell.reference_role) or refs.get("full")
        if mode != "local_ranking" and not report:
            raise ValueError("Global evaluator replay requires the declared role's reference")
        candidates = root / data["candidates"] if data.get("candidates") else None
        if mode == "local_ranking" and report and candidates:
            from exact.core.actions.evaluation import materialize_local_ranking_inputs

            alignment, candidates = materialize_local_ranking_inputs(
                alignment, candidates, root / report, self.cell.output_dir / "evaluation/inputs"
            )
        run_evaluation(
            alignment,
            self.cell.output_dir / "evaluation",
            error_on_fail=True,
            K=options.get("k"),
            train_reference_file_path=root / train if train else None,
            full_reference_file_path=root / report if report and mode != "local_ranking" else None,
            reference_candidates=candidates if mode == "local_ranking" else None,
            backends=options.get("backends"),
            backend_options={"bioml": options.get("bioml") or {}},
            run_stats_path=self.cell.output_dir / "stats" / "run_stats.json",
        )

    def finish(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """Publish successful standard outputs and preserve immutable attempt lineage."""
        status = str(manifest["status"])
        artifacts = {}
        if status == "complete":
            for stage, directories in (
                (
                    "extraction",
                    (
                        "alignment",
                        "dataset",
                        "explanations",
                        "stats",
                        "timings.json",
                        "published",
                        "evaluation_inputs",
                        "fitting",
                        "diagnostics",
                        "source_decisions.json",
                    ),
                ),
                ("evaluation", ("evaluation",)),
            ):
                if stage not in self.reuse:
                    self.store.publish(
                        self.identities[stage], _files(self.cell.output_dir, directories)
                    )
                artifacts[stage] = self.identities[stage]["artifact_id"]
            artifacts["inputs"] = self.identities["inputs"]["artifact_id"]
            _write(
                self.index,
                {
                    "artifacts": artifacts,
                    "attempt_id": self.attempt["attempt_id"],
                    "output_dir": str(self.cell.output_dir),
                },
            )
        finish_attempt(
            self.store.root,
            self.attempt["attempt_id"],
            status=status,
            continuation=self.metadata.get("continuation_command")
            or f"{__import__('sys').executable} tools/run_exact_job.py --run-config {self.cell.output_dir / '_inputs/job.yaml'}",
            summary={
                "cell": self.cell.cell_id,
                "artifacts": artifacts,
                "reuse_plan": self.plan["plan_id"],
            },
        )
        return {
            "attempt_id": self.attempt["attempt_id"],
            "artifacts": artifacts,
            "plan": self.plan,
            "reused_stages": sorted(self.reuse),
        }


def runtime_checkpoint(runner: Any, checkpoint_path: Path, processed: int) -> None:
    """Commit real inference explanations and exact ordered sample IDs at a boundary."""
    path = os.getenv("EXACT_EXPERIMENT_RUNTIME")
    if not path:
        return
    runtime = json.loads(Path(path).read_text())
    root = Path(path).parent
    frame = runner.dataset._active_dataframe()
    columns = [name for name in ("Src", "SrcKind", "Tgt", "TgtKind") if name in frame]
    ids = [
        json.dumps(list(row), default=str)
        for row in frame.iloc[:processed][columns].itertuples(index=False, name=None)
    ]
    if len(ids) != processed or len(set(ids)) != len(ids):
        raise ValueError("Inference checkpoint cannot establish unique completed sample IDs")
    outputs = _files(
        root,
        (
            "checkpoints",
            "explanations",
            "dataset",
            "fitting",
            "timings.json",
            "source_decisions.json",
        ),
    )
    outputs[checkpoint_path.relative_to(root).as_posix()] = checkpoint_path
    ArtifactStore(Path(runtime["root"])).checkpoint(
        runtime["identity"],
        completed_ids=ids,
        cursor={"next_pair": processed, "dataset_rows": len(frame)},
        outputs=outputs,
    )
    if (
        runtime.get("stop_after_checkpoint")
        or Path(os.getenv("EXACT_EXPERIMENT_STOP_FILE", str(root / "STOP"))).exists()
    ):
        _write(root / "interrupted.json", {"status": "interrupted", "completed_pairs": processed})
        raise KeyboardInterrupt("Experiment stopped after committed checkpoint")


def cached_encoder_rows(
    scorer: Any,
    tokenizer: Any,
    model: Any,
    texts: list[str],
    max_len: int,
    compute: Callable[[list[str]], Any],
) -> Any:
    """Reuse vectors by exact text, pinned encoder/tokenizer, precision, and role.

    Opaque/unpinned model weights fail closed to ordinary encoding. The shared
    cache is independent of candidate selection, fusion, and reporting labels.
    """
    directory = os.getenv("EXACT_EMBEDDING_CACHE_DIR")
    if not directory:
        return compute(texts)
    import torch

    config = getattr(model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    if (
        config is None
        or not revision
        or len(str(revision)) != 40
        or getattr(model, "training", False)
    ):
        return compute(texts)
    cached = getattr(scorer, "_stage_encoder_keys", {})
    local_key = (id(model), id(tokenizer), max_len)
    identity = cached.get(local_key)
    if identity is None:
        if not hasattr(tokenizer, "get_vocab"):
            return compute(texts)
        identity = _hash(
            {
                "schema": 1,
                "revision": revision,
                "model_config": {
                    key: value
                    for key, value in config.to_dict().items()
                    if key not in {"_name_or_path", "name_or_path"}
                },
                "tokenizer_vocab": tokenizer.get_vocab(),
                "tokenizer_backend": (
                    tokenizer.backend_tokenizer.to_str()
                    if hasattr(tokenizer, "backend_tokenizer")
                    else None
                ),
                "encode_code": inspect.getsource(scorer._encode_texts),
                "pool_code": inspect.getsource(scorer._pool),
                "tokenizer_config": {
                    key: value
                    for key, value in getattr(tokenizer, "init_kwargs", {}).items()
                    if key not in {"name_or_path", "cache_dir"}
                },
                "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
                "padding_side": getattr(tokenizer, "padding_side", None),
                "truncation_side": getattr(tokenizer, "truncation_side", None),
                "max_len": max_len,
                "pooling": scorer.pooling_method.value,
                "fp16": scorer.fp16,
                "stored_dtype": str(scorer._cache_tensor_dtype),
                "device": scorer.device_type,
                "hardware": (
                    torch.cuda.get_device_name(scorer.device)
                    if scorer.device_type == "cuda"
                    else "cpu"
                ),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "tokenizers": __import__("tokenizers").__version__,
                "role": os.getenv("EXACT_EXPERIMENT_ROLE", "unspecified"),
                "implementation": sha256_file(
                    Path(__file__).parents[1] / "impl/models/scorer_common.py"
                ),
            }
        )
        cached[local_key] = identity
        scorer._stage_encoder_keys = cached
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path / "vectors.sqlite3", timeout=30)
    keys = [_hash([identity, text]) for text in texts]
    try:
        db.execute("PRAGMA synchronous=FULL")
        db.execute(
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, shape TEXT, raw BLOB, sha256 TEXT)"
        )
        rows = [
            db.execute("SELECT shape,raw,sha256 FROM vectors WHERE key=?", (key,)).fetchone()
            for key in keys
        ]
        # The caller computes the requested missing text batch. Avoid batch-dependent cache IDs.
        if any(row is None for row in rows):
            missing = [index for index, row in enumerate(rows) if row is None]
            tensors = (
                compute([texts[index] for index in missing])
                .detach()
                .to("cpu")
                .to(scorer._cache_tensor_dtype)
            )
            for index, tensor in zip(missing, tensors):
                raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
                db.execute(
                    "INSERT OR IGNORE INTO vectors VALUES (?,?,?,?)",
                    (
                        keys[index],
                        json.dumps(list(tensor.shape)),
                        raw,
                        hashlib.sha256(raw).hexdigest(),
                    ),
                )
            db.commit()
            rows = [
                db.execute("SELECT shape,raw,sha256 FROM vectors WHERE key=?", (key,)).fetchone()
                for key in keys
            ]
        tensors = []
        for shape, raw, digest in rows:
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Corrupted durable embedding vector")
            tensors.append(
                torch.frombuffer(bytearray(raw), dtype=scorer._cache_tensor_dtype).reshape(
                    json.loads(shape)
                )
            )
        return torch.stack(tensors).to(scorer.device)
    finally:
        db.close()


def validate_campaign_results(
    manifests: Sequence[Mapping[str, Any]],
    expected_cells: Sequence[Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Select current approved attempts and reject stale paired execution revisions."""
    expected = {cell.cell_id: cell for cell in expected_cells}
    if len(expected) != len(expected_cells):
        raise ValueError("Duplicate cells in frozen execution plan")
    store = ArtifactStore(root)
    rows = []
    seen: set[str] = set()
    paired: dict[tuple[str, str, int], tuple[str, str | None]] = {}
    for manifest in manifests:
        cell_id = "/".join(
            (str(manifest[key]) for key in ("experiment_id", "stage", "arm_id", "task_id"))
        )
        cell_id += f"/seed-{manifest['seed']}"
        if cell_id in seen or cell_id not in expected:
            raise ValueError(f"Duplicate or unplanned current result cell: {cell_id}")
        seen.add(cell_id)
        cell = expected[cell_id]
        if manifest.get("status") != "complete":
            raise ValueError(f"Current result cell is incomplete: {cell_id}")
        if (
            manifest.get("design_declaration_hash") != cell.design_hash
            or manifest.get("selection_record_hash") != cell.selection_hash
            or manifest.get("resolved_config_hash") != cell.config_hash
        ):
            raise ValueError(f"Changed design or stale selection record for {cell_id}")
        recovery = manifest.get("recovery") or {}
        index_path = store.root / "recovery/cells" / f"{_hash([cell.suite_id, cell.cell_id])}.json"
        index = json.loads(index_path.read_text())
        if index.get("attempt_id") != recovery.get("attempt_id") or index.get(
            "artifacts"
        ) != recovery.get("artifacts"):
            raise ValueError(f"Superseded execution attempt included for {cell_id}")
        artifacts = recovery["artifacts"]
        prediction_payload = store.verify(artifacts["extraction"])
        evaluation_payload = store.verify(artifacts["evaluation"])
        _verify_materialized(store, prediction_payload, cell.output_dir)
        _verify_materialized(store, evaluation_payload, cell.output_dir)
        prediction = prediction_payload["identity"]
        evaluation = evaluation_payload["identity"]
        semantic_id = _hash(
            {
                "prediction": prediction["implementation"],
                "evaluation": evaluation["implementation"],
                "prediction_dependencies": prediction["dependencies"],
                "evaluation_dependencies": evaluation["dependencies"],
            }
        )
        universe = prediction["inputs"].get("source_universe")
        group = (cell.experiment_id, cell.task_id, cell.seed)
        previous = paired.setdefault(group, (semantic_id, universe))
        if previous != (semantic_id, universe):
            raise ValueError(
                f"Paired arms mix incompatible execution revisions or source universes: {group}"
            )
        rows.append(
            {
                "cell_id": cell_id,
                "attempt_id": recovery["attempt_id"],
                "artifacts": artifacts,
                "semantics_id": semantic_id,
                "source_universe_id": universe,
                "design_id": cell.design_hash,
                "selection_id": cell.selection_hash,
            }
        )
    if seen != set(expected):
        raise ValueError(
            f"Current result set lacks planned cells or controls: {sorted(set(expected) - seen)}"
        )
    rows.sort(key=lambda row: row["cell_id"])
    return {
        "schema_version": 2,
        "selection_rule": "current approved repair lineage",
        "result_set_id": _hash(rows),
        "cells": rows,
    }


def _portable_metadata(value: Any) -> Any:
    """Ignore relocated provenance paths, retaining all hashes and scientific values."""
    if isinstance(value, dict):
        return {
            key: _portable_metadata(item)
            for key, item in value.items()
            if key != "evaluation_inputs" and not (key == "path" and "sha256" in value)
        }
    if isinstance(value, list):
        return [_portable_metadata(item) for item in value]
    return value


def _verify_materialized(
    store: ArtifactStore, payload: Mapping[str, Any], output_dir: Path
) -> None:
    for name, output in payload["outputs"].items():
        actual = output_dir / name
        if not actual.is_file():
            raise ValueError(f"Current result is missing a consumed output: {name}")
        if sha256_file(actual) == output["sha256"]:
            continue
        if actual.suffix == ".json":
            original = store.root / output["path"]
            if _portable_metadata(json.loads(actual.read_text())) == _portable_metadata(
                json.loads(original.read_text())
            ):
                continue
        raise ValueError(f"Current result has a changed consumed output: {name}")
