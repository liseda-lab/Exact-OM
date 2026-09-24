"""Bounded, checksummed numerical reuse within one immutable experiment campaign."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import sqlite3
import time
import zlib
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from pathlib import Path

import torch

from exact.utils.fitted_artifacts import fingerprint


def _tree(value, tensor):
    if isinstance(value, torch.Tensor):
        return tensor(value)
    if isinstance(value, dict):
        return {key: _tree(item, tensor) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_tree(item, tensor) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported numerical cache value: {type(value).__name__}")


def _restore(value, device):
    if isinstance(value, dict) and set(value) == {"__exact_tensor__", "device"}:
        target = "cpu" if value["device"] == "cpu" else device
        return value["__exact_tensor__"].to(target)
    if isinstance(value, dict):
        return {key: _restore(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_restore(item, device) for item in value)
    return value


def _arguments(value):
    return _tree(
        value,
        lambda item: {
            "tensor_dtype": str(item.dtype),
            "shape": list(item.shape),
            "sha256": hashlib.sha256(
                item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        },
    )


def _scope(model, *, channels=False):
    """Retain all upstream identities; exclude only post-score treatment controls."""
    runtime_path = os.environ.get("EXACT_EXPERIMENT_RUNTIME")
    if not runtime_path or os.environ.get("EXACT_NUMERICAL_CACHE", "1") == "0":
        return None
    batch_scopes = getattr(model, "_numerical_channel_scopes", None)
    # Only the typed reversal diagnostic mutates channel settings inside forward.
    # Full runtime/model fingerprints are still rebuilt at every forward boundary.
    batch_key = fingerprint(
        {"tau": getattr(model, "tau", None), "difference": getattr(model, "diff_config", {})}
    )
    if channels and batch_scopes is not None and batch_key in batch_scopes:
        return batch_scopes[batch_key]
    cached = getattr(model, "_numerical_runtime_binding", None)
    if cached is None or cached[0] != runtime_path:
        record = json.loads(Path(runtime_path).read_text())
        model._numerical_runtime_binding = (runtime_path, record)
    else:
        record = cached[1]
    identity = record["identity"]
    parameters = deepcopy(identity["parameters"])
    parameters.pop("selector", None)
    parameters.pop("supervision", None)
    parameters.get("matching", {}).pop("extraction", None)
    state = deepcopy(model.runtime_fingerprint_payload())
    state["effective_scalars"] = {
        name: getattr(model, name, None)
        for name in ("tau", "gamma", "beta", "threshold", "use_lexical", "use_context", "use_llm")
    }
    if channels:
        # tau affects neutral channels and signed identifiers and MUST remain.
        for mapping in (
            state["effective_scalars"],
            parameters.get("matching", {}).get("fusion", {}),
            state.get("pair_adaptive_channels", {}).get("experiments", {}).get("fusion", {}),
            state.get("pair_adaptive_channels", {})
            .get("experiments", {})
            .get("fusion_effective", {}),
        ):
            mapping.pop("gamma", None)
            mapping.pop("beta", None)
    hardware = {
        "device": str(getattr(model, "device", "cpu")),
        "gpu": (
            torch.cuda.get_device_name(model.device)
            if str(getattr(model, "device", "cpu")).startswith("cuda")
            else None
        ),
        "cuda": torch.version.cuda,
        "fp16": bool(getattr(model, "fp16", False)),
        "autocast": torch.is_autocast_enabled("cuda"),
        "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    artifacts = {}
    for name in (
        "_graph_artifact",
        "_fusion_artifact",
        "_student_artifact",
        "_gate_artifact",
        "_exemplar_artifact",
    ):
        artifact = getattr(model, name, None)
        if artifact is not None:
            artifacts[name] = getattr(artifact, "provenance", artifact)
    scope = {
        "version": 2,
        "parameters": parameters,
        "state": state,
        "inputs": identity["inputs"],
        "implementation": identity["implementation"],
        "dependencies": identity["dependencies"],
        "role": identity["role"],
        "scoring_role": getattr(model, "_numerical_scoring_role", identity["role"]),
        "kind": identity["entity_kind"],
        "seed": identity["seed"],
        "hardware": hardware,
        "artifacts": artifacts,
        "dataset": getattr(getattr(model, "_attached_dataset", None), "cache_fingerprint", None),
    }
    result = Path(record["root"]), fingerprint(scope)
    if channels and batch_scopes is not None:
        batch_scopes[batch_key] = result
    return result


class NumericalCache:
    """One batched SQLite writer; a full or damaged cache falls back to computation."""

    def __init__(self, root):
        directory = Path(root) / "numerical-cache"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "evidence.sqlite"
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS entries (scope TEXT, operation TEXT, key TEXT, payload BLOB, sha256 TEXT, size INTEGER, PRIMARY KEY(scope, operation, key))"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY, entries INTEGER, bytes INTEGER)"
        )
        if self.connection.execute("SELECT 1 FROM usage WHERE id=1").fetchone() is None:
            self.connection.execute(
                "INSERT OR IGNORE INTO usage SELECT 1, count(*), coalesce(sum(size), 0) FROM entries"
            )
            self.connection.commit()
        self.entries, self.bytes = self.connection.execute(
            "SELECT entries, bytes FROM usage WHERE id=1"
        ).fetchone()
        self.limit = max(
            0, int(os.environ.get("EXACT_NUMERICAL_CACHE_MAX_BYTES", str(8 * 1024**3)))
        )
        self.hits = self.misses = self.skipped = 0
        self.depth = 0

    def get(self, scope, operation, key, device):
        row = self.connection.execute(
            "SELECT payload, sha256 FROM entries WHERE scope=? AND operation=? AND key=?",
            (scope, operation, key),
        ).fetchone()
        if row is None or hashlib.sha256(row[0]).hexdigest() != row[1]:
            self.misses += 1
            return None
        try:
            value = _restore(
                torch.load(
                    io.BytesIO(zlib.decompress(row[0])), map_location="cpu", weights_only=True
                ),
                device,
            )
        except (RuntimeError, ValueError, EOFError, pickle.UnpicklingError, zlib.error):
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, scope, operation, key, value):
        if self.bytes >= self.limit:
            self.skipped += 1
            return
        stream = io.BytesIO()
        torch.save(
            _tree(
                value,
                lambda item: {"__exact_tensor__": item.detach().cpu(), "device": item.device.type},
            ),
            stream,
        )
        # Repeated evidence text is compressible; never persist encoder weights or embeddings.
        payload = zlib.compress(stream.getvalue(), level=1)
        if self.bytes + len(payload) > self.limit:
            self.skipped += 1
            return
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO entries SELECT ?, ?, ?, ?, ?, ? "
            "WHERE (SELECT bytes FROM usage WHERE id=1) + ? <= ?",
            (
                scope,
                operation,
                key,
                payload,
                hashlib.sha256(payload).hexdigest(),
                len(payload),
                len(payload),
                self.limit,
            ),
        )
        if cursor.rowcount:
            self.connection.execute(
                "UPDATE usage SET entries=entries+1, bytes=bytes+? WHERE id=1", (len(payload),)
            )
            self.entries += 1
            self.bytes += len(payload)
        else:
            self.skipped += 1

    @contextmanager
    def batch(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if not self.depth:
                try:
                    self.connection.commit()
                except sqlite3.Error:
                    self.connection.rollback()
                    self.skipped += 1

    def stats(self):
        try:
            self.entries, self.bytes = self.connection.execute(
                "SELECT entries, bytes FROM usage WHERE id=1"
            ).fetchone()
            disk_bytes = sum(
                path.stat().st_size for path in self.path.parent.glob("evidence.sqlite*")
            )
        except (OSError, sqlite3.Error):
            disk_bytes = None
        return {
            "hits": self.hits,
            "misses": self.misses,
            "skipped_writes": self.skipped,
            "entries": self.entries,
            "payload_bytes": self.bytes,
            "limit_bytes": self.limit,
            "bytes_per_entry": self.bytes / self.entries if self.entries else 0,
            "disk_bytes": disk_bytes,
        }


def cached_numerical(*, channels=False):
    """Memoize pure channel payloads, or a decision-off scorer's complete output."""

    def decorate(function):
        @wraps(function)
        def call(model, *args, **kwargs):
            # Hosted responses retain their own request ledger and are never hidden
            # behind full-forward replay. Raw channels have no decision calls.
            gate = getattr(model, "llm_experiment_config", {}).get("gate", {}).get("mode")
            allowed = channels or not getattr(model, "use_llm", False) or gate == "off"
            if not allowed or torch.is_grad_enabled():
                return function(model, *args, **kwargs)
            try:
                scope = _scope(model, channels=channels)
                if scope is None:
                    return function(model, *args, **kwargs)
                root, identity = scope
                cache = getattr(model, "_numerical_cache", None)
                if cache is None or cache.path.parent.parent != root:
                    cache = model._numerical_cache = NumericalCache(root)
                key = fingerprint(_arguments([args, kwargs]))
            except (OSError, sqlite3.Error, TypeError):
                return function(model, *args, **kwargs)
            with cache.batch():
                try:
                    value = cache.get(
                        identity, function.__name__, key, getattr(model, "device", "cpu")
                    )
                except (OSError, sqlite3.Error, TypeError):
                    value = None
                replayed = value is not None
                if value is None:
                    # Errors from the actual scorer are not swallowed or retried.
                    if channels:
                        value = function(model, *args, **kwargs)
                    else:
                        model._numerical_channel_scopes = {}
                        try:
                            value = function(model, *args, **kwargs)
                        finally:
                            del model._numerical_channel_scopes
                    try:
                        cache.put(identity, function.__name__, key, value)
                    except (OSError, sqlite3.Error, TypeError):
                        cache.skipped += 1
            if not channels and isinstance(value, dict):
                value["numerical_cache"] = {
                    **cache.stats(),
                    "full_forward_replayed": replayed,
                    "new_forward_calls": int(not replayed),
                }
                if replayed and "backend_usage" in value:
                    value["backend_usage"] = {}
                now = time.monotonic()
                if now - getattr(model, "_numerical_stats_written_at", float("-inf")) >= 10:
                    try:
                        from exact.experiments.runtime import _write

                        _write(
                            Path(os.environ["EXACT_EXPERIMENT_RUNTIME"]).parent
                            / "diagnostics/numerical-cache.json",
                            cache.stats(),
                        )
                        model._numerical_stats_written_at = now
                    except OSError:
                        pass
            return value

        return call

    return decorate


def shared_training_scores(model, identity, application, batch_size, *, rows=None):
    """Reuse feature rows, never fitted heads; retain label/source/pool bindings."""
    old_role = getattr(model, "_numerical_scoring_role", None)
    model._numerical_scoring_role = "train"
    try:
        scope = _scope(model)
        if scope is None:
            return None
        root, model_identity = scope
        cache = getattr(model, "_numerical_cache", None)
        if cache is None or cache.path.parent.parent != root:
            cache = model._numerical_cache = NumericalCache(root)
        key = fingerprint(
            {"training_identity": identity, "application": application, "batch_size": batch_size}
        )
        with cache.batch():
            if rows is None:
                return cache.get(model_identity, "training_scores", key, "cpu")
            cache.put(model_identity, "training_scores", key, rows)
        return None
    except (OSError, sqlite3.Error, TypeError):
        return None
    finally:
        if old_role is None:
            del model._numerical_scoring_role
        else:
            model._numerical_scoring_role = old_role
