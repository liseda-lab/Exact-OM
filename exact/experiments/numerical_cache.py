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


def _encoder_bindings(model, *, label_only=False):
    from exact.experiments.runtime import encoder_identity

    bindings = {}
    for name, enabled, length in (
        ('lexical', bool(getattr(model, 'use_lexical', False)), 'max_input_tokens_lexical'),
        ('context', bool(getattr(model, 'use_context', False)) and (
            not label_only or (getattr(model, 'lex_enabled', False)
                               and getattr(model, 'lex_config', {}).get('quality') == 'encoder_agreement')),
         'max_input_tokens_context'),
    ):
        if not enabled:
            continue
        prefix = 'lex' if name == 'lexical' else 'ctx'
        encoder = getattr(model, prefix + '_model', None)
        tokenizer = getattr(model, prefix + '_tok', None)
        if encoder is None or tokenizer is None:
            return None
        identity = encoder_identity(model, tokenizer, encoder, getattr(model, length))
        if identity is None:
            return None
        bindings[name] = identity[0]
    return bindings


def _scope(model, *, channels=False, operation=None):
    """Retain all upstream identities; exclude only post-score treatment controls."""
    runtime_path = os.environ.get("EXACT_EXPERIMENT_RUNTIME")
    if not runtime_path or os.environ.get("EXACT_NUMERICAL_CACHE", "1") == "0":
        return None
    batch_scopes = getattr(model, "_numerical_channel_scopes", None)
    operation = operation if operation in {'_score_string_channel', '_score_label_channel'} else None
    # Cheap mutable settings invalidate the otherwise batch-scoped identity.
    settings = {name: value for name, value in vars(model).items()
         if name.endswith(("_config", "_enabled"))
         or name.startswith(("max_", "top_"))
         or name in {"tau", "use_lexical", "use_context", "fp16", "request_seed",
                     "label_pair_pooling", "pooling_method", "_numerical_scoring_role",
                     "hierarchical_relation_families", "ctx_sentence_delimiter"}}
    if channels:
        settings.pop("fusion_config", None)
    batch_key = fingerprint([operation, settings])
    if channels and batch_scopes is not None and batch_key in batch_scopes:
        return batch_scopes[batch_key]
    cached = getattr(model, "_numerical_runtime_binding", None)
    if cached is None or cached[0] != runtime_path:
        record = json.loads(Path(runtime_path).read_text())
        model._numerical_runtime_binding = (runtime_path, record)
    else:
        record = cached[1]
    identity = record["identity"]
    if channels and operation in {'_score_string_channel', '_score_label_channel'}:
        encoders = {} if operation == '_score_string_channel' else _encoder_bindings(model, label_only=True)
        if encoders is not None:
            pure = {
                'schema': 'exact-pure-channel-v1', 'operation': operation,
                'tau': model.tau, 'encoders': encoders,
                'options': (model.strsim_config if operation == '_score_string_channel' else {
                    'use_lexical': model.use_lexical, 'lex_enabled': model.lex_enabled,
                    'lex_config': model.lex_config, 'label_pair_pooling': model.label_pair_pooling.value}),
                'implementation': identity['implementation'], 'dependencies': identity['dependencies'],
                'device': str(model.device), 'torch': torch.__version__,
                'namespace': os.getenv('EXACT_PRIMITIVE_NAMESPACE', record.get('primitive_namespace', 'scientific')),
            }
            result = Path(os.environ.get('EXACT_NUMERICAL_CACHE_ROOT', record['root'])), fingerprint(pure)
            model._numerical_scope_rebuilds = getattr(model, '_numerical_scope_rebuilds', 0) + 1
            if batch_scopes is not None:
                batch_scopes[batch_key] = result
            return result
    parameters = deepcopy(identity["parameters"])
    parameters.pop("selector", None)
    parameters.pop("supervision", None)
    parameters.get("matching", {}).pop("extraction", None)
    parameters.get("matching", {}).pop("calibration", None)
    # Selectors consume scorer output; their parameters cannot change raw scores.
    parameters["pipeline"] = [
        component
        for component in parameters.get("pipeline", [])
        if component.get("name") != "CandidateSetSelector"
    ]
    state = deepcopy(model.runtime_fingerprint_payload())
    state["effective_channel_settings"] = settings
    state["exact_anchors"] = {
        name: {str(k): sorted(map(str, values)) for k, values in getattr(model, name, {}).items()}
        for name in ("_exact_anchor_src_to_tgt", "_exact_anchor_tgt_to_src")
    }
    encoders = _encoder_bindings(model)
    if encoders is None:
        return None
    model._numerical_scope_rebuilds = getattr(model, "_numerical_scope_rebuilds", 0) + 1
    state["effective_scalars"] = {
        name: getattr(model, name, None)
        for name in ("tau", "gamma", "beta", "threshold", "use_lexical", "use_context", "use_llm")
    }
    if channels:
        # A fusion fit consumes these channels; it does not create their evidence.
        # tau still affects neutral channels and signed identifiers and MUST remain.
        for name in ("gamma", "beta", "threshold"):
            state["effective_scalars"].pop(name, None)
        fusion = parameters.get("matching", {}).get("fusion", {})
        parameters.get("matching", {})["fusion"] = {"tau": fusion.get("tau")}
        experiments = state.get("pair_adaptive_channels", {}).get("experiments", {})
        for name in ("fusion", "fusion_effective", "fusion_artifact"):
            experiments.pop(name, None)
        for component in parameters.get("pipeline", []):
            if component.get("name") == "PairAdaptiveSemanticScorer":
                component.get("params", {}).pop("gamma", None)
                component.get("params", {}).pop("beta", None)
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
        "pair_context_batching": os.getenv("EXACT_PAIR_CONTEXT_BATCHING", "0"),
        "pair_context_block_pairs": os.getenv("EXACT_PAIR_CONTEXT_BLOCK_PAIRS", "32"),
        "pair_context_text_batch": os.getenv("EXACT_PAIR_CONTEXT_TEXT_BATCH", "64"),
        "pair_context_matrix_elements": os.getenv("EXACT_PAIR_CONTEXT_MATRIX_ELEMENTS", "1048576"),
    }
    artifacts = {}
    for name in (
        "_graph_artifact",
        "_fusion_artifact",
        "_student_artifact",
        "_gate_artifact",
        "_exemplar_artifact",
    ):
        if channels and name == "_fusion_artifact":
            continue
        artifact = getattr(model, name, None)
        if artifact is not None:
            artifacts[name] = getattr(artifact, "provenance", artifact)
    scope = {
        "version": 3,
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
        "encoders": encoders,
        "dataset": getattr(getattr(model, "_attached_dataset", None), "cache_fingerprint", None),
    }
    result = Path(os.environ.get("EXACT_NUMERICAL_CACHE_ROOT", record["root"])), fingerprint(scope)
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
        self.disk_limit = max(0, int(os.getenv("EXACT_NUMERICAL_CACHE_MAX_DISK_BYTES", str(self.limit))))
        self.hits = self.misses = self.skipped = 0
        self.depth = 0
        self.pid = os.getpid()
        self.pending = {}
        self.pending_bytes = 0
        self.transactions = 0
        self.bytes_read = self.bytes_written = 0

    def get(self, scope, operation, key, device):
        row = self.pending.get((scope, operation, key))
        if row is None:
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
        self.bytes_read += len(row[0])
        return value

    def put(self, scope, operation, key, value):
        if self.bytes >= self.limit or self._disk_bytes() >= self.disk_limit:
            self.skipped += 1
            return
        stream = io.BytesIO()
        torch.save(
            _tree(
                value,
                lambda item: {"__exact_tensor__": item.detach().cpu().contiguous().clone(), "device": item.device.type},
            ),
            stream,
        )
        # Repeated evidence text is compressible; never persist encoder weights or embeddings.
        payload = zlib.compress(stream.getvalue(), level=1)
        if self.bytes + self.pending_bytes + len(payload) > self.limit:
            self.skipped += 1
            return
        address = (scope, operation, key)
        if address not in self.pending:
            self.pending[address] = (payload, hashlib.sha256(payload).hexdigest())
            self.pending_bytes += len(payload)
        # Bound RAM, never retain a SQLite writer while hosted inference waits.
        if self.pending_bytes >= 8 * 1024**2 or len(self.pending) >= 256 or not self.depth:
            self.flush()

    def _disk_bytes(self):
        return sum(path.stat().st_size for path in self.path.parent.glob(self.path.name + "*"))

    def _insert(self, scope, operation, key, payload, digest):
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO entries SELECT ?, ?, ?, ?, ?, ? "
            "WHERE (SELECT bytes FROM usage WHERE id=1) + ? <= ?",
            (
                scope,
                operation,
                key,
                payload,
                digest,
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
            self.bytes_written += len(payload)
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
                self.flush()

    def flush(self):
        if not self.pending:
            return
        pending, self.pending = self.pending, {}
        self.pending_bytes = 0
        before = self.entries, self.bytes, self.bytes_written
        try:
            # Serialize quota admission with other writers. The reservation covers
            # database pages, their WAL copies, indexes and per-row page overhead.
            self.connection.execute("BEGIN IMMEDIATE")
            reserved = self._disk_bytes()
            for address, (payload, digest) in pending.items():
                growth = 2 * (len(payload) + sum(len(part.encode()) for part in address) + 32768)
                if reserved + growth > self.disk_limit:
                    self.skipped += 1
                    continue
                self._insert(*address, payload, digest)
                reserved += growth
            self.connection.commit()
            self.transactions += 1
        except (OSError, sqlite3.Error):
            self.entries, self.bytes, self.bytes_written = before
            try:
                self.connection.rollback()
            except sqlite3.Error:
                pass
            self.skipped += len(pending)

    def close(self):
        self.flush()
        self.connection.close()

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
            "physical_limit_bytes": self.disk_limit,
            "bytes_per_entry": self.bytes / self.entries if self.entries else 0,
            "disk_bytes": disk_bytes,
            "transactions": self.transactions,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
        }


def cached_numerical(*, channels=False):
    """Memoize pure channel payloads, or a decision-off scorer's complete output."""

    def decorate(function):
        def invoke(model, *args, **kwargs):
            if (not channels and function.__name__ == "forward"
                    and function.__module__ == "exact.impl.models.pair_adaptive_scorer"):
                from exact.impl.models.pair_adaptive_batch import validate_batching_admission

                # Admission also precedes cache hits and the cache-disabled path.
                validate_batching_admission()
            # Hosted responses retain their own request ledger and are never hidden
            # behind full-forward replay. Raw channels have no decision calls.
            gate = getattr(model, "llm_experiment_config", {}).get("gate", {}).get("mode")
            allowed = channels or not getattr(model, "use_llm", False) or gate == "off"
            if torch.is_grad_enabled():
                return function(model, *args, **kwargs)
            try:
                scope = _scope(model, channels=channels or not allowed, operation=function.__name__ if channels else None)
                if scope is not None:
                    root, identity = scope
                    cache = getattr(model, "_numerical_cache", None)
                    if cache is None or cache.path.parent.parent != root or cache.pid != os.getpid():
                        cache = model._numerical_cache = NumericalCache(root)
                    key = fingerprint(_arguments([args, kwargs])) if allowed else None
            except (OSError, sqlite3.Error, TypeError, ValueError, KeyError):
                scope = None
            if scope is None:
                return function(model, *args, **kwargs)
            with cache.batch():
                try:
                    value = cache.get(
                        identity, function.__name__, key, getattr(model, "device", "cpu")
                    ) if allowed else None
                except (OSError, sqlite3.Error, TypeError):
                    value = None
                replayed = value is not None
                if value is None:
                    # Errors from the actual scorer are not swallowed or retried.
                    value = function(model, *args, **kwargs)
                    try:
                        if allowed:
                            cache.put(identity, function.__name__, key, value)
                    except (OSError, sqlite3.Error, TypeError, RuntimeError):
                        cache.skipped += 1
            if not channels and isinstance(value, dict):
                value["numerical_cache"] = {
                    **cache.stats(),
                    "full_forward_replayed": replayed,
                    "new_forward_calls": int(not replayed),
                    "scope_rebuilds": getattr(model, "_numerical_scope_rebuilds", 0),
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

        @wraps(function)
        def call(model, *args, **kwargs):
            if channels or torch.is_grad_enabled():
                return invoke(model, *args, **kwargs)
            marker = object()
            previous = getattr(model, "_numerical_channel_scopes", marker)
            model._numerical_channel_scopes = {}
            try:
                return invoke(model, *args, **kwargs)
            finally:
                if previous is marker:
                    del model._numerical_channel_scopes
                else:
                    model._numerical_channel_scopes = previous

        return call

    return decorate


def training_content_contract(model):
    """Bind known train-only primitives independently of the reporting population.

    This narrow contract is intentionally unavailable for unfamiliar scorers or
    datasets. Actual train rows, labels and batching are bound by the caller.
    """
    dataset = getattr(model, "_attached_dataset", None)
    runtime_path = os.getenv("EXACT_EXPERIMENT_RUNTIME")
    if (not runtime_path or os.getenv("EXACT_NUMERICAL_CACHE", "1") == "0"
            or type(model).__module__ != "exact.impl.models.pair_adaptive_scorer"
            or type(dataset).__module__ != "exact.impl.datasets.pair_adaptive_context"):
        return None
    try:
        record = json.loads(Path(runtime_path).read_text())
        identity = record["identity"]
        if not all(side in identity["inputs"] for side in ("source", "target")):
            return None
        templates = getattr(dataset, "_verbalization_templates", None)
        if getattr(dataset, "verbalization_mode", None) == "deterministic":
            templates = {}
        if templates is None:
            return None  # Never generate templates while discovering a cache.
        data = deepcopy(dataset._cache_fingerprint_payload())
        for name in ("filter_exact_matches", "drop_exact_match_sources", "filter_ignored_alignment_classes",
                     "cardinality", "candidate_generation_version",
                     "exact_prefilter_materialization_version", "ignored_alignment_filter_version",
                     "candidate_generation_params", "retrieval_artifacts", "candidate_data_lock",
                     "candidate_spec_lock", "candidate_model_lock", "request_seed"):
            data.pop(name, None)
        data["template_contents"] = deepcopy(templates)
        encoders = _encoder_bindings(model)
        if encoders is None:
            return None
        settings = {name: deepcopy(value) for name, value in vars(model).items()
                    if name.endswith(("_config", "_enabled")) or name.startswith(("max_", "top_"))
                    or name in {"tau", "gamma", "beta", "threshold", "use_lexical", "use_context", "fp16",
                                "request_seed", "label_pair_pooling", "pooling_method",
                                "hierarchical_relation_families", "ctx_sentence_delimiter"}}
        original_llm = model.use_llm
        model.use_llm = False
        try:
            state = deepcopy(model.runtime_fingerprint_payload())
        finally:
            model.use_llm = original_llm
        deterministic = (getattr(model, "graph_config", {}).get("mode", "off") == "off"
                         and not getattr(model, "graph_config", {}).get("hierarchy_removal", False)
                         and not getattr(model, "property_config", {}).get("relations_shuffled", False)
                         and not getattr(model, "instance_config", {}).get("relations_shuffled", False))
        if deterministic:
            state.pop("request_seed", None)
            settings.pop("request_seed", None)
        artifacts = {}
        for name in ("_graph_artifact", "_fusion_artifact", "_gate_artifact", "_student_artifact", "_exemplar_artifact"):
            artifact = getattr(model, name, None)
            if artifact is None:
                continue
            if isinstance(artifact, dict):
                artifacts[name] = deepcopy(artifact)
            elif isinstance(getattr(artifact, "payload", None), dict) and isinstance(getattr(artifact, "provenance", None), dict):
                artifacts[name] = {"payload": deepcopy(artifact.payload), "provenance": deepcopy(artifact.provenance)}
            else:
                return None
        contract = {
            "schema": "exact-training-content-v1", "role": "train", "kind": identity["entity_kind"],
            "dataset": data, "model": state, "effective_settings": settings, "encoders": encoders,
            "inputs": {side: identity["inputs"][side] for side in ("source", "target")},
            "implementation": identity["implementation"], "dependencies": identity["dependencies"],
            "artifacts": artifacts,
            "exact_anchors": {name: {str(key): sorted(map(str, values)) for key, values in getattr(model, name, {}).items()}
                              for name in ("_exact_anchor_src_to_tgt", "_exact_anchor_tgt_to_src")},
            "hardware": {"device": str(model.device), "fp16": bool(model.fp16), "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(model.device) if str(model.device).startswith("cuda") else None,
                         "autocast": torch.is_autocast_enabled("cuda"),
                         "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                         "tf32": torch.backends.cuda.matmul.allow_tf32,
                         **{name: value for name, value in os.environ.items() if name.startswith("EXACT_PAIR_CONTEXT_")}},
            "namespace": os.getenv("EXACT_PRIMITIVE_NAMESPACE", record.get("primitive_namespace", "scientific")),
        }
        fingerprint(contract)  # Unknown opaque dependencies fail closed.
        return contract
    except (OSError, TypeError, ValueError, KeyError, AttributeError):
        return None


def publish_training_reference(path, reference):
    """Allow relocation only when every scientific binding and checksum is identical."""
    from exact.experiments.runtime import _write
    from exact.utils.fitted_artifacts import freeze_json
    from exact.utils.provenance import sha256_file

    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text())
        if ({key: value for key, value in previous.items() if key != "aggregate_uri"}
                != {key: value for key, value in reference.items() if key != "aggregate_uri"}):
            raise ValueError("Training aggregate reference identity changed")
        if sha256_file(Path(reference["aggregate_uri"])) != reference["sha256"]:
            raise ValueError("Relocated training aggregate checksum mismatch")
        if previous != reference:
            _write(path, reference)
        return reference
    return freeze_json(path, reference)


def training_score_reference(model, identity, application, batch_size, *, aggregate_path=None):
    """Discover a verified durable aggregate without duplicating its row payload."""
    from exact.experiments.runtime import _write
    from exact.utils.provenance import sha256_file

    marker = object()
    old_role = getattr(model, "_numerical_scoring_role", marker)
    old_llm = getattr(model, "use_llm", marker)
    model._numerical_scoring_role = "train"
    model.use_llm = False
    try:
        primitive = training_content_contract(model)
        if primitive is None:
            scope = _scope(model)
        else:
            record = json.loads(Path(os.environ["EXACT_EXPERIMENT_RUNTIME"]).read_text())
            scope = (Path(os.getenv("EXACT_NUMERICAL_CACHE_ROOT", record["root"])), fingerprint(primitive))
        if scope is None:
            return None
        root, model_identity = scope
        contract = {
            "training_identity": identity, "model_identity": model_identity,
            "application": {k: v for k, v in application.items() if k != "source_ids"},
            "batch_size": int(batch_size), "scoring_role": "train",
        }
        key = fingerprint(contract)
        index = root / "numerical-cache" / "training-aggregates" / (key + ".json")
        if aggregate_path is not None:
            path = Path(aggregate_path).resolve()
            payload = json.loads(path.read_text())
            if payload.get("identity") != identity or not isinstance(payload.get("rows"), list):
                return None
            rows = payload["rows"]
            if {str(row["Src"]) for row in rows} & set(map(str, application.get("source_ids", []))):
                raise ValueError("Training aggregate overlaps reporting source groups")
            reference = {
                "schema": "exact-training-aggregate-reference-v1",
                "semantic_key": key, "aggregate_uri": str(path),
                "sha256": sha256_file(path), "row_count": len(rows),
                "source_count": len({str(row["Src"]) for row in rows}),
                "upstream": contract, "retention": "retain_until_all_consumers_complete",
            }
            if index.exists():
                previous = json.loads(index.read_text())
                if (previous.get("semantic_key") != key or previous.get("upstream") != contract
                        or previous.get("sha256") != reference["sha256"]):
                    return None
            publish_training_reference(index, reference)
        else:
            reference = json.loads(index.read_text())
            if (reference.get("schema") != "exact-training-aggregate-reference-v1"
                    or reference.get("semantic_key") != key
                    or reference.get("upstream") != contract):
                return None
            path = Path(reference["aggregate_uri"])
            if sha256_file(path) != reference["sha256"]:
                return None
            payload = json.loads(path.read_text())
            rows = payload["rows"]
            if payload.get("identity") != identity or len(rows) != reference["row_count"]:
                return None
            if len({str(row["Src"]) for row in rows}) != reference["source_count"]:
                return None
        if {str(row["Src"]) for row in rows} & set(map(str, application.get("source_ids", []))):
            raise ValueError("Training aggregate overlaps reporting source groups")
        return reference
    except (OSError, sqlite3.Error, TypeError, KeyError, json.JSONDecodeError):
        return None
    finally:
        if old_role is marker:
            del model._numerical_scoring_role
        else:
            model._numerical_scoring_role = old_role
        if old_llm is marker:
            del model.use_llm
        else:
            model.use_llm = old_llm


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
