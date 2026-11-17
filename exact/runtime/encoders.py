"""Identity-checked, bounded persistent encoder reuse."""
from __future__ import annotations
import hashlib
import inspect
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable, Mapping
from exact.utils.provenance import sha256_file


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


def _compatible_encoder_source(scorer: Any) -> str | None:
    """Permit a bound cache migration only when encoder/pooling code is identical.

    This applies to immutable embedding rows only, never predictions or fits.
    Model, tokenizer, precision, role and hardware remain in both identities.
    """
    path = os.getenv("EXACT_ENCODER_COMPATIBLE_SOURCE")
    expected = os.getenv("EXACT_ENCODER_COMPATIBLE_SOURCE_SHA256")
    if not path and not expected:
        return None
    if not path or not expected or sha256_file(Path(path)) != expected:
        raise ValueError("Encoder compatibility source differs from its reviewed binding")
    import ast
    import textwrap

    tree = ast.parse(Path(path).read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = {
        node.name: node
        for node in classes["ScorerCommonMixin"].body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("_encode_texts", "_pool"):
        current = ast.parse(textwrap.dedent(inspect.getsource(getattr(scorer, name)))).body[0]
        if ast.dump(methods[name], include_attributes=False) != ast.dump(
            current, include_attributes=False
        ):
            raise ValueError(f"Encoder compatibility changes numerical method {name}")
    from exact.impl.models.scorer_common import PoolingMethod

    current_enum = ast.parse(textwrap.dedent(inspect.getsource(PoolingMethod))).body[0]
    if ast.dump(classes["PoolingMethod"], include_attributes=False) != ast.dump(
        current_enum, include_attributes=False
    ):
        raise ValueError("Encoder compatibility changes pooling semantics")
    return expected


def _stable_tokenizer_identity_enabled():
    enabled = os.getenv("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", "0")
    if enabled not in {"0", "1"}:
        raise ValueError("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY must be 0 or 1")
    if enabled == "0":
        return False
    namespace = os.getenv("EXACT_PRIMITIVE_NAMESPACE", "")
    if (
        os.getenv("EXACT_EXPERIMENT_ROLE") != "throughput_fixture"
        or not namespace.startswith("qualification-")
        or namespace == "qualification-"
    ):
        raise RuntimeError(
            "Stable tokenizer identity requires GPU discrete parity and is only admitted "
            "for throughput_fixture runs in an isolated qualification-* namespace."
        )
    return True


def _explicit_tokenization(scorer):
    from exact.impl.models.scorer_common import ScorerCommonMixin

    # inference_mode and instrumentation using functools.wraps retain the actual
    # implementation contract. Arbitrary custom encoders may use backend defaults.
    return inspect.unwrap(scorer._encode_texts) is inspect.unwrap(ScorerCommonMixin._encode_texts)


def _tokenizer_backend(tokenizer, *, explicit):
    if not hasattr(tokenizer, "backend_tokenizer"):
        return None
    backend = tokenizer.backend_tokenizer.to_str()
    if explicit:
        state = json.loads(backend)
        # _encode_texts explicitly sets padding=True, truncation=True and max_len.
        # Their backend state changes after a call; sides and special tokens are
        # independently bound. Every semantic backend component remains intact.
        state.pop("padding", None)
        state.pop("truncation", None)
        backend = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return backend


def encoder_identity(scorer, tokenizer, model, max_len):
    """Actual encoder weights and execution contract, independent of reporting labels."""
    import torch
    from exact.impl.models.encoder_graph import GRAPH_EXECUTION, graph_enabled

    stable_tokenizer = _stable_tokenizer_identity_enabled()
    graph_enabled()
    if os.getenv("EXACT_PAIR_CONTEXT_RECIPE") is not None:
        from exact.impl.models.pair_adaptive_batch import validate_batching_admission

        validate_batching_admission()
    config = getattr(model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    if (
        config is None
        or not revision
        or len(str(revision)) != 40
        or getattr(model, "training", False)
    ):
        return None
    namespace = os.getenv("EXACT_PRIMITIVE_NAMESPACE", "scientific")
    cached = getattr(scorer, "_stage_encoder_keys", {})
    state_tensors = list(model.named_parameters()) if hasattr(model, "named_parameters") else []
    if hasattr(model, "named_buffers"):
        state_tensors += list(model.named_buffers())
    try:
        weight_versions = tuple(
            (name, id(value), value.data_ptr(), value._version, str(value.dtype), tuple(value.shape))
            for name, value in state_tensors
        )
    except RuntimeError:
        # Tensors created in inference mode have no mutation counter. Their
        # weights cannot safely reuse a memoized identity after in-place edits.
        return None
    if not hasattr(tokenizer, "get_vocab"):
        return None
    tokenizer_settings = {
        "config": {key: value for key, value in getattr(tokenizer, "init_kwargs", {}).items()
                   if key not in {"name_or_path", "cache_dir"}},
        "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "added_vocab": tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {},
        "backend_id": id(getattr(tokenizer, "backend_tokenizer", None)),
    }
    batch = getattr(scorer, "_numerical_channel_scopes", None)
    explicit_tokenization = stable_tokenizer and _explicit_tokenization(scorer)
    tokenizer_key = (
        "encoder_tokenizer", id(tokenizer), _hash(tokenizer_settings), explicit_tokenization,
    )
    tokenization = batch.get(tokenizer_key) if batch is not None else None
    if tokenization is None:
        tokenization = {
            "vocab": tokenizer.get_vocab(),
            "backend": _tokenizer_backend(tokenizer, explicit=explicit_tokenization),
        }
        # Store copies: a mutable Python vocabulary must not alter its past identity.
        tokenization = json.loads(json.dumps(tokenization))
        tokenization["sha256"] = _hash(tokenization)
        if batch is not None:
            batch[tokenizer_key] = tokenization
    local_key = (os.getpid(), id(model), id(tokenizer), max_len, revision, scorer.fp16,
                 str(scorer._cache_tensor_dtype), scorer.pooling_method.value,
                 scorer.device_type, namespace)
    execution = {
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "autocast": torch.is_autocast_enabled("cuda"),
        "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        **{
            name: os.getenv(name)
            for name in (
                "EXACT_PAIR_CONTEXT_BATCHING",
                "EXACT_PAIR_CONTEXT_BLOCK_PAIRS",
                "EXACT_PAIR_CONTEXT_TEXT_BATCH",
                "EXACT_PAIR_CONTEXT_MATRIX_ELEMENTS",
                "EXACT_PAIR_CONTEXT_SCHEDULE",
                "EXACT_PAIR_CONTEXT_ENCODER",
                "EXACT_PAIR_CONTEXT_MATRIX_KERNEL",
                "EXACT_PAIR_CONTEXT_ARITHMETIC",
                "EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY",
                *GRAPH_EXECUTION,
            )
        },
    }
    if recipe := os.getenv("EXACT_PAIR_CONTEXT_RECIPE"):
        execution["EXACT_PAIR_CONTEXT_RECIPE"] = recipe
    model_config = {key: value for key, value in config.to_dict().items()
                    if key not in {"_name_or_path", "name_or_path"}}
    local_key = (*local_key, weight_versions, tuple(execution.items()),
                 _hash([model_config, tokenizer_settings, tokenization["sha256"]]))
    identities = cached.get(local_key)
    if identities is None:
        # Tokenizer/namespace changes require a new identity, not another copy
        # of GPU weights. Keep exactly the existing versioned byte digest.
        weight_key = (os.getpid(), weight_versions)
        weight_cache = getattr(model, "_exact_encoder_weight_digest", None)
        if weight_cache is None or weight_cache[0] != weight_key:
            weights = hashlib.sha256()
            for name, value in state_tensors:
                weights.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
                weights.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            weight_cache = (weight_key, weights.hexdigest() if state_tensors else None)
            model._exact_encoder_weight_digest = weight_cache
        payload = {
            "schema": 1,
            "revision": revision,
            "weights_sha256": weight_cache[1],
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "autocast": torch.is_autocast_enabled("cuda"),
            "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
            "model_config": model_config,
            "tokenizer_vocab": tokenization["vocab"],
            "tokenizer_backend": tokenization["backend"],
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
                torch.cuda.get_device_name(scorer.device) if scorer.device_type == "cuda" else "cpu"
            ),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "role": "label_independent_text_v1",
            "namespace": namespace,
            "execution": execution,
            "implementation": sha256_file(
                Path(__file__).parents[1] / "impl/models/scorer_common.py"
            ),
        }
        identity = _hash(payload)
        previous = _compatible_encoder_source(scorer)
        legacy = _hash({**payload, "implementation": previous}) if previous else None
        identities = (identity, legacy)
        cached[local_key] = identities
        scorer._stage_encoder_keys = cached
    return identities


def cached_encoder_rows(
    scorer: Any,
    tokenizer: Any,
    model: Any,
    texts: list[str],
    max_len: int,
    compute: Callable[[list[str]], Any],
    *, return_cpu: bool = False,
) -> Any:
    """Reuse vectors by exact text, pinned encoder/tokenizer, precision, and role.

    Opaque/unpinned model weights fail closed to ordinary encoding. The shared
    cache is independent of candidate selection, fusion, and reporting labels.
    """
    directory = os.getenv("EXACT_EMBEDDING_CACHE_DIR")
    if not directory or not texts:
        return compute(texts)
    identities = encoder_identity(scorer, tokenizer, model, max_len)
    return _cached_encoder_rows_from_identity(
        scorer, model, texts, compute, identities, return_cpu=return_cpu
    )


def _cached_encoder_rows_from_identity(
    scorer: Any,
    model: Any,
    texts: list[str],
    compute: Callable[[list[str]], Any],
    identities: tuple[str, str | None] | None,
    *, return_cpu: bool = False,
) -> Any:
    """Use an identity just checked by the same resident/persistent lookup call.

    This private path must not retain an identity across encoder invocations.
    Public callers use cached_encoder_rows, which validates its own inputs.
    """
    directory = os.getenv("EXACT_EMBEDDING_CACHE_DIR")
    if not directory or not texts:
        return compute(texts)
    if identities is None:
        return compute(texts)
    import torch

    config = model.config
    identity, legacy = identities
    from exact.runtime.encoder_store import encoder_store

    unique = list(dict.fromkeys(texts))
    keys = [_hash([identity, text]) for text in unique]
    store = None
    found = {}
    dimension = config.to_dict().get("hidden_size")
    try:
        store = encoder_store(Path(directory) / "vectors.sqlite3")
        found = store.lookup(keys, scorer._cache_tensor_dtype, dimension)
        if legacy and legacy != identity:
            absent = [i for i, key in enumerate(keys) if key not in found]
            old_keys = [_hash([legacy, unique[i]]) for i in absent]
            prior = store.lookup(old_keys, scorer._cache_tensor_dtype, dimension)
            migrated = {keys[i]: prior[old] for i, old in zip(absent, old_keys) if old in prior}
            found.update(migrated)
            store.publish(migrated)
    except (OSError, sqlite3.Error):
        pass
    missing = [i for i, key in enumerate(keys) if key not in found]
    if missing:
        fresh = compute([unique[i] for i in missing]).detach().to("cpu").to(scorer._cache_tensor_dtype)
        if fresh.ndim != 2 or len(fresh) != len(missing):
            raise ValueError("Encoder returned an invalid row count or shape")
        additions = {keys[i]: row for i, row in zip(missing, fresh)}
        found.update(additions)
        if store is not None:
            try:
                store.publish(additions)
            except (OSError, sqlite3.Error, RuntimeError):
                pass
    by_text = {text: found[key] for text, key in zip(unique, keys)}
    result = torch.stack([by_text[text] for text in texts])
    return result if return_cpu else result.to(scorer.device)
