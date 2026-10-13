"""Qualification-only, bounded CUDA graphs for the pinned BERT encoder path.

No padding, regrouping, precision or pooling policy is changed here. A graph
replays exactly one logical encoder call; unsupported executions remain eager.
"""

from collections import Counter, OrderedDict
from functools import lru_cache
import hashlib
import inspect
import json
import os
import threading
import time
from types import FunctionType

import torch


GRAPH_EXECUTION = (
    "EXACT_ENCODER_CUDA_GRAPHS",
    "EXACT_ENCODER_GRAPH_MAX_SHAPES",
    "EXACT_ENCODER_GRAPH_MAX_BYTES",
    "EXACT_ENCODER_GRAPH_MIN_CALLS",
)


def graph_enabled():
    enabled = os.getenv(GRAPH_EXECUTION[0], "0")
    if enabled not in {"0", "1"}:
        raise ValueError("EXACT_ENCODER_CUDA_GRAPHS must be 0 or 1")
    if enabled == "0":
        return False
    namespace = os.getenv("EXACT_PRIMITIVE_NAMESPACE", "")
    if (
        os.getenv("EXACT_EXPERIMENT_ROLE") != "throughput_fixture"
        or not namespace.startswith("qualification-")
        or namespace == "qualification-"
    ):
        raise RuntimeError(
            "CUDA graphs are only admitted in throughput_fixture qualification-* runs"
        )
    return True


def _limits():
    values = tuple(
        int(os.getenv(name, str(default)))
        for name, default in zip(GRAPH_EXECUTION[1:], (16, 512 * 1024**2, 8))
    )
    if any(value < 1 for value in values):
        raise ValueError("CUDA graph shape, byte and reuse limits must be positive")
    return values


@lru_cache(maxsize=1)
def _pinned_functions():
    from transformers import __version__
    from transformers.models.bert import modeling_bert as bert
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    functions = (
        (
            bert.BertModel.forward,
            "de674bd4a6dbab7598c8f8f08e3af3b69289aeea55559c1b55bcbcad712aa14b",
        ),
        (
            bert._prepare_4d_attention_mask_for_sdpa,
            "830aef6b922ae7a033cfedc360591a2ce9bfd99fdb22d29d253ce7962cdfec76",
        ),
        (
            AttentionMaskConverter._expand_mask,
            "3ce51cd726f2a91f30251603ca842bb4312bc2f59a6c0d45824051986a9b10ef",
        ),
    )
    if __version__ != "4.57.6" or torch.__version__ != "2.7.0+cu128":
        return None
    try:
        if any(
            hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() != digest
            for fn, digest in functions
        ):
            return None
    except (OSError, TypeError):
        return None
    return bert, functions[0][0], functions[1][0], functions[2][0]


def _specialized_forward(original, expand, *, shape, dtype, all_ones):
    """Clone just this function's globals; never modify a module or model method."""

    def prepare(mask, actual_dtype, tgt_len=None):
        if tuple(mask.shape) != shape or actual_dtype != dtype:
            raise RuntimeError("CUDA graph mask shape/dtype differs from its branch")
        target = shape[1] if tgt_len is None else tgt_len
        return None if all_ones else expand(mask=mask, dtype=actual_dtype, tgt_len=target)

    namespace = dict(original.__globals__)
    namespace["_prepare_4d_attention_mask_for_sdpa"] = prepare
    cloned = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    cloned.__kwdefaults__ = original.__kwdefaults__
    return cloned


def _module_scalars(module):
    def scalar(value):
        return value is None or type(value) in (bool, int, float, str)

    return tuple(
        sorted(
            (name, value)
            for name, value in vars(module).items()
            if scalar(value) or (isinstance(value, tuple) and all(scalar(item) for item in value))
        )
    )


def _model_signature(scorer, model):
    from exact.impl.models.scorer_common import ScorerCommonMixin
    from torch.nn.modules import module

    pinned = _pinned_functions()
    if pinned is None or type(model) is not pinned[0].BertModel:
        return None
    if (
        getattr(model.forward, "__func__", None) is not pinned[1]
        or pinned[0]._prepare_4d_attention_mask_for_sdpa is not pinned[2]
        or getattr(inspect.unwrap(scorer._pool), "__func__", None) is not ScorerCommonMixin._pool
        or model.config.is_decoder
        or model.config.output_attentions
        or not model.config.use_return_dict
        or model.attn_implementation != "sdpa"
        or model.position_embedding_type != "absolute"
        or len(str(getattr(model.config, "_commit_hash", ""))) != 40
        or not torch.is_inference_mode_enabled()
        or torch.is_grad_enabled()
    ):
        return None
    hook_names = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
    if any(getattr(module, "_global" + name, {}) for name in hook_names):
        return None
    for child in model.modules():
        if child.training or any(getattr(child, name, {}) for name in hook_names):
            return None
        if "forward" in child.__dict__ or getattr(child, "_compiled_call_impl", None) is not None:
            return None
        if isinstance(child, torch.nn.Embedding) and child.max_norm is not None:
            return None  # This forward mutates weights, so graph replay is not read-only.
    if any(
        type(layer.attention.self) is not pinned[0].BertSdpaSelfAttention
        for layer in model.encoder.layer
    ):
        return None
    try:
        tensors = list(model.named_parameters()) + list(model.named_buffers())
        device = torch.device(scorer.device)
        if any(
            value.device.type != device.type
            or (device.index is not None and value.device.index != device.index)
            for _, value in tensors
        ):
            return None
        versions = tuple(
            (
                name,
                id(value),
                value.data_ptr(),
                value._version,
                str(value.dtype),
                tuple(value.shape),
                str(value.device),
            )
            for name, value in tensors
        )
    except RuntimeError:
        return None
    backend = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
        torch.are_deterministic_algorithms_enabled(),
        torch.get_float32_matmul_precision(),
        torch.get_autocast_dtype("cuda"),
        torch.is_autocast_cache_enabled(),
        os.getenv("CUBLAS_WORKSPACE_CONFIG"),
    )
    return (
        os.getpid(),
        id(model),
        versions,
        json.dumps(model.config.to_dict(), sort_keys=True),
        scorer.fp16,
        scorer.pooling_method.value,
        id(ScorerCommonMixin._pool),
        str(scorer.device),
        backend,
        tuple(
            (name, id(child), id(type(child).forward), _module_scalars(child))
            for name, child in model.named_modules()
        ),
    )


def _input_key(inputs):
    if (
        not {"input_ids", "attention_mask"}
        <= inputs.keys()
        <= {"input_ids", "attention_mask", "token_type_ids"}
    ):
        return None
    ids = inputs["input_ids"]
    if ids.ndim != 2 or not 1 <= ids.shape[0] <= 64 or not 1 <= ids.shape[1] <= 512:
        return None
    if ids.numel() > 8192 or any(
        value.device.type != "cpu"
        or value.dtype != torch.int64
        or value.shape != ids.shape
        or not value.is_contiguous()
        for value in inputs.values()
    ):
        return None
    mask = inputs["attention_mask"]
    if not bool(((mask == 0) | (mask == 1)).all()):
        return None
    return tuple(
        (name, tuple(value.shape), value.dtype) for name, value in sorted(inputs.items())
    ), bool((mask == 1).all())


class _Entry:
    def __init__(self, graph, inputs, output, size):
        self.graph, self.inputs, self.output, self.size = graph, inputs, output, size

    def replay(self, inputs):
        for name, value in inputs.items():
            self.inputs[name].copy_(value)
        self.graph.replay()
        return self.output.clone()  # A later replay must not overwrite the caller's vectors.

    def close(self):
        self.graph.reset()
        self.graph, self.inputs, self.output = None, {}, None


class _Cache:
    def __init__(self):
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.entries, self.seen, self.models = OrderedDict(), OrderedDict(), {}
        self.stats, self.reasons = Counter(), Counter()
        self.bytes = 0

    def remove(self, key):
        entry = self.entries.pop(key)
        self.bytes -= entry.size
        entry.close()

    def clear(self):
        if self.entries:
            torch.cuda.synchronize()
        for key in list(self.entries):
            self.remove(key)
        self.seen.clear()
        self.models.clear()


def clear_encoder_graphs(scorer, *, reset_stats=False):
    cache = getattr(scorer, "_encoder_graph_cache", None)
    if cache is not None:
        if cache.pid != os.getpid():
            raise RuntimeError("Inherited CUDA graph state must not be used after fork")
        with cache.lock:
            cache.clear()
            if reset_stats:
                cache.stats.clear()
                cache.reasons.clear()


def encoder_graph_stats(scorer):
    cache = getattr(scorer, "_encoder_graph_cache", None)
    if cache is None:
        return {"cached_shapes": 0, "retained_bytes": 0, "fallback_reasons": {}}
    return {
        **cache.stats,
        "cached_shapes": len(cache.entries),
        "retained_bytes": cache.bytes,
        "fallback_reasons": dict(cache.reasons),
    }


def _same_tensor(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and bool(torch.isfinite(left).all())
        and bool(torch.isfinite(right).all())
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


class _CaptureRejected(RuntimeError):
    def __init__(self, reason, eager_result):
        super().__init__(reason)
        self.eager_result = eager_result


def _capture(scorer, model, inputs, all_ones, eager, stats):
    pinned = _pinned_functions()
    assert pinned is not None
    before_free = torch.cuda.mem_get_info(scorer.device)[0]
    static = {name: value.to(scorer.device) for name, value in inputs.items()}
    expected = eager(static)
    stats["verification_forward_calls"] += 1
    forward = _specialized_forward(
        pinned[1],
        pinned[3],
        shape=tuple(static["input_ids"].shape),
        dtype=model.embeddings.word_embeddings.weight.dtype,
        all_ones=all_ones,
    )

    def run():
        with torch.amp.autocast(device_type="cuda", enabled=scorer.fp16):
            out = forward(model, **static).last_hidden_state
        return scorer._pool(out, static["attention_mask"])

    stream = torch.cuda.Stream(device=scorer.device)
    stream.wait_stream(torch.cuda.current_stream(scorer.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
            stats["warmup_forward_calls"] += 1
    torch.cuda.current_stream(scorer.device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    capture_oom = None
    try:
        with torch.cuda.graph(graph, stream=stream):
            try:
                output = run()
            except torch.cuda.OutOfMemoryError as error:
                capture_oom = error
                raise
        stats["capture_forward_calls"] += 1
        graph.replay()
        actual = output.clone()
        torch.cuda.synchronize(scorer.device)
        if not _same_tensor(expected, actual):
            raise _CaptureRejected("output_mismatch", expected)
        pool = tuple(graph.pool())
        segments = torch.cuda.memory_snapshot()
        pool_bytes = sum(
            row["total_size"] for row in segments if tuple(row.get("segment_pool_id", ())) == pool
        )
        if pool_bytes <= 0:
            raise _CaptureRejected("allocator_pool_unavailable", expected)
        static_bytes = sum(value.untyped_storage().nbytes() for value in static.values())
        size = max(
            pool_bytes + static_bytes, before_free - torch.cuda.mem_get_info(scorer.device)[0]
        )
        return _Entry(graph, static, output, size), actual
    except BaseException:
        graph.reset()
        if capture_oom is not None:
            raise capture_oom
        raise


def encode_with_graph(scorer, model, inputs, eager):
    """Execute one original encoder call; the callback retains the eager path."""
    if not graph_enabled():
        return eager(inputs.to(scorer.device))
    limits = _limits()
    cache = getattr(scorer, "_encoder_graph_cache", None)
    if cache is None:
        cache = scorer._encoder_graph_cache = _Cache()
    if cache.pid != os.getpid():
        raise RuntimeError("Inherited CUDA graph state must not be used after fork")
    with cache.lock:
        cache.stats["requests"] += 1

        def fallback(reason):
            cache.stats["eager_fallbacks"] += 1
            cache.reasons[reason] += 1
            return eager(inputs.to(scorer.device))

        if torch.device(scorer.device).type != "cuda":
            return fallback("non_cuda")
        signature, shape = _model_signature(scorer, model), _input_key(inputs)
        if signature is None or shape is None:
            return fallback("unsupported_model_or_input")
        model_id = id(model)
        if cache.models.get(model_id) != signature:
            for key in list(cache.entries):
                if key[0] == model_id:
                    torch.cuda.synchronize(scorer.device)
                    cache.remove(key)
            cache.seen.clear()
            cache.models[model_id] = signature
        key = (model_id, shape, torch.cuda.current_stream(scorer.device).cuda_stream)
        while cache.entries and (len(cache.entries) > limits[0] or cache.bytes > limits[1]):
            torch.cuda.synchronize(scorer.device)
            cache.remove(next(iter(cache.entries)))
        if key in cache.entries:
            cache.entries.move_to_end(key)
            cache.stats["replays"] += 1
            return cache.entries[key].replay(inputs)
        if cache.seen.get(key) == -1:
            return fallback("rejected_capture")
        cache.seen[key] = cache.seen.get(key, 0) + 1
        cache.seen.move_to_end(key)
        while len(cache.seen) > 256:
            cache.seen.popitem(last=False)
        if cache.seen[key] < limits[2]:
            return fallback("amortization")
        if len(cache.entries) >= limits[0]:
            return fallback("shape_limit")
        if torch.cuda.mem_get_info(scorer.device)[0] < limits[1] + 256 * 1024**2:
            return fallback("free_memory_reserve")
        started = time.perf_counter()
        cache.stats["capture_attempts"] += 1
        capture_oom = False
        rejected = None
        try:
            entry, actual = _capture(scorer, model, inputs, shape[1], eager, cache.stats)
        except _CaptureRejected as error:
            cache.stats["capture_rejections"] += 1
            cache.stats["eager_fallbacks"] += 1
            cache.reasons[str(error)] += 1
            cache.seen[key] = -1
            rejected = error.eager_result
        except torch.cuda.OutOfMemoryError:
            cache.stats["capture_ooms"] += 1
            cache.clear()
            cache.models[model_id] = signature
            cache.seen[key] = -1
            capture_oom = True
        finally:
            cache.stats["capture_seconds"] += time.perf_counter() - started
        if rejected is not None:
            return rejected
        if capture_oom:
            # Leave the exception scope first: its traceback can retain tensors
            # from the failed capture and otherwise cause a second avoidable OOM.
            torch.cuda.empty_cache()
            return fallback("capture_oom")
        cache.stats["captures"] += 1
        if cache.bytes + entry.size > limits[1]:
            entry.close()
            cache.seen[key] = -1
            cache.stats["uncached_captures"] += 1
            cache.reasons["retained_byte_limit"] += 1
            # The independently checked ordinary result is already computed.
            return actual
        cache.entries[key] = entry
        cache.bytes += entry.size
        cache.stats["peak_retained_bytes"] = max(cache.stats["peak_retained_bytes"], cache.bytes)
        return actual
