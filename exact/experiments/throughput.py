"""Bounded, label-free scoring measurement contracts (never scientific results)."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any, Iterable, Mapping


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_workload(queries: Iterable[Mapping], *, pair: str, minimum_pairs: int = 5000) -> dict:
    """Select entire source groups by hash; preserve every original query and candidate."""
    groups = defaultdict(list)
    seen = set()
    for query in queries:
        qid, source = str(query["qid"]), str(query["source"])
        if qid in seen:
            raise ValueError("Duplicate original query identity")
        seen.add(qid)
        candidates = list(map(str, query["candidates"]))
        if len(candidates) != len(set(candidates)):
            raise ValueError("Repeated candidate within an original query")
        groups[source].append({"qid": qid, "source": source, "candidates": candidates})
    order = sorted(groups, key=lambda s: (identity(["throughput-20261009", pair, s]), s))
    selected, unique = [], set()
    for source in order:
        selected.extend(groups[source])
        unique.update((source, target) for q in groups[source] for target in q["candidates"])
        if len(unique) >= minimum_pairs:
            break
    result = {
        "schema_version": 1, "namespace": "qualification_only", "pair": pair,
        "selection": "fixed_source_hash_complete_original_groups", "reference_labels_used": False,
        "queries": selected, "source_count": len({q["source"] for q in selected}),
        "query_count": len(selected), "unique_candidate_pairs": len(unique),
        "candidate_occurrences": sum(len(q["candidates"]) for q in selected),
        "minimum_distinct_model_scored_pairs": minimum_pairs,
        "candidate_count_is_not_computed_pair_count": True,
        "stress_strata": ["long_uneven_evidence", "high_candidate_count", "missing_channels"],
    }
    result["identity"] = identity(result)
    return result


def group_chunks(queries: Iterable[Mapping], maximum_pairs: int):
    """Pack complete queries; an oversized query stays whole for group-dependent decisions."""
    if maximum_pairs < 1:
        raise ValueError("Chunk size must be positive")
    chunk, count, sources = [], 0, set()
    for query in queries:
        width = len(query["candidates"])
        # The scorer's group operations key by source. Repeated-source queries
        # must be separate calls, even when the combined batch would fit.
        if chunk and (count + width > maximum_pairs or query["source"] in sources):
            yield chunk
            chunk, count, sources = [], 0, set()
        chunk.append(query)
        count += width
        sources.add(query["source"])
    if chunk:
        yield chunk


def stress_coverage(records: Iterable[Mapping], group_sizes: Iterable[int]) -> dict:
    """Report predeclared label-free stress predicates without changing sampling."""
    counts = Counter()
    for row in records:
        lengths = []
        for value in row.get("context_sentences", {}).values():
            texts = [text for family in value.values() for text in family] if isinstance(value, dict) else value
            lengths.append(sum(len(text) for text in texts))
        counts["long_context_1024_characters"] += bool(lengths and max(lengths) >= 1024)
        positive = [n for n in lengths if n]
        counts["uneven_nonempty_contexts_4x"] += bool(positive and max(positive) >= 4 * min(positive))
        qualities = row.get("qualities", {})
        counts["missing_natural_channel"] += any(qualities.get(key) == 0 for key in ("q_hier", "q_sim", "q_attr", "q_diff"))
    return {"pair_counts": dict(counts), "high_candidate_queries_100": sum(n >= 100 for n in group_sizes),
            "selection_changed": False, "separate_stress_timing": None}


class PhaseCounters:
    """Opt-in inclusive host timers; no synchronization or hooks when not installed."""

    def __init__(self):
        self.seconds = Counter()
        self.calls = Counter()

    @contextmanager
    def phase(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1

    @contextmanager
    def instrument(self, owner, methods: Mapping[str, str]):
        originals = {}
        try:
            for method, phase in methods.items():
                if not hasattr(owner, method):
                    continue
                original = getattr(owner, method)
                originals[method] = (method in owner.__dict__, original)

                def measured(*args, _fn=original, _phase=phase, **kwargs):
                    with self.phase(_phase):
                        return _fn(*args, **kwargs)

                setattr(owner, method, measured)
            yield self
        finally:
            for method, (local, original) in originals.items():
                if local:
                    setattr(owner, method, original)
                else:
                    delattr(owner, method)

    def receipt(self):
        # Inclusive host times overlap; they are never summed as GPU elapsed time.
        return {"clock": "inclusive_host_perf_counter", "additive": False,
                "seconds": dict(self.seconds), "calls": dict(self.calls)}


class Measurement:
    """Count complete natural primitives once across chunks, never cache/fixture replays."""

    def __init__(self, *, mode: str, workload: dict, device: str):
        if mode not in {"cold", "warm", "replay", "hosted", "end_to_end"}:
            raise ValueError("Unknown measurement mode")
        self.mode, self.workload, self.device = mode, workload, device
        self.seen = set()
        self.chunks = []

    def commit_chunk(self, *, elapsed: float, computed: Iterable[tuple], available: int = 0,
                     duplicate_rows: int = 0, durable: bool, **metrics):
        if not durable or elapsed <= 0:
            raise ValueError("Only durably completed chunks with positive elapsed time count")
        computed = list(computed)
        keys = set(computed)
        fresh = keys - self.seen
        self.seen.update(keys)
        self.chunks.append({"elapsed_seconds": elapsed, "distinct_computed_pairs": len(fresh),
                            "already_available_pairs": available,
                            "duplicate_rows": duplicate_rows + len(computed) - len(fresh), **metrics})

    def receipt(self) -> dict:
        elapsed = sum(c["elapsed_seconds"] for c in self.chunks)
        count = sum(c["distinct_computed_pairs"] for c in self.chunks)
        rates = [c["distinct_computed_pairs"] / c["elapsed_seconds"] for c in self.chunks]
        valid = self.mode == "cold" and count >= 5000 and elapsed >= 30 and len(rates) >= 3
        rate = count / elapsed if elapsed else 0.0
        return {"schema_version": 1, "namespace": "qualification_only", "mode": self.mode,
                "device": self.device, "workload_identity": self.workload["identity"],
                "chunks": self.chunks, "elapsed_seconds": elapsed,
                "distinct_computed_pairs": count, "pairs_per_second": rate,
                "processed_rows_per_second": sum(c.get("rows", 0) for c in self.chunks) / elapsed if elapsed else 0.0,
                "median_chunk_pairs_per_second": statistics.median(rates) if rates else None,
                "worst_chunk_pairs_per_second": min(rates) if rates else None,
                "minimum_measurement_contract_met": valid,
                "throughput_target_status": ("met" if rate >= 200 else "missed") if valid else "unqualified",
                "execution_readiness": "requires_separate_parity_scientific_and_operational_admission"}


def forecast(work: Mapping[str, Mapping[str, Any]]) -> dict:
    """Sum counted work/rate intervals; unknown phases remain unknown, never zero."""
    phases, total_low, total_high = {}, 0.0, 0.0
    unknown = []
    for name, item in work.items():
        count, rate = item.get("count"), item.get("rate_per_second")
        if count is None or rate is None:
            phases[name] = None
            unknown.append(name)
            continue
        low, high = count if isinstance(count, (tuple, list)) else (count, count)
        slow, fast = rate if isinstance(rate, (tuple, list)) else (rate, rate)
        if low < 0 or high < low or slow <= 0 or fast < slow:
            raise ValueError("Invalid count/rate interval")
        phases[name] = [low / fast, high / slow]
        total_low += low / fast
        total_high += high / slow
    return {"phase_seconds": phases, "known_seconds": [total_low, total_high],
            "remaining_seconds": None if unknown else [total_low, total_high],
            "unknown_phases": unknown, "forecast_is_kill_timer": False}
