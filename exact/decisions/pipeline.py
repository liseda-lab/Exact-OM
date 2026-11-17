"""Source-gated Elephant decisions over already scored, reference-free candidates."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

MODEL = "baselabs/elephant-1"
REVISION = "b3f8a93462fb1051a9de031e94a7ff380e254072"
RUNTIME_COMMIT = "81ce62f1597c91e46767d4d02d6ac2e18534fe94"
INSTRUCTIONS = (
    "You are a binary classifier for ontology alignment. "
    "The source and target entities are equivalent."
)
POLICIES = ("global", "global-original", "local-application", "local-train")


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text())


@contextmanager
def atomic_stream(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write(path: Path, value: Any) -> None:
    with atomic_stream(path) as stream:
        stream.write(encoded(value).decode() + "\n")


def records(path: Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Candidate/request JSONL rows must be objects")
                yield row


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _groups(path: Path, sources: set[str]):
    seen = set()
    for source, group in itertools.groupby(records(path), key=lambda row: row["Src"]):
        if source not in sources or source in seen:
            raise ValueError("Rows must be contiguous by source and inside the declared universe")
        seen.add(source)
        rows = list(group)
        if len({row["Tgt"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate candidate pair for {source}")
        for row in rows:
            if not isinstance(row["Tgt"], str) or not row["Tgt"]:
                raise ValueError("Candidate target must be a nonempty string")
            _number(row["S_base"], "S_base")
            _number(row["U"], "U")
        yield source, rows


def request(row: dict, *, policy: str) -> dict:
    """Render the unchanged binary predicate and historical request identifier."""
    if "decision_request" in row:
        result = row["decision_request"]
        payload = result["payload"]
    else:
        labels = row["selected_labels"]
        field = (
            "llm_structured_evidence_packet"
            if policy.startswith("global")
            else "llm_evidence_packet"
        )
        text = (
            f"Source entity: {labels['source']}\nTarget entity: {labels['target']}\n\n"
            f"Pair brief:\n{row[field]}"
        )
        payload = {
            "state": {"evidence": text},
            "questions": {"equivalent": {"type": "noul", "instructions": INSTRUCTIONS}},
        }
    expected_question = {"equivalent": {"type": "noul", "instructions": INSTRUCTIONS}}
    if (
        set(payload) != {"state", "questions"}
        or set(payload.get("state", {})) != {"evidence"}
        or payload.get("questions") != expected_question
        or not isinstance(payload.get("state", {}).get("evidence"), str)
    ):
        raise ValueError("Candidate has an incompatible Elephant predicate")
    rid = fingerprint({"model": MODEL, "revision": REVISION, "payload": payload})
    if "decision_request" in row and (
        result["id"] != rid or result["pair"] != [row["Src"], row["Tgt"]]
    ):
        raise ValueError("Candidate request identity or pair changed")
    return {"id": rid, "pair": [row["Src"], row["Tgt"]], "payload": payload}


def prepare(
    candidates: Path,
    sources_path: Path,
    output: Path,
    *,
    policy: str = "global",
    anchors_path: Path | None = None,
) -> Path:
    """Freeze the measured 20%/five-candidate route; no reference labels are read.

    Corrected global uses primitive scores for every candidate, including former
    exact anchors. Original local includes score-one exact anchors in ambiguity,
    skips their requests, and routes every TRAIN source.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown decision policy: {policy}")
    candidates, sources_path, output = map(Path, (candidates, sources_path, output))
    sources = sources_path.read_text().splitlines()
    if any(not source for source in sources) or len(sources) != len(set(sources)):
        raise ValueError("Source universe must contain unique nonempty IRIs")
    source_set = set(sources)
    anchors = {}
    if anchors_path is not None:
        if policy == "global":
            raise ValueError("Corrected global uses scored anchors, not an anchor sidecar")
        for row in read(anchors_path):
            source, target = row["Src"], row["Tgt"]
            if source not in source_set or not isinstance(target, str) or not target:
                raise ValueError("Anchor endpoint outside the declared source universe")
            if target in anchors.get(source, ()):
                raise ValueError("Duplicate native anchor pair")
            if _number(row.get("S_final", row.get("Score", float("nan"))), "anchor score") != 1.0:
                raise ValueError("Native anchors must have an explicit score of one")
            anchors.setdefault(source, set()).add(target)
    identity = {
        "schema": "exact-decisions-plan/1",
        "implementation_sha256": file_hash(Path(__file__)),
        "candidates_sha256": file_hash(candidates),
        "sources_sha256": file_hash(sources_path),
        "anchors_sha256": file_hash(anchors_path) if anchors_path is not None else None,
        "policy": policy,
        "model": MODEL,
        "revision": REVISION,
        "runtime": fingerprint(read(Path(__file__).with_name("runtime.json"))),
        "fraction": 1.0 if policy == "local-train" else 0.2,
        "depth": 5,
    }
    plan_path = output / "plan.json"
    if plan_path.exists():
        old = read(plan_path)
        if (
            old["identity"] != identity
            or file_hash(output / "requests.jsonl") != old["requests_sha256"]
        ):
            raise ValueError("Decision plan inputs changed; use a new output directory")
        return plan_path
    ambiguities = {source: float(len(anchors.get(source, ())) > 1) for source in sources}
    for source, rows in _groups(candidates, source_set):
        scores_by_target = {
            row["Tgt"]: (
                1.0 if policy != "global" and row.get("exact_anchor") else float(row["S_base"])
            )
            for row in rows
        }
        scores_by_target.update({target: 1.0 for target in anchors.get(source, ())})
        scores = sorted(scores_by_target.values())
        gap = scores[-1] - (scores[-2] if len(scores) > 1 else 0.0)
        ambiguities[source] = 1.0 - min(1.0, max(0.0, gap))
    selected = sorted(sources, key=lambda source: (-ambiguities[source], source))[
        : math.ceil(identity["fraction"] * len(sources))
    ]
    admitted = set(selected)
    count = 0
    with atomic_stream(output / "requests.jsonl") as stream:
        for source, rows in _groups(candidates, source_set):
            if source not in admitted:
                continue
            eligible = [
                row
                for row in rows
                if policy == "global"
                or (not row.get("exact_anchor") and row["Tgt"] not in anchors.get(source, ()))
            ]
            for row in sorted(eligible, key=lambda row: (-float(row["S_base"]), row["Tgt"]))[:5]:
                stream.write(encoded(request(row, policy=policy)).decode() + "\n")
                count += 1
    if file_hash(candidates) != identity["candidates_sha256"]:
        raise ValueError("Candidates changed while freezing decisions")
    write(
        plan_path,
        {
            "identity": identity,
            "selected_sources": selected,
            "source_count": len(sources),
            "request_count": count,
            "requests_sha256": file_hash(output / "requests.jsonl"),
            "training_application_same_route": policy == "global",
        },
    )
    return plan_path


def load_plan(path: Path) -> dict:
    plan = read(path)
    expected_runtime = fingerprint(read(Path(__file__).with_name("runtime.json")))
    identity = plan["identity"]
    if (
        identity.get("implementation_sha256") != file_hash(Path(__file__))
        or identity["model"] != MODEL
        or identity["revision"] != REVISION
        or identity["runtime"] != expected_runtime
    ):
        raise ValueError("Decision plan implementation/model/runtime identity differs")
    if file_hash(Path(path).parent / "requests.jsonl") != plan["requests_sha256"]:
        raise ValueError("Decision requests changed")
    return plan


def _database(cache: Path):
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT NOT NULL, sha256 TEXT NOT NULL)"
    )
    return connection


def cache_key(plan: dict, rec: dict) -> str:
    if rec["id"] != fingerprint({"model": MODEL, "revision": REVISION, "payload": rec["payload"]}):
        raise ValueError("Request payload identity changed")
    return fingerprint({"request": rec["id"], "runtime": plan["identity"]["runtime"]})


def _probability(value):
    probability = _number(value, "probability")
    if not 0 <= probability <= 1:
        raise ValueError("Elephant probability must be within [0, 1]")
    return probability


def run(
    plan_path: Path,
    cache: Path,
    *,
    engine=None,
    model_dir: Path | None = None,
    device: str = "cuda:0",
    cache_only: bool = False,
) -> dict:
    """Commit each decision atomically; reruns reuse compatible completed requests."""
    import fcntl

    plan = load_plan(plan_path)
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(cache.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        connection = _database(cache)
        complete = reused = inferred = 0
        try:
            for rec in records(Path(plan_path).parent / "requests.jsonl"):
                key = cache_key(plan, rec)
                saved = connection.execute(
                    "SELECT value, sha256 FROM answers WHERE key=?", (key,)
                ).fetchone()
                if saved:
                    result = json.loads(saved[0])
                    if fingerprint(result) != saved[1]:
                        raise ValueError("Cached response checksum differs")
                    _probability(result["probability"])
                    reused += 1
                else:
                    if cache_only:
                        raise ValueError(f"No cached decision for {rec['id']}")
                    if engine is None:
                        if model_dir is None:
                            raise ValueError("A local pinned model directory is required")
                        from .runtime import ElephantEngine

                        engine = ElephantEngine(model_dir, device=device)
                    result = dict(engine.predict(rec["payload"]))
                    result["probability"] = _probability(result["probability"])
                    connection.execute(
                        "INSERT INTO answers VALUES (?,?,?)",
                        (key, encoded(result).decode(), fingerprint(result)),
                    )
                    connection.commit()
                    inferred += 1
                complete += 1
                if complete % 64 == 0:
                    write(
                        Path(plan_path).parent / "status.json",
                        {
                            "status": "running",
                            "completed": complete,
                            "total": plan["request_count"],
                            "cached_response_reused": reused,
                            "local_model_calls": inferred,
                            "provider_calls": 0,
                            "selector_stage_consultation": False,
                        },
                    )
        finally:
            connection.close()
    if complete != plan["request_count"]:
        raise ValueError("Decision plan request count differs from its records")
    result = {
        "status": "complete",
        "plan_sha256": file_hash(plan_path),
        "completed": complete,
        "cached_response_reused": reused,
        "local_model_calls": inferred,
        "provider_calls": 0,
    }
    write(Path(plan_path).parent / "status.json", result)
    return result


def apply(
    candidates: Path, plan_path: Path, cache: Path, output: Path, *, beta: float = 0.8
) -> dict:
    """Apply the deployed clip(beta*U) mixture only to admitted pairs."""
    beta = _number(beta, "beta")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    plan = load_plan(plan_path)
    if file_hash(candidates) != plan["identity"]["candidates_sha256"]:
        raise ValueError("Application candidates differ from frozen decision inputs")
    request_map = {}
    connection = _database(cache)
    try:
        for rec in records(Path(plan_path).parent / "requests.jsonl"):
            pair = tuple(rec["pair"])
            if pair in request_map:
                raise ValueError("Duplicate decision pair")
            saved = connection.execute(
                "SELECT value, sha256 FROM answers WHERE key=?", (cache_key(plan, rec),)
            ).fetchone()
            if saved is None:
                raise ValueError(f"Incomplete decision cache: {rec['id']}")
            answer = json.loads(saved[0])
            if fingerprint(answer) != saved[1]:
                raise ValueError("Cached response checksum differs")
            request_map[pair] = (rec["id"], _probability(answer["probability"]))
    finally:
        connection.close()
    count = changed = applied = 0
    with atomic_stream(output) as stream:
        for original in records(candidates):
            row = dict(original)
            base = _number(row["S_base"], "S_base")
            answer = request_map.get((row["Src"], row["Tgt"]))
            weight = min(1.0, max(0.0, beta * _number(row["U"], "U"))) if answer else 0.0
            probability = answer[1] if answer else None
            score = (1.0 - weight) * base + weight * probability if answer else base
            row.update(
                S_pair_final=score,
                S_final=score,
                p_llm=probability,
                llm_used=answer is not None,
                llm_weight=weight,
                decision_request_id=answer[0] if answer else None,
                cached_response_reused=answer is not None,
                local_model_call_attempted=False,
                live_provider_call_attempted=False,
                effective_pair_score_contribution=score != base,
                selector_stage_consultation=False,
            )
            stream.write(encoded(row).decode() + "\n")
            count += 1
            changed += score != base
            applied += answer is not None
        if applied != len(request_map):
            raise ValueError("An admitted decision pair is missing from application")
        if file_hash(candidates) != plan["identity"]["candidates_sha256"]:
            raise ValueError("Candidate input changed while applying decisions")
    receipt = {
        "schema": "exact-decisions-application/1",
        "plan_sha256": file_hash(plan_path),
        "candidates_sha256": file_hash(candidates),
        "output_sha256": file_hash(output),
        "rows": count,
        "judgments_applied": applied,
        "changed_scores": changed,
        "fusion": {"mode": "beta_u", "beta": beta},
        "provider_calls": 0,
    }
    write(Path(str(output) + ".receipt.json"), receipt)
    return receipt
