"""Bounded, read-only detection of software failures inside completed repair rows.

Opt in with ``science_report_relative`` on a run. A successful launcher is not
evidence that every nested worker succeeded. Timeouts, unknowns and unavailable
scientific rows are valid recorded outcomes and never become software incidents.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

_MAX_BYTES = 4 * 1024 * 1024
_MAX_ROWS = 256
_MAX_CACHED_BYTES = 16 * 1024
_SCHEMAS = frozenset({"exact-repair/fresh-evaluation/v1", "exact-repair/schema-recovery/v1"})
_LEGACY_UNSUPPORTED_ENDPOINT = (
    "ValueError: cannot infer complete original relation for endpoint retrieval"
)
_SOFTWARE_STATUSES = frozenset(
    {
        "error",
        "worker_error",
        "generation_error",
        "verification_error",
        "evaluator_error",
    }
)


def _read(path, digest=None):
    path = Path(path)
    state = path.stat()
    # Every check still stats all bound evidence. Reuse parsing/hash work only
    # while size, inode, modification and change times remain unchanged.
    reader = _read_cached if state.st_size <= _MAX_CACHED_BYTES else _read_uncached
    return reader(
        str(path),
        digest,
        state.st_dev,
        state.st_ino,
        state.st_size,
        state.st_mtime_ns,
        state.st_ctime_ns,
    )


def _read_uncached(path, digest, *stat_identity):
    with Path(path).open("rb") as stream:
        raw = stream.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError("scientific metadata exceeds 4 MiB")
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("scientific evidence digest mismatch: " + str(path))
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("scientific metadata must be an object")
    return value


# At most 8 MiB of raw document content, plus bounded parsed-object overhead.
# Large reports/output indexes are checked directly without retained copies.
_read_cached = lru_cache(maxsize=512)(_read_uncached)


def software_failure(status, detail=""):
    """Recognize software errors without relabeling explicit scientific scope gaps.

    One exact legacy generation exception represents the same unsupported
    mapping-bundle condition now returned as a typed unavailable outcome. Do not
    exempt other ValueErrors or infer support from arbitrary message fragments.
    """
    return status in _SOFTWARE_STATUSES and not (
        status == "generation_error" and str(detail).strip() == _LEGACY_UNSUPPORTED_ENDPOINT
    )


def _inside(ref, work):
    child = Path(ref["path"]).resolve()
    if child == work or not child.is_relative_to(work):
        raise ValueError("scientific payload escapes registered work directory")
    return _read(child, ref["sha256"])


def _result_payload(saved, work, depth=0):
    """Read an external reused result only through its complete original receipt chain."""
    ref = saved["result"]
    if Path(ref["path"]).resolve().is_relative_to(work):
        return _inside(ref, work)
    if depth >= 8 or saved.get("recovery_action") != "reused_original_unchanged":
        raise ValueError("External scientific payload has no bounded unchanged-reuse chain")
    source, old_ref = saved["reuse_source"], saved["reuse_of"]
    completion = _read(source["completion"]["path"], source["completion"]["sha256"])
    if (
        completion.get("status") != "complete"
        or completion.get("exit_code", 0) != 0
        or completion.get("step_id") != source["step_id"]
        or completion.get("dispatch_nonce") != source["nonce"]
    ):
        raise ValueError("Reused scientific completion ownership differs")
    if Path(source["outputs"]["path"]).resolve() != Path(
        source["completion"]["path"]
    ).resolve().with_name("outputs.json"):
        raise ValueError("Reused output index does not belong to completion")
    old_work = Path(completion["work"]).resolve()
    outputs = _read(source["outputs"]["path"], source["outputs"]["sha256"])
    for child in (source["report"], old_ref):
        path = Path(child["path"]).resolve()
        if (
            not path.is_relative_to(old_work)
            or outputs.get(str(path.relative_to(old_work))) != child["sha256"]
        ):
            raise ValueError("Reused report/row is absent from original completed outputs")
    report = _inside(source["report"], old_work)
    if report.get("schema") not in _SCHEMAS or report.get("status") != "complete":
        raise ValueError("Reused source report is not a supported completion")
    rows = report["rows"]
    if not isinstance(rows, list) or len(rows) > _MAX_ROWS:
        raise ValueError("Reused report row metadata is invalid or exceeds 256 rows")
    matched = [
        r for r in rows if r.get("path") == old_ref["path"] and r.get("sha256") == old_ref["sha256"]
    ]
    if len(matched) != 1 or matched[0].get("row_id") != saved["row"]["id"]:
        raise ValueError("Reused row does not belong to original report")
    previous = _inside(old_ref, old_work)
    if previous.get("row") != saved["row"] or previous.get("result") != ref:
        raise ValueError("Reused scientific row/result binding changed")
    return _result_payload(previous, old_work, depth + 1)


def inspect_science(run, completion):
    """Return grouped software failures, or retryable evidence-reading errors.

    Only completed, ownership-matched runs are read. The output index pins the
    report, whose row bindings pin each nested payload. Do not use row IDs/counts
    in retry incident identities: one error must retain identity across shards
    and replacement jobs. ``signature`` contains its stable status/detail pair.
    """
    result = dict(failures=[], errors=[])
    relative = run.get("science_report_relative")
    if not relative or not completion or completion.get("status") != "complete":
        return result
    try:
        if completion.get("step_id") != run["step_id"] or (
            run.get("dispatch_nonce") is not None
            and completion.get("dispatch_nonce") != run["dispatch_nonce"]
        ):
            raise ValueError("scientific completion ownership mismatch")
        work = Path(completion["work"]).resolve()
        path = (work / relative).resolve()
        if path == work or not path.is_relative_to(work):
            raise ValueError("scientific report escapes registered work directory")
        outputs = _read(Path(run["completion_path"]).with_name("outputs.json"))
        digest = outputs.get(str(path.relative_to(work)))
        if not isinstance(digest, str):
            raise ValueError("scientific report missing from completed output index")
        report = _read(path, digest)
        if report.get("schema") == "exact-repair/corrective-study/v1":
            rows = report.get("rows")
            if (
                not isinstance(rows, list)
                or len(rows) > _MAX_ROWS
                or len(rows) != report.get("expected_rows")
                or len({row["id"] for row in rows}) != len(rows)
                or sum(row.get("status") != "not_attempted" for row in rows)
                != report.get("completed_rows")
            ):
                raise ValueError("Corrective scientific denominator differs")
            grouped = {}
            for row in rows:
                inner = row.get("result") or {}
                for status, detail in (
                    (row.get("status"), row.get("detail", "")),
                    (inner.get("status"), inner.get("detail", "")),
                ):
                    if software_failure(status, detail):
                        lines = str(detail).strip().splitlines()
                        message = (
                            lines[-1] if lines else "nested worker reported a software error"
                        ).replace(str(work), "<science-work>")
                        signature = dict(status=status, detail=message[:4096])
                        group = grouped.setdefault(
                            json.dumps(signature, sort_keys=True),
                            dict(
                                signature=signature,
                                row_ids=[],
                                evidence=dict(path=str(path), sha256=digest),
                            ),
                        )
                        if row["id"] not in group["row_ids"]:
                            group["row_ids"].append(row["id"])
            result["failures"] = list(grouped.values())
            return result
        if report.get("schema") not in _SCHEMAS:
            raise ValueError("unsupported declared scientific report schema")
        rows = report.get("rows")
        if not isinstance(rows, list) or len(rows) > _MAX_ROWS:
            raise ValueError("scientific row metadata is invalid or exceeds 256 rows")
        if (
            report.get("status") != "complete"
            or len(rows) != report.get("scheduled")
            or len(rows) != report.get("recorded", len(rows))
        ):
            raise ValueError("scientific completion denominator differs")
        grouped, seen = {}, set()

        for ref in rows:
            saved = _inside(ref, work)
            row_id = saved["row"]["id"]
            if row_id != ref.get("row_id") or row_id in seen:
                raise ValueError("scientific row identity is missing or duplicated")
            seen.add(row_id)
            candidates = [(saved.get("status"), saved.get("detail", ""), ref)]
            if saved.get("result"):
                payload = _result_payload(saved, work)
                if payload.get("row_id") != row_id:
                    raise ValueError("scientific result row identity mismatch")
                candidates.extend(
                    [
                        (payload.get("status"), payload.get("detail", ""), saved["result"]),
                        (
                            payload.get("semantic_status"),
                            payload.get("semantic_detail", ""),
                            saved["result"],
                        ),
                    ]
                )
            for status, detail, evidence in candidates:
                if not software_failure(status, detail):
                    continue
                # Exception type/message, not traceback line numbers or paths,
                # identifies an error when workers include a traceback.
                lines = str(detail).strip().splitlines()
                message = lines[-1] if lines else "nested worker reported a software error"
                message = re.sub(
                    r"(?<![\w.-])" + re.escape(str(work)) + r"(?![\w.-])",
                    "<science-work>",
                    message,
                )
                signature = dict(status=status, detail=message[:4096])
                key = json.dumps(signature, sort_keys=True)
                group = grouped.setdefault(
                    key, dict(signature=signature, row_ids=[], evidence=evidence)
                )
                if row_id not in group["row_ids"]:
                    group["row_ids"].append(row_id)
        result["failures"] = list(grouped.values())
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        result["errors"].append("scientific evidence: " + type(exc).__name__ + ": " + str(exc))
    return result
