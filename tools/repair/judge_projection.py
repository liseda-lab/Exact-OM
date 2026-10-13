"""Deterministic, provenance-bound views of complete semantic evidence.

Original packets remain immutable. Unary views retain common meaning and only
the selected theory's facts; pair views explicitly name each fact's owner.
No semantic summary, new logical conclusion or quality label is generated here.
"""

import json

from exact.repair.records import canonical_hash

VERSION = "exact-repair/judge-evidence-projection/v1"
THEORY_PREFIX = "Complete reconstructed theory:\n"
INDEX_PREFIX = "Checked fact index. Plan IDs identify complete theories; ENTAILS and SATISFIABLE answer different questions.\n"


def structured_evidence(row):
    """Decode native JSON once, rejecting contradictory outer/inner metadata."""
    if not row["kind"].startswith("native_"):
        return dict(row)
    result = dict(row)
    if "text" in result:
        fact = json.loads(result.pop("text"))
        if not isinstance(fact, dict):
            raise ValueError("Native evidence must be a structured fact")
        for key, value in fact.items():
            if key in result and result[key] != value:
                raise ValueError("Contradictory native evidence metadata: " + key)
            result[key] = value
    return result


def typed_plan(plan):
    """Preserve the complete theory and distinguish selected/original bundles."""
    current, original, metadata, theories = [], [], [], []
    for item in plan["content"]:
        if item.startswith(THEORY_PREFIX):
            theories.append(item[len(THEORY_PREFIX) :])
            continue
        row = json.loads(item)
        if not isinstance(row, dict):
            raise ValueError("Unrecognized plan content")
        if "original" in row or "selected" in row:
            if not {"original", "selected", "object_id"} <= row.keys():
                raise ValueError("Incomplete original/selected object provenance")
            identity = {k: v for k, v in row.items() if k not in ("original", "selected")}
            current.append({**identity, "selected_axioms": row["selected"]})
            original.append({**identity, "observed_axioms": row["original"]})
        else:
            metadata.append(row)
    if len(theories) != 1:
        raise ValueError("Exactly one complete current theory is required")
    return {
        **{k: v for k, v in plan.items() if k != "content"},
        "current_theory": theories[0],
        "current_revision_objects": current,
        "original_observations": {
            "role": "fallible history; not current assertions or meaning authority",
            "objects": original,
        },
        "verification_metadata": metadata,
        "source_plan_hash": canonical_hash(plan),
    }


def project(payload, side=None):
    plans = {s: payload["plan_" + s] for s in ("a", "b")}
    by_id = {}
    for plan in plans.values():
        previous = by_id.setdefault(plan["plan_id"], plan)
        if previous != plan:
            raise ValueError("One plan identity refers to different complete theories")
    context = payload.get("context", {})
    common, native, pair = {}, {}, {}
    for key, source in payload["evidence"].items():
        row = structured_evidence(source)
        owner = row.get("plan_id")
        if row["kind"] == "native_asserted_text_identity":
            for position in ("first", "second"):
                pid = row[position + "_plan_id"]
                if (
                    pid not in by_id
                    or row[position + "_plan_theory_hash"] != by_id[pid]["theory_hash"]
                ):
                    raise ValueError("Comparison fact belongs to a different theory")
            pair[key] = row
        elif owner is not None:
            if owner not in by_id:
                raise ValueError("Native evidence belongs to another plan outside this case")
            plan = by_id[owner]
            expected = {
                "theory_hash": plan["theory_hash"],
                "policy_hash": plan.get("policy_hash", context.get("policy_hash")),
                "query_basis_hash": plan.get("query_basis_hash", context.get("query_basis_hash")),
                "verification_report_id": plan.get("report_id"),
            }
            for field, value in expected.items():
                if field in row and value is not None and row[field] != value:
                    raise ValueError("Native evidence context mismatch: " + field)
            if (
                "symbolic_value" in row
                and row.get("outcome", row.get("status")) != row["symbolic_value"]
            ):
                raise ValueError("Native evidence outcome disagrees with source")
            native[key] = row
        elif row["kind"].startswith("native_"):
            raise ValueError("Native evidence has no qualified plan owner")
        else:
            common[key] = row
    # This legacy field duplicates native facts as prose, including both owners.
    # Keep arbitrary local context; omit only the explicitly identified index.
    local = payload.get("local_context", [])
    retained_local = [x for x in local if not x.startswith(INDEX_PREFIX)]
    selected = set(by_id) if side is None else {plans[side]["plan_id"]}
    evidence = {**common, **{k: v for k, v in native.items() if v["plan_id"] in selected}}
    if side is None:
        evidence.update(pair)
        for row in evidence.values():
            if "plan_id" in row:
                row["applies_to"] = [
                    "plan_" + s for s, p in plans.items() if p["plan_id"] == row["plan_id"]
                ]
    view = {
        **{
            k: v
            for k, v in payload.items()
            if k not in ("plan_a", "plan_b", "evidence", "context", "local_context")
        },
        "evidence": evidence,
        "local_context": retained_local,
        "projection": {
            "version": VERSION,
            "source_payload_hash": canonical_hash(payload),
            "source_evidence_hash": canonical_hash(payload["evidence"]),
            "source_local_context_hash": canonical_hash(local),
            "excluded_evidence_ids": sorted(set(payload["evidence"]) - set(evidence)),
            "excluded_reason": "other-plan and pair-only facts are outside a unary assessment",
            "duplicate_fact_indexes_removed": len(local) - len(retained_local),
            "complete_current_theories_preserved": True,
        },
    }
    for s in (side,) if side is not None else ("a", "b"):
        plan = typed_plan(plans[s])
        plan["native_evidence_ids"] = [
            k for k, v in native.items() if v["plan_id"] == plan["plan_id"]
        ]
        view["plan" if side is not None else "plan_" + s] = plan
    if side is None:
        view["context"] = context
    else:
        view["source_context"] = {
            k: v for k, v in context.items() if k not in ("plan_a_id", "plan_b_id")
        }
    return view
