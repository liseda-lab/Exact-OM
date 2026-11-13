"""Read retained matcher records for a private selection; no generation or bundle copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.prepare_final_study_catalogue import read_shard, sha


def bind_retained_record(pair):
    """Keep scored features and the explicit exact anchor as separate provenance.

    A protected anchor may override scores, but must not erase the scored row's
    original-fact features. Unexplained variants are never silently collapsed.
    """
    variants = pair["records"]
    if len(variants) == 1:
        return {"status": "single_record", "record": variants[0], "exact_anchor": None}
    rows = [
        r
        for r, origins in zip(variants, pair["record_origins"])
        if {o["role"] for o in origins} == {"row"}
    ]
    anchors = [
        r
        for r, origins in zip(variants, pair["record_origins"])
        if {o["role"] for o in origins} == {"anchor"}
    ]
    if (
        len(variants) == 2
        and len(rows) == len(anchors) == 1
        and pair["protected_exact"]
        and anchors[0].get("S_final") == pair["score"]
        and (rows[0].get("Src"), rows[0].get("Tgt"))
        == (anchors[0].get("Src"), anchors[0].get("Tgt"))
    ):
        return {
            "status": "scored_row_plus_exact_anchor",
            "record": {**rows[0], **anchors[0]},
            "exact_anchor": anchors[0],
        }
    return {"status": "unresolved_multiple_records", "record": None, "exact_anchor": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new private output")
    prior = json.loads(args.catalogue.read_bytes())
    inventory = json.loads(args.inventory.read_bytes())
    if inventory["predecessor_sha256"] != sha(args.catalogue):
        raise ValueError("Inventory is not bound to this catalogue")
    requested = args.case_ids.split(",")
    available = {c["case_id"]: c for c in inventory["cases"]}
    if len(set(requested)) != len(requested) or any(cid not in available for cid in requested):
        raise ValueError("Require unique inventoried cases")
    cases = [available[cid] for cid in requested]
    pairs = {(c["source"], t["target"]) for c in cases for t in c["candidates"]}
    identity = prior["native_result"]
    if sha(identity["path"]) != identity["sha256"]:
        raise ValueError("Frozen native result changed")
    result = json.loads(Path(identity["path"]).read_bytes())
    records = {}
    record_origins = {}
    for artifact in result["shards"]:
        receipt = Path(artifact["path"])
        if sha(receipt) != artifact["sha256"]:
            raise ValueError("Native receipt changed")
        data = json.loads(receipt.read_bytes())["data"]
        if sha(data["path"]) != data["sha256"]:
            raise ValueError("Native evidence changed")
        shard = read_shard(data["path"])
        for role, row in [("row", r) for r in shard["rows"]] + [
            ("anchor", r) for r in shard.get("anchors", [])
        ]:
            pair = row["Src"], row["Tgt"]
            if pair not in pairs:
                continue
            variants = records.setdefault(pair, [])
            origins = record_origins.setdefault(pair, [])
            if row not in variants:
                variants.append(row)
                origins.append([])
            origins[variants.index(row)].append({"receipt": artifact, "role": role})
    output = {
        "schema": "exact-final-study-retained-evidence/1",
        "private": True,
        "native_result": identity,
        "inventory_sha256": sha(args.inventory),
        "status": "retained_records_not_workspace_admission",
        "cases": {},
    }
    for case in cases:
        output["cases"][case["case_id"]] = [
            {
                "target": t["target"],
                "original_rank": t["original_rank"],
                "score": t["score"],
                "records": records.get((case["source"], t["target"]), []),
                "record_origins": record_origins.get((case["source"], t["target"]), []),
                "protected_exact": t.get("protected_exact", False),
            }
            for t in case["candidates"]
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x") as stream:
        json.dump(output, stream, ensure_ascii=False)
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "pairs": len(pairs),
                "recorded": len(records),
                "multiple_retained_variants": sum(len(v) > 1 for v in records.values()),
                "output_bytes": args.output.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
