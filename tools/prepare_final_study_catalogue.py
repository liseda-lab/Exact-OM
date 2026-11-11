"""Private owner review of a frozen Exact 2 local run; never a participant asset.

Reference answers are used ONLY to form study strata. No fitting, inference, provider
calls, score editing or candidate replacement is performed. Outputs are created exclusively.
Run on the data server, with the completed run, its matching gold file and verified contexts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read_shard(path):
    import zstandard

    with Path(path).open("rb") as stream:
        return json.load(zstandard.ZstdDecompressor().stream_reader(stream))


def eligible_queries(queries, references):
    """A source may have several queries; retain one original query, never merge lists."""
    by_source = defaultdict(set)
    for ref in references:
        by_source[ref["SrcEntity"]].add(ref["TgtEntity"])
    result, excluded = [], Counter()
    for query in sorted(queries, key=lambda q: q["qid"]):
        source = query["source"]
        candidates = query["candidates"][:5]
        targets = [c["target"] for c in candidates]
        if len(set(targets)) != 5:
            excluded["fewer_than_five_unique_candidates"] += 1
            continue
        accepted = by_source[source] & set(targets)
        if len(accepted) != 1:
            excluded["no_reference_in_five" if not accepted else "multiple_references_in_five"] += 1
            continue
        if any(not math.isfinite(c["score"]) for c in candidates):
            raise ValueError("Nonfinite final score")
        answer = next(iter(accepted))
        result.append(
            {
                "case_id": f"ncit-doid-q{query['qid']:04d}",
                "query_id": query["qid"],
                "source": source,
                "reference_rank": targets.index(answer) + 1,
                "reference_target": answer,
                "other_reference_targets_outside_five": sorted(by_source[source] - accepted),
                "candidates": [dict(c, original_rank=i + 1) for i, c in enumerate(candidates)],
            }
        )
    return result, dict(excluded)


def choose_pool(cases, quotas):
    """Rare strata first; stable quality ordering, unique sources, untouched candidates."""
    chosen, used = [], set()
    for rank in sorted(quotas, key=lambda r: (sum(c["reference_rank"] == r for c in cases), r)):
        pool = sorted(
            (c for c in cases if c["reference_rank"] == rank and c["source"] not in used),
            key=lambda c: (tuple(-v for v in c["quality"]), c["source"], c["query_id"]),
        )
        for case in pool:
            if case["source"] in used:
                continue
            chosen.append(case)
            used.add(case["source"])
            if sum(c["reference_rank"] == rank for c in chosen) == quotas[rank]:
                break
        if sum(c["reference_rank"] == rank for c in chosen) != quotas[rank]:
            raise ValueError(
                f"Not enough distinct sources for rank {rank}: requested {quotas[rank]}"
            )
    return sorted(
        chosen, key=lambda c: (c["reference_rank"], tuple(-v for v in c["quality"]), c["source"])
    )


def context_inventory(path, iris):
    """Small indexed reads; no copied database, inferred absence or hidden xrefs."""
    result = {}
    with sqlite3.connect(f"file:{Path(path).resolve()}/context.sqlite?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        for iri in sorted(iris):
            counts = dict(
                db.execute(
                    "SELECT a.category,count(*) FROM refs r JOIN axioms a ON a.id=r.axiom_id "
                    "WHERE r.iri=? AND r.kind='class' AND a.category IN "
                    "('definitions','hierarchy','restrictions','synonyms') GROUP BY a.category",
                    (iri,),
                )
            )
            label = db.execute(
                "SELECT term FROM terms WHERE iri=? AND kind='class' AND category='labels' "
                "ORDER BY CASE language WHEN 'en' THEN 0 WHEN '' THEN 1 ELSE 2 END,term LIMIT 1",
                (iri,),
            ).fetchone()
            present = db.execute(
                "SELECT 1 FROM entities WHERE iri=? AND kind='class'", (iri,)
            ).fetchone()
            result[iri] = {
                "label": label[0] if label else iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1],
                "counts": counts,
                "present": bool(present),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="H0 directory")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-context", type=Path, required=True)
    parser.add_argument("--target-context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory; never overwrite a review")
    receipt_path = args.run / "local-apply/submission-complete.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "complete" or receipt["reference_labels_loaded"]:
        raise ValueError("Require a completed, reference-blind matcher run")
    recipe = json.loads((args.run / "local-apply/recipe.json").read_text())
    input_path = Path(recipe["manifest"]["path"])
    if sha(input_path) != recipe["manifest"]["sha256"]:
        raise ValueError("Frozen input manifest changed")
    task = next(t for t in json.loads(input_path.read_text())["tasks"] if t["id"] == recipe["task"])
    for side, path in [("source", args.source_context), ("target", args.target_context)]:
        manifest = json.loads((path / "manifest.json").read_text())
        if (
            manifest["identity"]["root_sha256"].removeprefix("sha256:")
            != task["ontology"][side]["sha256"]
        ):
            raise ValueError("Context does not match the ontology in this run")
    queries = []
    for artifact in receipt["outputs"]:
        p = Path(artifact["path"])
        if sha(p) != artifact["sha256"]:
            raise ValueError("Output receipt changed")
        shard_receipt = json.loads(p.read_text())
        # The receipt binds the compressed output; do not trust a sibling by filename alone.
        candidates = [
            v
            for v in shard_receipt.values()
            if isinstance(v, dict) and "path" in v and "sha256" in v
        ]
        shard = p.with_name(p.name.replace(".receipt.json", ".json.zst"))
        bound = next((v for v in candidates if Path(v["path"]).resolve() == shard.resolve()), None)
        if bound is None or sha(shard) != bound["sha256"]:
            raise ValueError(f"Unbound/changed output shard {p}")
        queries.extend(read_shard(shard)["queries"])
    queries.sort(key=lambda q: q["qid"])
    references = list(csv.DictReader(args.reference.open(), delimiter="\t"))
    if len(queries) != len(references) or [q["qid"] for q in queries] != list(range(len(queries))):
        raise ValueError("Query/reference count or identities differ")
    if any(q["source"] != r["SrcEntity"] for q, r in zip(queries, references)):
        raise ValueError("Reference/query source order differs")
    submission = args.run / "local-apply/ncit-doid.tsv"
    submitted = list(csv.DictReader(submission.open(), delimiter="\t"))
    if len(submitted) != len(queries) or any(
        q["source"] != row["SrcEntity"]
        or [c["target"] for c in q["candidates"]] != ast.literal_eval(row["TgtCandidates"])
        for q, row in zip(queries, submitted)
    ):
        raise ValueError("Scored query order differs from the submitted final ranking")
    cases, excluded = eligible_queries(queries, references)
    print("Validated final ranks; checking original ontology coverage", flush=True)
    source_info = context_inventory(args.source_context, {c["source"] for c in cases})
    target_info = context_inventory(
        args.target_context, {t["target"] for c in cases for t in c["candidates"]}
    )
    for case in cases:
        infos = [
            source_info[case["source"]],
            *(target_info[c["target"]] for c in case["candidates"]),
        ]
        case["source_summary"] = infos[0]
        for c, info in zip(case["candidates"], infos[1:]):
            c["context_summary"] = info
        case["quality"] = [
            sum(i["counts"].get("definitions", 0) > 0 for i in infos),
            sum(i["counts"].get("hierarchy", 0) > 0 for i in infos),
            sum(i["counts"].get("restrictions", 0) > 0 for i in infos),
            sum(min(12, i["counts"].get("hierarchy", 0)) for i in infos),
            sum(min(6, i["counts"].get("synonyms", 0)) for i in infos),
        ]
    missing = [
        c
        for c in cases
        if not c["source_summary"]["present"]
        or any(not t["context_summary"]["present"] for t in c["candidates"])
    ]
    cases = [c for c in cases if c not in missing]
    excluded["entity_missing_from_context"] = len(missing)
    capacities = {
        rank: len({c["source"] for c in cases if c["reference_rank"] == rank})
        for rank in range(1, 6)
    }
    per_rank = min(8, min(capacities.values()))
    balanced = choose_pool(cases, {r: per_rank for r in capacities})
    quotas = {r: min(8, n) for r, n in capacities.items()}
    while sum(quotas.values()) < 40:
        available = [r for r in quotas if quotas[r] < capacities[r]]
        if not available:
            break
        quotas[min(available, key=lambda r: (quotas[r], r))] += 1
    pool40 = choose_pool(cases, quotas)
    selected = {c["case_id"]: c for c in balanced + pool40}
    selected_pairs = {
        (c["source"], t["target"]) for c in selected.values() for t in c["candidates"]
    }
    print("Selected candidate pools; loading retained matcher evidence", flush=True)
    native = {}
    native_result_path = args.run / "local-native/test/result.json"
    native_result = json.loads(native_result_path.read_text())
    for artifact in native_result["shards"]:
        rp = Path(artifact["path"])
        if sha(rp) != artifact["sha256"]:
            raise ValueError("Native receipt changed")
        shard_ref = json.loads(rp.read_text())["data"]
        p = Path(shard_ref["path"])
        if sha(p) != shard_ref["sha256"]:
            raise ValueError("Native evidence shard changed")
        shard = read_shard(p)
        for row in shard["rows"] + shard.get("anchors", []):
            pair = row["Src"], row["Tgt"]
            if pair in selected_pairs:
                native[pair] = row
    # Import the established policy-aware context reader only after the small selection.
    from exact_inspect.context import OntologyContext
    from exact_inspect.contracts import EntityRef, VisibilityPolicy

    print("Reading verified original facts for the owner catalogue", flush=True)
    contexts = {
        "source": OntologyContext(args.source_context),
        "target": OntologyContext(args.target_context),
    }
    policy = VisibilityPolicy(
        policy_id="final-study-review-20261110",
        ontology_ids=tuple(c.ontology_version_id for c in contexts.values()),
    )
    entities = {}
    for side, iris in [
        ("source", {c["source"] for c in selected.values()}),
        ("target", {t["target"] for c in selected.values() for t in c["candidates"]}),
    ]:
        context = contexts[side]
        for iri in sorted(iris):
            ref = EntityRef(ontology_version_id=context.ontology_version_id, iri=iri, kind="class")
            facts = {}
            for category in ["labels", "definitions", "hierarchy", "restrictions", "synonyms"]:
                page = context.facts(ref, category=category, limit=20, policy=policy)
                facts[category] = {
                    "items": page["items"],
                    "has_more": bool(page.get("next_cursor")),
                }
            entities[iri] = {
                "entity": ref.model_dump(mode="json"),
                "facts": facts,
                "generated_description": None,
                "generated_status": "not_prepared_for_this_selection",
            }
    for case in selected.values():
        for candidate in case["candidates"]:
            row = native.get((case["source"], candidate["target"]))
            candidate["matcher_record"] = row
            candidate["evidence_status"] = (
                "recorded_selected_evidence"
                if row and row.get("selected_attributions")
                else "exact_match" if candidate.get("protected_exact") else "not_recorded"
            )
    args.output.mkdir(parents=True, mode=0o700)
    data = {
        "schema": "exact-final-study-owner-review/1",
        "private": True,
        "status": "owner_selection_review_not_published",
        "run": binding(receipt_path),
        "submission": binding(submission),
        "reference": binding(args.reference),
        "input_manifest": binding(input_path),
        "native_result": binding(native_result_path),
        "contexts": {
            side: {
                "manifest": binding(context.path / "manifest.json"),
                "ontology_version_id": context.ontology_version_id,
                "source_sha256": context.manifest["identity"]["root_sha256"],
                "database_sha256": context.manifest["artifacts"]["context.sqlite"],
            }
            for side, context in contexts.items()
        },
        "policy": policy.model_dump(mode="json"),
        "query_count": len(queries),
        "excluded_queries": excluded,
        "eligible_distinct_sources_by_rank": capacities,
        "requested_balanced_40_feasible": all(n >= 8 for n in capacities.values()),
        "quality_order": [
            "entities_with_definitions",
            "entities_with_hierarchy",
            "entities_with_restrictions",
            "bounded_hierarchy_count",
            "bounded_synonym_count",
        ],
        "options": {
            "balanced": [c["case_id"] for c in balanced],
            "forty_unequal": [c["case_id"] for c in pool40],
        },
        "option_quotas": {"balanced": {r: per_rank for r in capacities}, "forty_unequal": quotas},
        "cases": list(selected.values()),
        "entities": entities,
        "limitations": [
            "Purposive case selection supports a controlled tool comparison, not an estimate of overall matcher accuracy.",
            "Reference answers are private selection data and never enter generation or participant resources.",
            "Multiple-reference sources are eligible only when exactly one known reference occurs in the untouched five candidates; other references are retained privately.",
            "Related sources and repeated candidate concepts still need review for learning/transfer effects.",
            "Recorded matcher evidence is not yet admitted into the study workspace; original-fact links and generated prose still require preparation and validation.",
            "Original ontology facts are bounded excerpts from matching root documents; an absent excerpt is not a claim of no ontology information.",
        ],
    }
    with (args.output / "catalogue.json").open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "eligible": capacities,
                "quotas": data["option_quotas"],
                "cases_for_review": len(selected),
                "entities": len(entities),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
