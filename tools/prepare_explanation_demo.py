"""Prepare a bounded real backend handoff from existing native contexts and one run.

No matching, evaluation or training is launched. --generate explicitly enables
OpenRouter preparation; subsequent identical requests replay the durable ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exact.runs import RunReader
from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import canonical_hash
from exact_inspect.generation import ExplanationOutput, FactPacket, grounding
from exact_inspect.preparation import Preparation, bind_execution_lock


def main() -> None:
    """Freeze actual inputs, prepare the portable package and retain a claim audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--additional-pairs",
        type=Path,
        help="JSON list of at most four explicit source/target IRI coverage pairs; no scoring",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reader = RunReader.open(args.run)
    records = list(reader.iter_explanations())
    if not 1 <= len(records) <= 12:
        parser.error("The operational demonstration must contain 1..12 saved pairs")
    entities, pairs = {}, []
    for record in records:
        pair = {}
        for side, prefix, name in (("source", "src", "NCIT"), ("target", "tgt", "DOID")):
            entry = {
                "ontology": name,
                "iri": record[prefix + "_iri"],
                "kind": record.get(prefix + "_kind", "class"),
            }
            key = canonical_hash(entry)
            entities[key] = {"id": key, **entry}
            pair[side] = key
        pairs.append(pair)
    if args.additional_pairs:
        additional = json.loads(args.additional_pairs.read_bytes())
        sources = {record["src_iri"] for record in records}
        if not isinstance(additional, list) or len(additional) > 4:
            parser.error("Coverage is limited to four additional explicit pairs")
        for entry in additional:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"source", "target"}
                or entry["source"] not in sources
                or not isinstance(entry["target"], str)
                or not entry["target"].startswith("http://purl.obolibrary.org/obo/DOID_")
            ):
                parser.error("Coverage pairs require a saved source and an explicit DOID IRI")
            pair = {}
            for side, name in (("source", "NCIT"), ("target", "DOID")):
                entity = {"ontology": name, "iri": entry[side], "kind": "class"}
                key = canonical_hash(entity)
                entities[key] = {"id": key, **entity}
                pair[side] = key
            if pair not in pairs:
                pairs.append(pair)
    ontologies = []
    for name, source in (("NCIT", args.source), ("DOID", args.target)):
        manifest_path = args.contexts / name.lower() / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        ontologies.append(
            {
                "name": name,
                "root": {"path": str(source.resolve())},
                "scope": "root",
                "prepared_context": {"path": str(manifest_path.resolve())},
                "source_derivation": manifest.get("source_derivation"),
            }
        )
    template = {
        "design_revision": "explanation-framework/2026-09-24-operational-root-scope",
        "ontologies": ontologies,
        "run": {
            "binding": {"path": str(reader.layout.manifest_path.resolve())},
            "source_ontology": "NCIT",
            "target_ontology": "DOID",
        },
        "entities": list(entities.values()),
        "pairs": pairs if args.generate else [],
    }
    if args.generate:
        template["profile"] = {
            "name": "openrouter_gpt4o_mini_pinned",
            "model": "openai/gpt-4o-mini-2024-07-18",
            "revision": "2024-07-18",
            "routing": {"only": ["OpenAI"], "allow_fallbacks": False, "require_parameters": True},
            "max_tokens": 3000,
            "temperature": 0,
        }
    lock = bind_execution_lock(template, input_root=Path.cwd())
    lock_path = args.output / ("execution-" + canonical_hash(lock)[7:] + ".json")
    atomic_json(lock_path, lock)
    report = Preparation(
        lock, args.output, input_root=Path.cwd(), resume_from=args.resume_from
    ).run()
    # Audit only the generations selected by this execution, not superseded
    # validators or older prompt/model outputs retained for reproducibility.
    selected = {}
    for stage in ("profiles", "comparisons"):
        if stage in report["outputs"]:
            rows = json.loads(
                (args.output / report["outputs"][stage] / (stage + ".json")).read_bytes()
            )
            selected.update((row["explanation_id"], row) for row in rows.values())
    audit = []
    for explanation_id, result in sorted(selected.items()):
        request_path = args.output / "generations" / explanation_id[7:] / "request.json"
        request = json.loads(request_path.read_bytes())
        packet = FactPacket.model_validate(request["packet"])
        status, reasons = grounding(ExplanationOutput(claims=result["claims"]), packet)
        audit.append(
            {
                "explanation_id": result["explanation_id"],
                "task": packet.task,
                "entities": [e.model_dump(mode="json") for e in packet.entities],
                "definition_counts": [
                    sum(
                        f["category"] == "definitions" and f["subject"] == e.model_dump()
                        for f in packet.facts
                    )
                    for e in packet.entities
                ],
                "packet_missingness": packet.missingness,
                "claim_count": len(result["claims"]),
                "grounding_status": status,
                "review_reasons": reasons,
                "generation_status": result["manifest"]["status"],
                "requested_model": result["manifest"]["requested_model"],
                "returned_model": result["manifest"]["returned_model"],
                "response_hash": result["manifest"]["response_hash"],
                "semantic_review": "Exact original excerpts and deterministic conservative comparison templates; free paraphrases are not admitted",
            }
        )
    atomic_json(
        args.output / "claim-audit.json",
        {
            "schema": "exact-explain-claim-audit/1",
            "records": audit,
            "saved_matcher_pair_count": len(records),
            "comparison_worklist_size": len(lock.pairs),
            "additional_pairs": additional if args.additional_pairs else [],
            "all_supported": all(row["grounding_status"] == "validated" for row in audit),
            "nonempty": bool(audit),
            "provider_accounting": provider_accounting(args.output / "generations"),
        },
    )
    print(
        json.dumps(
            {
                "lock": str(lock_path),
                "package": str(args.output / report["outputs"]["portable-export"] / "package.json"),
                "claim_audit_records": len(audit),
            },
            indent=2,
        )
    )


def provider_accounting(root: Path) -> dict:
    """Count actual wire attempts once, including superseded responses and costs."""
    import sqlite3
    from contextlib import closing

    path = root / "provider-ledger" / "requests.sqlite3"
    if not path.exists():
        return {"wire_attempts": 0, "returned_cost_usd": 0, "status": "not_requested"}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT state,usage FROM attempts").fetchall()
        columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
        durations = (
            [
                row[0]
                for row in db.execute(
                    "SELECT elapsed_seconds FROM attempts WHERE elapsed_seconds IS NOT NULL"
                )
            ]
            if "elapsed_seconds" in columns
            else []
        )
    durations.sort()
    usages = [json.loads(usage) for _, usage in rows if usage]
    return {
        "wire_attempts": len(rows),
        "attempt_states": {state: sum(r[0] == state for r in rows) for state, _ in rows},
        "returned_cost_usd": sum(u.get("cost", 0) or 0 for u in usages),
        "attempts_without_returned_cost": sum(
            not u or u.get("cost") is None
            for _, u in ((state, json.loads(usage) if usage else None) for state, usage in rows)
        ),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
        "validator_replays_without_dispatch": len(list(root.glob("*/revalidation.json"))),
        "provider_latency": {
            "measured_attempts": len(durations),
            "unmeasured_attempts": len(rows) - len(durations),
            "total_seconds": sum(durations) if durations else None,
            "mean_seconds": sum(durations) / len(durations) if durations else None,
            "p95_seconds": (
                durations[min(len(durations) - 1, int(len(durations) * 0.95))]
                if durations
                else None
            ),
            "scope": "Actual HTTP attempts only; old attempts remain unmeasured. Replay, validation and read API latency are excluded.",
        },
        "status": "recorded",
        "scope": "All actual wire attempts in this preparation ledger; copied response checkpoints are not charged again.",
    }


if __name__ == "__main__":
    main()
