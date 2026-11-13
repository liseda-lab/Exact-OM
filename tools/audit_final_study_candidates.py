"""Private difficulty screening of unchanged final-run five-candidate sets.

Lexical heuristics are triage, not expert difficulty measurements or medical
adjudication. The reference is used only for private study selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from tools.prepare_final_study_catalogue import (
    context_inventory,
    eligible_queries,
    read_shard,
    sha,
)


def label_key(value):
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold())
    return " ".join(
        {"disorder": "disease", "disorders": "disease", "diseases": "disease"}.get(w, w)
        for w in words
    )


def lexical_screen(case):
    source = label_key(case["source_summary"]["label"])
    labels = [label_key(c["context_summary"]["label"]) for c in case["candidates"]]
    answer_index = case["reference_rank"] - 1
    # Reordering the same words is just as easy as repeating them. Preserve all
    # qualifiers, numbers and negation; this is not a medical synonym expander.
    bags = [Counter(label.split()) for label in labels]
    source_bag = Counter(source.split())
    twins = [i + 1 for i, bag in enumerate(bags) if bag == source_bag]
    trivial = answer_index + 1 in twins and len(twins) == 1
    generic = {"disease", "syndrome", "of", "the", "a", "an", "and", "type", "system"}
    words = lambda label: set(label.split()) - generic
    similarities = []
    for index, label in enumerate(labels):
        if index == answer_index:
            continue
        left, right = words(labels[answer_index]), words(label)
        overlap = len(left & right) / max(1, len(left | right))
        similar = SequenceMatcher(None, labels[answer_index], label).ratio()
        similarities.append(
            {
                "rank": index + 1,
                "token_overlap": round(overlap, 4),
                "character_similarity": round(similar, 4),
                "family_overlap": bool(left & right),
            }
        )
    competitors = sum(
        row["family_overlap"]
        and (row["token_overlap"] >= 0.25 or row["character_similarity"] >= 0.6)
        for row in similarities
    )
    infos = [case["source_summary"], *(c["context_summary"] for c in case["candidates"])]
    definitions = [bool(info["counts"].get("definitions")) for info in infos]
    return {
        "screen_version": "challenge-triage/2",
        "obvious_reference_label_twin": trivial,
        "label_twin_candidate_ranks": twins,
        "lexical_competitor_count": competitors,
        "candidate_similarities_to_reference": similarities,
        "entities_with_original_definitions": sum(definitions),
        "source_and_reference_defined": definitions[0] and definitions[answer_index + 1],
        "top_two_score_gap": case["candidates"][0]["score"] - case["candidates"][1]["score"],
        "manual_semantic_review_required": True,
    }


def detail_catalogue(path, case_ids):
    """Read original, policy-filtered facts for a private replacement shortlist."""
    from exact_inspect.context import OntologyContext
    from exact_inspect.contracts import EntityRef, VisibilityPolicy

    data = json.loads(path.read_text())
    cases = {c["case_id"]: c for c in data["cases"]}
    requested = case_ids.split(",")
    if len(set(requested)) != len(requested) or any(i not in cases for i in requested):
        raise ValueError("Require unique case identities from the verified inventory")
    selected = [cases[i] for i in requested]
    contexts = {}
    for side, identity in data["input_bindings"]["contexts"].items():
        manifest = Path(identity["manifest"]["path"])
        if sha(manifest) != identity["manifest"]["sha256"]:
            raise ValueError("Original context manifest changed")
        context = OntologyContext(manifest.parent)
        if context.manifest["artifacts"]["context.sqlite"] != identity["database_sha256"]:
            raise ValueError("Original context database changed")
        contexts[side] = context
    policy = VisibilityPolicy(
        policy_id="final-study-review-20261110",
        ontology_ids=tuple(c.ontology_version_id for c in contexts.values()),
    )
    entities = {}
    for side, iris in [
        ("source", {c["source"] for c in selected}),
        ("target", {t["target"] for c in selected for t in c["candidates"]}),
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
    for case in selected:
        case["challenge_screen"] = lexical_screen(case)
        for candidate in case["candidates"]:
            candidate["matcher_record"] = None
            candidate["evidence_status"] = "not_extracted_for_shortlist"
    return {
        "schema": "exact-final-study-challenge-shortlist/1",
        "private": True,
        "status": "replacement_proposals_not_final_pool",
        "inventory_sha256": sha(path),
        "input_bindings": data["input_bindings"],
        "cases": selected,
        "entities": entities,
        "limitations": [
            "Replacement proposals, not a new published study pool. Original five candidates, scores and ranks are unchanged.",
            "Difficulty screening is heuristic; expert suitability and reference-equivalence review remain separate.",
            "Original facts only. Matcher evidence admission, generated prose and inferred hierarchy are not prepared for these new cases.",
            "Reference identities are private. They must never enter provider packets or participant materials.",
        ],
    }


def inventory(catalogue):
    prior = json.loads(catalogue.read_text())
    for field in ("run", "submission", "reference", "input_manifest"):
        if sha(prior[field]["path"]) != prior[field]["sha256"]:
            raise ValueError(f"Frozen {field} changed")
    receipt = json.loads(Path(prior["run"]["path"]).read_text())
    queries = []
    for artifact in receipt["outputs"]:
        path = Path(artifact["path"])
        if sha(path) != artifact["sha256"]:
            raise ValueError("Output receipt changed")
        record = json.loads(path.read_text())
        shard = path.with_name(path.name.replace(".receipt.json", ".json.zst"))
        bound = next(
            v
            for v in record.values()
            if isinstance(v, dict)
            and v.get("path")
            and Path(v["path"]).resolve() == shard.resolve()
        )
        if sha(shard) != bound["sha256"]:
            raise ValueError("Final ranking shard changed")
        queries.extend(read_shard(shard)["queries"])
    refs = list(csv.DictReader(open(prior["reference"]["path"]), delimiter="\t"))
    cases, excluded = eligible_queries(queries, refs)
    info = {}
    for side, iris in [
        ("source", {c["source"] for c in cases}),
        ("target", {x["target"] for c in cases for x in c["candidates"]}),
    ]:
        manifest = prior["contexts"][side]["manifest"]
        if sha(manifest["path"]) != manifest["sha256"]:
            raise ValueError("Original context manifest changed")
        context_path = Path(manifest["path"]).parent
        if sha(context_path / "context.sqlite") != prior["contexts"][side][
            "database_sha256"
        ].removeprefix("sha256:"):
            raise ValueError("Original context database changed")
        info[side] = context_inventory(context_path, iris)
    selected = set(prior["options"]["forty_unequal"])
    for c in cases:
        c["source_summary"] = info["source"][c["source"]]
        for row in c["candidates"]:
            row["context_summary"] = info["target"][row["target"]]
        c["previously_selected"] = c["case_id"] in selected
        c["challenge_screen"] = lexical_screen(c)
    return {
        "schema": "exact-final-study-challenge-inventory/1",
        "private": True,
        "predecessor_sha256": sha(catalogue),
        "input_bindings": {k: prior[k] for k in ("run", "submission", "reference", "contexts")},
        "excluded": excluded,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--catalogue", type=Path)
    source.add_argument("--inventory", type=Path)
    parser.add_argument("--case-ids", help="Comma-separated private shortlist, with --inventory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new private output, preserving previous reviews")
    if bool(args.inventory) != bool(args.case_ids):
        parser.error("--inventory and --case-ids must be used together")
    data = (
        detail_catalogue(args.inventory, args.case_ids)
        if args.inventory
        else inventory(args.catalogue)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False)
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "cases": len(data["cases"]),
                "label_twins": dict(
                    Counter(
                        c["reference_rank"]
                        for c in data["cases"]
                        if c["challenge_screen"]["obvious_reference_label_twin"]
                    )
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
