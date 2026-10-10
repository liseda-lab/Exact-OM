"""Read-only independent audit of the final typed-renderer review publication.

The separate audit rules below do not call the production template renderer or
grounding matcher. Production admission is an additional check, not the oracle.
This creates no provider requests or independent provider-prose approvals.

Run with the maintained runtime checkout explicitly selected::

    python tools/audit_review_publication.py --runtime-root /path/to/Exact-OM \
        /path/to/review-input /path/to/new-audit-receipt.json

The output and adjacent catalogue files must be outside the immutable input.
Each receipt binds this script and the imported runtime module bytes. Keep earlier
receipts and choose a new output path for a repeat verification. If the portable
service input omits the public selection metadata, supply its separately delivered
hash-bound copy with --selection-file /path/to/selection.json.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_runtime_root = None


def load_runtime(root):
    """Select installed verification code without an expiring preparation path."""
    global _runtime_root, OntologyContext, VisibilityPolicy, canonical_hash, file_hash
    global validate_prepared_support, PairEvidenceMetadata, ExplanationResource
    root = root.resolve(strict=True)
    if not (root / "exact_inspect" / "__init__.py").is_file():
        raise ValueError("--runtime-root must name the Exact-OM repository checkout")
    sys.path.insert(0, str(root))
    from exact_inspect.context import OntologyContext
    from exact_inspect.contracts import VisibilityPolicy, canonical_hash, file_hash
    from exact_inspect.descriptions import validate_prepared_support
    from exact_inspect.evidence_metadata import PairEvidenceMetadata
    from exact_inspect.study.resources import ExplanationResource

    _runtime_root = root
    runtime_identity()  # Reject pre-imported modules from a different checkout.


def runtime_identity():
    modules = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("exact_inspect") and path is not None:
            path = Path(path).resolve()
            if not path.is_relative_to(_runtime_root):
                raise ValueError("Imported runtime module is outside --runtime-root")
            modules[str(path.relative_to(_runtime_root))] = file_hash(path)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_runtime_root, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"checkout_commit": commit, "imported_module_sha256": modules}


def named(node, allowed):
    if not isinstance(node, dict) or node.get("type") not in allowed:
        return None
    value = node.get("iri", {}).get("value")
    return value if isinstance(value, str) and value else None


def expression(node, depth=0):
    """Independent exact wording and the OWL semantic obligations it preserves."""
    assert isinstance(node, dict) and depth <= 8, "Unbounded/invalid expression"
    kind = node.get("type")
    value = named(node, {"Class", "ObjectProperty", "DataProperty", "Datatype", "NamedIndividual"})
    if value:
        return f"<{value}>", {"exact_named_IRI"}
    if kind == "ObjectInverseOf":
        text, rules = expression(node["property"], depth + 1)
        return "inverse of " + text, rules | {"inverse_property_direction"}
    if kind in {"ObjectIntersectionOf", "ObjectUnionOf"}:
        values = [expression(n, depth + 1) for n in node["operands"]]
        assert len(values) >= 2
        join = " and " if kind == "ObjectIntersectionOf" else " or "
        return "(" + join.join(t for t, _ in values) + ")", set().union(*(r for _, r in values)) | {
            kind
        }
    if kind == "ObjectComplementOf":
        text, rules = expression(node["operand"], depth + 1)
        return f"outside ({text})", rules | {"class_complement"}
    property_text, property_rules = expression(node["property"], depth + 1)
    filler_text, filler_rules = expression(node["filler"], depth + 1)
    rules = property_rules | filler_rules
    if kind == "ObjectSomeValuesFrom":
        return f"at least one {property_text} relationship to {filler_text}", rules | {
            "existential_witness"
        }
    if kind == "ObjectAllValuesFrom":
        return (
            f"only {filler_text} as {property_text} values (without requiring any such value)",
            rules | {"universal_without_existential_import"},
        )
    quantifier = {
        "ObjectMinCardinality": "at least",
        "ObjectMaxCardinality": "at most",
        "ObjectExactCardinality": "exactly",
    }[kind]
    n = node["cardinality"]
    assert type(n) is int and n >= 0
    return (
        f"{quantifier} {n} distinct {property_text} values belonging to {filler_text}",
        rules | {"qualified_distinct_cardinality", kind},
    )


def assess_claim(claim, result):
    packet = result["support_packet"]
    by_id = {f["fact_id"]: f for f in packet["facts"]}
    cited = [by_id[fid] for fid in claim["fact_ids"]]
    assert cited and len(claim["fact_ids"]) == len(
        set(claim["fact_ids"])
    ), "Missing/duplicate citations"
    if len(cited) == 1:
        fact = cited[0]
        value = fact.get("value") or {}
        if value.get("term_type") == "literal":
            if claim["text"] == value["lexical_form"] and claim["category"] == "key_fact":
                return "exact_original_literal", {"no_interpretive_extension"}
            if fact["category"] in {"definitions", "alternate_definitions"}:
                expected = "The ontology describes this entity as: “" + value["lexical_form"] + "”."
                if claim["text"] == expected and claim["category"] == "meaning":
                    return "attributed_original_definition", {
                        "exact_quotation",
                        "ontology_attribution",
                    }
        ast = value.get("ast") or {}
        if fact["subject"]["kind"] == "class" and claim["category"] == "scope":
            iri = fact["subject"]["iri"]
            if ast.get("type") == "SubClassOf" and named(ast["sub_class"], {"Class"}) == iri:
                text, rules = expression(ast["super_class"])
                if (
                    claim["text"]
                    == "Every instance satisfies this recorded necessary condition: " + text + "."
                ):
                    return "asserted_subclass_necessary_condition", rules | {
                        "necessary_not_sufficient",
                        "recorded_not_inferred",
                    }
            if ast.get("type") == "EquivalentClasses":
                values = ast["expressions"]
                assert any(named(n, {"Class"}) == iri for n in values)
                other = [expression(n) for n in values if named(n, {"Class"}) != iri]
                expected = (
                    "The ontology records class equivalence with: "
                    + "; ".join(t for t, _ in other)
                    + "."
                )
                if other and claim["text"] == expected:
                    return "asserted_within_ontology_class_equivalence", set().union(
                        *(r for _, r in other)
                    ) | {"recorded_equivalence_not_cross_ontology_mapping"}
    if result["task"] == "pair_comparison":
        defs = [
            next(
                (
                    f
                    for f in packet["facts"]
                    if f["subject"] == e
                    and f["category"] == "definitions"
                    and (f.get("value") or {}).get("term_type") == "literal"
                ),
                None,
            )
            for e in result["entities"]
        ]
        a, b = defs
        if a and b:
            source, target = a["value"]["lexical_form"], b["value"]["lexical_form"]
            excerpt = lambda text: text if len(text) <= 180 else text[:180] + "…"  # noqa: E731
            expected = f"Source definition excerpt: “{excerpt(source)}”. Target definition excerpt: “{excerpt(target)}”. These excerpts alone do not establish equivalence or incompatibility."
            if (
                claim["text"] == expected
                and claim["fact_ids"] == [a["fact_id"], b["fact_id"]]
                and claim["category"] == ("agreement" if source == target else "difference")
            ):
                return "ordered_definition_excerpts", {
                    "ordered_source_target_roles",
                    "explicit_excerpt_truncation",
                    "no_equivalence_or_incompatibility_inference",
                }
        elif a or b:
            side, other, fact = ("source", "target", a) if a else ("target", "source", b)
            expected = f"The {side} has a recorded definition in this fact packet; the {other} does not. One-sided information does not establish incompatibility."
            if (
                claim["text"] == expected
                and claim["fact_ids"] == [fact["fact_id"]]
                and claim["category"] == "difference"
            ):
                return "scoped_packet_missingness", {
                    "packet_absence_not_global_absence",
                    "no_incompatibility_inference",
                }
    raise AssertionError("Claim does not match an independently checked admitted rendering rule")


def contained(root, name):
    path = (root / name).resolve()
    assert path.is_relative_to(root.resolve()), "Path escapes bound artifact root"
    return path


def verify_label(entity, label, contexts, policy):
    rows = contexts[entity["ontology_version_id"]].labels([entity["iri"]], policy=policy)
    matching = [
        r for r in rows if r["entity"] == entity and r["preferred_label"]["status"] == "available"
    ]
    expected = matching[0]["preferred_label"]["value"] if matching else entity["iri"]
    assert label == expected[:512], "Display label differs from admitted preferred label"


def catalogue(report, output):
    """Owner-readable safe inventory: no keys, invitations or provider data."""
    value = {
        k: report[k]
        for k in (
            "recorded_at",
            "publication_sha256",
            "package_manifest_sha256",
            "selection_sha256",
            "policy_sha256",
            "case_count",
            "pair_count",
            "unique_results",
            "unique_claims",
            "claim_methods",
            "coverage",
            "missing_coverage",
            "inference_status",
            "evidence_coverage",
            "limitations",
            "cases",
        )
    }
    value.update(
        schema="exact-reviewed-case-catalogue/1",
        audit_receipt_sha256=file_hash(output),
        score_meaning="Saved matcher S_final; not probability of semantic equivalence.",
        position_meaning="Display position is the shown order. Saved source-decision row position is the recorded producer field, not a reconstructed production rank.",
    )
    json_path = output.with_name(output.stem + "-catalogue.json")
    md_path = output.with_name(output.stem + "-catalogue.md")
    json_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def cell(text):
        return str(text).replace("|", "\\|").replace("\n", " ")

    def compact(counts):
        return (
            ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in sorted(counts.items()) if v)
            or "none"
        )

    lines = [
        "# Audited NCIT–DOID review catalogue",
        "",
        f"{value['case_count']} cases, {value['pair_count']} displayed pairs. The immutable publication contains {value['unique_results']} unique deterministic explanation results and {value['unique_claims']} unique supported claims. This catalogue checks the served assets and their source facts; browser layout and usefulness remain separate review tasks.",
        "",
        f"Publication: `{value['publication_sha256']}`. Audit receipt: `{value['audit_receipt_sha256']}`.",
        "",
        value["score_meaning"],
        "",
        value["position_meaning"],
        "",
        "Recorded comparison resolution concerns saved matcher identity links. It is separate from whether the linked original facts resolve or support graph projection. A recorded bridge does not establish class equivalence.",
        "",
        "Channel coverage distinguishes all saved selected records, records with admitted focal original links, and records explicitly marked graph-supported. The first is a matcher inventory; it does not establish visible graph content. Unrecorded projection is unknown support, not proof of a supported or unsupported rendering.",
        "",
        "Evidence coverage: " + compact(value["evidence_coverage"]["pair_channels"]) + ".",
        "",
        "## Open gaps",
        "",
        *["- " + text for text in value["limitations"]],
        "",
        "- Earlier failed or rejected provider outputs remain separate; none is admitted as provider prose in this typed-only publication.",
        "- Scope and quantifier support below describes the selected original ontology statements, not a biomedical adjudication.",
        "",
        "## Cases",
        "",
    ]
    for case in value["cases"]:
        source = case["source"]
        lines += [
            f"### {case['case_id']} — {cell(case['source_label'])}",
            "",
            f"Source: `{source['iri']}`.",
            "",
            "Coverage: " + ", ".join(t.replace("_", " ") for t in case["coverage_strata"]) + ".",
            "",
            f"Descriptions: {case['profile_claim_occurrences']} profile claims; {case['comparison_claim_occurrences']} comparison claims. Results: {compact(case['description_kinds'])}. Support: {compact(case['description_support'])}. Original definition counts in source/target packet order: {case['definition_counts']}; packet-label occurrences over 80 characters: {case['packet_long_label_count']}; displayed preferred labels over 80 characters: {len(case['long_preferred_labels'])}.",
            "",
            "| Candidate / target | Display / saved row | Saved score | Original links (resolved / unresolved / absent / withheld) | Projection (supported / unsupported / unrecorded) | Recorded comparisons (resolved / unresolved / ambiguous / unsupported) | Bridges |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for pair in case["pairs"]:
            target = pair["target"]
            identifier = target["iri"].rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            origins, graph, comparisons = (
                pair["origin_resolution"],
                pair["graph_projection"],
                pair["comparison_resolution"] or {},
            )
            original_text = " / ".join(
                str(origins.get(k, 0))
                for k in ("resolved", "unresolved", "not_exported", "withheld")
            )
            graph_text = " / ".join(
                str(graph.get(k, 0)) for k in ("supported", "unsupported", "not_recorded")
            )
            comparison_text = " / ".join(
                str(comparisons.get(k, 0))
                for k in ("resolved", "unresolved", "ambiguous", "unsupported")
            )
            if pair["evidence_status"] == "not_exported":
                original_text = "not exported (unknown)"
                graph_text = "not exported (unknown)"
                comparison_text = "not recorded"
            lines.append(
                f"| {cell(pair['candidate_id'])}<br>{cell(pair['target_label'])}<br>[{identifier}]({target['iri']}) | {pair['display_position']} / {pair['saved_row_position']} | {pair['saved_score']:.8g} | {original_text} | {graph_text} | {comparison_text} | {pair['bridge_count']} ({pair['bridge_status']}); cross-channel: {pair['cross_channel_bridges']} |"
            )
        lines += [
            "",
            "Channel coverage (selected inventory / admitted original links / explicitly graph-supported): source-only pairs "
            + " / ".join(
                str(case["channel_coverage"][k]["source_only_pairs"])
                for k in ("selected", "resolved", "graph_supported")
            )
            + "; both-sided multiple-channel pairs "
            + " / ".join(
                str(case["channel_coverage"][k]["both_sides_multiple_channel_pairs"])
                for k in ("selected", "resolved", "graph_supported")
            )
            + ". Claim support rules: "
            + compact(case["claim_methods"])
            + ".",
            "",
        ]
        for label in case["long_preferred_labels"]:
            lines += [
                f"Long preferred label ({label['length']} characters): `{label['entity']['iri']}` — {cell(label['label'])}.",
                "",
            ]
        for pair in case["pairs"]:
            if pair["available_expression_originals"]:
                lines += [
                    f"{pair['candidate_id']} has {len(pair['available_expression_originals'])} linked original expression axioms available for inspection (graph support is reported separately): "
                    + ", ".join(
                        f"`{row['fact_id']}` ({'/'.join(row['constructors'])})"
                        for row in pair["available_expression_originals"]
                    )
                    + ".",
                    "",
                ]
    md_path.write_text("\n".join(lines))
    print(str(json_path), file_hash(json_path))
    print(str(md_path), file_hash(md_path))


def read_selection(root, definition, selection_file=None):
    """Read an explicit public sidecar only when its frozen publication hash agrees."""
    path = selection_file if selection_file is not None else root / "selection.json"
    selection = json.loads(path.read_bytes())
    assert (
        definition["review_provenance"]["selection_manifest_sha256"] == file_hash(path)[7:]
    ), "Selection metadata does not match the frozen publication"
    return path, selection


def audit(root, output, selection_file=None):
    started = time.monotonic()
    assert not output.is_relative_to(root), "Write audit receipts outside immutable input"
    publication_path = root / "publication.json"
    publication = json.loads(publication_path.read_bytes())
    definition = publication["definition"]
    assert (
        definition["purpose"] == "development_review"
        and definition["correctness_metrics"] == "disabled"
    )
    package_root = root / "package"
    package_path = package_root / "package.json"
    package = json.loads(package_path.read_bytes())
    package_artifacts = {a["path"]: a for a in package["artifacts"]}
    assert package["provenance"]["generation_mode"] == "typed-only"
    selection_path, selection = read_selection(root, definition, selection_file)
    contexts = {
        version: OntologyContext(contained(package_root, path))
        for version, path in package["ontologies"].items()
    }
    policy = VisibilityPolicy.model_validate(package["policy"])
    print("Verified context bytes", round(time.monotonic() - started, 2), flush=True)
    results, claims, records, methods, semantic_rules = {}, {}, [], Counter(), Counter()
    original_keys = set()
    for key, locator in package["explanations"].items():
        path = contained(package_root, locator)
        assert file_hash(path) == package_artifacts[locator]["sha256"]
        assert path.stat().st_size == package_artifacts[locator]["size"]
        result = json.loads(path.read_bytes())
        assert result["explanation_id"] == key
        manifest = result["manifest"]
        assert (
            manifest["requested_model"] == "none-typed-renderer"
            and manifest["provider"] is None
            and manifest["response_hash"] is None
        )
        assert result.get("independent_review") is None
        packet = validate_prepared_support(result, contexts, policy)
        assert result["limitations"] == packet.missingness
        for f in packet.facts:
            original_keys.add((canonical_hash(f["subject"]), f["fact_id"]))
        audited = []
        for claim in result["claims"]:
            method, rules = assess_claim(claim, result)
            methods[method] += 1
            semantic_rules.update(rules)
            record = {
                "claim_id": claim["claim_id"],
                "claim_hash": canonical_hash(claim),
                "fact_ids": claim["fact_ids"],
                "method": method,
                "semantic_obligations": sorted(rules),
            }
            audited.append(record)
            expected = {
                **claim,
                "entities": result["entities"],
                "preparation_kind": result["preparation_kind"],
                "support_status": result["support_status"],
                "generation_manifest_sha256": canonical_hash(manifest)[7:],
                "method": method,
            }
            assert claims.setdefault(claim["claim_id"], expected) == expected
        results[key] = result
        records.append(
            {
                "explanation_id": key,
                "artifact_sha256": file_hash(path),
                "path": locator,
                "task": result["task"],
                "preparation_kind": result["preparation_kind"],
                "manifest_status": manifest["status"],
                "claims": audited,
            }
        )
    assets = {a["asset_id"]: a for a in definition["assets"]}
    scopes = {s["case_id"]: s for s in definition["workspace_scopes"] if s["kind"] == "case"}
    planned = {c["case_id"]: c for c in selection["cases"]}
    cases, displayed, served_results = [], Counter(), set()
    for case in definition["cases"]:
        case_id = case["case_id"]
        selected = planned[case_id]
        verify_label(case["source"], case["source_label"], contexts, policy)
        assert (
            case["source"] == selected["source"]
            and [c["entity"] for c in case["candidates"]] == selected["targets"]
        )
        asset = assets[case["explanation_refs"][0]]
        path = contained(root, asset["path"])
        assert file_hash(path)[7:] == asset["sha256"] and path.stat().st_size == asset["size_bytes"]
        resource = ExplanationResource.model_validate_json(path.read_bytes())
        assert not resource.independent_reviews
        counts = Counter()
        for claim in [*resource.entity_profiles, *resource.pair_comparison]:
            expected = claims[claim.claim_id]
            for field in (
                "text",
                "fact_ids",
                "category",
                "preparation_kind",
                "support_status",
                "generation_manifest_sha256",
            ):
                assert (
                    getattr(claim, field) == expected[field]
                ), f"Resource changes generated claim {field}"
            assert [e.model_dump() for e in claim.scoped_entities] == expected["entities"]
            counts[expected["method"]] += 1
            displayed[claim.claim_id] += 1
        scope = scopes[case_id]
        index = contained(root, scope["index_path"])
        assert file_hash(index)[7:] == scope["index_sha256"]
        with sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as db:
            saved = [json.loads(row[0]) for row in db.execute("SELECT payload FROM explanations")]
            for result in saved:
                original = results[result["explanation_id"]]
                projected = dict(result)
                # The strict workspace response model materializes this optional
                # absent field as null. Claims, manifests and every other field
                # still require exact equality with the main saved artifact.
                if "independent_review" not in original:
                    assert projected.pop("independent_review", None) is None
                assert projected == original
                served_results.add(result["explanation_id"])
            expected_claims = {c["claim_id"] for result in saved for c in result["claims"]}
            assert expected_claims == {
                c.claim_id for c in [*resource.entity_profiles, *resource.pair_comparison]
            }
            pairs = []
            for position, candidate in enumerate(case["candidates"]):
                cid = candidate["candidate_id"]
                verify_label(candidate["entity"], candidate["label"], contexts, policy)
                assert candidate["score"] == selected["recorded_candidates"][position]["S_final"]
                assert candidate["display_position"] == position + 1
                rows = [
                    json.loads(row[0])
                    for row in db.execute(
                        "SELECT payload FROM evidence WHERE candidate_id=?", (cid,)
                    )
                ]
                saved_metadata = db.execute(
                    "SELECT payload FROM pair_metadata WHERE candidate_id=?", (cid,)
                ).fetchone()
                assert saved_metadata is not None, "New review pair lacks recorded metadata status"
                envelope = json.loads(saved_metadata[0])
                assert (
                    envelope["source"] == case["source"]
                    and envelope["target"] == candidate["entity"]
                )
                metadata = PairEvidenceMetadata.model_validate(
                    envelope["evidence_metadata"]
                ).model_dump(mode="json")
                channel_sides, resolved_channels, graph_channels = {}, {}, {}
                available_expressions = {}
                origins, graph = Counter(), Counter()
                by_id = {r["evidence_id"]: r for r in rows}
                assert len(by_id) == len(rows)
                for row in rows:
                    channel_sides.setdefault(row["channel"], set()).add(row["side"])
                    assert row["entity"] == (
                        case["source"] if row["side"] == "source" else candidate["entity"]
                    )
                    origins[row["origin_resolution"]] += 1
                    graph[row["graph_projection"]] += 1
                    if row["origin_resolution"] == "resolved":
                        resolved_channels.setdefault(row["channel"], set()).add(row["side"])
                        if row["graph_projection"] == "supported":
                            graph_channels.setdefault(row["channel"], set()).add(row["side"])
                    for fact_id in row["fact_ids"]:
                        context = contexts[row["entity"]["ontology_version_id"]]
                        context.fact(fact_id, row["entity"], policy=policy)
                        key = (row["entity"]["ontology_version_id"], fact_id)
                        if key not in available_expressions:
                            ast = context.axiom(fact_id, policy=policy)["ast"]
                            constructors = set()

                            def walk(node):
                                if isinstance(node, dict):
                                    kind = node.get("type", "")
                                    if kind in {
                                        "EquivalentClasses",
                                        "ObjectIntersectionOf",
                                        "ObjectUnionOf",
                                        "ObjectComplementOf",
                                        "ObjectSomeValuesFrom",
                                        "ObjectAllValuesFrom",
                                        "ObjectMinCardinality",
                                        "ObjectMaxCardinality",
                                        "ObjectExactCardinality",
                                        "ObjectHasValue",
                                        "ObjectHasSelf",
                                        "ObjectOneOf",
                                        "ObjectInverseOf",
                                        "DataSomeValuesFrom",
                                        "DataAllValuesFrom",
                                        "DataMinCardinality",
                                        "DataMaxCardinality",
                                        "DataExactCardinality",
                                        "DataHasValue",
                                        "DataIntersectionOf",
                                        "DataUnionOf",
                                        "DataComplementOf",
                                        "DataOneOf",
                                        "DatatypeRestriction",
                                    }:
                                        constructors.add(kind)
                                    for child in node.values():
                                        walk(child)
                                elif isinstance(node, list):
                                    for child in node:
                                        walk(child)

                            walk(ast)
                            available_expressions[key] = {
                                "ontology_version_id": key[0],
                                "fact_id": fact_id,
                                "constructors": sorted(constructors),
                            }
                expected_counts = {
                    "selected": len(rows),
                    **{
                        k: origins[k]
                        for k in ("resolved", "unresolved", "not_exported", "withheld")
                    },
                    **{
                        "graph_" + k: graph[k] for k in ("supported", "unsupported", "not_recorded")
                    },
                }
                assert metadata["counts"] == expected_counts
                for bridge in metadata["bridges"]:
                    assert (
                        bridge["source"] == case["source"]
                        and bridge["target"] == candidate["entity"]
                    )
                    unavailable = False
                    for side in ("source", "target"):
                        linked_facts = set()
                        for eid in bridge[side + "_evidence_ids"]:
                            item = by_id[eid]
                            assert item["side"] == side and item["channel"] == (
                                bridge[side + "_channel"] or bridge["channel"]
                            )
                            if bridge[side + "_family"] is not None:
                                assert item["family"] == bridge[side + "_family"]
                            linked_facts.update(item["fact_ids"])
                            unavailable |= item["status"] != "available"
                        assert set(bridge[side + "_fact_ids"]) <= linked_facts
                    if unavailable:
                        assert (
                            bridge["status"] == "withheld"
                            and not bridge["source_fact_ids"]
                            and not bridge["target_fact_ids"]
                        )
                pairs.append(
                    {
                        "candidate_id": cid,
                        "target": candidate["entity"],
                        "target_label": candidate["label"],
                        "display_position": candidate["display_position"],
                        "saved_row_position": selected["recorded_candidates"][position][
                            "recorded_position"
                        ],
                        "saved_score": candidate["score"],
                        "selected_count": len(rows),
                        "selected_channels": {k: sorted(v) for k, v in channel_sides.items()},
                        "resolved_channels": {k: sorted(v) for k, v in resolved_channels.items()},
                        "graph_supported_channels": {
                            k: sorted(v) for k, v in graph_channels.items()
                        },
                        "available_expression_originals": [
                            row for row in available_expressions.values() if row["constructors"]
                        ],
                        "origin_resolution": dict(origins),
                        "graph_projection": dict(graph),
                        "pair_metadata_sha256": canonical_hash(metadata),
                        "evidence_status": metadata["status"],
                        "bridge_status": metadata["bridge_status"],
                        "bridge_count": len(metadata["bridges"]),
                        "bridge_levels": dict(Counter(b["level"] for b in metadata["bridges"])),
                        "bridge_statuses": dict(Counter(b["status"] for b in metadata["bridges"])),
                        "comparison_resolution": metadata["comparison_resolution"],
                        "cross_channel_bridges": sum(
                            bool(b["source_evidence_ids"] and b["target_evidence_ids"])
                            and (b["source_channel"] or b["channel"])
                            != (b["target_channel"] or b["channel"])
                            for b in metadata["bridges"]
                        ),
                    }
                )
        cases.append(
            {
                "case_id": case_id,
                "source": case["source"],
                "source_label": case["source_label"],
                "coverage_strata": selected["coverage"],
                "asset_sha256": file_hash(path),
                "scope_index_sha256": file_hash(index),
                "profile_claim_occurrences": len(resource.entity_profiles),
                "comparison_claim_occurrences": len(resource.pair_comparison),
                "claim_methods": dict(counts),
                "description_kinds": dict(Counter(r["preparation_kind"] for r in saved)),
                "description_support": dict(Counter(r["support_status"] for r in saved)),
                "definition_counts": selected["definition_counts"],
                "packet_long_label_count": selected["long_label_count"],
                "long_preferred_labels": [
                    {"entity": entity, "label": label, "length": len(label)}
                    for entity, label in [
                        (case["source"], case["source_label"]),
                        *[(c["entity"], c["label"]) for c in case["candidates"]],
                    ]
                    if len(label) > 80
                ],
                "channel_coverage": {
                    scope: {
                        "source_only_pairs": sum(
                            any(v == ["source"] for v in p[scope + "_channels"].values())
                            for p in pairs
                        ),
                        "both_sides_multiple_channel_pairs": sum(
                            sum(
                                set(v) == {"source", "target"}
                                for v in p[scope + "_channels"].values()
                            )
                            >= 2
                            for p in pairs
                        ),
                    }
                    for scope in ("selected", "resolved", "graph_supported")
                },
                "pairs": pairs,
            }
        )
    assert served_results == set(results), "Published main/scoped explanation inventories differ"
    assert set(displayed) == set(claims), "Displayed/generated claim inventories differ"
    report = {
        "schema": "exact-independent-published-claim-audit/1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reviewer_id": "exact-s1-independent-B",
        "scope": "Final immutable typed-only publication and scope indexes, checked against admitted original ontology facts and independent rendering rules; actual browser display is a separate verification",
        "provider_prose_approval": False,
        "publication_sha256": file_hash(publication_path),
        "package_manifest_sha256": file_hash(package_path),
        "selection_sha256": file_hash(selection_path),
        "selection_input": "explicit_sidecar" if selection_file is not None else "publication_root",
        "policy_sha256": policy.policy_hash,
        "source_contexts": {
            v: {
                "manifest_sha256": file_hash(c.path / "manifest.json"),
                "database_sha256": c.manifest["artifacts"]["context.sqlite"],
            }
            for v, c in contexts.items()
        },
        "case_count": len(cases),
        "inference_status": {
            v: c.manifest["capabilities"]["reasoner_inferred"]["status"]
            for v, c in contexts.items()
        },
        "pair_count": sum(len(c["pairs"]) for c in cases),
        "unique_results": len(results),
        "unique_claims": len(claims),
        "displayed_claim_occurrences": sum(displayed.values()),
        "distinct_typed_original_facts_verified": len(original_keys),
        "claim_methods": dict(methods),
        "semantic_obligations_checked": dict(semantic_rules),
        "coverage": selection["coverage"],
        "missing_coverage": selection["missing_coverage"],
        "evidence_coverage": {
            "pair_channels": {
                scope + "_" + metric: sum(c["channel_coverage"][scope][metric] for c in cases)
                for scope in ("selected", "resolved", "graph_supported")
                for metric in ("source_only_pairs", "both_sides_multiple_channel_pairs")
            },
            "origin_resolution": {
                state: sum(p["origin_resolution"].get(state, 0) for c in cases for p in c["pairs"])
                for state in ("resolved", "unresolved", "not_exported", "withheld")
            },
            "graph_projection": {
                state: sum(p["graph_projection"].get(state, 0) for c in cases for p in c["pairs"])
                for state in ("supported", "unsupported", "not_recorded")
            },
            "pairs_with_available_expression_originals": sum(
                bool(p["available_expression_originals"]) for c in cases for p in c["pairs"]
            ),
            "preferred_labels_over_80_character_occurrences": sum(
                len(c["long_preferred_labels"]) for c in cases
            ),
            "packet_labels_over_80_character_occurrences": sum(
                c["packet_long_label_count"] for c in cases
            ),
        },
        "results": records,
        "cases": cases,
        "failures": [],
        "limitations": [
            "No provider prose or provider semantic-review receipt is approved by this deterministic audit.",
            "Ontology source correctness, completeness, biomedical usefulness and mapping equivalence are not established.",
            "Full native inferred coverage remains failed; asserted facts and navigation retain their declared scope.",
            "Sparse selected evidence coverage remains absent; missing exports are not zero-evidence observations.",
            "Selected channel inventory is not linked-original or graph-visible coverage; unresolved neighbour citations and unrecorded projection remain explicit gaps.",
            "This validates immutable served artifacts; the independent HTTPS/browser check determines actual rendering and interaction.",
        ],
        "audit_script_sha256": file_hash(Path(__file__)),
        "runtime": runtime_identity(),
        "elapsed_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), "Preserve the earlier audit receipt; choose another output name"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    catalogue(report, output)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "case_count",
                    "pair_count",
                    "unique_results",
                    "unique_claims",
                    "displayed_claim_occurrences",
                    "claim_methods",
                    "elapsed_seconds",
                )
            }
        )
    )
    print(str(output), file_hash(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-root",
        type=Path,
        required=True,
        help="Maintained Exact-OM checkout providing the runtime admission checks",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="Separately delivered public selection metadata; its frozen publication hash is required",
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    load_runtime(args.runtime_root)
    audit(
        args.root.resolve(),
        args.output.resolve(),
        args.selection_file.resolve() if args.selection_file is not None else None,
    )
