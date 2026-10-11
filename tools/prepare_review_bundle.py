"""Offline development review preparation from completed, recorded matcher output.

Reference labels are never read for case selection. Serving never calls this module.
All paths are explicit; an existing publication is reused only with identical inputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from exact.runs.reader import RunReader
from exact_inspect.artifacts import atomic_json, publish_bundle, validate_bundle
from exact_inspect.context import OntologyContext
from exact_inspect.context_resources import export_ontology_resource
from exact_inspect.contracts import EntityRef, canonical_hash, file_hash
from exact_inspect.decisions import (
    _feature_items,
)
from exact_inspect.decisions_export import export_policy_run
from exact_inspect.descriptions import (
    DESCRIPTION_VERSION,
    description_templates,
    prepare_description_packet,
)
from exact_inspect.generation import (
    GenerationProfile,
    comparison_packet,
    comparison_templates,
    entity_packet,
    factual_fallback,
)
from exact_inspect.study.builder import build_explanation_resource
from exact_inspect.study.models import Publish
from exact_inspect.study.workspace import WorkspaceScope, build_workspace_scope
from tools.build_explanation_v2_fixture import publication_v2
from tools.review_generation_attempt import ReviewExplanationJobs as ExplanationJobs
from tools.review_generation_attempt import (
    attempt_paths,
    attempt_scope_id,
    bind_attempt,
    bind_generation_contract,
    previous_attempt_summaries,
)
from tools.review_generation_summary import generation_limitations, generation_summary
from tools.review_pair_inputs import frozen_case_pair_inputs
from tools.review_private_run_cache import prepared_private_run
from tools.review_run_cache import prepared_policy_run

CATEGORIES = (
    "rich_definitions",
    "absent_or_asymmetric_definitions",
    "multiple_parents",
    "restrictions_or_equivalences",
    "close_alternatives",
    "sparse_evidence",
)


def review_scope_limitations(plan, private_counts):
    """Describe the actual preparation and recorded decision coverage without score inference."""
    limitations = [
        "Matcher-to-original evidence resolution was prepared only for the selected typed review entities. "
        "All recorded candidates remain browsable; this preparation does not resolve evidence origins outside that selected set."
    ]
    selected = private_counts.get("selected_evidence", 0)
    if selected and private_counts.get("provenance_unavailable", 0) > selected / 2:
        limitations.append(
            "Most run-wide selected-evidence origins remain unresolved; exact counts are retained in the preparation receipt."
        )
    displayed = [row for case in plan["cases"] for row in case["recorded_candidates"]]
    if displayed and all(
        row.get("threshold_positive") is True and row.get("emitted") is True for row in displayed
    ):
        limitations.append(
            "All displayed candidate pairs have recorded threshold-positive and emitted decisions. "
            "Rejected candidates and differing final decisions are not represented in the review cases."
        )
    return limitations


def packet_for(entity, contexts, policy):
    context = contexts[entity.ontology_version_id]
    facts = {}
    counts = {}
    for category, limit in [
        ("definitions", 8),
        ("alternate_definitions", 4),
        ("hierarchy", 12),
        ("restrictions", 12),
        ("labels", 3),
        ("synonyms", 6),
        ("annotations", 6),
    ]:
        page = context.facts(entity, category=category, limit=limit, policy=policy)
        counts[category] = len(page["items"])
        facts.update({f["fact_id"]: f for f in page["items"]})
    missingness = (
        []
        if counts["definitions"]
        else ["No original definition is available in the admitted context."]
    )
    packet = entity_packet(
        entity,
        list(facts.values()),
        context_hash=context.manifest["artifacts"]["context.sqlite"],
        policy=policy,
        missingness=missingness,
    )
    packet = prepare_description_packet(packet, context, policy)
    return (
        packet.model_copy(update={"selection": {**packet.selection, "description_prose": True}}),
        counts,
    )


def literal_label(entity, contexts, policy):
    rows = contexts[entity.ontology_version_id].labels([entity.iri], policy=policy)
    for row in rows:
        if row["entity"] == entity.model_dump() and row["preferred_label"]["status"] == "available":
            return row["preferred_label"]["value"][:512]
    return entity.iri[:512]


def plan_cases(run, contexts, source_version, target_version, policy, count):
    experiment = json.loads((run / "experiment_manifest.json").read_bytes())
    if experiment.get("status") != "complete" or experiment.get("split_role") != "development":
        raise ValueError("Review input must be an explicitly completed development run")
    source_records = json.loads((run / "source_decisions.json").read_bytes())["records"]
    feature_summary = {}
    for saved in RunReader.open(run).iter_explanations():
        channels = {}
        feature_count = 0
        for channel, side, _, _ in _feature_items(saved):
            channels.setdefault(channel, set()).add(side)
            feature_count += 1
        feature_summary[(saved["src_iri"], saved["tgt_iri"])] = {
            "selected": feature_count,
            "source_only_channel": any(sides == {"source"} for sides in channels.values()),
            "both_sides_channels": sum(
                sides == {"source", "target"} for sides in channels.values()
            ),
        }
    inventory, packet_cache, excluded = [], {}, []
    for record in sorted(source_records, key=lambda row: row["Src"]):
        candidates = sorted(
            enumerate(record["candidates"]),
            key=lambda item: (-item[1].get("S_final", float("-inf")), item[0]),
        )
        if len(candidates) < 5:
            excluded.append(
                {"source": record["Src"], "reason": "fewer_than_five_recorded_candidates"}
            )
            continue
        top = candidates[:5]
        source = EntityRef(
            ontology_version_id=source_version,
            iri=record["Src"],
            kind=top[0][1].get("SrcKind", "class"),
        )
        targets = [
            EntityRef(
                ontology_version_id=target_version,
                iri=row["target"],
                kind=row.get("TgtKind", "class"),
            )
            for _, row in top
        ]
        try:
            for entity in [source, *targets]:
                if canonical_hash(entity) not in packet_cache:
                    packet_cache[canonical_hash(entity)] = packet_for(entity, contexts, policy)
        except (KeyError, ValueError):
            excluded.append(
                {"source": record["Src"], "reason": "top_five_outside_admitted_context"}
            )
            continue
        packs = [packet_cache[canonical_hash(entity)] for entity in [source, *targets]]
        defs = [counts["definitions"] + counts["alternate_definitions"] for _, counts in packs]
        tags = []
        if sum(bool(n) for n in defs) >= 4:
            tags.append("rich_definitions")
        if not all(defs):
            tags.append("absent_or_asymmetric_definitions")
        if any(
            contexts[e.ontology_version_id].hierarchy(
                e, policy=policy, direction="parents", basis="literal_asserted", limit=3
            )["total_count"]
            > 1
            for e in [source, *targets]
        ):
            tags.append("multiple_parents")
        if any(counts["restrictions"] >= 3 for _, counts in packs):
            tags.append("restrictions_or_equivalences")
        if abs(top[0][1]["S_final"] - top[1][1]["S_final"]) <= 0.05:
            tags.append("close_alternatives")
        recorded_features = [feature_summary.get((source.iri, target.iri)) for target in targets]
        if any(row is not None and row["selected"] <= 2 for row in recorded_features):
            tags.append("sparse_evidence")
        inventory.append(
            {
                "source": source.model_dump(),
                "targets": [e.model_dump() for e in targets],
                "recorded_candidates": [
                    {**row, "recorded_position": index + 1} for index, row in top
                ],
                "full_pool_count": len(candidates),
                "coverage": tags,
                "feature_export_summary": recorded_features,
                "definition_counts": defs,
                "restriction_counts": [counts["restrictions"] for _, counts in packs],
                "long_label_count": sum(
                    len(f["value"].get("lexical_form", "")) > 80
                    for p, _ in packs
                    for f in p.facts
                    if f.get("category") == "labels"
                ),
                "original_annotation_count": sum(
                    f.get("value", {}).get("term_type") == "literal"
                    for p, _ in packs
                    for f in p.facts
                ),
            }
        )
    chosen, coverage = [], Counter()
    remaining = list(inventory)
    while remaining and len(chosen) < count:
        row = max(
            remaining,
            key=lambda value: (
                sum(coverage[tag] < 3 for tag in value["coverage"]),
                -inventory.index(value),
            ),
        )
        chosen.append(row)
        remaining.remove(row)
        coverage.update(row["coverage"])
    if len(chosen) != count:
        raise ValueError(f"Only {len(chosen)} eligible sources; requested {count}")
    for i, row in enumerate(chosen):
        row["case_id"] = f"review-{i + 1:02}"
    return {
        "schema": "exact-review-selection/1",
        "run_manifest_sha256": file_hash(run / "run_manifest.json"),
        "experiment_manifest_sha256": file_hash(run / "experiment_manifest.json"),
        "selection_rule": "Greedy coverage quotas without reference labels; top five descending recorded S_final, ties retain recorded source-decision position.",
        "recorded_position_meaning": "One-based position in the saved source_decisions candidate array, which is target-ordered; this is not a production rank.",
        "source_commit": experiment.get("git"),
        "split_role": experiment["split_role"],
        "reference_role": experiment.get("reference_role"),
        "source_sampling": experiment.get("source_sampling"),
        "candidate_pool": experiment.get("candidate_pool"),
        "available_coverage": dict(Counter(tag for row in inventory for tag in row["coverage"])),
        "coverage_definitions": {
            "rich_definitions": "At least four of six compared entities have original definitions.",
            "restrictions_or_equivalences": "At least one compared entity has three admitted restriction/equivalence records.",
            "close_alternatives": "Two highest recorded final scores differ by at most0.05; no semantic adjudication is implied.",
            "sparse_evidence": "At most two selected exported features in at least one presented pair; absent exports are distinct.",
        },
        "cases": chosen,
        "excluded": excluded,
        "coverage": dict(coverage),
        "missing_coverage": {tag: max(0, 3 - coverage[tag]) for tag in CATEGORIES},
    }, packet_cache


def deterministic_result(packet):
    templates = description_templates(packet)
    if packet.task == "pair_comparison":
        templates = comparison_templates(packet)[:3]
    output = factual_fallback(packet) if not templates else None
    claims = templates[:4] if templates else output.claims
    key = canonical_hash(
        {
            "contract": DESCRIPTION_VERSION,
            "packet": packet.model_dump(mode="json"),
            "renderer": "typed-owl/1",
        }
    )
    return dict(
        artifact_type="generated_explanation",
        contract_version="exact-explain/1.0",
        explanation_id=key,
        task=packet.task,
        entities=[e.model_dump() for e in packet.entities],
        claims=[
            c.model_copy(
                update={
                    "claim_id": canonical_hash(
                        {"generation": key, "claim": c.model_dump(mode="json")}
                    )
                }
            ).model_dump()
            for c in claims
        ],
        grounding_status="validated",
        limitations=packet.missingness,
        fixture_provenance="Real ontology facts; deterministic typed rendering, no model request.",
        preparation_kind="grounded_description" if templates else "factual_fallback",
        support_status="typed_template_support" if templates else "fallback",
        support_reasons=[] if templates else ["No supported non-label description available."],
        support_packet=packet.model_dump(mode="json"),
        manifest=dict(
            ontology_context_hashes=packet.context_hashes,
            visibility_policy_hash=packet.policy_hash,
            packet_hash=canonical_hash(packet),
            prompt_hash=canonical_hash("typed-owl/1"),
            output_schema="exact-explain-generation/1",
            requested_model="none-typed-renderer",
            returned_model=None,
            provider=None,
            parameters_hash=canonical_hash({}),
            language=packet.language,
            response_hash=None,
            status="validated" if templates else "failed",
            description_contract=DESCRIPTION_VERSION,
        ),
    )


def unresolved_review_key(case):
    """Keep unrecorded producer ranks absent; array positions belong in the catalogue."""
    return dict(
        case_id=case["case_id"],
        case_kind="unresolved",
        acceptable_candidate_ids=[],
        adjudication_version="unresolved-development-review/1",
        criterion="Correctness metrics disabled; no reference labels selected or shown.",
        evidence=[
            "Recorded development matcher outputs only; biomedical adjudication not performed.",
            "Production ranking is not reconstructed from candidate array positions.",
        ],
        origin="unresolved",
        original_production_ranks={},
    )


def prepare(
    root,
    run,
    count=24,
    generate_pairs=12,
    revision=None,
    *,
    inputs_only=False,
    generation_mode="provider",
    generation_attempt=1,
    generation_cache_only=False,
):
    if generation_mode not in {"provider", "typed-only"}:
        raise ValueError("Generation mode must be provider or typed-only")
    root, run = Path(root).resolve(), Path(run).resolve()
    if (root / "composition-inputs.json").exists():
        raise ValueError("Use the bound composition/resume command for a layered publication")
    attempt = attempt_paths(root, generation_attempt)
    package = root / "package"
    publication_path = root / "publication.json"
    # One root owns one exact offline preparation. Restart may reuse its outputs,
    # but it must never silently bind a different run, ontology or case selection.
    inputs = {
        "schema": "exact-review-preparation/1",
        "run_manifest": file_hash(run / "run_manifest.json"),
        "source_decisions": file_hash(run / "source_decisions.json"),
        "cases": count,
        "generate_pairs": generate_pairs,
        "revision": revision,
        "description_preparation": (
            "independent-prose/1" if generation_mode == "provider" else "typed-rendering-only/1"
        ),
        "contexts": {
            str(path.relative_to(package)): file_hash(path)
            for path in sorted((package / "ontologies").glob("*/manifest.json"))
        },
    }
    binding = root / "preparation-inputs.json"
    prior_inputs_bound = binding.is_file()
    if binding.exists() and json.loads(binding.read_bytes()) != inputs:
        raise ValueError("Preparation inputs changed; use a new review root")
    atomic_json(binding, inputs)
    if publication_path.exists():
        existing = json.loads(publication_path.read_bytes())
        publication = Publish.model_validate(existing)
        if (
            existing["definition"]["review_provenance"]["run_manifest_sha256"]
            != file_hash(run / "run_manifest.json")[7:]
        ):
            raise ValueError("Existing review publication has different input; use a new root")
        completed_bundle = validate_bundle(package / "package.json")
        if not (root / "catalogue.json").is_file():
            raise ValueError("Published review is missing its coverage catalogue")
        catalogue = json.loads((root / "catalogue.json").read_bytes())
        if catalogue.get("publication_sha256") != canonical_hash(publication):
            raise ValueError("Completed review publication differs from its coverage catalogue")
        if any(
            value.get("generation_attempt", 1) != generation_attempt
            for value in (catalogue, completed_bundle.provenance)
        ):
            raise ValueError(
                "Completed publication uses another generation attempt; use a new root/revision"
            )
        from tools.review_environment import verify_assets, verify_workspaces

        # Assets and scoped indexes live outside the exploration package. Reusing
        # its valid manifest alone would not verify the completed study's inputs.
        verify_assets(existing, root)
        verify_workspaces(existing, root)
        return publication_path
    bundle = validate_bundle(package / "package.json")
    policy = bundle.policy
    contexts = {
        version: OntologyContext(package / locator)
        for version, locator in bundle.ontologies.items()
    }
    source_version = next(v for v, c in contexts.items() if "NCIT" in c.manifest["name"].upper())
    target_version = next(v for v, c in contexts.items() if "DOID" in c.manifest["name"].upper())
    plan, cache = plan_cases(run, contexts, source_version, target_version, policy, count)
    atomic_json(root / "selection.json", plan)
    selected_entities = [e for case in plan["cases"] for e in [case["source"], *case["targets"]]]
    private_run = prepared_private_run(
        root,
        run,
        contexts=contexts,
        source_version=source_version,
        target_version=target_version,
        selected_entities=selected_entities,
    )
    portable_run = prepared_policy_run(
        private_run,
        package / "runs" / "review",
        contexts=contexts,
        policy=policy,
        checkpoint=root / "policy-export-inputs.json",
        adopt_existing=prior_inputs_bound,
        exporter=export_policy_run,
    )
    # The full gold-free producer candidate universe remains private preparation input.
    fixture_path = root / "practice-publication.json"
    if fixture_path.is_file():
        fixture = json.loads(fixture_path.read_bytes())
    else:
        fixture = publication_v2(
            root, forms_version="exact-study-forms/3", visibility_policy=policy
        ).model_dump(mode="json")
        atomic_json(fixture_path, fixture)
    definition = fixture["definition"]
    definition["assets"] = [
        a for a in definition["assets"] if a["asset_id"].startswith("practice-")
    ]
    definition["workspace_scopes"] = [
        s for s in definition["workspace_scopes"] if s["kind"] == "tutorial"
    ]
    for label, version in [("source", source_version), ("target", target_version)]:
        context = contexts[version]
        path = root / f"{label}-review.ofn"
        receipt = export_ontology_resource(context, path, policy)
        definition["assets"].append(
            dict(
                asset_id=label,
                path=path.name,
                sha256=file_hash(path)[7:],
                size_bytes=path.stat().st_size,
                kind="ontology",
                media_type="application/owl-functional",
                policy_hash=policy.policy_hash[7:],
                admission_receipt_path=receipt.name,
                admission_receipt_sha256=file_hash(receipt)[7:],
                title=context.manifest["name"],
                role=label,
                ontology_version_id=version,
                version_label=version,
                license_note="Local development review; source ontology licence applies.",
                information_notice="Policy-admitted prepared context; consult manifest for import/derivation limitations.",
            )
        )
    if inputs_only:
        return binding
    bind_attempt(root, generation_attempt, inputs)
    profile = GenerationProfile(
        name="review-description-v2",
        model="openai/gpt-4o-mini-2024-07-18",
        revision="2024-07-18",
        routing={"only": ["OpenAI"], "allow_fallbacks": False, "require_parameters": True},
    )
    if generation_mode == "provider":
        bind_generation_contract(root, generation_attempt, profile)
    jobs = ExplanationJobs(attempt["generation"]) if generation_mode == "provider" else None
    if jobs is not None:
        jobs.cache_only = generation_cache_only
    prepared_profiles, generated_index, cases, keys, catalogue = {}, {}, [], [], []
    dependency_index = {}
    source_artifacts, served_artifacts = {}, {}
    pair_number = 0
    pending_reviews = set()
    from exact_inspect.description_review import (
        IndependentDescriptionReview,
        admit_independent_review,
        build_review_request,
    )

    for case_number, selected in enumerate(plan["cases"]):
        entities = [EntityRef.model_validate(e) for e in [selected["source"], *selected["targets"]]]
        source = entities[0]
        candidates = [
            dict(
                candidate_id=f"{selected['case_id']}-candidate-{i + 1}",
                entity=entity.model_dump(),
                label=literal_label(entity, contexts, policy),
                score=row["S_final"],
                score_meaning="Recorded matcher S_final; not probability of semantic equivalence.",
                display_position=i + 1,
            )
            for i, (entity, row) in enumerate(zip(entities[1:], selected["recorded_candidates"]))
        ]
        case = dict(
            case_id=selected["case_id"],
            source=source.model_dump(),
            source_label=literal_label(source, contexts, policy),
            transfer_group=canonical_hash(source),
            package_version="development-review/1",
            candidates=candidates,
            ontology_resource_ids=["source", "target"],
            explanation_refs=[selected["case_id"] + "-explanation"],
        )
        packets, results = [], []
        # At least twelve distinct pairs are provider prepared; every entity without
        # an original definition also gets an explicit provider attempt.
        for entity in entities:
            identity = canonical_hash(entity)
            packet, counts = cache[identity]
            if identity not in prepared_profiles:
                result = (
                    jobs.generate(packet, profile, max_repairs=0)
                    if jobs is not None
                    and (case_number < generate_pairs or not counts["definitions"])
                    else deterministic_result(packet)
                )
                prepared_profiles[identity] = result
            packets.append(packet)
            results.append(prepared_profiles[identity])
        profile_packets = list(packets)
        for index, entity in enumerate(entities[1:]):
            packet = comparison_packet(
                profile_packets[0], profile_packets[index + 1], [results[0], results[index + 1]]
            )
            result = (
                jobs.generate(packet, profile, max_repairs=0)
                if jobs is not None and case_number < generate_pairs and index == 0
                else deterministic_result(packet)
            )
            packets.append(packet)
            results.append(result)
            pair_number += 1
        reviewed = []
        case_pending = False
        for result in results:
            source_artifacts[result["explanation_id"]] = result
            key = result["explanation_id"][7:]
            locator = str((attempt["dependencies"] / (key + ".json")).relative_to(package))
            atomic_json(package / locator, result)
            dependency_index[result["explanation_id"]] = locator
            responses = (
                sorted((jobs.root / key).glob("response-*.json"))
                if generation_mode == "provider"
                else []
            )
            raw = next(
                (
                    saved["response"]
                    for path in responses
                    if (saved := json.loads(path.read_bytes()))["sha256"]
                    == result["manifest"]["response_hash"]
                ),
                None,
            )
            if raw is not None:
                try:
                    request = build_review_request(result, raw)
                except ValueError as exc:
                    atomic_json(
                        attempt["requests"] / (key + ".unreviewable.json"),
                        {"explanation_id": result["explanation_id"], "reason": str(exc)},
                    )
                else:
                    atomic_json(attempt["requests"] / (key + ".json"), request)
                    receipt = attempt["reviews"] / (key + ".json")
                    if receipt.exists():
                        decisions = IndependentDescriptionReview.model_validate_json(
                            receipt.read_bytes()
                        )
                        values = decisions.model_dump(mode="json")
                        if any(values[k] != v for k, v in request["bindings"].items()):
                            raise ValueError(
                                "Independent review belongs to another generation input"
                            )
                        if any(d.decision == "approved" for d in decisions.decisions):
                            result = admit_independent_review(result, raw, values)
                    else:
                        pending_reviews.add(key)
                        case_pending = True
            reviewed.append(result)
        results = reviewed
        served_artifacts.update({r["explanation_id"]: r for r in results})
        if case_pending:
            continue
        evidence, pair_metadata = frozen_case_pair_inputs(
            portable_run, source.model_dump(mode="json"), candidates
        )
        # The same policy filter used by the exploration API preserves missingness
        # tombstones and suppresses unadmitted display strings.
        from exact_inspect.evidence_metadata import filter_evidence_items

        evidence = {
            key: filter_evidence_items(items, contexts, policy) for key, items in evidence.items()
        }
        resource = build_explanation_resource(
            contexts,
            entities,
            policy=policy,
            packets=packets,
            explanations=results,
            evidence=evidence,
        )
        path = attempt["assets"] / (case["explanation_refs"][0] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resource.model_dump_json())
        definition["assets"].append(
            dict(
                asset_id=case["explanation_refs"][0],
                path=str(path.relative_to(root)),
                sha256=file_hash(path)[7:],
                size_bytes=path.stat().st_size,
                kind="explanation",
                media_type="application/json",
                policy_hash=policy.policy_hash[7:],
            )
        )
        scope_id = attempt_scope_id(case["case_id"], generation_attempt)
        scope_path = attempt["scopes"] / (case["case_id"] + ".json")
        scope_input = canonical_hash(
            {
                "case": case,
                "policy": policy.model_dump(mode="json"),
                "results": results,
                "evidence": evidence,
                "pair_metadata": pair_metadata,
                "contexts": inputs["contexts"],
            }
        )
        scope_binding = scope_path.with_suffix(".input.json")
        if scope_path.exists():
            if not scope_binding.exists() or json.loads(scope_binding.read_bytes()) != scope_input:
                raise ValueError("Prepared scope inputs changed; use a new review root")
            scope = WorkspaceScope.model_validate_json(scope_path.read_bytes())
        else:
            # A failed unpublished build may leave its empty immutable scope
            # directory. Remove only this exact task-owned staging index.
            import hashlib

            staging = root / "workspaces" / hashlib.sha256(scope_id.encode()).hexdigest()
            if staging.exists():
                shutil.rmtree(staging)
            scope = build_workspace_scope(
                root,
                scope_id=scope_id,
                kind="case",
                case=case,
                contexts=contexts,
                policy=policy,
                explanations=results,
                evidence=evidence,
                pair_metadata=pair_metadata,
                components=["context", "hierarchy", "profiles", "comparison", "evidence"],
                reuse_prepared_contexts=True,
            )
            atomic_json(scope_binding, scope_input)
            atomic_json(scope_path, scope)
        definition["workspace_scopes"].append(scope.model_dump(mode="json"))
        cases.append(case)
        keys.append(unresolved_review_key(case))
        per_pair = []
        for candidate in candidates:
            rows = evidence[candidate["candidate_id"]]
            per_pair.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "selected_features": len(rows),
                    "resolved_original_facts": len(
                        {fid for row in rows for fid in row.get("fact_ids", [])}
                    ),
                    "expression_original_facts": len(
                        {
                            fid
                            for row in rows
                            for fid in row.get("fact_ids", [])
                            if any(
                                f.fact_id == fid and f.value.term_type == "expression_ref"
                                for f in resource.facts
                            )
                        }
                    ),
                    "origin_resolution": dict(
                        Counter(row.get("origin_resolution", "not_recorded") for row in rows)
                    ),
                    "graph_projection": dict(
                        Counter(row.get("graph_projection", "not_recorded") for row in rows)
                    ),
                    "feature_sides": sorted({row["side"] for row in rows}),
                }
            )
        catalogue.append(
            {
                **selected,
                "pairs": per_pair,
                "descriptions": [
                    {
                        "id": r["explanation_id"],
                        "kind": r["preparation_kind"],
                        "status": r["manifest"]["status"],
                    }
                    for r in results
                ],
            }
        )
        for result in results:
            locator = str(
                (attempt["explanations"] / (result["explanation_id"][7:] + ".json")).relative_to(
                    package
                )
            )
            atomic_json(package / locator, result)
            generated_index[result["explanation_id"]] = locator
        atomic_json(attempt["partial_catalogue"], catalogue)
    preparation_summary = generation_summary(
        jobs.root if jobs is not None else attempt["generation"],
        source_artifacts.values(),
        served_artifacts.values(),
        generation_mode,
    )
    preparation_summary.update(
        generation_attempt=generation_attempt,
        previous_attempts=previous_attempt_summaries(root, generation_attempt),
    )
    atomic_json(attempt["summary"], preparation_summary)
    if pending_reviews:
        status_path = attempt["status"]
        atomic_json(
            status_path,
            {
                "status": "independent_review_required",
                "pending_jobs": sorted(pending_reviews),
                "request_count": len(pending_reviews),
                "generation_summary": preparation_summary,
                "generation_attempt": generation_attempt,
                "review_receipts_directory": attempt["reviews"].name,
                "instructions": "Review exact claims against cited packet facts; save bound receipts in the indicated review_receipts_directory and rerun this same preparation attempt.",
            },
        )
        return status_path
    blocks = [
        {"condition": "explanation", "case_ids": [c["case_id"] for c in cases[::2]]},
        {"condition": "ontology_baseline", "case_ids": [c["case_id"] for c in cases[1::2]]},
    ]
    schedules = []
    for form in range(2):
        ordered = (
            blocks
            if not form
            else [
                {**blocks[0], "case_ids": blocks[1]["case_ids"]},
                {**blocks[1], "case_ids": blocks[0]["case_ids"]},
            ]
        )
        schedules += [
            dict(schedule_id=f"form-{form}-forward", blocks=ordered),
            dict(schedule_id=f"form-{form}-reverse", blocks=list(reversed(ordered))),
        ]
    definition.update(
        study_revision=revision
        or "development-review-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        purpose="development_review",
        synthetic=False,
        correctness_metrics="disabled",
        information_text="Provisional real development-run review. Test sessions only; no biomedical correctness claims or research launch.",
        consent_text="This is a local development review, not research participation.",
        closes_at=(datetime.now(timezone.utc) + timedelta(days=14)).isoformat(),
        analysis_plan="Development review only. All case correctness is unresolved; quality metrics disabled.",
        cases=cases,
        schedules=schedules,
        review_provenance=dict(
            run_manifest_sha256=file_hash(run / "run_manifest.json")[7:],
            selection_manifest_sha256=file_hash(root / "selection.json")[7:],
            source_run_id=portable_run.manifest()["run_id"],
            limitation="Provisional development outputs. Original DOID input and import-closure gates remain open.",
        ),
    )
    counts = Counter(tag for case in catalogue for tag in case["coverage"])
    from exact_inspect.review_metadata import ReviewBundleSummary

    about = ReviewBundleSummary(
        title="Provisional NCIT–DOID development review",
        prepared_at=datetime.now(timezone.utc).isoformat(),
        source_run_id=portable_run.manifest()["run_id"],
        run_manifest_sha256=file_hash(run / "run_manifest.json"),
        selection_sha256=file_hash(root / "selection.json"),
        policy_sha256=policy.policy_hash,
        source_count=len(cases),
        presented_pairs=pair_number,
        coverage=dict(counts),
        cases=[
            dict(
                case_id=c["case_id"],
                source=c["source"],
                targets=[t["entity"] for t in c["candidates"]],
            )
            for c in cases
        ],
        limitations=[
            "Development-run candidate selection is for interface review, not effectiveness evaluation.",
            "Correctness is unresolved and correctness metrics are disabled.",
            *generation_limitations(preparation_summary),
            *review_scope_limitations(plan, private_run.manifest().get("counts", {})),
            "DOID uses a declared derivative and root-only context; the ignored ext.owl import is not admitted.",
            *[
                f"Prepared inference is {c.manifest["capabilities"]["reasoner_inferred"]["status"]} for {c.manifest["name"]}; asserted navigation remains available."
                for c in contexts.values()
                if c.manifest["capabilities"]["reasoner_inferred"]["status"] != "complete"
            ],
            *[
                f"Coverage below three sources: {tag} ({counts[tag]})."
                for tag in CATEGORIES
                if counts[tag] < 3
            ],
        ],
        ontologies=[
            dict(
                ontology_version_id=v,
                name=c.manifest["name"],
                context_sha256=c.manifest["artifacts"]["context.sqlite"],
                root_sha256=c.manifest["identity"]["root_sha256"],
                scope=c.manifest["scope"],
                imports_complete=c.manifest["completeness"]["imports_complete"],
                entity_count=c.manifest["completeness"]["entity_count"],
                axiom_count=c.manifest["completeness"]["axiom_count"],
                inference_status=c.manifest["capabilities"]["reasoner_inferred"]["status"],
            )
            for v, c in contexts.items()
        ],
    )
    definition["review_bundle"] = about.model_dump(mode="json")
    publication = Publish.model_validate(dict(definition=definition, case_keys=keys))
    metadata = bundle.model_dump(mode="json", exclude={"package_id", "artifacts"})
    metadata["runs"] = {portable_run.manifest()["run_id"]: "runs/review"}
    metadata["explanations"] = generated_index
    metadata["provenance"] = {
        **metadata.get("provenance", {}),
        "review_selection_sha256": file_hash(root / "selection.json"),
        "purpose": "development_review",
        "source_run_manifest_sha256": file_hash(run / "run_manifest.json"),
        "review_bundle": about.model_dump(mode="json"),
        "description_dependencies": dependency_index,
        "generation_mode": generation_mode,
        "generation_summary": preparation_summary,
        "generation_attempt": generation_attempt,
    }
    metadata["capabilities"].update(
        review_bundle="exact-review-about/1",
        hierarchy="exact-hierarchy/1",
        descriptions="exact-description/2",
    )
    publish_bundle(package, **metadata)
    atomic_json(
        root / "catalogue.json",
        {
            "schema": "exact-review-catalogue/1",
            "cases": catalogue,
            "coverage": dict(counts),
            "missing_coverage": {tag: max(0, 3 - counts[tag]) for tag in CATEGORIES},
            "recorded_position_meaning": plan["recorded_position_meaning"],
            "generation_mode": generation_mode,
            "provider_batch_status": preparation_summary["provider_batch_status"],
            "generation_summary": preparation_summary,
            "generation_attempt": generation_attempt,
            "publication_sha256": canonical_hash(publication),
        },
    )
    # This is the completion marker, written only after the portable package and
    # coverage report exist. Serving never sees a partially prepared publication.
    atomic_json(publication_path, publication)
    return publication_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--generate-pairs", type=int, default=12)
    parser.add_argument("--revision")
    parser.add_argument(
        "--generation-cache-only",
        action="store_true",
        help="Assemble saved provider jobs only; fail rather than dispatch any missing identity",
    )
    parser.add_argument(
        "--generation-attempt",
        type=int,
        default=1,
        help="Explicit isolated provider attempt; 1 preserves the original generation cache, 2+ use separate generation-attempt-N directories",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["provider", "typed-only"],
        default="provider",
        help="Use typed-only in a separate root/revision for deterministic local preparation without provider dispatch",
    )
    parser.add_argument(
        "--inputs-only",
        action="store_true",
        help="Prepare real inputs without provider dispatch or publication",
    )
    args = parser.parse_args()
    if args.cases < 24 or args.generate_pairs < 12:
        parser.error("Full S1 preparation requires >=24 cases and >=12 real provider comparisons")
    print(
        prepare(
            args.root,
            args.run,
            args.cases,
            args.generate_pairs,
            args.revision,
            inputs_only=args.inputs_only,
            generation_mode=args.generation_mode,
            generation_attempt=args.generation_attempt,
            generation_cache_only=args.generation_cache_only,
        )
    )


if __name__ == "__main__":
    main()
