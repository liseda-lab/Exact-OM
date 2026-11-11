"""Prepare final-run study inputs and isolated review resources using existing contexts.

Private keys and provider packets stay outside the asset root. This command does not
publish to the live database, copy ontology databases, classify, or run a matcher.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from exact_inspect.artifacts import atomic_json
from exact_inspect.context import OntologyContext
from exact_inspect.contracts import (
    EntityRef,
    VisibilityPolicy,
    canonical_hash,
    file_hash,
)
from exact_inspect.decisions import selected_evidence
from exact_inspect.description_review import validate_reviewed_artifact
from exact_inspect.evidence_metadata import (
    filter_evidence_items,
    filter_pair_metadata,
    pair_evidence_metadata,
)
from exact_inspect.generation import FactPacket, comparison_packet
from exact_inspect.readable_descriptions import revise
from exact_inspect.study.builder import build_explanation_resource
from exact_inspect.study.models import Publish
from exact_inspect.study.workspace import build_workspace_scope
from tools.final_study_copy import OWNER_DETAILS, participant_information
from tools.study_preparation_support import (
    deterministic_result,
    literal_label,
    packet_for,
)


def read(path):
    return json.loads(Path(path).read_bytes())


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen preparation changed: {path.name}")
    else:
        atomic_json(path, value)


def open_contexts(assets, predecessor, catalogue):
    selection_policy = VisibilityPolicy.model_validate(catalogue["policy"])
    policy = VisibilityPolicy.model_validate(predecessor["definition"]["visibility_policy"])
    # The private selector gave its policy a new name and named the two roots.
    # Reuse the existing physical filter only when the allowed categories and
    # mapping-withholding rule are identical; scopes still bind exact versions.
    versions = {c["ontology_version_id"] for c in catalogue["contexts"].values()}
    if (
        selection_policy.categories != policy.categories
        or selection_policy.allow_mapping_xrefs != policy.allow_mapping_xrefs
        or any(
            not policy.allows_ontology(v) or not selection_policy.allows_ontology(v)
            for v in versions
        )
    ):
        raise ValueError("Predecessor and selection permit different ontology facts")
    refs = next(s for s in predecessor["definition"]["workspace_scopes"] if s["kind"] == "case")[
        "contexts"
    ]
    contexts = {}
    for ref in refs:
        path = (assets / ref["path"]).resolve(strict=True)
        if not path.is_relative_to(assets):
            raise ValueError("Context escapes asset root")
        for name, expected in [
            ("manifest.json", ref["manifest_sha256"]),
            ("context.sqlite", ref["database_sha256"]),
        ]:
            if file_hash(path / name).removeprefix("sha256:") != expected.removeprefix("sha256:"):
                raise ValueError(f"Reused context {name} changed")
        context = OntologyContext(
            path, verified_database_hash="sha256:" + ref["database_sha256"].removeprefix("sha256:")
        )
        wanted = next(
            c
            for c in catalogue["contexts"].values()
            if c["ontology_version_id"] == context.ontology_version_id
        )
        if context.manifest["identity"]["root_sha256"] != wanted["source_sha256"]:
            raise ValueError("Final run and prepared ontology root differ")
        contexts[context.ontology_version_id] = context
    return contexts, policy


def prepare_inputs(args):
    catalogue = read(args.catalogue)
    predecessor = read(args.predecessor)
    binding = {
        "catalogue_sha256": file_hash(args.catalogue),
        "predecessor_sha256": file_hash(args.predecessor),
        "option": "forty_unequal",
    }
    freeze(args.private / "inputs-binding.json", binding)
    if (args.private / "prepared-inputs.json").exists():
        return read(args.private / "prepared-inputs.json")
    contexts, policy = open_contexts(args.assets, predecessor, catalogue)
    selected = {c["case_id"]: c for c in catalogue["cases"]}
    selected = [selected[cid] for cid in catalogue["options"]["forty_unequal"]]
    entities = {
        iri: EntityRef.model_validate(value["entity"])
        for iri, value in catalogue["entities"].items()
    }
    profiles, packets, results = {}, {}, {}
    for iri, entity in entities.items():
        packet, _ = packet_for(entity, contexts, policy)
        result = revise(deterministic_result(packet), contexts, policy)
        profiles[iri] = result
        packets[result["manifest"]["packet_hash"]] = result["support_packet"]
        results[result["explanation_id"]] = result
        # Enrich the private review with actual ASTs and referenced labels, never
        # substitute expression hashes for a readable hierarchy.
        info = catalogue["entities"][iri]
        for page in info["facts"].values():
            for fact in page["items"]:
                if fact["value"]["term_type"] == "expression_ref":
                    fact["value"]["ast"] = contexts[entity.ontology_version_id].axiom(
                        fact["axiom_ref"], policy=policy
                    )["ast"]
        info["prepared_profile"] = result
        info["generated_status"] = "typed_rendering_prepared_provider_prose_pending"
    cases, keys, evidence, metadata, case_results = [], [], {}, {}, {}
    for selected_case in selected:
        source = entities[selected_case["source"]]
        candidates = []
        ids = [profiles[selected_case["source"]]["explanation_id"]]
        for row in selected_case["candidates"]:
            target = entities[row["target"]]
            candidate_id = f"{selected_case['case_id']}-candidate-{row['original_rank']}"
            candidates.append(
                dict(
                    candidate_id=candidate_id,
                    entity=target.model_dump(),
                    label=literal_label(target, contexts, policy),
                    score=row["score"],
                    score_meaning="Exact 2.0 recorded candidate score.",
                    display_position=row["original_rank"],
                )
            )
            target_profile, source_profile = (
                profiles[row["target"]],
                profiles[selected_case["source"]],
            )
            ids.append(target_profile["explanation_id"])
            packet = comparison_packet(
                FactPacket.model_validate(source_profile["support_packet"]),
                FactPacket.model_validate(target_profile["support_packet"]),
                [source_profile, target_profile],
            )
            result = revise(deterministic_result(packet), contexts, policy)
            packets[result["manifest"]["packet_hash"]] = result["support_packet"]
            results[result["explanation_id"]] = result
            ids.append(result["explanation_id"])
            row["prepared_comparison"] = result
            native = row.get("matcher_record") or {}
            record = {**native, **native.get("selected_attributions", {})}
            raw = selected_evidence(
                record,
                source.model_dump(),
                target.model_dump(),
                resolver=lambda item, entity: contexts[
                    entity["ontology_version_id"]
                ].resolve_feature(item, entity),
            )
            safe = filter_evidence_items(raw, contexts, policy, physically_filtered=True)
            events = (
                [{"stage": "prefilter", "reason_code": "exact_match_protected"}]
                if row["evidence_status"] == "exact_match"
                else []
            )
            pair = pair_evidence_metadata(
                record,
                raw,
                source.model_dump(),
                target.model_dump(),
                [catalogue["native_result"]["sha256"]],
                events=events,
            )
            pair = filter_pair_metadata(pair, safe)
            evidence[candidate_id] = safe
            metadata[candidate_id] = {
                "source": source.model_dump(),
                "target": target.model_dump(),
                "evidence_metadata": pair,
            }
            row["prepared_evidence"] = safe
            row["prepared_evidence_metadata"] = pair
        cid = selected_case["case_id"]
        case = dict(
            case_id=cid,
            source=source.model_dump(),
            source_label=literal_label(source, contexts, policy),
            transfer_group=canonical_hash(source),
            package_version="exact2-final-20261110",
            candidates=candidates,
            ontology_resource_ids=["source", "target"],
            explanation_refs=[cid + "-explanation"],
        )
        cases.append(case)
        case_results[cid] = list(dict.fromkeys(ids))
        keys.append(
            dict(
                case_id=cid,
                case_kind="answer_present",
                acceptable_candidate_ids=[
                    c["candidate_id"]
                    for c in candidates
                    if c["entity"]["iri"] == selected_case["reference_target"]
                ],
                adjudication_version="bioml-private-test-reference/20261110",
                criterion="Exactly one known reference match among the five untouched final candidates.",
                evidence=[
                    "Reference SHA256 " + catalogue["reference"]["sha256"],
                    "Additional reference targets outside these five: "
                    + str(len(selected_case["other_reference_targets_outside_five"])),
                ],
                origin="natural",
                original_production_ranks={
                    c["candidate_id"]: c["display_position"] for c in candidates
                },
                construction_provenance=None,
            )
        )
        print(json.dumps({"prepared_case": cid, "cases_done": len(cases)}), flush=True)
    data = {
        "binding": binding,
        "cases": cases,
        "case_keys": keys,
        "packets": packets,
        "results": results,
        "case_results": case_results,
        "evidence": evidence,
        "pair_metadata": metadata,
    }
    freeze(args.private / "catalogue-prepared.json", catalogue)
    freeze(args.private / "prepared-inputs.json", data)
    # Only facts and independent profiles are provider inputs. No private selection
    # key, rank, score, original native evidence, or query ID enters this file.
    freeze(
        args.private / "provider-inputs.json",
        {"schema": "exact-final-prose-inputs/1", "results": results},
    )
    return data


def assemble(args, inputs):
    catalogue, predecessor = read(args.catalogue), read(args.predecessor)
    contexts, policy = open_contexts(args.assets, predecessor, catalogue)
    overrides = read(args.reviewed_results)["results"] if args.reviewed_results else {}
    for old_id, value in overrides.items():
        if old_id not in inputs["results"]:
            raise ValueError("Reviewed result does not belong to this preparation")
        if value["support_packet"] != inputs["results"][old_id]["support_packet"]:
            raise ValueError("Reviewed result changed the original fact packet")
        validate_reviewed_artifact(value)
    d = copy.deepcopy(predecessor["definition"])
    copy_text = participant_information(OWNER_DETAILS)
    d.update(
        study_revision=args.revision,
        software_version="final-study-review/1",
        information_version="final-study-information/20261111",
        information_text=copy_text["information_text"],
        consent_text=copy_text["consent_text"],
        instructions=copy_text["instructions"],
        setup_instructions=copy_text["setup_instructions"],
        tutorial_skip_allowed=True,
        consultation_scope="baseline_cases",
        schedules=[],
        cases=inputs["cases"],
        closes_at=(datetime.now(timezone.utc) + timedelta(days=31)).isoformat(),
    )
    d["protocol_versions"].update(
        information=d["information_version"], software=d["software_version"]
    )
    d["tutorial"]["compatible_builds"] = list(
        dict.fromkeys(d["tutorial"]["compatible_builds"] + [d["software_version"]])
    )
    # A changed build allowlist is part of the frozen tutorial identity.
    d["tutorial"]["hash"] = ""
    d["analysis_plan"] = (
        "Private owner review of the 40-case final-run pool. Ten distinct cases per participant: one reference rank 1–5 in each condition; counterbalanced condition order. Inspection methods collected after no-tool cases only. Correctness metrics disabled until owner/protocol review and a separately approved research publication."
    )
    d["assets"] = [
        a
        for a in d["assets"]
        if a["asset_id"].startswith("practice-") or a["asset_id"] in {"source", "target"}
    ]
    d["workspace_scopes"] = [s for s in d["workspace_scopes"] if s["kind"] == "tutorial"]
    d["sampling"] = dict(
        version="rank-balanced/1",
        selection_manifest_sha256=file_hash(args.catalogue)[7:],
        reference_alignment_sha256=catalogue["reference"]["sha256"],
        strata=[
            dict(
                reference_rank=rank,
                case_ids=[
                    c["case_id"]
                    for c in catalogue["cases"]
                    if c["reference_rank"] == rank
                    and c["case_id"] in catalogue["options"]["forty_unequal"]
                ],
            )
            for rank in range(1, 6)
        ],
    )
    d["review_provenance"] = {
        **d["review_provenance"],
        "run_manifest_sha256": catalogue["run"]["sha256"],
        "selection_manifest_sha256": file_hash(args.catalogue)[7:],
        "source_run_id": "exact2-ncit-doid-final-20261110-H0",
        "limitation": "Private owner review of the final Exact 2.0 run; research publication and recruitment have not started.",
    }
    summary = d["review_bundle"]
    summary.update(
        title="Final Exact 2.0 NCIT–DOID pool · owner review",
        prepared_at=datetime.now(timezone.utc).isoformat(),
        source_run_id="exact2-ncit-doid-final-20261110-H0",
        run_manifest_sha256="sha256:" + catalogue["run"]["sha256"],
        selection_sha256=file_hash(args.catalogue),
        source_count=40,
        presented_pairs=200,
        coverage={
            "original_definition_all_six": sum(c["quality"][0] == 6 for c in catalogue["cases"])
        },
        cases=[
            dict(
                case_id=c["case_id"],
                source=c["source"],
                targets=[x["entity"] for x in c["candidates"]],
            )
            for c in inputs["cases"]
        ],
        limitations=[
            "Private owner review; not a research launch.",
            "Reference-balanced individual assignments from an unequal 40-source pool; rare strata are sampled more often.",
            "Partial ELK navigation retains its original scope and incompleteness.",
            (
                "Luna prose has individually recorded independent claim reviews; rejected or unavailable drafts retain typed original-fact descriptions."
                if args.reviewed_results
                else "Provider prose is pending; this revision uses validated typed descriptions and original facts."
            ),
        ],
        composition_identity=None,
        delivered_layers=None,
        selected_evidence_counts=None,
        recorded_bridge_counts=None,
    )
    out = args.assets / args.revision
    out.mkdir(exist_ok=False)
    for case in inputs["cases"]:
        values = [
            overrides.get(key, inputs["results"][key])
            for key in inputs["case_results"][case["case_id"]]
        ]
        candidates = {c["candidate_id"] for c in case["candidates"]}
        evidence = {k: v for k, v in inputs["evidence"].items() if k in candidates}
        metadata = {k: v for k, v in inputs["pair_metadata"].items() if k in candidates}
        # Each independently prepared description retains its whole support packet.
        # Combining all eleven into one resource can exceed the frozen 64 KiB
        # label limit; partition resources instead of dropping citations or
        # weakening admission. The workspace still presents one coherent case.
        case["explanation_refs"] = []
        for number, value in enumerate(values, 1):
            value_entities = [EntityRef.model_validate(e) for e in value["entities"]]
            pair_ids = {
                c["candidate_id"]
                for c in case["candidates"]
                if value["task"] == "pair_comparison" and c["entity"] == value["entities"][1]
            }
            resource = build_explanation_resource(
                contexts,
                value_entities,
                policy=policy,
                packets=[FactPacket.model_validate(value["support_packet"])],
                explanations=[value],
                evidence={k: v for k, v in evidence.items() if k in pair_ids},
                pair_metadata={k: v for k, v in metadata.items() if k in pair_ids},
            )
            asset_id = f"{case['case_id']}-explanation-{number}"
            path = out / (asset_id + ".json")
            atomic_json(path, resource)
            case["explanation_refs"].append(asset_id)
            d["assets"].append(
                dict(
                    asset_id=asset_id,
                    path=path.relative_to(args.assets).as_posix(),
                    sha256=file_hash(path)[7:],
                    size_bytes=path.stat().st_size,
                    kind="explanation",
                    media_type="application/json",
                    policy_hash=policy.policy_hash[7:],
                )
            )
        scope = build_workspace_scope(
            args.assets,
            scope_id=args.revision + ":" + case["case_id"],
            kind="case",
            case=case,
            contexts=contexts,
            policy=policy,
            explanations=values,
            evidence=evidence,
            pair_metadata=metadata,
            components=["context", "hierarchy", "profiles", "comparison", "evidence"],
            reuse_prepared_contexts=True,
        )
        d["workspace_scopes"].append(scope.model_dump(mode="json"))
        print(
            json.dumps(
                {"assembled_case": case["case_id"], "completed": len(d["workspace_scopes"]) - 1}
            ),
            flush=True,
        )
    # Preserve the real reference keys privately. Development-review sessions keep
    # unresolved evaluation keys and cannot be exported as scored research data.
    review_keys = [
        dict(
            case_id=c["case_id"],
            case_kind="unresolved",
            acceptable_candidate_ids=[],
            adjudication_version="owner-review/1",
            criterion="Owner review only; correctness metrics disabled.",
            evidence=["Real reference keys are separately retained for review."],
            origin="unresolved",
            original_production_ranks={},
            construction_provenance=None,
        )
        for c in inputs["cases"]
    ]
    publication = Publish.model_validate(dict(definition=d, case_keys=review_keys))
    freeze(args.private / (args.revision + ".json"), publication.model_dump(mode="json"))
    freeze(args.private / "reference-keys.json", inputs["case_keys"])
    print(
        json.dumps(
            {
                "publication": str(args.private / (args.revision + ".json")),
                "cases": len(d["cases"]),
                "evidence_counts": dict(
                    Counter(
                        r["origin_resolution"] for rows in inputs["evidence"].values() for r in rows
                    )
                ),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["assets", "catalogue", "predecessor", "private"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--revision", default="final-pool-owner-review-20261111")
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--reviewed-results", type=Path)
    args = parser.parse_args()
    args.assets, args.private = args.assets.resolve(), args.private.resolve()
    if args.private.is_relative_to(args.assets) or args.catalogue.resolve().is_relative_to(
        args.assets
    ):
        raise ValueError("Private preparation and catalogue must be outside served assets")
    args.private.mkdir(parents=True, exist_ok=True)
    inputs = prepare_inputs(args)
    if not args.inputs_only:
        assemble(args, inputs)


if __name__ == "__main__":
    main()
