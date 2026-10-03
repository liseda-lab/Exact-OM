"""Real indexed synthetic v2 package preparation; no runtime/provider generation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from exact_inspect.context import build_context_package
from exact_inspect.context_resources import export_ontology_resource
from exact_inspect.contracts import EntityRef, VisibilityPolicy, canonical_hash
from exact_inspect.generation import FactPacket, factual_fallback
from exact_inspect.study.builder import build_explanation_resource
from exact_inspect.study.models import Publish
from exact_inspect.study.v2_models import TutorialDefinition
from exact_inspect.study.workspace import build_workspace_scope


def _base_publication():
    cases, keys, schedules = [], [], []
    for index in range(4):
        candidates = [
            dict(
                candidate_id=f"candidate-{i}",
                entity=dict(ontology_version_id="pending", iri=f"urn:target:{i}", kind="class"),
                label=f"Synthetic candidate {i}",
                score=1 - i / 10,
                score_meaning="Invented fixture score; not probability of correctness",
                display_position=i,
            )
            for i in range(1, 6)
        ]
        cases.append(
            dict(
                case_id=f"case-{index}",
                source=dict(ontology_version_id="pending", iri=f"urn:source:{index}", kind="class"),
                source_label=f"Synthetic source {index}",
                transfer_group=f"source-{index}",
                package_version="synthetic-package-v2",
                candidates=candidates,
                ontology_resource_ids=["source", "target"],
                explanation_refs=[],
            )
        )
        keys.append(
            dict(
                case_id=f"case-{index}",
                case_kind="answer_present" if index < 2 else "answer_absent",
                acceptable_candidate_ids=[f"candidate-{index+1}"] if index < 2 else [],
                adjudication_version="synthetic-key-v2",
                criterion="Invented fixture equivalence only",
                evidence=["Synthetic test oracle; not biomedical adjudication"],
                origin="natural",
                original_production_ranks={f"candidate-{i}": i for i in range(1, 6)},
            )
        )
    for form in range(2):
        blocks = [
            dict(
                condition="explanation",
                case_ids=["case-0", "case-2"] if form == 0 else ["case-1", "case-3"],
            ),
            dict(
                condition="ontology_baseline",
                case_ids=["case-1", "case-3"] if form == 0 else ["case-0", "case-2"],
            ),
        ]
        schedules.extend(
            [
                dict(schedule_id=f"form-{form}-forward", blocks=blocks),
                dict(schedule_id=f"form-{form}-reverse", blocks=list(reversed(blocks))),
            ]
        )
    return {
        "definition": dict(
            study_revision="synthetic-v2-" + uuid4().hex,
            information_text="Synthetic development session. No human study is launched.",
            consent_text="Synthetic consent flow only, not research consent wording.",
            instructions="Rank plausible equivalents, select None of these or Insufficient information. External inspection is optional, methods may be combined or changed, and you report actual methods after each case.",
            setup_instructions="Tool-neutral synthetic setup.",
            tutorial_steps=[],
            closes_at=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
            synthetic=True,
            analysis_plan="Synthetic integration tests only; not a participant study.",
            cases=cases,
            schedules=schedules,
        ),
        "case_keys": keys,
    }


def publication_v2(
    root, *, navigation_size=0, export_version="exact-study-analysis/2", paged_facts=0
):
    import pyowl_core as core

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    legacy = _base_publication()
    study = legacy["definition"]
    authored = json.loads(
        (Path(__file__).resolve().parent / "fixtures/study_tutorial_v2.json").read_text()
    )
    tutorial = authored["tutorial"]
    contexts = {}
    for name, indices in (("source", range(4)), ("target", range(1, 6))):
        axioms = [f"Declaration(Class(<urn:{name}:parent>))"]
        for i in indices:
            iri = f"urn:{name}:{i}"
            axioms.extend(
                [
                    f"Declaration(Class(<{iri}>))",
                    f'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <{iri}> "Synthetic {name} {i}")',
                    f'AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <{iri}> "Invented test concept {name} {i}")',
                    f"SubClassOf(<{iri}> <urn:{name}:parent>)",
                ]
            )
        # Optional full-scope fixture: these original entities are deliberately
        # outside focal explanation packets, with paged children and facts.
        if navigation_size:
            for i in range(navigation_size):
                iri = f"urn:{name}:navigation:{i:03}"
                axioms.extend(
                    [
                        f"Declaration(Class(<{iri}>))",
                        f'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <{iri}> "Navigation {name} {i:03}")',
                        f"SubClassOf(<{iri}> <urn:{name}:parent>)",
                    ]
                )
                for focal in indices:
                    axioms.extend(
                        [
                            f"SubClassOf(<{iri}> <urn:{name}:{focal}>)",
                            f'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#comment> <urn:{name}:{focal}> "Original navigation fact {i:03}")',
                        ]
                    )
        # Optional frontend paging fixture (spec 19 F16): one non-focal source class whose
        # definitions, alternate definitions, synonyms, restrictions, comments and parents
        # each exceed the entity context's first page. Default 0 leaves fixtures unchanged.
        if paged_facts and name == "source":
            axioms.extend(_paged_facts_axioms(paged_facts))
        context = build_context_package(
            core.load_snapshot(("Ontology(" + " ".join(axioms) + ")").encode()),
            root / f"{name}-v2-context",
        )
        contexts[name] = context
    for case in study["cases"]:
        case["source"]["ontology_version_id"] = contexts["source"].ontology_version_id
        for candidate in case["candidates"]:
            candidate["entity"]["ontology_version_id"] = contexts["target"].ontology_version_id
    replacement = {}
    for index, (old_version, document) in enumerate(authored["ontology_documents"].items()):
        name = f"practice-{index}"
        contexts[name] = build_context_package(
            core.load_snapshot(
                document.encode(),
                options=core.LoadOptions(
                    imports=core.ImportPolicy.IGNORE, preserve_source_map=True
                ),
            ),
            root / f"{name}-context",
        )
        replacement[old_version] = contexts[name].ontology_version_id
    policy = VisibilityPolicy(
        policy_id="synthetic-study-v2",
        ontology_ids=tuple(c.ontology_version_id for c in contexts.values()),
    )
    policy_hash = policy.policy_hash.removeprefix("sha256:")
    by_version = {c.ontology_version_id: c for c in contexts.values()}
    assets = []
    for name, context in contexts.items():
        path = root / f"{name}-v2.ofn"
        receipt = Path(export_ontology_resource(context, path, policy))
        assets.append(
            dict(
                asset_id=name,
                path=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
                kind="ontology",
                media_type="application/owl-functional",
                policy_hash=policy_hash,
                admission_receipt_path=str(receipt.relative_to(root)),
                admission_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                title=f"{name} synthetic ontology",
                role="source" if name in {"source", "practice-0"} else "target",
                ontology_version_id=context.ontology_version_id,
                version_label="synthetic/2",
                license_note="Invented test material; CC0",
                information_notice="Only policy-admitted ontology facts are included.",
            )
        )
    for entity in [
        tutorial["case"]["source"],
        *(c["entity"] for c in tutorial["case"]["candidates"]),
    ]:
        entity["ontology_version_id"] = replacement[entity["ontology_version_id"]]
    tutorial["case"]["case_id"] = "practice-case"
    tutorial["case"]["ontology_resources"] = [
        {
            k: v
            for k, v in a.items()
            if k not in {"path", "admission_receipt_path", "admission_receipt_sha256"}
        }
        for a in assets
        if a["asset_id"].startswith("practice-")
    ]
    tutorial.update(
        version="tutorial/2.0-native-2" + (f"-nav{navigation_size}" if navigation_size else ""),
        practice_id="practice-case",
        transfer_group="synthetic-tutorial-transfer/2",
        assessment_version="assessment/2",
        compatible_builds=["exact-explain-ui-1.2"],
        grading=authored["grading"],
    )
    scopes = []
    cases = [*study["cases"], tutorial["case"]]
    for case_index, case in enumerate(cases):
        practice = case_index == len(cases) - 1
        entities = [
            EntityRef.model_validate(e)
            for e in [case["source"], *(c["entity"] for c in case["candidates"])]
        ]
        # All practice nodes are included so the tutorial can genuinely navigate the full graphs.
        resource_entities = list(entities)
        if practice:
            for ctx in contexts.values():
                if ctx.ontology_version_id in {e.ontology_version_id for e in entities}:
                    with ctx._connection() as db:
                        for row in db.execute("SELECT iri,kind FROM entities"):
                            entity = EntityRef(
                                ontology_version_id=ctx.ontology_version_id,
                                iri=row[0],
                                kind=row[1],
                            )
                            if entity not in resource_entities and entity.kind in {
                                "class",
                                "object_property",
                                "data_property",
                                "individual",
                            }:
                                resource_entities.append(entity)
        packets, generated = [], []
        for pair in [[e] for e in entities] + [[entities[0], target] for target in entities[1:]]:
            facts = [
                f
                for e in pair
                for f in by_version[e.ontology_version_id].facts(e, limit=100, policy=policy)[
                    "items"
                ]
            ]
            packet = FactPacket(
                task="entity_profile" if len(pair) == 1 else "pair_comparison",
                entities=pair,
                context_hashes=[
                    by_version[e.ontology_version_id].manifest["artifacts"]["context.sqlite"]
                    for e in pair
                ],
                policy_hash=policy.policy_hash,
                facts=facts[:100],
                missingness=[],
                selection=(
                    {
                        "limit": 100,
                        "available_count": len(facts),
                        "selected_count": min(100, len(facts)),
                    }
                    if navigation_size
                    else {}
                ),
            )
            output = factual_fallback(packet)
            if not output.claims:
                continue
            manifest = dict(
                ontology_context_hashes=packet.context_hashes,
                visibility_policy_hash=policy.policy_hash,
                packet_hash=canonical_hash(packet),
                prompt_hash=canonical_hash("synthetic offline original-excerpt preparation"),
                output_schema="exact-explain-generation/1",
                requested_model="none-synthetic-extracts",
                returned_model=None,
                provider=None,
                parameters_hash=canonical_hash({}),
                language="en",
                response_hash=None,
                status="failed",
            )
            generated.append(
                dict(
                    artifact_type="generated_explanation",
                    contract_version="exact-explain/1.0",
                    explanation_id=canonical_hash(
                        {"packet": packet.model_dump(), "fixture": "study-v2"}
                    ),
                    task=packet.task,
                    entities=[e.model_dump() for e in pair],
                    claims=[
                        {
                            **c.model_dump(),
                            "claim_id": canonical_hash(
                                {
                                    "packet": canonical_hash(packet),
                                    "claim": c.model_dump(exclude={"claim_id"}),
                                }
                            ),
                        }
                        for c in output.claims
                    ],
                    grounding_status="validated",
                    manifest=manifest,
                    limitations=output.limitations,
                    fixture_provenance="Synthetic offline original-excerpt fallback; no matcher or model run",
                )
            )
            packets.append(packet)
        asset_id = "practice-explanations" if practice else f"case-explanations-{case_index}"
        evidence = {}
        for candidate in case["candidates"]:
            entity = EntityRef.model_validate(candidate["entity"])
            facts = by_version[entity.ontology_version_id].facts(entity, limit=100, policy=policy)[
                "items"
            ]
            if facts:
                evidence[candidate["candidate_id"]] = [
                    dict(
                        evidence_id="synthetic-evidence-" + candidate["candidate_id"],
                        entity=entity.model_dump(),
                        fact_ids=[facts[0]["fact_id"]],
                        channel="lexical",
                        side="target",
                    )
                ]
        resource = build_explanation_resource(
            by_version,
            resource_entities,
            policy=policy,
            packets=packets,
            explanations=generated,
            evidence=evidence,
        )
        payload = resource.model_dump_json().encode()
        path = root / f"{asset_id}.json"
        path.write_bytes(payload)
        assets.append(
            dict(
                asset_id=asset_id,
                path=path.name,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                kind="explanation",
                media_type="application/json",
                policy_hash=policy_hash,
            )
        )
        case["explanation_refs"] = [asset_id]
        workspace_evidence = {
            cid: [
                {
                    **row,
                    "role": "synthetic practice feature" if practice else "synthetic test feature",
                    "feature_id": row["evidence_id"],
                    "source_axiom_refs": row["fact_ids"],
                    "axiom_origins": [],
                    "semantic_terms": {},
                    "historical_item_alias": None,
                    "display": {},
                    "values": {},
                    "interpretation": "projected",
                    "provenance_status": "synthetic_fixture",
                    "status": "available",
                    "reason": None,
                }
                for row in rows
            ]
            for cid, rows in evidence.items()
        }
        scope = build_workspace_scope(
            root,
            scope_id=f"synthetic-v2-{case_index}",
            kind="tutorial" if practice else "case",
            case=case,
            contexts={e.ontology_version_id: by_version[e.ontology_version_id] for e in entities},
            policy=policy,
            explanations=generated,
            evidence=workspace_evidence,
            components=["context", "hierarchy", "profiles", "comparison", "evidence"],
        )
        scopes.append(scope.model_dump(mode="json"))
    tutorial = TutorialDefinition.model_validate(tutorial).model_dump(mode="json")
    study.update(
        contract_version="exact-study/2.0",
        software_version="exact-explain-ui-1.2",
        form_version="exact-study-forms/2",
        information_version="synthetic-information-v2",
        policy_hash=policy_hash,
        visibility_policy=policy.model_dump(mode="json"),
        tutorial=tutorial,
        assets=assets,
        workspace_scopes=scopes,
        resource_scope_analysis_rule="Describe supplied-only, different/additional, unsure and unanswered separately; no outcome-dependent exclusion.",
        protocol_versions=dict(
            setup="setup/2",
            tutorial=tutorial["version"],
            assessment="assessment/2",
            forms="exact-study-forms/2",
            information="synthetic-information-v2",
            resource_policy="synthetic-study-v2",
            software="exact-explain-ui-1.2",
            export=export_version,
        ),
        setup_instructions="Read the instructions and confirm access to the supplied resources. External tools are optional; any combination of methods may be used.",
    )
    return Publish.model_validate(legacy)


def _paged_facts_axioms(count, parents=55):
    """Axioms for `urn:source:paged`: every displayed category spans more than one page."""
    iri = "urn:source:paged"
    relation = "urn:source:paged-relation"
    axioms = [
        f"Declaration(Class(<{iri}>))",
        f'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <{iri}> "Paged facts source")',
        f"Declaration(ObjectProperty(<{relation}>))",
        f'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <{relation}> "paged relation")',
    ]
    for i in range(count):
        filler = f"urn:source:paged:filler:{i:03}"
        axioms += [
            f'AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <{iri}> "Paged definition {i:03}")',
            f'AnnotationAssertion(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P325> <{iri}> "Paged alternate definition {i:03}")',
            f'AnnotationAssertion(<http://www.geneontology.org/formats/oboInOwl#hasExactSynonym> <{iri}> "paged synonym {i:03}")',
            f'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#comment> <{iri}> "Paged comment {i:03}")',
            f"Declaration(Class(<{filler}>))",
            f'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <{filler}> "Paged filler {i:03}")',
            f"SubClassOf(<{iri}> ObjectSomeValuesFrom(<{relation}> <{filler}>))",
        ]
    for i in range(parents):
        parent = f"urn:source:paged:parent:{i:03}"
        axioms += [
            f"Declaration(Class(<{parent}>))",
            f'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <{parent}> "Paged parent {i:03}")',
            f"SubClassOf(<{iri}> <{parent}>)",
        ]
    return axioms


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a synthetic exact-study/2.0 package for local integration checks; does not publish or recruit"
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--navigation-size",
        type=int,
        default=0,
        help="Add non-focal entities and paged context for integration checks (0-200)",
    )
    parser.add_argument(
        "--paged-facts",
        type=int,
        default=0,
        help="Add one non-focal source class with this many facts per displayed category and 55 parents (0-200)",
    )
    parser.add_argument(
        "--export-version",
        choices=["exact-study-analysis/2", "exact-study-analysis/3"],
        default="exact-study-analysis/3",
        help="Freeze the export default explicitly; /2 retains historical compatibility semantics",
    )
    args = parser.parse_args()
    if not 0 <= args.navigation_size <= 200:
        parser.error("Navigation size must be between 0 and 200")
    if not 0 <= args.paged_facts <= 200:
        parser.error("Paged facts must be between 0 and 200")
    if args.destination.exists() and any(args.destination.iterdir()):
        parser.error("Destination must be new or empty; frozen packages are never overwritten")
    frozen = publication_v2(
        args.destination,
        navigation_size=args.navigation_size,
        export_version=args.export_version,
        paged_facts=args.paged_facts,
    )
    path = args.destination / "publication.json"
    path.write_text(frozen.model_dump_json(indent=2) + "\n")
    print(path.resolve())


if __name__ == "__main__":
    main()
