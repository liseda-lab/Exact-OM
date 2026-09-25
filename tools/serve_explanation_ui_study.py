"""Serve a synthetic three-case ranking study locally for frontend development.

Builds study resources from a package made by ``tools.build_explanation_ui_fixture``
with the real study builder, store and API (SQLite test mode), publishes it as a
synthetic study and prints test invitation links. Nothing here is a real study.

The study API only accepts its configured HTTPS origin. This harness binds to loopback
and presents the browser's ``http://localhost:<port>`` origin as that HTTPS origin; this
rewrite exists only in this development tool and never in a deployment.

    python -m tools.build_explanation_ui_fixture --output /tmp/exact-ui-fixture
    (cd explanations_visualizer && npm run build)
    python -m tools.serve_explanation_ui_study --fixture /tmp/exact-ui-fixture

Requires ``pyowl-core`` (ontology downloads are rendered from the fixture contexts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

INSTRUCTIONS = (
    "Rank the candidates you consider plausible equivalents of the source concept, with the "
    "best first. You may use fewer than five. If none appears equivalent, choose None of these; "
    "if you cannot judge, choose Insufficient information. The initial order and system scores "
    "are suggestions, not answers."
)
TUTORIAL = [
    "Each case shows one source concept and five candidates from the other ontology, in the "
    "system's initial order.",
    "Matching scores are suggestions from the matcher, not correctness. A high score can still "
    "be wrong.",
    "Read the definitions, synonyms, parents and defining facts. Generated descriptions are "
    "marked as generated and cite the original facts.",
    "Rank only the candidates you consider plausible equivalents. If none fits, choose None of "
    "these; if you cannot judge, choose Insufficient information.",
]


def publication(fixture: Path, out: Path) -> dict:
    """Build per-pair explanation resources, ontology downloads and a synthetic study."""
    from exact_inspect.context import OntologyContext
    from exact_inspect.context_resources import export_ontology_resource
    from exact_inspect.contracts import EntityRef, VisibilityPolicy
    from exact_inspect.decisions import DecisionStore
    from exact_inspect.generation import FactPacket
    from exact_inspect.study.builder import build_explanation_resource

    prepared = fixture / "prepared"
    stages = {}
    for directory in (prepared / "stages").iterdir():
        stages[json.loads((directory / "stage.json").read_text())["stage"]] = directory
    names = json.loads((stages["context-index"] / "contexts.json").read_text())
    contexts = {n: OntologyContext(stages["context-index"] / loc) for n, loc in names.items()}
    by_id = {c.ontology_version_id: c for c in contexts.values()}
    # The fixture's generations used the default exploration policy; reuse it so the
    # frozen packets and grounded claims stay valid.
    policy = VisibilityPolicy()
    policy_hash = policy.policy_hash.removeprefix("sha256:")
    store = DecisionStore(stages["run-import"] / "run")

    explanations = []
    for stage in ("profiles", "comparisons"):
        explanations.extend(json.loads((stages[stage] / f"{stage}.json").read_text()).values())
    packets = {}
    for result in explanations:
        request = prepared / "generations" / result["explanation_id"][7:] / "request.json"
        packets[result["explanation_id"]] = FactPacket.model_validate(
            json.loads(request.read_text())["packet"]
        )

    assets = []
    for name in ("NCIT", "DOID"):
        path = out / f"{name.lower()}.ofn"
        receipt = Path(export_ontology_resource(contexts[name], path, policy))
        content = path.read_bytes()
        assets.append(
            {
                "asset_id": f"{name.lower()}-ontology",
                "path": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "kind": "ontology",
                "media_type": "application/owl-functional",
                "policy_hash": policy_hash,
                "admission_receipt_path": receipt.name,
                "admission_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            }
        )

    def label(ref):
        return by_id[ref.ontology_version_id]._label(ref.model_dump(), None, policy)["value"]

    cases, keys = [], []
    for number, source in enumerate(store.sources(limit=20)["items"]):
        source_ref = EntityRef.model_validate(source["entity"])
        rows = sorted(
            store.candidates(source_ref.iri, limit=20)["items"],
            key=lambda row: row["ordinal_ranks"]["candidate_joint_rank"],
        )[:5]
        refs, candidates = [], []
        for index, row in enumerate(rows):
            target = EntityRef.model_validate(row["target"])
            # One resource per source-candidate pair: a per-case resource currently fails
            # when a case contains an entity and its asserted parent (see the F1 handoff).
            members = {(e.ontology_version_id, e.iri, e.kind) for e in (source_ref, target)}
            chosen = [
                r
                for r in explanations
                if all(
                    (e["ontology_version_id"], e["iri"], e["kind"]) in members
                    for e in r["entities"]
                )
            ]
            resource = build_explanation_resource(
                by_id,
                [source_ref, target],
                policy=policy,
                packets=[packets[r["explanation_id"]] for r in chosen],
                explanations=chosen,
                evidence={f"c{index + 1}": store.evidence(row["pair_id"])},
            )
            asset_id = f"explanation-{number}-c{index + 1}"
            content = resource.model_dump_json().encode()
            (out / f"{asset_id}.json").write_bytes(content)
            refs.append(asset_id)
            assets.append(
                {
                    "asset_id": asset_id,
                    "path": f"{asset_id}.json",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                    "kind": "explanation",
                    "media_type": "application/json",
                    "policy_hash": policy_hash,
                }
            )
            score = next(s for s in row["scores"] if s["name"] == "S_final")
            candidates.append(
                {
                    "candidate_id": f"c{index + 1}",
                    "entity": target.model_dump(),
                    "label": label(target),
                    "score": score["value"],
                    "score_meaning": score["meaning"],
                    "display_position": index + 1,
                }
            )
        case_id = f"case-{number}"
        cases.append(
            {
                "case_id": case_id,
                "source": source_ref.model_dump(),
                "source_label": label(source_ref),
                "transfer_group": f"group-{number}",
                "package_version": "ui-fixture/1",
                "candidates": candidates,
                "ontology_resource_ids": ["ncit-ontology", "doid-ontology"],
                "explanation_refs": refs,
            }
        )
        keys.append(
            {
                "case_id": case_id,
                "case_kind": "unresolved",
                "acceptable_candidate_ids": [],
                "adjudication_version": "not-adjudicated",
                "criterion": "Synthetic development fixture; no adjudication exists.",
                "evidence": ["None: development fixture"],
                "origin": "natural",
                "original_production_ranks": {
                    c["candidate_id"]: c["display_position"] for c in candidates
                },
            }
        )

    ids = [case["case_id"] for case in cases]
    forms = [
        {"explanation": ids[:2], "ontology_baseline": ids[2:]},
        {"explanation": ids[2:], "ontology_baseline": ids[:2]},
    ]
    schedules = [
        {
            "schedule_id": f"form{number}-{order[0]}",
            "blocks": [{"condition": c, "case_ids": form[c]} for c in order],
        }
        for number, form in enumerate(forms)
        for order in (("explanation", "ontology_baseline"), ("ontology_baseline", "explanation"))
    ]
    definition = {
        "study_revision": "ui-fixture-study/1",
        "software_version": "exact-explain-ui-1.0",
        "information_version": "dev-1",
        "information_text": "[Participant information supplied by the study owner.]",
        "consent_text": "[Consent statement supplied by the study owner.]",
        "instructions": INSTRUCTIONS,
        "setup_instructions": "[Setup instructions supplied with the study, including the "
        "Protégé version and the practice class to look up.]",
        "tutorial_steps": TUTORIAL,
        "closes_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "synthetic": True,
        "policy_hash": policy_hash,
        "visibility_policy": policy.model_dump(mode="json"),
        "analysis_plan": "Development fixture only; not analysed.",
        "cases": cases,
        "assets": assets,
        "schedules": schedules,
    }
    return {"definition": definition, "case_keys": keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--frontend-dir", type=Path, default=Path("explanations_visualizer/out"))
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--invitations", type=int, default=3)
    args = parser.parse_args()
    frontend = args.frontend_dir.resolve()
    if not (frontend / "participate" / "index.html").is_file():
        parser.error("Build the frontend first: cd explanations_visualizer && npm run build")
    out = args.fixture.resolve() / "study"
    out.mkdir(exist_ok=True)
    database = out / "study.sqlite3"
    database.unlink(missing_ok=True)

    from exact_inspect.study import create_study_app
    from exact_inspect.study.models import Publish

    origin = "https://localhost:8443"
    researcher = secrets.token_urlsafe(32)
    app = create_study_app(
        f"sqlite:///{database}",
        secrets.token_urlsafe(32),
        researcher,
        origin,
        assets_dir=out,
        allow_test_sqlite=True,
        frontend_dir=frontend,
    )
    frozen = publication(args.fixture.resolve(), out)
    (out / "publication.json").write_text(json.dumps(frozen, indent=1))
    store = app.state.study_store
    store.publish(Publish.model_validate(frozen))
    issued = store.issue(frozen["definition"]["study_revision"], args.invitations, test=True)

    class LoopbackOrigin:
        """Development only: present http://localhost as the configured HTTPS origin."""

        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                scope = {
                    **scope,
                    "headers": [
                        (
                            (k, origin.encode())
                            if k == b"origin" and v.startswith(b"http://localhost")
                            else (k, v)
                        )
                        for k, v in scope["headers"]
                    ],
                }
            await self.inner(scope, receive, send)

    print("Synthetic study (development only). Test invitation links:")
    for item in issued["invitations"]:
        print(f"  http://localhost:{args.port}{item['invitation']}")
    print(f"Researcher page: http://localhost:{args.port}/admin/  token: {researcher}")
    import uvicorn

    uvicorn.run(LoopbackOrigin(app), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
