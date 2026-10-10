"""Package a completed diagnostic run separately from the fixed study publication."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from fastapi.testclient import TestClient

from exact_inspect.artifacts import atomic_json, publish_bundle
from exact_inspect.context import OntologyContext
from exact_inspect.contracts import VisibilityPolicy, file_hash
from exact_inspect.decisions import DecisionStore, import_run
from exact_inspect.decisions_export import export_policy_run
from exact_inspect.service import create_prepared_app
from tools.refresh_review_evidence import clone_file


def prepare(run, context, root):
    run, root = Path(run), Path(root)
    if root.exists():
        raise ValueError("Use a new independent preparation directory")
    status = json.loads((run / "smoke-status.json").read_bytes())
    if status["phase"] != "complete" or status["pairs"] != 9:
        raise ValueError("The bounded smoke has not completed with its nine recorded pairs")
    package = root / "package"
    shutil.copytree(context, package / "ontology", copy_function=clone_file)
    ctx = OntologyContext(package / "ontology")
    policy = VisibilityPolicy.model_validate(ctx.manifest["policy_filter"]["policy"])
    version = ctx.ontology_version_id
    import_run(
        run,
        root / "private",
        source_ontology_version_id=version,
        target_ontology_version_id=version,
        source_root_sha256=ctx.manifest["identity"]["root_sha256"],
        target_root_sha256=ctx.manifest["identity"]["root_sha256"],
        context_revision=ctx.manifest["artifacts"]["context.sqlite"],
        evidence_resolver=ctx.resolve_feature,
    )
    store = export_policy_run(
        DecisionStore(root / "private"), package / "run", contexts={version: ctx}, policy=policy
    )
    rid = store.manifest()["run_id"]
    path = publish_bundle(
        package,
        policy=policy,
        ontologies={version: "ontology"},
        runs={rid: "run"},
        provenance={
            "purpose": "development_review_trace_smoke",
            "study_pool": False,
            "smoke_status_sha256": file_hash(run / "smoke-status.json"),
            "limitation": "Nine new diagnostic DOID-to-DOID pairs, not retained NCIT/DOID study history or effectiveness evaluation",
        },
    )
    events, decisions, responses = Counter(), Counter(), []
    with store._connect() as db, TestClient(
        create_prepared_app(path, profile="local_app")
    ) as client:
        for pid, trace in db.execute("SELECT id,trace FROM pairs ORDER BY id"):
            response = client.get(
                "/api/v1/runs/" + quote(rid, safe="") + "/pair", params={"pair_id": pid}
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["pair_id"] == pid
            for event in result["events"]:
                events[event["stage"] + ":" + event["reason_code"]] += 1
            decisions[str(json.loads(trace)["saved_alignment_member"])] += 1
            responses.append(
                {"pair_id": pid, "status": response.status_code, "bytes": len(response.content)}
            )
    receipt = {
        "schema": "exact-review-trace-smoke-package/1",
        "package_sha256": file_hash(path),
        "run_id": rid,
        "ontology_version_id": version,
        "events": dict(events),
        "saved_membership": dict(decisions),
        "api": responses,
        "boundary": "Actual ASGI API with native real ontology and completed bounded matcher output; no rendered frontend or PostgreSQL study claim",
    }
    atomic_json(root / "verification.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("context", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    prepare(args.run, args.context, args.root)
