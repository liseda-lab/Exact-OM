"""Verify real original-fact admission before exercising the frozen frontend model."""

import argparse
import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path

from exact_inspect.context import OntologyContext
from exact_inspect.contracts import VisibilityPolicy, file_hash


def verify(root, frontend_module, node):
    root = Path(root)
    publication = json.loads((root / "publication.json").read_bytes())
    definition = publication["definition"]
    policy = VisibilityPolicy.model_validate(definition["visibility_policy"])
    contexts, axioms, pairs = {}, {}, []
    for case in definition["cases"]:
        scope = next(
            s
            for s in definition["workspace_scopes"]
            if s["case_id"] == case["case_id"] and s["kind"] == "case"
        )
        for ref in scope["contexts"]:
            contexts.setdefault(ref["ontology_version_id"], None)
            if contexts[ref["ontology_version_id"]] is None:
                contexts[ref["ontology_version_id"]] = OntologyContext(root / ref["path"])
        with sqlite3.connect((root / scope["index_path"]).as_uri() + "?mode=ro", uri=True) as db:
            for candidate in case["candidates"]:
                items = [
                    json.loads(r[0])
                    for r in db.execute(
                        "SELECT payload FROM evidence WHERE candidate_id=? ORDER BY id",
                        (candidate["candidate_id"],),
                    )
                ]
                used = {}
                for item in items:
                    assert item["entity"] == (
                        case["source"] if item["side"] == "source" else candidate["entity"]
                    )
                    context = contexts[item["entity"]["ontology_version_id"]]
                    for fid in item["fact_ids"]:
                        context.fact(fid, item["entity"], policy=policy)
                        key = (context.ontology_version_id, fid)
                        if key not in axioms:
                            axioms[key] = context.axiom(fid, policy=policy)
                        used[fid] = axioms[key]
                pairs.append(
                    {
                        "case_id": case["case_id"],
                        "candidate_id": candidate["candidate_id"],
                        "source": case["source"],
                        "target": candidate["entity"],
                        "items": items,
                        "axioms": used,
                    }
                )
    # Materialize the identical pure module outside cloud-backed checkout paths.
    # .mts makes ESM explicit without consulting unrelated parent package files.
    with tempfile.TemporaryDirectory(prefix="exact-review-projection-") as temporary:
        module = Path(temporary) / "projection.mts"
        worker = Path(temporary) / "worker.mjs"
        module.write_bytes(Path(frontend_module).read_bytes())
        worker.write_bytes(Path(__file__).with_suffix(".mjs").read_bytes())
        assert file_hash(module) == file_hash(frontend_module)
        return json.loads(
            subprocess.check_output(
                [str(node), str(worker), str(module)],
                input=json.dumps(
                    {"publication_sha256": file_hash(root / "publication.json"), "pairs": pairs}
                ),
                text=True,
                timeout=60,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("frontend_module", type=Path)
    parser.add_argument("node", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("Preserve the existing receipt")
    args.receipt.write_text(
        json.dumps(verify(args.root, args.frontend_module, args.node), indent=2) + "\n"
    )
