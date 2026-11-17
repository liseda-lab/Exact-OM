"""Add recovered decisions to new immutable study indexes; never copy ontology contexts."""

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from exact_inspect.contracts import canonical_json, file_hash
from exact_inspect.models import Candidate
from exact_inspect.study.models import Publish
from tools.reviewer_decision_trace import recovered_pair


def prepare(assets, publication, recovery_path, output, revision):
    raw = json.loads(publication.read_bytes())
    original = json.loads(publication.read_bytes())
    recovery = json.loads(recovery_path.read_bytes())
    d = raw["definition"]
    if set(recovery["cases"]) != {c["case_id"] for c in d["cases"]}:
        raise ValueError("Recovery must cover exactly the study cases")
    destination = assets / "study-decisions" / revision
    destination.mkdir(parents=True, exist_ok=False)
    cases = {c["case_id"]: c for c in d["cases"]}
    pairs = 0
    for scope in d["workspace_scopes"]:
        if scope["kind"] != "case":
            continue
        case = cases[scope["case_id"]]
        index = destination / (case["case_id"] + ".sqlite")
        shutil.copyfile(assets / scope["index_path"], index)
        with sqlite3.connect(index) as db:
            for c in case["candidates"]:
                row = db.execute(
                    "SELECT payload FROM pair_metadata WHERE candidate_id=?", (c["candidate_id"],)
                ).fetchone()
                if row is None:
                    raise ValueError("Candidate metadata missing")
                metadata = json.loads(row[0])
                scores, trace = recovered_pair(
                    recovery, case, c, "exact2-final-20261110", metadata["evidence_metadata"]
                )
                details = Candidate.model_validate(
                    dict(
                        pair_id=c["candidate_id"],
                        source=case["source"],
                        target=c["entity"],
                        scores=scores,
                        retrieval_channels=[],
                        ordinal_ranks={
                            "candidate_joint_rank": c["display_position"],
                            "joint_ordering": "Recorded final candidate order",
                        },
                        evidence_counts={
                            "visible_selected": metadata["evidence_metadata"]["counts"]["selected"],
                            "scope": "prepared_visibility_policy",
                        },
                        nil={},
                        saved_alignment_member=None,
                        membership_provenance={"status": "unavailable", "artifact_hash": None},
                        status="available",
                        decision_status="available",
                    )
                ).model_dump(mode="json")
                metadata.update(decision_trace=trace, score_details=details)
                db.execute(
                    "UPDATE pair_metadata SET payload=? WHERE candidate_id=?",
                    (canonical_json(metadata).decode(), c["candidate_id"]),
                )
                pairs += 1
        scope["index_path"] = index.relative_to(assets).as_posix()
        scope["index_sha256"] = file_hash(index).removeprefix("sha256:")
        scope["components"].append("decisions")
    d["study_revision"] = revision
    d["form_version"] = "exact-study-forms/3-copy-20261117-traces"
    d["protocol_versions"]["forms"] = d["form_version"]
    d["components"] += ["decision_trace", "scores"]
    Publish.model_validate(raw)
    for key in original["definition"]:
        if key not in {
            "study_revision",
            "form_version",
            "software_version",
            "protocol_versions",
            "components",
            "workspace_scopes",
        }:
            assert d[key] == original["definition"][key], key
    assert raw["case_keys"] == original["case_keys"]
    assert pairs == len(cases) * 5
    output.write_bytes(canonical_json(raw) + b"\n")
    print(
        json.dumps(
            {
                "revision": revision,
                "cases": len(cases),
                "pairs": pairs,
                "bytes": sum(p.stat().st_size for p in destination.iterdir()),
                "publication_sha256": file_hash(output),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("assets", "publication", "recovery", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--revision", required=True)
    a = p.parse_args()
    prepare(a.assets, a.publication, a.recovery, a.output, a.revision)
