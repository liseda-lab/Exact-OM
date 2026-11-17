"""Build a fixed exploration package from a published study's immutable workspaces.

No provider, matcher, study database or reference keys are used at serving time.
Existing ontology files are referenced in place, never copied or hardlinked.
"""

import argparse
import json
import sqlite3
from pathlib import Path

from exact_inspect.artifacts import (
    Artifact,
    InspectionBundle,
    atomic_json,
    relative_path,
    validate_bundle,
)
from exact_inspect.contracts import canonical_hash, file_hash
from exact_inspect.decisions import (
    DEPENDENCY_INVENTORY_SCHEMA,
    _missing_event,
    dependency_inventory_hash,
)
from exact_inspect.models import Candidate, GeneratedExplanationResponse, PairResponse

STAGES = (
    "retrieval",
    "prefilter",
    "pair_scoring",
    "llm_signal",
    "selection",
    "threshold",
    "cardinality",
    "extraction",
    "relation_typing",
    "repair",
)

# Owner-reviewed cases, selected for information, clinical diversity and challenge.
# The two rank-4 cases are the only two available in the deployed 40-source pool.
SELECTION = frozenset(
    {
        "ncit-doid-q0030",
        "ncit-doid-q0051",
        "ncit-doid-q0059",
        "ncit-doid-q0060",
        "ncit-doid-q0378",
        "ncit-doid-q0556",
        "ncit-doid-q0615",
        "ncit-doid-q1176",
        "ncit-doid-q1194",
        "ncit-doid-q1426",
    }
)


def prepare(assets, publication, catalogue, name, report):
    assets = assets.resolve()
    old = json.loads(publication.read_bytes())
    d = old["definition"]
    if d["study_revision"] != "exact2-ncit-doid-production-20261114-r2":
        raise ValueError("This selection is bound to the latest approved r2 publication")
    if set(SELECTION) - {c["case_id"] for c in d["cases"]}:
        raise ValueError("Demo cases must belong to the deployed revision")
    cat = json.loads(catalogue.read_bytes())
    selected = {c["case_id"]: c for c in cat["cases"] if c["case_id"] in SELECTION}
    assert set(selected) == SELECTION
    assert sorted(c["reference_rank"] for c in selected.values()) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    dest = relative_path(assets, name)
    dest.mkdir(exist_ok=False)
    run = dest / "run"
    run.mkdir()
    dbpath = run / "decisions.sqlite"
    explanation_paths = {}
    contexts = {}
    artifacts = {}
    counts = {"sources": 10, "pairs": 50}

    def bind(path):
        rel = path.relative_to(assets).as_posix()
        artifacts[rel] = Artifact(path=rel, sha256=file_hash(path), size=path.stat().st_size)

    summary = []
    identities = {}
    with sqlite3.connect(dbpath) as db:
        db.executescript(
            "CREATE TABLE pairs(id TEXT PRIMARY KEY,source TEXT,source_kind TEXT,target TEXT,target_kind TEXT,candidate TEXT,trace TEXT,evidence TEXT); CREATE INDEX pair_source ON pairs(source,source_kind,id); CREATE TABLE sources(id TEXT PRIMARY KEY,iri TEXT,kind TEXT,payload TEXT); CREATE INDEX source_order ON sources(id);"
        )
        for case in d["cases"]:
            cid = case["case_id"]
            if cid not in SELECTION:
                continue
            scope = next(
                s for s in d["workspace_scopes"] if s.get("case_id") == cid and s["kind"] == "case"
            )
            index = relative_path(assets, scope["index_path"])
            if file_hash(index).removeprefix("sha256:") != scope["index_sha256"]:
                raise ValueError("Changed workspace")
            for ref in scope["contexts"]:
                contexts[ref["ontology_version_id"]] = ref["path"]
                for filename, expected in [
                    ("manifest.json", ref["manifest_sha256"]),
                    ("context.sqlite", ref["database_sha256"]),
                ]:
                    path = relative_path(assets, ref["path"] + "/" + filename)
                    st = path.stat()
                    identities[str(path)] = [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
                    if path.relative_to(assets).as_posix() not in artifacts:
                        bind(path)
                        assert artifacts[path.relative_to(assets).as_posix()].sha256.removeprefix(
                            "sha256:"
                        ) == expected.removeprefix("sha256:")
            with sqlite3.connect(f"file:{index}?mode=ro", uri=True) as src:
                for eid, payload in src.execute("SELECT id,payload FROM explanations"):
                    value = json.loads(payload)
                    GeneratedExplanationResponse.model_validate(value)
                    path = dest / "explanations" / (eid.removeprefix("sha256:") + ".json")
                    if path.exists():
                        assert json.loads(path.read_bytes()) == value
                    else:
                        atomic_json(path, value)
                    bind(path)
                    explanation_paths[eid] = path.relative_to(assets).as_posix()
                source = case["source"]
                sid = canonical_hash(source)
                db.execute(
                    "INSERT INTO sources VALUES(?,?,?,?)",
                    (
                        sid,
                        source["iri"],
                        source["kind"],
                        json.dumps({"entity_id": sid, "entity": source}),
                    ),
                )
                for c in case["candidates"]:
                    evidence = [
                        json.loads(v[0])
                        for v in src.execute(
                            "SELECT payload FROM evidence WHERE candidate_id=? ORDER BY id",
                            (c["candidate_id"],),
                        )
                    ]
                    metadata = json.loads(
                        src.execute(
                            "SELECT payload FROM pair_metadata WHERE candidate_id=?",
                            (c["candidate_id"],),
                        ).fetchone()[0]
                    )
                    scores = [
                        dict(
                            name="S_final",
                            stage="selection",
                            value=c["score"],
                            meaning="Recorded final Exact 2.0 candidate ranking score.",
                            calibration_status="not_established",
                        )
                    ]
                    candidate = Candidate.model_validate(
                        dict(
                            pair_id=c["candidate_id"],
                            source=source,
                            target=c["entity"],
                            scores=scores,
                            retrieval_channels=[],
                            ordinal_ranks={
                                "candidate_joint_rank": c["display_position"],
                                "joint_ordering": "Recorded final candidate order",
                            },
                            evidence_counts={
                                "visible_selected": len(evidence),
                                "scope": "prepared_visibility_policy",
                            },
                            nil={},
                            saved_alignment_member=None,
                            membership_provenance={"status": "unavailable", "artifact_hash": None},
                            status="available",
                            decision_status="not_exported",
                        )
                    ).model_dump(mode="json")
                    # The study records scores and evidence, not the full pipeline event history.
                    # Preserve that distinction rather than inventing stage decisions.
                    trace = PairResponse.model_validate(
                        dict(
                            artifact_type="pair_decision_trace",
                            contract_version="exact-explain/1.0",
                            run_id=name,
                            pair_id=c["candidate_id"],
                            source=source,
                            target=c["entity"],
                            events=[_missing_event(s) for s in STAGES],
                            saved_alignment_member=None,
                            relation=None,
                            relation_basis="unavailable",
                            assumption_refs=[],
                            evidence_metadata=metadata["evidence_metadata"],
                            fixture_provenance="Recorded Exact 2.0 run; curated reviewer demonstration.",
                        )
                    ).model_dump(mode="json")
                    db.execute(
                        "INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?)",
                        (
                            c["candidate_id"],
                            source["iri"],
                            source["kind"],
                            c["entity"]["iri"],
                            c["entity"]["kind"],
                            json.dumps(candidate),
                            json.dumps(trace),
                            json.dumps(evidence),
                        ),
                    )
            summary.append(
                {
                    "case_id": cid,
                    "source_label": case["source_label"],
                    "reference_rank": selected[cid]["reference_rank"],
                    "original_definitions": selected[cid]["quality"][0],
                }
            )
            print(json.dumps({"prepared_case": cid}), flush=True)
    dependencies = {"published-study": file_hash(publication)}
    manifest = dict(
        contract_version="exact-explain/1.0",
        artifact_type="run_decision_package",
        implementation="study-demo-adapter/1",
        run_id=name,
        revision=canonical_hash(dependencies),
        source_ontology_version_id=source["ontology_version_id"],
        target_ontology_version_id=c["entity"]["ontology_version_id"],
        counts=counts,
        status="available",
        database="decisions.sqlite",
        database_hash=file_hash(dbpath),
        dependencies=dependencies,
        dependency_inventory_schema=DEPENDENCY_INVENTORY_SCHEMA,
        dependency_inventory_hash=dependency_inventory_hash(dependencies),
    )
    atomic_json(run / "decisions.manifest.json", manifest)
    bind(dbpath)
    bind(run / "decisions.manifest.json")
    bundle = InspectionBundle(
        package_id="pending",
        audience="development_demo",
        policy=d["visibility_policy"],
        artifacts=list(artifacts.values()),
        ontologies=contexts,
        runs={name: run.relative_to(assets).as_posix()},
        explanations=explanation_paths,
        capabilities={"matching": "not_requested", "generation": "prepared", "imports": "disabled"},
        provenance={
            "source_revision": d["study_revision"],
            "publication_sha256": file_hash(publication),
            "selection": "Ten curated examples from the deployed study; private reference answers are not included.",
        },
        portable=False,
    )
    bundle = bundle.model_copy(
        update={
            "package_id": canonical_hash(bundle.model_dump(mode="json", exclude={"package_id"}))
        }
    )
    package = assets / (name + "-package.json")
    atomic_json(package, bundle.model_dump(mode="json"))
    validate_bundle(package)
    for p, before in identities.items():
        st = Path(p).stat()
        assert before == [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    atomic_json(
        report,
        {
            "package_id": bundle.package_id,
            "package": str(package),
            "source_revision": d["study_revision"],
            "sources": summary,
            "pairs": 50,
            "explanations": len(explanation_paths),
            "shared_context_identity_preserved": True,
            "publication_sha256": file_hash(publication),
        },
    )
    print(json.dumps({"package": str(package), "sources": 10, "pairs": 50}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["assets", "publication", "catalogue", "report"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--name", default="reviewer-demo-20261117")
    a = p.parse_args()
    prepare(a.assets, a.publication, a.catalogue, a.name, a.report)
