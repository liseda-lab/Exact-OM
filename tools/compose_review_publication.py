"""Compose frozen evidence, independently reviewed prose and partial ELK layers.

Uses only retained artifacts. Unchanged large files are APFS clones; shared protected
source inodes are never changed. All mutable staging writes are private to a new revision.
Run under tools.review_storage_budget; no providers or reasoners are called.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from exact_inspect.artifacts import atomic_json, publish_bundle, validate_bundle
from exact_inspect.context import OntologyContext
from exact_inspect.contracts import EntityRef, canonical_hash, file_hash
from exact_inspect.decisions import DecisionStore
from exact_inspect.generation import FactPacket
from exact_inspect.partial_elk import admit, materialize
from exact_inspect.readable_descriptions import RENDER_REVISION, presentation, revise
from exact_inspect.study.builder import build_explanation_resource
from exact_inspect.study.models import Publish
from exact_inspect.study.workspace import build_workspace_scope

VERSION = "exact-review-composition/1"


def private_copy(source, destination, *, writable=False):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Private destination must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    if sys.platform == "darwin":
        subprocess.run(["/bin/cp", "-c", str(source), str(destination)], check=True)
    else:
        shutil.copyfile(source, destination)
    after = destination.stat()
    if os.path.samefile(source, destination) or after.st_nlink != 1:
        raise ValueError("Copy is not private")
    if writable:
        if hasattr(os, "chflags"):
            os.chflags(destination, after.st_flags & ~stat.UF_IMMUTABLE)
        destination.chmod(after.st_mode | stat.S_IWUSR)
    now = source.stat()
    if (
        before.st_ino,
        before.st_nlink,
        before.st_mode,
        getattr(before, "st_flags", 0),
        before.st_ctime_ns,
    ) != (now.st_ino, now.st_nlink, now.st_mode, getattr(now, "st_flags", 0), now.st_ctime_ns):
        raise ValueError("Shared source identity/protection changed")
    return str(destination)


def stage_tree(source, destination, checkpoint):
    """Resume an interrupted clone without overwriting completed or shared files."""
    expected = file_hash(source / "package.json")
    if checkpoint.exists():
        if json.loads(checkpoint.read_bytes()) != {"source_package": expected}:
            raise ValueError("Staged package identity changed")
        return
    for item in sorted(source.rglob("*")):
        if item.is_symlink():
            raise ValueError("Package source cannot contain symlinks")
        if not item.is_file():
            continue
        target = destination / item.relative_to(source)
        if target.exists():
            if (
                target.is_symlink()
                or os.path.samefile(item, target)
                or target.stat().st_nlink != 1
                or file_hash(item) != file_hash(target)
            ):
                raise ValueError("Interrupted private stage differs from source")
        else:
            private_copy(item, target)
    atomic_json(checkpoint, {"source_package": expected})


def _resume_case(checkpoint, destination, binding):
    if not checkpoint.exists():
        return None
    value = json.loads(checkpoint.read_bytes())
    if value["binding"] != canonical_hash(binding):
        raise ValueError("Case resume identity changed")
    for locator, digest in value["files"].items():
        if file_hash(destination / locator) != digest:
            raise ValueError("Completed case changed before resume")
    return value


def _decision_identity(row):
    candidate, trace = json.loads(row["candidate"]), json.loads(row["trace"])
    candidate = {
        k: v for k, v in candidate.items() if k not in ("evidence_counts", "evidence_status")
    }
    trace = {k: v for k, v in trace.items() if k != "evidence_metadata"}
    return canonical_hash(
        [
            row["id"],
            row["source"],
            row["source_kind"],
            row["target"],
            row["target_kind"],
            candidate,
            trace,
        ]
    )


def merge_evidence(recovered, latest):
    """Preserve exact feature identity, recorded numbers and producer projection flags."""
    old = {x["evidence_id"]: x for x in recovered}
    new = {x["evidence_id"]: x for x in latest}
    if len(old) != len(recovered) or len(new) != len(latest) or old.keys() != new.keys():
        raise ValueError("Selected feature identities differ")
    mutable = {
        "fact_ids",
        "feature_id",
        "semantic_terms",
        "source_axiom_refs",
        "axiom_origins",
        "provenance_status",
        "status",
        "origin_resolution",
        "reason",
        "display",
        "derivation",
    }
    out = []
    for current in recovered:
        other = new[current["evidence_id"]]
        if {k: v for k, v in current.items() if k not in mutable} != {
            k: v for k, v in other.items() if k not in mutable
        }:
            raise ValueError("Producer evidence identity, score or projection changed")
        if current["fact_ids"] and other["fact_ids"] and current["fact_ids"] != other["fact_ids"]:
            raise ValueError("Conflicting admitted fact mappings need explicit review")
        out.append(current if current["fact_ids"] or not other["fact_ids"] else other)
    return out


def source_identity(row):
    return canonical_hash(
        {**{k: row[k] for k in ("id", "iri", "kind")}, "payload": json.loads(row["payload"])}
    )


def compare_runs(recovered, latest):
    counts = Counter()
    rows = []
    with recovered._connect() as a, latest._connect() as b:
        ai = a.execute("SELECT * FROM pairs ORDER BY id")
        bi = b.execute("SELECT * FROM pairs ORDER BY id")
        for old, new in zip(ai, bi, strict=True):
            if _decision_identity(old) != _decision_identity(new):
                raise ValueError("Matcher decisions or pool changed")
            merged = merge_evidence(json.loads(old["evidence"]), json.loads(new["evidence"]))
            # A new supported layer absent in the recovery branch must never be silently dropped.
            if merged != json.loads(old["evidence"]):
                raise ValueError(
                    "Recovery predecessor lacks newer links; explicit merged index required"
                )
            counts["pairs"] += 1
            counts["selected_features"] += len(merged)
            rows.append(_decision_identity(old))
        source_rows = []
        for old, new in zip(
            a.execute("SELECT * FROM sources ORDER BY iri,kind"),
            b.execute("SELECT * FROM sources ORDER BY iri,kind"),
            strict=True,
        ):
            if source_identity(old) != source_identity(new):
                raise ValueError("Recorded source inventory changed")
            source_rows.append(source_identity(old))
        counts["sources"] = len(source_rows)

    return {"counts": dict(counts), "matcher_identity": canonical_hash(rows)}


def _bundle_metadata(bundle):
    return bundle.model_dump(mode="json", exclude={"artifacts", "package_id"})


def compose(prose, evidence, elk, destination, revision):
    prose, evidence, elk, destination = map(
        lambda p: Path(p).resolve(), (prose, evidence, elk, destination)
    )
    pb, eb = validate_bundle(prose / "package/package.json"), validate_bundle(
        evidence / "package/package.json"
    )
    if pb.policy != eb.policy or pb.ontologies.keys() != eb.ontologies.keys():
        raise ValueError("Predecessor policy or ontology identity changed")
    proseraw = json.loads((prose / "publication.json").read_bytes())
    oldraw = json.loads((evidence / "publication.json").read_bytes())
    if proseraw["definition"]["purpose"] != "development_review":
        raise ValueError("Only development review may be composed")
    if proseraw["definition"]["cases"] != oldraw["definition"]["cases"]:
        raise ValueError("Presented cases changed")
    ph = {
        v: json.loads((prose / "package" / p / "manifest.json").read_bytes())["artifacts"][
            "context.sqlite"
        ]
        for v, p in pb.ontologies.items()
    }
    eh = {
        v: json.loads((evidence / "package" / p / "manifest.json").read_bytes())["artifacts"][
            "context.sqlite"
        ]
        for v, p in eb.ontologies.items()
    }
    if ph != eh:
        raise ValueError("Context change requires new original-fact admission")
    pins = {
        name: {
            "manifest": file_hash(elk / "manifest.json"),
            "result": file_hash(elk / f"results/14455-run/{name}/result.json"),
            "taxonomy": file_hash(elk / f"results/14455-run/{name}/taxonomy.partial.json"),
            "resource_receipt": file_hash(elk / f"inputs/{name}.receipt.json"),
        }
        for name in ["ncit", "doid"]
    }
    # The retained cluster output pins are independent of the fetched filenames.
    if (
        pins["ncit"]["taxonomy"]
        != "sha256:68cc52ef6e78d356d923741614b28cc50c876665db4725ec914a6032916c93a4"
        or pins["doid"]["taxonomy"]
        != "sha256:8563d5690fd1a52669f76460c57f4e12e69e65b5910c333d7e22dd838c3d1cd5"
    ):
        raise ValueError("Retained ELK output differs from independent review")
    known = {
        "ncit": {
            "manifest": "e12aed8e7fd110e017d2b54153d97a75ba5fab4b8a047be4eb1213e6afa1cb91",
            "result": "8cd61822b26fb42332b2ac37f9fc829c39c669669a4cef54b32ed4c6dcfd1701",
            "resource_receipt": "f8835f4ddf83aa67f3eab6fbc97ff4b2e53f1cfe41b3f8737a10be3cc2b9fe8c",
        },
        "doid": {
            "manifest": "e12aed8e7fd110e017d2b54153d97a75ba5fab4b8a047be4eb1213e6afa1cb91",
            "result": "63234a4f7d11d18ada3bfa10bb98417480eb9eb26674de0eab0c4a351346623d",
            "resource_receipt": "9cad3ae40290d9d4aa169022180d6ced0433cb46513180d76c8165810312987d",
        },
    }
    if any(
        pins[name][key] != "sha256:" + value
        for name, fields in known.items()
        for key, value in fields.items()
    ):
        raise ValueError("Retained ELK admission receipts differ from reviewed inputs")
    binding = {
        "schema": VERSION,
        "revision": revision,
        "prose_publication": file_hash(prose / "publication.json"),
        "evidence_publication": file_hash(evidence / "publication.json"),
        "prose_package": file_hash(prose / "package/package.json"),
        "evidence_package": file_hash(evidence / "package/package.json"),
        "contexts": ph,
        "policy": pb.policy.policy_hash,
        "elk": pins,
        "renderer": RENDER_REVISION,
    }
    marker = destination / "composition-inputs.json"
    if marker.exists():
        if json.loads(marker.read_bytes()) != binding:
            raise ValueError("Composition resume inputs changed")
        if (destination / "publication.json").exists():
            validate_bundle(destination / "package/package.json")
            from tools.review_environment import verify_assets, verify_workspaces

            raw = json.loads((destination / "publication.json").read_bytes())
            verify_assets(raw, destination)
            verify_workspaces(raw, destination)
            receipt = json.loads((destination / "composition.json").read_bytes())
            if (
                receipt["publication_sha256"] != file_hash(destination / "publication.json")
                or receipt["package_sha256"] != file_hash(destination / "package/package.json")
                or receipt["catalogue_sha256"] != file_hash(destination / "catalogue.json")
            ):
                raise ValueError("Composed publication, package or catalogue changed")
            return receipt
    elif destination.exists():
        raise ValueError("Use a new private revision")
    run_id = next(iter(pb.runs))
    oldrun = DecisionStore(evidence / "package" / eb.runs[run_id])
    newrun = DecisionStore(prose / "package" / pb.runs[run_id])
    destination.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        atomic_json(marker, binding)
    preservation = destination / "preservation.json"
    if preservation.exists():
        saved = json.loads(preservation.read_bytes())
        if saved["binding"] != canonical_hash(binding):
            raise ValueError("Preservation inputs changed")
        invariant = saved["result"]
    else:
        print("phase compare full matcher pool and predecessor layers", flush=True)
        invariant = compare_runs(oldrun, newrun)
        atomic_json(preservation, {"binding": canonical_hash(binding), "result": invariant})
    print("phase stage private revision", flush=True)
    package = destination / "package"
    # Large contexts are reused by APFS clone, without adding hardlinks under running readers.
    stage_tree(prose / "package", package, destination / "package-staged.json")
    runpath = package / pb.runs[run_id]
    # The staged clone is private. Replace it atomically; never touch a protected source inode.
    current = runpath / "decisions.sqlite"
    if current.exists() and file_hash(current) != file_hash(oldrun.database):
        if (
            current.stat().st_nlink != 1
            or os.path.samefile(current, newrun.database)
            or file_hash(current) != file_hash(newrun.database)
        ):
            raise ValueError("Staged decision index is not a private predecessor copy")
        if getattr(current.stat(), "st_flags", 0) & getattr(stat, "UF_IMMUTABLE", 0):
            os.chflags(current, current.stat().st_flags & ~stat.UF_IMMUTABLE)
        current.unlink()
    if not current.exists():
        private_copy(oldrun.database, current)
    runmanifest = copy.deepcopy(oldrun.manifest())
    runmanifest["delivered_layers"] = {
        "recovered_evidence": binding["evidence_publication"],
        "reviewed_prose": binding["prose_publication"],
        "composition": canonical_hash(binding),
    }
    runmanifest["revision"] = canonical_hash(
        {"previous": runmanifest["revision"], "layers": runmanifest["delivered_layers"]}
    )
    atomic_json(runpath / "decisions.manifest.json", runmanifest)
    raw = copy.deepcopy(proseraw)
    definition = raw["definition"]
    definition["study_revision"] = revision
    catalogue = json.loads((prose / "catalogue.json").read_bytes())
    catalogue_cases = {c["case_id"]: c for c in catalogue["cases"]}
    # Copy only delivered assets and tutorial scope inputs, not private imports/jobs or live databases.
    locators = set()
    for asset in definition["assets"]:
        locators.add(asset["path"])
        if asset.get("admission_receipt_path"):
            locators.add(asset["admission_receipt_path"])
    for scope in definition["workspace_scopes"]:
        if scope["kind"] == "tutorial":
            locators.add(scope["index_path"])
            for c in scope["contexts"]:
                for p in (prose / c["path"]).rglob("*"):
                    if p.is_file():
                        locators.add(str(p.relative_to(prose)))
    for path in sorted(locators):
        if not (destination / path).exists():
            private_copy(prose / path, destination / path)
    contexts = {v: OntologyContext(package / p) for v, p in pb.ontologies.items()}
    if any(c.manifest["artifacts"]["context.sqlite"] != ph[v] for v, c in contexts.items()):
        raise ValueError("Staged context differs from pinned predecessors")
    print("phase partial ELK admission", flush=True)
    for version, context in list(contexts.items()):
        name = context.manifest["name"].lower()
        role = "source" if name == "ncit" else "target"
        data, admission = admit(
            context,
            policy=pb.policy,
            manifest_path=elk / "manifest.json",
            result_path=elk / f"results/14455-run/{name}/result.json",
            taxonomy_path=elk / f"results/14455-run/{name}/taxonomy.partial.json",
            resource_receipt_path=elk / f"inputs/{name}.receipt.json",
            resource_path=evidence / f"{role}-review.ofn",
            expected=pins[name],
        )
        contexts[version] = materialize(context, policy=pb.policy, data=data, binding=admission)
        print("partial taxonomy ready", name, flush=True)
    print("phase descriptions and resources", flush=True)
    results = {}
    replacement = {}
    lineage = {}
    metadata = _bundle_metadata(pb)
    for old_id, locator in pb.explanations.items():
        previous = json.loads((prose / "package" / locator).read_bytes())
        updated = revise(previous, contexts, pb.policy)
        results[updated["explanation_id"]] = updated
        replacement[old_id] = updated["explanation_id"]
        if updated != previous:
            path = (
                "corrected-descriptions/"
                + updated["explanation_id"].removeprefix("sha256:")
                + ".json"
            )
            atomic_json(package / path, updated)
            metadata["explanations"].pop(old_id)
            metadata["explanations"][updated["explanation_id"]] = path
            lineage[old_id] = {
                "replacement": updated["explanation_id"],
                "reason": "versioned deterministic rendering; provider approvals retained separately",
            }
    preserved = sum(
        len(v["claims"])
        for v in results.values()
        if v.get("support_status") == "independently_reviewed"
    )
    if preserved != 68:
        raise ValueError("Approved prose was not preserved")
    run = DecisionStore(runpath)
    coverage = []
    oldscopes = {
        s["case_id"]: s for s in proseraw["definition"]["workspace_scopes"] if s["kind"] == "case"
    }
    definition["workspace_scopes"] = [
        s for s in definition["workspace_scopes"] if s["kind"] == "tutorial"
    ]
    with run._connect() as db:
        for case in definition["cases"]:
            checkpoint = destination / "composition-cases" / f"{case['case_id']}.json"
            saved = _resume_case(checkpoint, destination, binding)
            if saved:
                asset = next(
                    a for a in definition["assets"] if a["asset_id"] == saved["asset"]["asset_id"]
                )
                asset.update(saved["asset"])
                definition["workspace_scopes"].append(saved["scope"])
                coverage.append(saved["coverage"])
                catalogue_cases[case["case_id"]].update(saved["catalogue"])
                continue
            evidence_by_candidate = {}
            pairmeta = {}
            casepairs = []
            for candidate in case["candidates"]:
                e = candidate["entity"]
                row = db.execute(
                    "SELECT * FROM pairs WHERE source=? AND source_kind=? AND target=? AND target_kind=?",
                    (case["source"]["iri"], case["source"]["kind"], e["iri"], e["kind"]),
                ).fetchone()
                if row is None:
                    raise ValueError("Missing presented pair")
                ev = json.loads(row["evidence"])
                trace = json.loads(row["trace"])
                cid = candidate["candidate_id"]
                evidence_by_candidate[cid] = ev
                pairmeta[cid] = {
                    "source": trace["source"],
                    "target": trace["target"],
                    "evidence_metadata": trace["evidence_metadata"],
                }
                originals = {
                    fid: contexts[x["entity"]["ontology_version_id"]].fact(
                        fid, EntityRef.model_validate(x["entity"]), policy=pb.policy
                    )
                    for x in ev
                    for fid in x["fact_ids"]
                }
                casepairs.append(
                    {
                        "candidate_id": cid,
                        "pair_id": row["id"],
                        "selected_features": len(ev),
                        "origin_resolution": dict(Counter(x["origin_resolution"] for x in ev)),
                        "graph_projection": dict(Counter(x["graph_projection"] for x in ev)),
                        "feature_sides": sorted({x["side"] for x in ev}),
                        "resolved_original_facts": len(originals),
                        "expression_original_facts": sum(
                            f["value"]["term_type"] in {"expression", "expression_ref"}
                            for f in originals.values()
                        ),
                        "bridge_statuses": dict(
                            Counter(b["status"] for b in trace["evidence_metadata"]["bridges"])
                        ),
                    }
                )
            oldscope = oldscopes[case["case_id"]]
            with sqlite3.connect(
                "file:" + str(prose / oldscope["index_path"]) + "?mode=ro", uri=True
            ) as scope_db:
                ids = [
                    json.loads(row[0])["explanation_id"]
                    for row in scope_db.execute("SELECT payload FROM explanations ORDER BY id")
                ]
            explanations = [results[replacement[i]] for i in ids]
            packets = [FactPacket.model_validate(x["support_packet"]) for x in explanations]
            resource = build_explanation_resource(
                contexts,
                [
                    EntityRef.model_validate(e)
                    for e in [case["source"], *(c["entity"] for c in case["candidates"])]
                ],
                policy=pb.policy,
                packets=packets,
                explanations=explanations,
                evidence=evidence_by_candidate,
                pair_metadata=pairmeta,
            )
            asset = next(
                a for a in definition["assets"] if a["asset_id"] == case["explanation_refs"][0]
            )
            asset["path"] = "corrected-resources/" + case["case_id"] + ".json"
            atomic_json(destination / asset["path"], resource.model_dump(mode="json"))
            asset.update(
                sha256=file_hash(destination / asset["path"])[7:],
                size_bytes=(destination / asset["path"]).stat().st_size,
            )
            pending_index = (
                destination
                / "workspaces"
                / hashlib.sha256(("combined-" + case["case_id"]).encode()).hexdigest()
                / "workspace.sqlite"
            )
            if pending_index.exists():
                # Only our unpublished, uncheckpointed, private case output may be retried.
                if pending_index.is_symlink() or pending_index.stat().st_nlink != 1:
                    raise ValueError("Cannot resume shared workspace index")
                pending_index.unlink()
            scope = build_workspace_scope(
                destination,
                scope_id="combined-" + case["case_id"],
                kind="case",
                case=case,
                contexts=contexts,
                policy=pb.policy,
                explanations=explanations,
                evidence=evidence_by_candidate,
                pair_metadata=pairmeta,
                components=oldscope["components"],
                reuse_prepared_contexts=True,
            )
            scope_value = scope.model_dump(mode="json")
            case_coverage = {"case_id": case["case_id"], "pairs": casepairs}
            catalogue_update = {
                "pairs": casepairs,
                "descriptions": [
                    {
                        "id": x["explanation_id"],
                        "kind": x["preparation_kind"],
                        "status": x["manifest"]["status"],
                        "support_status": x["support_status"],
                        "presentation": presentation(x),
                    }
                    for x in explanations
                ],
            }
            catalogue_cases[case["case_id"]].update(catalogue_update)
            files = {
                asset["path"]: "sha256:" + asset["sha256"],
                scope.index_path: "sha256:" + scope.index_sha256,
            }
            files.update(
                {c.path + "/manifest.json": "sha256:" + c.manifest_sha256 for c in scope.contexts}
            )
            atomic_json(
                checkpoint,
                {
                    "binding": canonical_hash(binding),
                    "asset": asset,
                    "scope": scope_value,
                    "coverage": case_coverage,
                    "catalogue": catalogue_update,
                    "files": files,
                },
            )
            definition["workspace_scopes"].append(scope_value)
            coverage.append(case_coverage)
            print("case frozen", case["case_id"], flush=True)
    about = definition["review_bundle"]
    about.update(
        composition_identity=canonical_hash(binding),
        delivered_layers=runmanifest["delivered_layers"],
        prepared_at=datetime.fromtimestamp(marker.stat().st_mtime, timezone.utc).isoformat(),
        selected_evidence_counts=dict(
            sum((Counter(p["origin_resolution"]) for c in coverage for p in c["pairs"]), Counter())
        ),
        recorded_bridge_counts=dict(
            sum((Counter(p["bridge_statuses"]) for c in coverage for p in c["pairs"]), Counter())
        ),
    )
    for ont in about["ontologies"]:
        ont["inference_status"] = "partial"
    about["limitations"] = [x for x in about["limitations"] if "inference" not in x.lower()]
    about["limitations"].append(
        "ELK inferred hierarchy is partial; absent edges and whole-ontology consistency remain unknown."
    )
    catalogue["generation_summary"].update(
        served_artifacts=len(results),
        independently_reviewed_artifacts=sum(
            x["support_status"] == "independently_reviewed" for x in results.values()
        ),
        deterministic_artifacts=sum(
            x["support_status"] not in {"independently_reviewed", "fallback"}
            for x in results.values()
        ),
        factual_fallback_artifacts=sum(x["support_status"] == "fallback" for x in results.values()),
    )
    metadata["provenance"]["generation_summary"] = catalogue["generation_summary"]
    metadata["provenance"].update(
        composition=canonical_hash(binding),
        delivered_layers=runmanifest["delivered_layers"],
        review_bundle=about,
    )
    # Replace stale prepared explanation dependency records; immutable review packets keep their original hashes.
    dependencies = dict(pb.provenance.get("description_dependencies", {}))
    for key, result in results.items():
        path = "corrected-dependencies/" + key.removeprefix("sha256:") + ".json"
        atomic_json(package / path, result)
        dependencies[key] = path
    metadata["provenance"]["description_dependencies"] = dependencies
    changed = {
        str(Path(pb.runs[run_id]) / name)
        for name in ["decisions.sqlite", "decisions.manifest.json"]
    }
    changed.update(
        str(Path(p) / name)
        for p in pb.ontologies.values()
        for name in ["manifest.json", "context.sqlite"]
    )
    for artifact in pb.artifacts:
        if artifact.path not in changed and file_hash(package / artifact.path) != artifact.sha256:
            raise ValueError("An unchanged predecessor artifact changed in private staging")
    publish_bundle(package, **metadata)
    raw = Publish.model_validate(raw).model_dump(mode="json")
    catalogue.update(
        publication_sha256=canonical_hash(raw),
        composition=binding,
        composition_evidence=coverage,
        description_replacements=lineage,
    )
    from tools.review_environment import verify_assets, verify_workspaces

    verify_assets(raw, destination)
    verify_workspaces(raw, destination)
    atomic_json(destination / "catalogue.json", catalogue)
    receipt = {
        "schema": VERSION,
        "assembler_sha256": file_hash(Path(__file__)),
        "catalogue_sha256": file_hash(destination / "catalogue.json"),
        "binding": binding,
        "matcher_preservation": invariant,
        "approved_claims_preserved": preserved,
        "description_replacements": lineage,
        "cases": coverage,
        "package_sha256": file_hash(package / "package.json"),
        "publication_sha256": canonical_hash(raw),
        "scope": "development_review; frontend acceptance separate",
    }
    # file_hash includes the exact serialized publication bytes, not canonical model identity.
    atomic_json(destination / "composition.json", receipt)
    atomic_json(destination / "publication.json", raw)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["prose", "evidence", "elk", "destination"]:
        p.add_argument(key, type=Path)
    p.add_argument("--revision", required=True)
    a = p.parse_args()
    out = compose(a.prose, a.evidence, a.elk, a.destination, a.revision)
    print(json.dumps({"status": "prepared", "publication_sha256": out["publication_sha256"]}))
