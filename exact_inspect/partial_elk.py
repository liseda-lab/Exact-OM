"""Admit retained native ELK output as a separate, explicitly incomplete taxonomy.

No classification, parsing, remote access or on-request enrichment occurs here.
The complete-only hierarchy contract and historical receipts are left unchanged.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections import deque
from contextlib import closing
from pathlib import Path

from .artifacts import atomic_json
from .context_semantics import _json, _ref
from .contracts import canonical_hash, file_hash
from .navigation import PARTIAL_BASIS, PARTIAL_CONTRACT, PARTIAL_FILE
from .navigation_prepare import _BOTTOM, _SQL, _TOP, _entries

ADMISSION = "exact-elk-partial-admission/1"
VERSIONS = {"pyelk-reasoner": "0.2.1", "pyowl-core": "0.2.1"}
SEMANTICS = "direct_named_class_sub_to_super_edges_between_equivalence_groups"


def _load(path, digest):
    if file_hash(path) != digest:
        raise ValueError("Partial ELK input checksum mismatch")
    return json.loads(Path(path).read_bytes())


def validate_taxonomy(data):
    """Validate the exported group DAG without deriving new entailments."""
    groups, members = {}, set()
    for raw in data["nodes"]:
        if not isinstance(raw, list) or not raw or len(raw) > 4096:
            raise ValueError("Malformed or unbounded equivalence group")
        if any(not isinstance(x, str) or not x or len(x) > 2048 for x in raw):
            raise ValueError("Malformed named taxonomy member")
        group = tuple(sorted(raw))
        if len(set(raw)) != len(raw) or members.intersection(raw) or group in groups:
            raise ValueError("Overlapping taxonomy groups")
        groups[group] = canonical_hash(group)
        members.update(raw)
    top, bottom = tuple(sorted(data["top"])), tuple(sorted(data["bottom"]))
    if (
        top not in groups
        or bottom not in groups
        or top == bottom
        or _TOP not in top
        or _BOTTOM not in bottom
    ):
        raise ValueError("Missing bounds or inconsistent taxonomy")
    edges, outgoing, degree = set(), {g: [] for g in groups}, {g: 0 for g in groups}
    for raw in data["direct_edges"]:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("Malformed direct edge")
        child, parent = (tuple(sorted(x)) for x in raw)
        if (
            child not in groups
            or parent not in groups
            or child == parent
            or child == top
            or parent == bottom
        ):
            raise ValueError("Invalid taxonomy endpoint or orientation")
        if (child, parent) in edges:
            raise ValueError("Duplicate taxonomy edge")
        edges.add((child, parent))
        outgoing[child].append(parent)
        degree[parent] += 1
    pending = deque(g for g, n in degree.items() if n == 0)
    visited = 0
    while pending:
        g = pending.popleft()
        visited += 1
        for parent in outgoing[g]:
            degree[parent] -= 1
            if degree[parent] == 0:
                pending.append(parent)
    if visited != len(groups):
        raise ValueError("Cyclic taxonomy direct edges")
    return groups, edges, top, bottom, members


def admit(
    context,
    *,
    policy,
    manifest_path,
    result_path,
    taxonomy_path,
    resource_receipt_path,
    resource_path,
    expected,
):
    policy = context._effective_policy(policy)
    if context.manifest.get("policy_filter", {}).get("policy_hash") != policy.policy_hash:
        raise ValueError("Partial inference requires an exactly filtered context")
    paths = {
        "manifest": manifest_path,
        "result": result_path,
        "taxonomy": taxonomy_path,
        "resource_receipt": resource_receipt_path,
    }
    if set(expected) != set(paths):
        raise ValueError("Explicit independently recorded artifact pins are required")
    values = {key: _load(path, expected[key]) for key, path in paths.items()}
    manifest, result, data, receipt = (values[k] for k in paths)
    item = result["input"]
    if (
        manifest["schema"] != "exact-cluster-elk-inputs/1"
        or manifest["versions"] != VERSIONS
        or result["schema"] != "exact-cluster-elk-result/1"
        or result["versions"] != VERSIONS
        or result["manifest_sha256"] != expected["manifest"]
        or item not in manifest["inputs"]
        or result["runner_sha256"] != manifest["job_files"]["tools/cluster_elk_taxonomy.py"]
        or result["status"] != "completed_incomplete"
        or result["eligible_for_integration_review"] is not False
        or result["parse"]["complete"] is not True
        or result["parse"]["backend"] != "native"
        or result["parse"]["documents"] != 1
        or result["backend"]["name"] != "rust"
        or result["backend"]["native_available"] is not True
        or result["backend"]["implementation_version"] != VERSIONS["pyelk-reasoner"]
        or result["backend"]["fallback_reason"] is not None
        or result["options"]["backend"] != "rust"
        or result["options"]["require_native_pipeline"] is not True
        or result["options"]["unsupported"] != "ignore_with_explicit_incompleteness"
        or any(result["options"][k] != v for k, v in manifest["options"].items())
        or result["consistency"]["value"] is not True
        or result["classification"]["complete"] is not False
        or data["complete"] is not False
    ):
        raise ValueError("Unadmitted ELK execution, consistency or completeness")
    phases = {p["phase"]: p["status"] for p in result["phases"]}
    if any(
        phases.get(k) != "complete"
        for k in [
            "native_parse",
            "native_admission",
            "consistency",
            "classification",
            "taxonomy_export",
        ]
    ):
        raise ValueError("Interrupted or incomplete taxonomy export")
    if (
        data["schema"] != "exact-cluster-elk-taxonomy/1"
        or data["semantics"] != SEMANTICS
        or result["taxonomy"]["sha256"] != expected["taxonomy"]
        or item["receipt_sha256"] != expected["resource_receipt"]
        or file_hash(resource_path) != item["source_sha256"]
        or receipt["schema"] != "exact-ontology-resource/1"
        or receipt["logical_axioms_preserved"] is not True
        or receipt["imports"] != "flattened_declared_scope_no_external_resolution"
        or receipt["resource_sha256"] != item["source_sha256"]
        or receipt["resource_size_bytes"] != Path(resource_path).stat().st_size
        or receipt["source_context_sha256"]
        not in {
            context.manifest["artifacts"]["context.sqlite"],
            context.manifest.get("policy_filter", {}).get("source_context_sha256"),
        }
        or sorted(receipt["source_documents"]) != sorted(context.manifest["identity"]["documents"])
        or sorted(result["source_documents"]) != sorted(receipt["source_documents"])
    ):
        raise ValueError("Partial taxonomy differs from admitted exported resource")
    for key, value in {
        "ontology_version_id": context.ontology_version_id,
        "policy_hash": policy.policy_hash,
        "scope": context.manifest["scope"],
    }.items():
        if any(obj.get(key) != value for obj in [item, data, receipt]):
            raise ValueError("Partial taxonomy input policy or scope mismatch")
    if data["input_sha256"] != item["source_sha256"]:
        raise ValueError("Taxonomy input identity mismatch")
    groups, edges, top, bottom, members = validate_taxonomy(data)
    if (
        len(groups) != result["taxonomy"]["nodes"]
        or len(edges) != result["taxonomy"]["direct_edges"]
    ):
        raise ValueError("Truncated taxonomy")
    with context._connection() as db:
        classes = {r[0] for r in db.execute("SELECT iri FROM entities WHERE kind='class'")}
    if members - classes - {_TOP, _BOTTOM}:
        raise ValueError("Taxonomy contains a class outside admitted context")
    info = {
        "contract_version": PARTIAL_CONTRACT,
        "basis": PARTIAL_BASIS,
        "status": "partial",
        "complete": False,
        "completion": "partial_for_recorded_ELK_procedure",
        "reason": "Unsupported constructs remain; missing relationships are unknown.",
        "supported_kinds": ["class"],
        "reasoner": "pyelk",
        "versions": VERSIONS,
        "options": result["options"],
        "consistency": result["consistency"],
        "classification": result["classification"],
        "unsupported_constructs": result["classification"]["reasons"],
        "scope": context.manifest["scope"],
        "source_imports_complete": context.manifest["completeness"]["imports_complete"],
        "source_documents": receipt["source_documents"],
        "input_sha256": item["source_sha256"],
        "artifact_sha256": expected["taxonomy"],
        "policy_hash": policy.policy_hash,
        "semantics": SEMANTICS,
        "directness": "direct_in_returned_partial_taxonomy_only",
        "positive_edge_guarantee": "returned_by_pinned_ELK_supported_projection",
        "full_ontology_entailment": "unknown_without_exported_justification",
        "negative_entailment": "unknown",
        "satisfiability": "unknown",
        "justifications": "not_exported",
        "top": list(top),
        "bottom": list(bottom),
        "class_count": len(members & classes),
        "unrepresented_class_count": len(classes - members),
        "equivalence_group_count": len(groups),
        "reported_direct_edge_count": len(edges),
    }
    binding = {
        "schema": ADMISSION,
        "retained_complete_only_eligibility": False,
        "partial_admission": "verified",
        "pins": expected,
        "context_sha256": context.manifest["artifacts"]["context.sqlite"],
        "ontology_version_id": context.ontology_version_id,
        "policy_hash": policy.policy_hash,
        "inference": info,
    }
    info["admission_identity"] = canonical_hash(binding)
    from .postmigration_contracts import PartialInferenceMetadata

    PartialInferenceMetadata.model_validate(info)
    return data, binding


def materialize(context, *, policy, data, binding):
    """Write only a new sidecar/manifest in an unpublished private revision."""
    destination = context.path
    target = destination / PARTIAL_FILE
    if context.manifest.get("partial_navigation"):
        if (
            context.manifest["partial_navigation"]["preparation_key"]
            != binding["inference"]["admission_identity"]
        ):
            raise ValueError("Partial navigation resume identity mismatch")
        return context
    if any((parent / "publication.json").exists() for parent in list(destination.parents)[:4]):
        raise ValueError("Cannot change a published context; stage a private revision")
    checkpoint = destination / ".partial-preparation.json"
    identity = {"admission_identity": binding["inference"]["admission_identity"]}
    if checkpoint.exists() and json.loads(checkpoint.read_bytes()) != identity:
        raise ValueError("Partial preparation inputs changed before resume")
    if target.exists():
        if not checkpoint.exists() or target.is_symlink() or target.stat().st_nlink != 1:
            raise ValueError("Uncommitted partial output is not private owned staging")
        target.unlink()
    atomic_json(checkpoint, identity)
    groups, edges, top, bottom, members = validate_taxonomy(data)
    with context._connection() as source:
        classes = {row[0] for row in source.execute("SELECT iri FROM entities WHERE kind='class'")}
    manifest = dict(context.manifest)
    info = binding["inference"]
    navigation = {
        "contract_version": PARTIAL_CONTRACT,
        "preparation_key": info["admission_identity"],
        "context_sha256": manifest["artifacts"]["context.sqlite"],
        "ontology_version_id": context.ontology_version_id,
        "policy_hash": policy.policy_hash,
        "entry_points": "available",
        "order": "root_priority_label_iri",
        "inference": info,
    }
    fd, temp = tempfile.mkstemp(prefix=".partial-", dir=destination)
    os.close(fd)
    try:
        with closing(sqlite3.connect(temp)) as db:
            db.executescript(_SQL)
            for group, gid in groups.items():
                for iri in group:
                    if iri in classes:
                        db.execute(
                            "INSERT INTO members VALUES (?,?,?,?,?)",
                            (iri, "class", gid, int(group == top), int(group == bottom)),
                        )
            count = 0
            for child, parent in sorted(edges):
                for a in child:
                    for b in parent:
                        if a not in classes or b not in classes:
                            continue
                        count += 1
                        if count > 2_000_000:
                            raise ValueError("Bounded named-edge expansion exceeded")
                        edge = {
                            "child": _ref(context.ontology_version_id, a, "class"),
                            "parent": _ref(context.ontology_version_id, b, "class"),
                            "basis": PARTIAL_BASIS,
                            "relation": "subclass",
                            "direct": True,
                            "axiom_id": None,
                            "interpretation": {
                                "kind": "reasoner_inferred",
                                "provider": "pyelk",
                                "input_sha256": info["input_sha256"],
                                "premises": [],
                                "premises_unavailable": True,
                            },
                            "origins": [],
                        }
                        edge["id"] = canonical_hash(
                            {"artifact": info["artifact_sha256"], "edge": edge}
                        )
                        db.execute(
                            "INSERT INTO edges VALUES (?,?,?,?,?)",
                            (edge["id"], a, b, "class", _json(edge)),
                        )
            info["served_direct_edge_count"] = count
            _entries(
                context, db, policy, inferred=True, inferred_basis=PARTIAL_BASIS, only_inferred=True
            )
            db.execute("INSERT INTO metadata VALUES (?,?)", ("binding", canonical_hash(navigation)))
            db.execute("ANALYZE")
            db.commit()
        os.replace(temp, target)
        receipt = destination / "partial-admission.json"
        atomic_json(receipt, binding)
        manifest.update(
            partial_navigation=navigation,
            capabilities={**manifest["capabilities"], PARTIAL_BASIS: info},
            artifacts={
                **manifest["artifacts"],
                PARTIAL_FILE: file_hash(target),
                "partial-admission.json": file_hash(receipt),
            },
            portable={
                **manifest["portable"],
                "browsing": list(
                    dict.fromkeys(
                        [*manifest["portable"]["browsing"], PARTIAL_FILE, "partial-admission.json"]
                    )
                ),
            },
        )
        atomic_json(destination / "manifest.json", manifest)
        from .context import OntologyContext

        return OntologyContext(destination)
    finally:
        Path(temp).unlink(missing_ok=True)
