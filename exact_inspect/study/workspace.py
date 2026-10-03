"""Frozen participant workspace over the shared, policy-filtered read indexes.

A scope is a locator, never a capability. Authorization and cursor binding include
its participant, invitation generation, publication and current presentation on
*every* read. Preparation is separate from serving and never dispatches a provider.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from ..context import OntologyContext
from ..context_semantics import decode_payload
from ..contracts import (
    DomainError,
    EntityKind,
    EntityRef,
    Page,
    VisibilityPolicy,
    canonical_hash,
    canonical_json,
    decode_cursor,
    encode_cursor,
    file_hash,
)
from ..generation import Claim
from ..models import (
    AxiomResponse,
    EntityContextResponse,
    ExplanationSummary,
    Fact,
    GeneratedExplanationResponse,
    GenerationManifest,
    HierarchyPage,
    LabelsResponse,
    SelectedEvidence,
)
from ..sqlite_safety import readonly_connection
from .models import Identifier, StrictModel


def _error(status, message):
    from .store import StudyError

    return StudyError(status, message)


Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Component = Literal["context", "hierarchy", "profiles", "comparison", "evidence"]
_VERIFIED_INDEXES = OrderedDict()
_VERIFICATION_LOCK = Lock()
_TABLES = {
    "explanations": ("id", "task", "entities", "summary", "payload"),
    "subjects": ("explanation_id", "entity"),
    "evidence": ("candidate_id", "id", "payload"),
}


def _locator(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or str(path) == ".":
        raise ValueError("Workspace paths must be safe relative locators")
    return value


class WorkspaceContextRef(StrictModel):
    ontology_version_id: Identifier
    path: Annotated[str, Field(min_length=1, max_length=512)]
    manifest_sha256: Digest
    database_sha256: Digest

    @model_validator(mode="after")
    def safe_path(self):
        _locator(self.path)
        return self


class WorkspaceScope(StrictModel):
    scope_id: Identifier
    kind: Literal["case", "tutorial"]
    case_id: Identifier
    contexts: Annotated[list[WorkspaceContextRef], Field(min_length=1, max_length=20)]
    index_path: Annotated[str, Field(min_length=1, max_length=512)]
    index_sha256: Digest
    components: list[Component] = Field(default_factory=lambda: ["context", "hierarchy"])

    @model_validator(mode="after")
    def unique_resources(self):
        _locator(self.index_path)
        if len({c.ontology_version_id for c in self.contexts}) != len(self.contexts):
            raise ValueError("Duplicate workspace ontology")
        if len(set(self.components)) != len(self.components) or "context" not in self.components:
            raise ValueError("Workspace requires unique components and context")
        return self


class WorkspaceCapabilitiesResponse(StrictModel):
    contract_version: Literal["exact-explain/1.0"] = "exact-explain/1.0"
    scope_id: Identifier
    study_revision: Identifier
    policy_hash: str
    synthetic: bool
    focal_entities: list[EntityRef]
    components: list[Component]
    page_limit: int
    iri_limit: int
    ontology_versions: list[Identifier]
    bases: list[Literal["literal_asserted", "structural_navigation", "reasoner_inferred"]]
    reasoner: Literal["not_run"]
    reason: str
    completeness: dict[str, dict[str, Any]]
    coverage: Literal["complete_declared_context_scope"]
    continuation: Literal["query_bound_keyset"]
    original_format: str
    graph_form: str


def _digest(path: Path) -> str:
    return file_hash(path).removeprefix("sha256:")


def _identity(entity: Any) -> str:
    return canonical_hash(EntityRef.model_validate(entity).model_dump())


def _focal(case):
    return [case["source"], *(c["entity"] for c in case["candidates"])]


def _strict_projection(model, value):
    """Reject extra envelope fields even where the exploration model is additive."""
    if set(value) - set(model.model_fields):
        raise ValueError("Unadmitted workspace record field")
    return model.model_validate(value).model_dump(mode="json")


def _admit_explanation(value, case, policy, contexts):
    result = _strict_projection(GeneratedExplanationResponse, value)
    manifest = dict(value["manifest"])
    normalization = manifest.pop("category_alias_normalization", None)
    result["manifest"] = _strict_projection(GenerationManifest, manifest)
    if normalization is not None:
        if not isinstance(normalization, dict) or any(
            not isinstance(k, str) or not isinstance(v, int) or v < 0
            for k, v in normalization.items()
        ):
            raise ValueError("Invalid recorded category normalization")
        result["manifest"]["category_alias_normalization"] = normalization
    result["claims"] = [_strict_projection(Claim, c) for c in value["claims"]]
    focal = {_identity(e) for e in _focal(case)}
    if (
        result["grounding_status"] != "validated"
        or result["manifest"]["visibility_policy_hash"] != policy.policy_hash
        or any(_identity(e) not in focal for e in result["entities"])
        or len(result["entities"]) != (1 if result["task"] == "entity_profile" else 2)
    ):
        raise ValueError("Explanation differs from the frozen case or policy")
    if result["task"] == "pair_comparison" and (
        result["entities"][0] != case["source"]
        or result["entities"][1] not in [c["entity"] for c in case["candidates"]]
    ):
        raise ValueError("Comparison is not an active-case ordered pair")
    for claim in result["claims"]:
        if not claim["fact_ids"]:
            raise ValueError("Workspace claims require original fact citations")
        for fid in claim["fact_ids"]:
            for entity in result["entities"]:
                try:
                    contexts[entity["ontology_version_id"]].fact(fid, entity, policy=policy)
                    break
                except KeyError:
                    pass
            else:
                raise ValueError("Workspace citation has no permitted typed subject")
    return result


def _admit_evidence(value, candidate, case, policy, contexts):
    result = _strict_projection(SelectedEvidence, value)
    expected = case["source"] if result["side"] == "source" else candidate["entity"]
    if result["entity"] != expected:
        raise ValueError("Selected evidence differs from its active-case pair")
    context = contexts[expected["ontology_version_id"]]
    for fid in result["fact_ids"]:
        try:
            context.fact(fid, expected, policy=policy)
        except KeyError as exc:
            raise ValueError("Evidence citation is outside the admitted typed subject") from exc
    # The strict main evidence contract preserves recorded projection, missingness,
    # selection numbers and unsupported graph forms without inventing feature pairs.
    return result


def build_workspace_scope(
    root: str | Path,
    *,
    scope_id: str,
    kind: Literal["case", "tutorial"],
    case: dict[str, Any],
    contexts: dict[str, OntologyContext],
    policy: VisibilityPolicy,
    explanations=(),
    evidence=None,
    components: list[Component] | None = None,
) -> WorkspaceScope:
    """Prepare complete declared context coverage and indexed active-case records.

    Call before publication. Inputs are already prepared artifacts; this operation
    filters SQLite indexes and validates saved records, never parses an ontology.
    Scope directories are content-bound and cannot overwrite published resources.
    """
    from ..context_export import export_policy_context

    root = Path(root).resolve()
    base = root / "workspaces" / hashlib.sha256(scope_id.encode()).hexdigest()
    base.mkdir(parents=True, exist_ok=True)
    prepared = {}
    references = []
    required = {e["ontology_version_id"] for e in _focal(case)}
    for version in sorted(required):
        source = contexts[version]
        destination = base / hashlib.sha256(version.encode()).hexdigest()
        context = export_policy_context(source, destination, policy)
        prepared[version] = context
        references.append(
            WorkspaceContextRef(
                ontology_version_id=version,
                path=destination.relative_to(root).as_posix(),
                manifest_sha256=_digest(destination / "manifest.json"),
                database_sha256=_digest(destination / "context.sqlite"),
            )
        )
    database = base / "workspace.sqlite"
    if database.exists():
        raise ValueError("Workspace index already exists; prepare a new scope")
    try:
        with sqlite3.connect(database) as output:
            output.executescript(
                "CREATE TABLE explanations(id TEXT PRIMARY KEY, task TEXT NOT NULL, "
                "entities TEXT NOT NULL, summary TEXT NOT NULL, payload TEXT NOT NULL);"
                "CREATE TABLE subjects(explanation_id TEXT,entity TEXT,PRIMARY KEY(entity,explanation_id));"
                "CREATE TABLE evidence(candidate_id TEXT,id TEXT,payload TEXT,PRIMARY KEY(candidate_id,id));"
            )
            for value in explanations:
                result = _admit_explanation(value, case, policy, prepared)
                summary = {
                    k: result[k] for k in ("explanation_id", "task", "entities", "grounding_status")
                }
                summary["generation_status"] = result["manifest"]["status"]
                output.execute(
                    "INSERT INTO explanations VALUES (?,?,?,?,?)",
                    (
                        result["explanation_id"],
                        result["task"],
                        canonical_json(result["entities"]).decode(),
                        canonical_json(summary).decode(),
                        canonical_json(result).decode(),
                    ),
                )
                output.executemany(
                    "INSERT INTO subjects VALUES (?,?)",
                    [(result["explanation_id"], _identity(e)) for e in result["entities"]],
                )
            candidates = {c["candidate_id"]: c for c in case["candidates"]}
            for cid, rows in (evidence or {}).items():
                if cid not in candidates:
                    raise ValueError("Evidence belongs to a different case")
                for value in rows:
                    result = _admit_evidence(value, candidates[cid], case, policy, prepared)
                    output.execute(
                        "INSERT INTO evidence VALUES (?,?,?)",
                        (cid, result["evidence_id"], canonical_json(result).decode()),
                    )
    except Exception:
        database.unlink(missing_ok=True)
        raise
    return WorkspaceScope(
        scope_id=scope_id,
        kind=kind,
        case_id=case["case_id"],
        contexts=references,
        index_path=database.relative_to(root).as_posix(),
        index_sha256=_digest(database),
        components=components or ["context", "hierarchy"],
    )


def _open_resources(store, scope, policy):
    contexts = {}
    for ref in scope.contexts:
        manifest = store._resource_path(ref.path + "/manifest.json")
        database = store._resource_path(ref.path + "/context.sqlite")
        if not manifest.is_file() or not database.is_file():
            raise ValueError("Frozen workspace context is missing")
        if _digest(manifest) != ref.manifest_sha256:
            raise ValueError("Frozen workspace manifest changed")
        context = OntologyContext(manifest.parent)
        if (
            context.ontology_version_id != ref.ontology_version_id
            or context.manifest["artifacts"]["context.sqlite"] != "sha256:" + ref.database_sha256
            or context.manifest.get("policy_filter", {}).get("policy_hash") != policy.policy_hash
        ):
            raise ValueError("Workspace requires the frozen physically filtered context")
        # The path resolver separately rejects escaping links for both files.
        if context.database.resolve() != database.resolve():
            raise ValueError("Workspace index path differs")
        contexts[ref.ontology_version_id] = context
    index = store._resource_path(scope.index_path)
    if not index.is_file():
        raise ValueError("Frozen workspace index is missing")
    stat = index.stat()
    key = (
        str(index.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    with _VERIFICATION_LOCK:
        actual = _VERIFIED_INDEXES.get(key)
        if actual is None:
            actual = _digest(index)
            if index.stat() != stat:
                raise ValueError("Workspace index changed during verification")
            _VERIFIED_INDEXES[key] = actual
            while len(_VERIFIED_INDEXES) > 128:
                _VERIFIED_INDEXES.popitem(last=False)
    if actual != scope.index_sha256:
        raise ValueError("Frozen workspace index changed")
    return contexts, index


def case_workspace_descriptor(store, study, case):
    """Discover only the already-authorized case's frozen, usable locator."""
    try:
        matches = [
            s
            for s in study.get("workspace_scopes", [])
            if s.get("kind") == "case" and s.get("case_id") == case["case_id"]
        ]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous case workspace")
        scope = WorkspaceScope.model_validate(matches[0])
        if not {e["ontology_version_id"] for e in _focal(case)} <= {
            c.ontology_version_id for c in scope.contexts
        }:
            raise ValueError("Workspace does not cover the case ontologies")
        _open_resources(store, scope, VisibilityPolicy.model_validate(study["visibility_policy"]))
        return {"scope_id": scope.scope_id}
    except (ValueError, OSError, KeyError) as exc:
        raise _error(
            503, "Frozen case workspace is unavailable; retry or contact the study team"
        ) from exc


def validate_publication_workspaces(store, study, admitted_explanations):
    """Reject incomplete capabilities, alien case records and scored tutorial graphs."""
    scopes = [WorkspaceScope.model_validate(s) for s in study.get("workspace_scopes", [])]
    if len({s.scope_id for s in scopes}) != len(scopes):
        raise ValueError("Duplicate workspace scope")
    cases = {c["case_id"]: c for c in study["cases"]}
    tutorial = study.get("tutorial")
    if hasattr(tutorial, "model_dump"):
        tutorial = tutorial.model_dump(mode="json")
    tutorial_case = (tutorial or {}).get("case")
    policy = VisibilityPolicy.model_validate(study["visibility_policy"])
    scored_versions = {e["ontology_version_id"] for c in cases.values() for e in _focal(c)}
    scored_iris = {e["iri"] for c in cases.values() for e in _focal(c)}
    scored_contexts = []
    tutorial_contexts = []
    seen_cases = set()
    tutorial_count = 0
    for scope in scopes:
        if scope.kind == "tutorial":
            tutorial_count += 1
            if not tutorial_case or scope.case_id != tutorial_case["case_id"]:
                raise ValueError("Tutorial workspace is not its frozen synthetic case")
            case = tutorial_case
            if any(ref.ontology_version_id in scored_versions for ref in scope.contexts):
                raise ValueError("Tutorial cannot use a scored ontology index")
        else:
            if scope.case_id not in cases or scope.case_id in seen_cases:
                raise ValueError("Case workspace is missing or duplicated")
            seen_cases.add(scope.case_id)
            case = cases[scope.case_id]
        contexts, index = _open_resources(store, scope, policy)
        (tutorial_contexts if scope.kind == "tutorial" else scored_contexts).extend(
            contexts.values()
        )
        if set(contexts) != {e["ontology_version_id"] for e in _focal(case)}:
            raise ValueError("Workspace must cover source and all five candidates")
        for entity in _focal(case):
            try:
                contexts[entity["ontology_version_id"]]._entity(entity, policy)
            except KeyError as exc:
                raise ValueError("A focal entity is absent from the workspace index") from exc
        if scope.kind == "tutorial":
            for context in contexts.values():
                with context._connection() as db:
                    for row in db.execute("SELECT iri FROM entities"):
                        if row[0] in scored_iris:
                            raise ValueError("Tutorial graph reaches a scored entity")
        if scope.kind == "case" and study.get("contract_version") == "exact-study/2.0":
            names = {
                "original_context": "context",
                "entity_description": "profiles",
                "hierarchy": "hierarchy",
                "evidence_table": "evidence",
                "evidence_graph": "evidence",
                "pair_comparison": "comparison",
            }
            if any(
                names[c] not in scope.components for c in study.get("components", []) if c in names
            ):
                raise ValueError("Study promises a component outside the admitted workspace")
        counts = {"entity_profile": 0, "pair_comparison": 0, "evidence": 0}
        with readonly_connection(index, _TABLES) as db:
            for row in db.execute("SELECT id,task,entities,summary,payload FROM explanations"):
                value = _admit_explanation(json.loads(row[4]), case, policy, contexts)
                expected_summary = {
                    k: value[k] for k in ("explanation_id", "task", "entities", "grounding_status")
                }
                expected_summary["generation_status"] = value["manifest"]["status"]
                actual_subjects = {
                    r[0]
                    for r in db.execute(
                        "SELECT entity FROM subjects WHERE explanation_id=?", (row[0],)
                    )
                }
                if (
                    row[0] != value["explanation_id"]
                    or row[1] != value["task"]
                    or json.loads(row[2]) != value["entities"]
                    or json.loads(row[3]) != expected_summary
                    or actual_subjects != {_identity(e) for e in value["entities"]}
                ):
                    raise ValueError("Workspace discovery index differs from its admitted record")
                store._check_safe_json(value)
                counts[value["task"]] += 1
                # Saved prose must have independently admitted grounding, not only a
                # caller-supplied validated flag or a syntactically valid citation.
                admitted = [
                    admitted_explanations[r]
                    for r in case.get("explanation_refs", [])
                    if r in admitted_explanations
                ]
                allowed = {
                    (
                        claim.text,
                        tuple(claim.fact_ids),
                        claim.generation_manifest_sha256,
                        claim.category,
                        canonical_hash([e.model_dump() for e in claim.scoped_entities]),
                    )
                    for resource in admitted
                    for claim in resource.entity_profiles + resource.pair_comparison
                }
                manifest_hash = canonical_hash(value["manifest"]).removeprefix("sha256:")
                if any(
                    (
                        c["text"],
                        tuple(c["fact_ids"]),
                        manifest_hash,
                        c["category"],
                        canonical_hash(value["entities"]),
                    )
                    not in allowed
                    for c in value["claims"]
                ):
                    raise ValueError("Workspace prose lacks independently admitted grounding")
            candidates = {c["candidate_id"]: c for c in case["candidates"]}
            for row in db.execute("SELECT candidate_id,payload,id FROM evidence"):
                if row[0] not in candidates:
                    raise ValueError("Workspace evidence belongs to another case")
                value = _admit_evidence(
                    json.loads(row[1]), candidates[row[0]], case, policy, contexts
                )
                if row[2] != value["evidence_id"]:
                    raise ValueError("Workspace evidence index identity differs")
                store._check_safe_json(value)
                counts["evidence"] += 1
        for component, task in (
            ("profiles", "entity_profile"),
            ("comparison", "pair_comparison"),
            ("evidence", "evidence"),
        ):
            if component in scope.components and not counts[task]:
                raise ValueError("Promised workspace component has no prepared resources")
    # Validate the entire admitted ontology graph, including navigation targets;
    # do not let practice reveal an entity only used by an unrelated scored case.
    tutorial_iris = set()
    for context in tutorial_contexts:
        with context._connection() as db:
            tutorial_iris.update(row[0] for row in db.execute("SELECT iri FROM entities"))
            for row in db.execute("SELECT payload FROM axioms"):
                pending = [decode_payload(row[0])["ast"]]
                while pending:
                    node = pending.pop()
                    if isinstance(node, dict):
                        if node.get("type") == "IRI" and isinstance(node.get("value"), str):
                            tutorial_iris.add(node["value"])
                        pending.extend(node.values())
                    elif isinstance(node, list):
                        pending.extend(node)
    ordered = sorted(tutorial_iris)
    for context in scored_contexts:
        with context._connection() as db:
            for offset in range(0, len(ordered), 100):
                chunk = ordered[offset : offset + 100]
                marks = ",".join("?" for _ in chunk)
                if db.execute(
                    f"SELECT 1 FROM entities WHERE iri IN ({marks}) LIMIT 1", chunk
                ).fetchone():
                    raise ValueError("Tutorial graph intersects a scored ontology graph")
    if study.get("contract_version") == "exact-study/2.0" and (
        seen_cases != set(cases) or tutorial_count != 1
    ):
        raise ValueError("V2 requires one workspace for every case and its synthetic tutorial")


class ParticipantWorkspace:
    """One already-authorized request, with bounded index reads and scoped cursors."""

    def __init__(self, store, scope, study, case, binding):
        self.scope, self.study, self.case, self.binding = scope, study, case, binding
        self.policy = VisibilityPolicy.model_validate(study["visibility_policy"])
        self.contexts, self.index = _open_resources(store, scope, self.policy)

    def context(self, version):
        if version not in self.contexts:
            raise _error(404, "Workspace ontology unavailable")
        return self.contexts[version]

    def cursor_scope(self, operation, query):
        return {**self.binding, "operation": operation, "query": query}

    def page(self, operation, query, read, cursor=None):
        binding = self.cursor_scope(operation, query)
        underlying = decode_cursor(cursor, binding) if cursor else None
        if underlying is not None and not isinstance(underlying, str):
            raise DomainError("invalid_cursor", "Invalid workspace cursor")
        result = read(underlying)
        if result["next_cursor"]:
            result["next_cursor"] = encode_cursor(binding, result["next_cursor"])
        return result

    def entity_context(self, entity):
        result = self.context(entity["ontology_version_id"]).entity_context(
            entity, policy=self.policy
        )
        # Rewrite every embedded continuation to the same authorization-bound fact
        # route used for later pages, including the original default page limit.
        for category, page in result["categories"].items():
            if page["next_cursor"]:
                page["next_cursor"] = encode_cursor(
                    self.cursor_scope(
                        "facts", {"entity": entity, "category": category, "basis": "asserted"}
                    ),
                    page["next_cursor"],
                )
        for name in ("definitions", "synonyms"):
            result[name] = result["categories"][name]
        if result["parents"]["next_cursor"]:
            result["parents"]["next_cursor"] = encode_cursor(
                self.cursor_scope(
                    "hierarchy",
                    {"entity": entity, "direction": "parents", "basis": "literal_asserted"},
                ),
                result["parents"]["next_cursor"],
            )
        return result

    def indexed_page(self, table, where, args, *, cursor=None, limit=20):
        OntologyContext._bound(limit)
        binding = self.cursor_scope(table, [where, args])
        after = decode_cursor(cursor, binding) if cursor else ""
        if not isinstance(after, str):
            raise DomainError("invalid_cursor", "Invalid workspace cursor")
        column = "summary" if table == "explanations" else "payload"
        with readonly_connection(self.index, _TABLES) as db:
            total = db.execute(f"SELECT count(*) FROM {table} WHERE {where}", args).fetchone()[0]
            rows = db.execute(
                f"SELECT id,{column} FROM {table} WHERE {where} AND id>? ORDER BY id LIMIT ?",
                [*args, after, limit + 1],
            ).fetchall()
        more = len(rows) > limit
        return {
            "items": [json.loads(r[1]) for r in rows[:limit]],
            "returned_count": min(limit, len(rows)),
            "total_count": total,
            "truncated": more,
            "next_cursor": encode_cursor(binding, rows[limit - 1][0]) if more else None,
            "scope": {
                "ontology_version_id": self.case["source"]["ontology_version_id"],
                "context_revision": self.study["study_revision"],
                "basis": "prepared_explanations" if table == "explanations" else "run_selected",
                "visibility_policy_hash": self.policy.policy_hash,
                "filter_id": canonical_hash(binding),
            },
            "status": "available" if total else "not_exported",
            "reason": None if total else "No resource was prepared for this selection",
        }


@contextmanager
def authorized_workspace(store, sid, generation, scope_id):
    """Hold the session lock until the bounded read finishes, including transitions."""
    with store.transaction() as db:
        row, state, study = store._session(db, sid, generation)
        if not (state.get("consent") or {}).get("accepted") or state["stage"] in {
            "closed",
            "paused",
        }:
            raise _error(403, "Workspace unavailable in this stage")
        scope = next(
            (
                WorkspaceScope.model_validate(s)
                for s in study.get("workspace_scopes", [])
                if s["scope_id"] == scope_id
            ),
            None,
        )
        if scope is None:
            raise _error(404, "Workspace unavailable")
        current = store._current(state)
        if scope.kind == "case":
            if (
                not current
                or current["condition"] != "explanation"
                or state["stage"] not in {"case", "consultation"}
                or current["case_id"] != scope.case_id
            ):
                raise _error(403, "Workspace unavailable for the current presentation")
            case = next(c for c in study["cases"] if c["case_id"] == scope.case_id)
        else:
            if state["stage"] not in {
                "practice",
                "tutorial",
                "case",
                "consultation",
                "final",
                "completed",
            }:
                raise _error(403, "Synthetic help is unavailable in this stage")
            case = study["tutorial"]["case"]
        binding = {
            "session": sid,
            "generation": generation,
            "publication": row["study_revision"],
            "policy": study["policy_hash"],
            "scope": scope.scope_id,
            "index": scope.index_sha256,
            "presentation": current["presentation_id"] if current else None,
            "stage": state["stage"],
        }
        yield ParticipantWorkspace(store, scope, study, case, binding)


def install_workspace_routes(router, store, participant_dependency):
    """Mount only scoped reads; pass the existing signed-session dependency."""
    from fastapi import Depends, Query
    from fastapi.responses import JSONResponse

    prefix = "/study/workspace/{scope_id:path}"

    def read(scope_id, identity, operation):
        try:
            with authorized_workspace(store, *identity, scope_id) as workspace:
                result = operation(workspace)
                if len(canonical_json(result)) > 4 * 1024 * 1024:
                    raise _error(413, "Workspace response exceeds the inline budget")
                return result
        except KeyError as exc:
            raise _error(404, "Workspace resource unavailable") from exc
        except DomainError as exc:
            return JSONResponse(exc.envelope.model_dump(mode="json"), status_code=exc.status_code)
        except (ValueError, OSError, sqlite3.DatabaseError) as exc:
            raise _error(
                503, "Frozen workspace resources are unavailable; retry or contact the study team"
            ) from exc

    def entity(version, iri, kind):
        return EntityRef(ontology_version_id=version, iri=iri, kind=kind).model_dump()

    @router.get(prefix + "/capabilities", response_model=WorkspaceCapabilitiesResponse)
    @router.get(prefix + "/scope", response_model=WorkspaceCapabilitiesResponse)
    def capabilities(scope_id: str, identity=Depends(participant_dependency)):
        def result(w):
            return {
                "contract_version": "exact-explain/1.0",
                "scope_id": w.scope.scope_id,
                "study_revision": w.study["study_revision"],
                "policy_hash": w.policy.policy_hash,
                "synthetic": w.scope.kind == "tutorial",
                "focal_entities": _focal(w.case),
                "components": w.scope.components,
                "page_limit": 100,
                "iri_limit": 2048,
                "ontology_versions": list(w.contexts),
                "bases": ["literal_asserted", "structural_navigation"],
                "reasoner": "not_run",
                "reason": "No recorded inference artifact was prepared",
                "completeness": {
                    version: {
                        "scope": context.manifest["scope"],
                        **context.manifest["completeness"],
                        "domain_completeness": "not_established",
                    }
                    for version, context in w.contexts.items()
                },
                "coverage": "complete_declared_context_scope",
                "continuation": "query_bound_keyset",
                "original_format": "typed_ast_and_canonical_bytes_base64; base64 is not human-readable syntax",
                "graph_form": "recorded_features_only; channel bridges do not imply feature pairing",
            }

        return read(scope_id, identity, result)

    @router.get(prefix + "/entities", response_model=Page[dict[str, Any]])
    @router.get(prefix + "/entities/search", response_model=Page[dict[str, Any]])
    def entities(
        scope_id: str,
        ontology_version_id: str,
        kind: EntityKind | None = None,
        term: str = Query("", max_length=512),
        language: str | None = None,
        limit: int = Query(20, ge=1, le=100),
        cursor: str | None = None,
        identity=Depends(participant_dependency),
    ):
        query = {
            "ontology_version_id": ontology_version_id,
            "kind": kind,
            "term": term,
            "language": language,
        }
        return read(
            scope_id,
            identity,
            lambda w: w.page(
                "search",
                query,
                lambda c: w.context(ontology_version_id).search(
                    kind=kind, term=term, language=language, limit=limit, cursor=c, policy=w.policy
                ),
                cursor,
            ),
        )

    @router.get(prefix + "/entity-context", response_model=EntityContextResponse)
    @router.get(prefix + "/entities/context", response_model=EntityContextResponse)
    def context(
        scope_id: str,
        ontology_version_id: str,
        iri: str = Query(..., min_length=1, max_length=2048),
        kind: EntityKind = "class",
        identity=Depends(participant_dependency),
    ):
        return read(
            scope_id, identity, lambda w: w.entity_context(entity(ontology_version_id, iri, kind))
        )

    @router.get(prefix + "/entity-facts", response_model=Page[Fact])
    def facts(
        scope_id: str,
        ontology_version_id: str,
        iri: str = Query(..., min_length=1, max_length=2048),
        kind: EntityKind = "class",
        category: str | None = None,
        basis: Literal[
            "asserted", "literal_asserted", "structural_navigation", "reasoner_inferred"
        ] = "asserted",
        limit: int = Query(20, ge=1, le=100),
        cursor: str | None = None,
        identity=Depends(participant_dependency),
    ):
        ref = entity(ontology_version_id, iri, kind)
        query = {"entity": ref, "category": category, "basis": basis}
        return read(
            scope_id,
            identity,
            lambda w: w.page(
                "facts",
                query,
                lambda c: w.context(ontology_version_id).facts(
                    ref, category=category, basis=basis, limit=limit, cursor=c, policy=w.policy
                ),
                cursor,
            ),
        )

    @router.get(prefix + "/hierarchy", response_model=HierarchyPage)
    def hierarchy(
        scope_id: str,
        ontology_version_id: str,
        iri: str = Query(..., min_length=1, max_length=2048),
        kind: EntityKind = "class",
        direction: Literal["parents", "children"] = "parents",
        basis: Literal[
            "literal_asserted", "structural_navigation", "reasoner_inferred"
        ] = "literal_asserted",
        limit: int = Query(50, ge=1, le=100),
        cursor: str | None = None,
        identity=Depends(participant_dependency),
    ):
        ref = entity(ontology_version_id, iri, kind)
        query = {"entity": ref, "direction": direction, "basis": basis}
        return read(
            scope_id,
            identity,
            lambda w: w.page(
                "hierarchy",
                query,
                lambda c: w.context(ontology_version_id).hierarchy(
                    ref, direction=direction, basis=basis, limit=limit, cursor=c, policy=w.policy
                ),
                cursor,
            ),
        )

    @router.get(prefix + "/labels", response_model=LabelsResponse)
    def labels(
        scope_id: str,
        ontology_version_id: str,
        iri: list[str] = Query(..., min_length=1, max_length=100),
        language: str | None = None,
        identity=Depends(participant_dependency),
    ):
        def result(w):
            items = w.context(ontology_version_id).labels(iri, language=language, policy=w.policy)
            return {
                "ontology_version_id": ontology_version_id,
                "items": items,
                "returned_count": len(items),
            }

        return read(scope_id, identity, result)

    @router.get(prefix + "/axioms/{axiom_id}", response_model=AxiomResponse)
    def axiom(
        scope_id: str,
        axiom_id: str,
        ontology_version_id: str,
        iri: str | None = Query(None, max_length=2048),
        kind: EntityKind = "class",
        identity=Depends(participant_dependency),
    ):
        def result(w):
            context = w.context(ontology_version_id)
            if iri is not None:
                context.fact(axiom_id, entity(ontology_version_id, iri, kind), policy=w.policy)
            return context.axiom(axiom_id, policy=w.policy)

        return read(scope_id, identity, result)

    @router.get(prefix + "/facts/{fact_id}", response_model=Fact)
    def fact(
        scope_id: str,
        fact_id: str,
        ontology_version_id: str,
        iri: str = Query(..., min_length=1, max_length=2048),
        kind: EntityKind = "class",
        identity=Depends(participant_dependency),
    ):
        return read(
            scope_id,
            identity,
            lambda w: w.context(ontology_version_id).fact(
                fact_id, entity(ontology_version_id, iri, kind), policy=w.policy
            ),
        )

    @router.get(prefix + "/explanations", response_model=Page[ExplanationSummary])
    def explanations(
        scope_id: str,
        ontology_version_id: str,
        iri: str = Query(..., min_length=1, max_length=2048),
        kind: EntityKind = "class",
        task: Literal["entity_profile", "pair_comparison"] | None = None,
        counterpart_ontology_version_id: str | None = None,
        counterpart_iri: str | None = Query(None, max_length=2048),
        counterpart_kind: EntityKind = "class",
        limit: int = Query(20, ge=1, le=100),
        cursor: str | None = None,
        identity=Depends(participant_dependency),
    ):
        wanted = [entity(ontology_version_id, iri, kind)]
        if (counterpart_iri is None) != (counterpart_ontology_version_id is None):
            raise _error(422, "Counterpart requires ontology and IRI")
        if counterpart_iri is not None:
            wanted.append(
                entity(counterpart_ontology_version_id, counterpart_iri, counterpart_kind)
            )

        def result(w):
            focal = {_identity(e) for e in _focal(w.case)}
            if any(_identity(e) not in focal for e in wanted):
                raise _error(403, "Generated resources are restricted to the active case")
            admitted_tasks = [
                task_name
                for component, task_name in (
                    ("profiles", "entity_profile"),
                    ("comparison", "pair_comparison"),
                )
                if component in w.scope.components
            ]
            if not admitted_tasks or (task is not None and task not in admitted_tasks):
                raise _error(403, "Explanation component is outside the admitted workspace")
            conditions, args = ["task IN (" + ",".join("?" for _ in admitted_tasks) + ")"], list(
                admitted_tasks
            )
            for ref in wanted:
                conditions.append("id IN (SELECT explanation_id FROM subjects WHERE entity=?)")
                args.append(_identity(ref))
            if task:
                conditions.append("task=?")
                args.append(task)
            return w.indexed_page(
                "explanations", " AND ".join(conditions), args, cursor=cursor, limit=limit
            )

        return read(scope_id, identity, result)

    @router.get(
        prefix + "/explanations/{explanation_id}", response_model=GeneratedExplanationResponse
    )
    def explanation(scope_id: str, explanation_id: str, identity=Depends(participant_dependency)):
        def result(w):
            with readonly_connection(w.index, _TABLES) as db:
                row = db.execute(
                    "SELECT payload FROM explanations WHERE id=?", (explanation_id,)
                ).fetchone()
            if row is None:
                raise _error(404, "Workspace explanation unavailable")
            value = json.loads(row[0])
            if (
                "profiles" if value["task"] == "entity_profile" else "comparison"
            ) not in w.scope.components:
                raise _error(403, "Explanation component is outside the admitted workspace")
            return value

        return read(scope_id, identity, result)

    @router.get(prefix + "/evidence", response_model=Page[SelectedEvidence])
    def evidence(
        scope_id: str,
        candidate_id: str,
        limit: int = Query(20, ge=1, le=100),
        cursor: str | None = None,
        identity=Depends(participant_dependency),
    ):
        def result(w):
            if "evidence" not in w.scope.components:
                raise _error(403, "Evidence component is outside the admitted workspace")
            if candidate_id not in {c["candidate_id"] for c in w.case["candidates"]}:
                raise _error(403, "Evidence is outside the current case")
            return w.indexed_page(
                "evidence", "candidate_id=?", [candidate_id], cursor=cursor, limit=limit
            )

        return read(scope_id, identity, result)
