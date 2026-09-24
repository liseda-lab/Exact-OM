"""Versioned read APIs over prepared packages; no request-path models, parser or provider."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse

from .artifacts import (
    BoundedCache,
    BundleLibrary,
    read_metadata,
    relative_path,
    validate_bundle,
)
from .contracts import (
    CONTRACT_VERSION,
    DomainError,
    EntityKind,
    EntityRef,
    ErrorEnvelope,
    Page,
    Scope,
    WireModel,
    canonical_hash,
    canonical_json,
    decode_cursor,
    encode_cursor,
)
from .models import (
    AxiomResponse,
    Candidate,
    EntityContextResponse,
    Fact,
    GeneratedExplanationResponse,
    HierarchyPage,
    PairResponse,
    SelectedEvidence,
)


class Health(WireModel):
    """Public service readiness without filesystem or credential metadata."""

    status: Literal["available", "not_requested"]
    contract_version: str = CONTRACT_VERSION
    profile: Literal["local_app", "public_demo"]
    package_id: str | None
    capabilities: dict[str, str]


class PreparedService:
    """Lazy immutable indexes and bounded JSON caches isolated by package and policy identity."""

    def __init__(self, package: Path):
        self.path = Path(package)
        self.manifest = validate_bundle(self.path, verify_hashes=False)
        self.cache = BoundedCache()
        self._contexts: dict[str, Any] = {}
        self._runs: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._verified_files: dict[str, tuple[int, ...]] = {}
        # Verify immutable bytes once at startup; opening the first index is then cheap.
        # Actual SQLite connections and resource JSON stay lazy.
        for artifact in self.manifest.artifacts:
            self._verify_artifact(artifact)

    def _verify_artifact(self, artifact) -> None:
        from .contracts import file_hash

        path = relative_path(self.path.parent, artifact.path)
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._verified_files.get(artifact.path) == signature:
            return
        if stat.st_size != artifact.size or file_hash(path) != artifact.sha256:
            raise DomainError("corrupt_artifact", "Package artifact failed verification", 409)
        after = path.stat()
        if signature != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise DomainError(
                "artifact_changed", "Package artifact changed during verification", 409
            )
        self._verified_files[artifact.path] = signature

    @property
    def policy(self):
        """Return the server-owned frozen package policy."""
        return self.manifest.policy

    def _verify_resource_files(self, locator: str) -> dict[str, str]:
        hashes = {}
        for artifact in self.manifest.artifacts:
            if artifact.path == locator or artifact.path.startswith(locator.rstrip("/") + "/"):
                self._verify_artifact(artifact)
                hashes[Path(artifact.path).name] = artifact.sha256
        return hashes

    def ontology_metadata(self, ontology_id: str) -> dict[str, Any]:
        """Read bounded public manifest metadata without constructing a database reader."""
        if ontology_id not in self.manifest.ontologies or not self.policy.allows_ontology(
            ontology_id
        ):
            raise DomainError("not_found", "Resource unavailable", 404)
        locator = self.manifest.ontologies[ontology_id]
        self._verify_resource_files(locator)
        key = canonical_hash([self.manifest.package_id, "ontology_metadata", ontology_id])
        cached = self.cache.get(key)
        if cached is not None:
            return dict(cached)
        metadata = read_metadata(relative_path(self.path.parent, locator + "/manifest.json"))
        if metadata.get("ontology_version_id") != ontology_id:
            raise DomainError("invalid_manifest", "Ontology metadata identity does not match", 409)
        if any(
            not isinstance(metadata.get(name, {}), dict)
            for name in ("identity", "capabilities", "completeness")
        ) or not isinstance(metadata.get("source_derivation") or {}, dict):
            raise DomainError("invalid_manifest", "Ontology metadata has an invalid shape", 409)
        derivation = metadata.get("source_derivation")
        if derivation:
            # A legacy raw receipt is hash-bound only; paths/notes never become API fields.
            derivation = (
                {
                    name: derivation.get(name)
                    for name in ("status", "receipt_hash", "original_source_sha256")
                }
                if derivation.get("status") == "declared_derivative"
                else {
                    "status": "declared_derivative",
                    "receipt_hash": canonical_hash(derivation),
                    "original_source_sha256": None,
                }
            )
        result = {
            "ontology_version_id": ontology_id,
            "name": metadata.get("name"),
            "scope": metadata.get("scope"),
            "capabilities": {
                name: value
                for name, value in metadata.get("capabilities", {}).items()
                if name
                in {
                    "provider",
                    "kinds",
                    "categories",
                    "source_spans",
                    "typed_expressions",
                    "reasoner_inferred",
                }
            },
            "completeness": {
                name: value
                for name, value in metadata.get("completeness", {}).items()
                if name
                in {
                    "scope",
                    "imports_complete",
                    "extraction",
                    "entity_count",
                    "axiom_count",
                    "category_counts",
                    "domain_completeness",
                }
            },
            "source_root_sha256": metadata.get("identity", {}).get("root_sha256"),
            "source_derivation": derivation or None,
        }
        self.cache.put(key, result)
        return result

    def context(self, ontology_id: str):
        """Open only the requested verified context index; no ontology parser is imported."""
        if ontology_id not in self.manifest.ontologies or not self.policy.allows_ontology(
            ontology_id
        ):
            raise DomainError("not_found", "Resource unavailable", 404)
        with self._lock:
            if ontology_id not in self._contexts:
                from .context import OntologyContext

                hashes = self._verify_resource_files(self.manifest.ontologies[ontology_id])
                self._contexts[ontology_id] = OntologyContext(
                    relative_path(self.path.parent, self.manifest.ontologies[ontology_id]),
                    verified_database_hash=hashes.get("context.sqlite"),
                )
            return self._contexts[ontology_id]

    def run(self, run_id: str):
        """Open only one saved decision index, never an original training environment."""
        if run_id not in self.manifest.runs:
            raise DomainError("not_found", "Resource unavailable", 404)
        with self._lock:
            if run_id not in self._runs:
                from .decisions import DecisionStore

                self._verify_resource_files(self.manifest.runs[run_id])
                store = DecisionStore(relative_path(self.path.parent, self.manifest.runs[run_id]))
                if any(
                    not self.policy.allows_ontology(store.manifest()[side + "_ontology_version_id"])
                    for side in ("source", "target")
                ):
                    raise DomainError("not_found", "Resource unavailable", 404)
                self._runs[run_id] = store
            return self._runs[run_id]

    def resource(self, resource_id: str, family: str) -> dict[str, Any]:
        """Return checksum-verified prepared JSON with mandatory matching policy identity."""
        resources = getattr(self.manifest, family)
        locator = resources.get(resource_id)
        if locator is None:
            raise DomainError("not_found", "Resource unavailable", 404)
        key = canonical_hash(
            [self.manifest.package_id, self.policy.policy_hash, family, resource_id]
        )
        result = self.cache.get(key)
        if result is not None:
            return dict(result)
        path = relative_path(self.path.parent, locator)
        artifact = next(a for a in self.manifest.artifacts if a.path == locator)
        if artifact.size > 2 * 1024**2:
            raise DomainError(
                "response_too_large", "Prepared JSON exceeds the response budget", 413
            )
        self._verify_artifact(artifact)
        result = json.loads(path.read_bytes())
        if (
            family == "explanations"
            and result.get("manifest", {}).get("visibility_policy_hash") != self.policy.policy_hash
        ):
            raise DomainError("not_found", "Resource unavailable", 404)
        if family == "explanations" and result.get("grounding_status") != "validated":
            raise DomainError(
                "explanation_unverified", "Explanation is awaiting grounding review", 503
            )
        self.cache.put(key, result)
        return dict(result)


def create_prepared_app(
    package: Path | None,
    *,
    profile: Literal["local_app", "public_demo"] = "local_app",
    library_dir: Path | None = None,
) -> FastAPI:
    """Serve an immutable package with explicit deployment boundaries and inert local imports."""
    if profile not in {"local_app", "public_demo"}:
        raise ValueError("Study uses the isolated study application")
    if profile == "public_demo" and package is None:
        raise ValueError("Public demo requires a fixed prepared package")
    app = FastAPI(title="Exact Explain", version=CONTRACT_VERSION)
    library = BundleLibrary(library_dir) if profile == "local_app" and library_dir else None
    if package is None and library and (library.root / "selection.json").exists():
        selected = json.loads((library.root / "selection.json").read_bytes())["package_id"]
        package = library.select(selected)
    app.state.service = PreparedService(package) if package else None
    if profile == "public_demo" and app.state.service.manifest.audience != "development_demo":
        raise ValueError(
            "Public demo requires a bundle explicitly approved for development demonstration"
        )

    def service() -> PreparedService:
        if app.state.service is None:
            raise DomainError("package_unavailable", "Import or select a prepared package", 503)
        return cast(PreparedService, app.state.service)

    @app.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError):
        return JSONResponse(status_code=exc.status_code, content=exc.envelope.model_dump())

    @app.exception_handler(KeyError)
    async def unknown_resource(request: Request, exc: KeyError):
        return JSONResponse(
            status_code=404,
            content=ErrorEnvelope(code="not_found", message="Resource unavailable").model_dump(),
        )

    @app.exception_handler(ValueError)
    @app.exception_handler(RequestValidationError)
    async def invalid_query(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=ErrorEnvelope(
                code="invalid_query", message="Invalid request parameters"
            ).model_dump(),
        )

    @app.exception_handler(ResponseValidationError)
    async def invalid_prepared_resource(request: Request, exc: ResponseValidationError):
        return JSONResponse(
            status_code=409,
            content=ErrorEnvelope(
                code="invalid_prepared_resource",
                message="Prepared resource does not satisfy its declared contract",
            ).model_dump(),
        )

    @app.exception_handler(sqlite3.DatabaseError)
    @app.exception_handler(OSError)
    async def unavailable(request: Request, exc: Exception):
        return JSONResponse(
            status_code=503,
            content=ErrorEnvelope(
                code="artifact_unavailable",
                message="Prepared artifact is unavailable",
                retryable=True,
            ).model_dump(),
        )

    @app.middleware("http")
    async def boundaries(request: Request, call_next):
        # Local mutations must not be invocable through cross-origin browser requests.
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("origin"):
            origin = request.headers["origin"].rstrip("/")
            if origin != str(request.base_url).rstrip("/"):
                return JSONResponse(
                    status_code=403,
                    content={
                        "code": "origin_denied",
                        "message": "Request origin is not permitted",
                        "retryable": False,
                    },
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def bounded(value: Any) -> Any:
        if len(canonical_json(value)) > 2 * 1024**2:
            raise DomainError(
                "response_too_large", "Use a smaller page or explicit axiom detail", 413
            )
        return value

    @app.get("/api/v1/health", response_model=Health)
    def health():
        active = app.state.service
        return Health(
            status="available" if active else "not_requested",
            profile=profile,
            package_id=active.manifest.package_id if active else None,
            capabilities=active.manifest.capabilities if active else {},
        )

    @app.get("/api/v1/ontologies", response_model=Page[dict[str, Any]])
    def ontologies(limit: int = Query(20, ge=1, le=100), cursor: str | None = None):
        active = service()
        scope = Scope(
            ontology_version_id="package",
            context_revision=active.manifest.package_id,
            basis="resolved_closure",
            visibility_policy_hash=active.policy.policy_hash,
            filter_id="ontologies:id",
        )
        after = decode_cursor(cursor, scope)
        if after is None:
            after = ""
        if not isinstance(after, str):
            raise DomainError("invalid_cursor", "Invalid collection cursor")
        allowed = sorted(k for k in active.manifest.ontologies if active.policy.allows_ontology(k))
        remaining = [k for k in allowed if k > after]
        items = [active.ontology_metadata(k) for k in remaining[:limit]]
        more = len(remaining) > limit
        return bounded(
            Page(
                items=items,
                returned_count=len(items),
                total_count=len(allowed),
                next_cursor=encode_cursor(scope, remaining[limit - 1]) if more else None,
                truncated=more,
                scope=scope,
                status="available" if allowed else "absent_in_scope",
            )
        )

    @app.get("/api/v1/entities", response_model=Page[dict[str, Any]])
    def entities(
        ontology_version_id: str,
        kind: EntityKind | None = None,
        term: str = Query("", max_length=512),
        language: str | None = None,
        limit: int = Query(20, ge=1, le=100),
        cursor: str | None = None,
    ):
        active = service()
        return bounded(
            active.context(ontology_version_id).search(
                kind=kind,
                term=term,
                language=language,
                limit=limit,
                cursor=cursor,
                policy=active.policy,
            )
        )

    @app.get("/api/v1/entity-context", response_model=EntityContextResponse)
    def entity_context(
        ontology_version_id: str, iri: str, kind: EntityKind = "class", language: str | None = None
    ):
        active = service()
        return bounded(
            active.context(ontology_version_id).entity_context(
                EntityRef(ontology_version_id=ontology_version_id, iri=iri, kind=kind),
                language=language,
                policy=active.policy,
            )
        )

    @app.get("/api/v1/entity-facts", response_model=Page[Fact])
    def entity_facts(
        ontology_version_id: str,
        iri: str,
        kind: EntityKind = "class",
        category: str | None = None,
        basis: str = "asserted",
        limit: int = Query(20, ge=1, le=100),
        cursor: str | None = None,
    ):
        active = service()
        return bounded(
            active.context(ontology_version_id).facts(
                EntityRef(ontology_version_id=ontology_version_id, iri=iri, kind=kind),
                category=category,
                basis=basis,
                limit=limit,
                cursor=cursor,
                policy=active.policy,
            )
        )

    @app.get("/api/v1/hierarchy", response_model=HierarchyPage)
    def hierarchy(
        ontology_version_id: str,
        iri: str,
        kind: EntityKind = "class",
        direction: Literal["parents", "children", "ancestors"] = "parents",
        basis: Literal[
            "literal_asserted", "structural_navigation", "reasoner_inferred"
        ] = "literal_asserted",
        limit: int = Query(50, ge=1, le=1000),
        cursor: str | None = None,
    ):
        active = service()
        if direction != "ancestors" and limit > 200:
            raise DomainError("invalid_limit", "Hierarchy pages allow at most 200 edges")
        return bounded(
            active.context(ontology_version_id).hierarchy(
                EntityRef(ontology_version_id=ontology_version_id, iri=iri, kind=kind),
                direction=direction,
                basis=basis,
                limit=limit,
                cursor=cursor,
                policy=active.policy,
            )
        )

    @app.get("/api/v1/axioms/{axiom_id}/stream")
    def stream_axiom(axiom_id: str, ontology_version_id: str):
        from itertools import chain

        from fastapi.responses import StreamingResponse

        active = service()
        context = active.context(ontology_version_id)
        chunks = iter(context.axiom_chunks(axiom_id, policy=active.policy))
        # Resolve authorization/missing resources before HTTP headers are emitted.
        first = next(chunks)
        return StreamingResponse(chain((first,), chunks), media_type="application/json")

    @app.get("/api/v1/axioms/{axiom_id}", response_model=AxiomResponse)
    def axiom(axiom_id: str, ontology_version_id: str):
        active = service()
        return active.context(ontology_version_id).axiom(axiom_id, policy=active.policy)

    @app.get("/api/v1/runs/{run_id}/sources", response_model=Page[dict[str, Any]])
    def sources(run_id: str, limit: int = Query(20, ge=1, le=100), cursor: str | None = None):
        active = service()
        return bounded(
            active.run(run_id).sources(
                limit=limit, cursor=cursor, policy_hash=active.policy.policy_hash
            )
        )

    @app.get("/api/v1/runs/{run_id}/candidates", response_model=Page[Candidate])
    def candidates(
        run_id: str,
        source: str,
        source_kind: EntityKind = "class",
        limit: int = Query(20, ge=1, le=100),
        cursor: str | None = None,
    ):
        active = service()
        return bounded(
            active.run(run_id).candidates(
                source,
                source_kind=source_kind,
                limit=limit,
                cursor=cursor,
                policy_hash=active.policy.policy_hash,
            )
        )

    @app.get("/api/v1/runs/{run_id}/pair", response_model=PairResponse)
    def pair(run_id: str, pair_id: str):
        return bounded(service().run(run_id).pair(pair_id))

    @app.get("/api/v1/runs/{run_id}/pair-evidence", response_model=Page[SelectedEvidence])
    def evidence(
        run_id: str, pair_id: str, limit: int = Query(20, ge=1, le=100), cursor: str | None = None
    ):
        active = service()
        run = active.run(run_id)
        scope = Scope(
            ontology_version_id=run.manifest()["source_ontology_version_id"],
            context_revision=run.manifest()["revision"],
            basis="run_selected",
            visibility_policy_hash=active.policy.policy_hash,
            filter_id=canonical_hash([run_id, pair_id]),
        )
        after = decode_cursor(cursor, scope)
        if after is None:
            after = ""
        if not isinstance(after, str):
            raise DomainError("invalid_cursor", "Invalid collection cursor")
        allowed = []
        for row in run.evidence(pair_id):
            # Every original supporting fact must be visible before exposing raw selected text.
            fact_ids = row.get("fact_ids") or []
            if not fact_ids:
                continue
            try:
                context = active.context(row["entity"]["ontology_version_id"])
                for fact_id in fact_ids:
                    context.axiom(fact_id, policy=active.policy)
            except KeyError:
                continue
            except DomainError as exc:
                if exc.status_code == 404:
                    continue
                raise
            allowed.append(row)
        rows = sorted(allowed, key=lambda x: x["evidence_id"])
        eligible = [r for r in rows if r["evidence_id"] > after]
        more = len(eligible) > limit
        return bounded(
            Page(
                items=eligible[:limit],
                returned_count=min(limit, len(eligible)),
                total_count=len(rows),
                next_cursor=(
                    encode_cursor(scope, eligible[limit - 1]["evidence_id"]) if more else None
                ),
                truncated=more,
                scope=scope,
                status="available" if rows else "not_exported",
                reason=None if rows else "No selected evidence was exported",
            ).model_dump()
        )

    @app.get("/api/v1/explanations/{explanation_id}", response_model=GeneratedExplanationResponse)
    def explanation(explanation_id: str):
        return bounded(service().resource(explanation_id, "explanations"))

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str):
        return bounded(service().resource(job_id, "jobs"))

    if profile == "local_app":
        from .import_jobs import ImportJobs

        import_jobs = ImportJobs(library.root / ".jobs") if library else None

        def require_library() -> BundleLibrary:
            if library is None:
                raise DomainError("library_unavailable", "Configure a local bundle library", 503)
            return library

        @app.get("/api/v1/bundles")
        def bundles():
            return bounded(require_library().list())

        @app.post("/api/v1/bundles/import-jobs")
        def create_import():
            require_library()
            assert import_jobs is not None
            return import_jobs.create()

        @app.get("/api/v1/bundles/import-jobs/{job_id}")
        def import_progress(job_id: str):
            require_library()
            assert import_jobs is not None
            return import_jobs.get(job_id)

        @app.delete("/api/v1/bundles/import-jobs/{job_id}")
        def cancel_import(job_id: str):
            require_library()
            assert import_jobs is not None
            return import_jobs.cancel(job_id)

        @app.post("/api/v1/bundles/import")
        async def import_bundle(request: Request, job_id: str | None = None):
            store = require_library()
            assert import_jobs is not None
            if request.headers.get("content-type", "").split(";")[0] not in {
                "application/zip",
                "application/octet-stream",
            }:
                raise DomainError("invalid_archive", "Upload a ZIP as application/zip")
            from starlette.concurrency import run_in_threadpool

            job_id = job_id or import_jobs.create()["job_id"]
            import_jobs.claim(job_id)

            def cancelled():
                return import_jobs.get(job_id)["cancel_requested"]

            try:
                with tempfile.NamedTemporaryFile(dir=store.root, prefix=".upload-") as upload:
                    size = 0
                    async for block in request.stream():
                        if cancelled():
                            raise DomainError("import_cancelled", "Import was cancelled", 409)
                        size += len(block)
                        if size > store.max_bytes:
                            raise DomainError(
                                "payload_too_large", "Upload exceeds import limit", 413
                            )
                        upload.write(block)
                        import_jobs.update(job_id, received_bytes=size)
                    upload.flush()
                    import_jobs.update(job_id, status="validating")
                    manifest = await run_in_threadpool(
                        store.import_archive, Path(upload.name), cancelled=cancelled
                    )
                    return import_jobs.update(
                        job_id, package_id=manifest.package_id, status="available"
                    )
            except BaseException as exc:
                import_jobs.update(
                    job_id,
                    status="cancelled" if cancelled() else "failed",
                    reason="Upload interrupted or package validation failed",
                )
                raise exc

        @app.post("/api/v1/bundles/{package_id}/select")
        def select(package_id: str):
            path = require_library().select(package_id)
            active = PreparedService(path)
            app.state.service = (
                active  # Each in-flight request retains its previous immutable service.
            )
            return {"package_id": active.manifest.package_id, "status": "available"}

    return app
