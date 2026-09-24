"""Stage-scoped execution locks and resumable preparation of immutable inspection artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .artifacts import Artifact, atomic_json, publish_bundle, relative_path
from .contracts import (
    DomainError,
    EntityRef,
    VisibilityPolicy,
    WireModel,
    canonical_hash,
    file_hash,
)

STAGE_DEPENDENCIES = {
    "acquire-verify": ("ontologies",),
    "context-index": ("ontologies", "runtime"),
    "run-import": ("run", "context-index"),
    "profiles": ("context-index", "profile", "entities"),
    "comparisons": ("profiles", "pairs"),
    "study-export": ("study_definition",),
    "portable-export": ("context-index",),
}


class InputBinding(WireModel):
    """A verified input locator; its path is excluded from semantic stage identity."""

    path: str
    sha256: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    size: int = Field(gt=0)

    def verify(self, base: Path) -> Path:
        """Require the entire pinned nonempty file before any consuming stage executes."""
        path = Path(self.path).expanduser()
        path = path if path.is_absolute() else base / path
        if not path.is_file() or path.stat().st_size != self.size or file_hash(path) != self.sha256:
            raise DomainError(
                "unverified_input", "Input bytes do not match the execution binding", 409
            )
        return path


class OntologyBinding(WireModel):
    """Root and explicitly resolved import bindings, independent from future run/model inputs."""

    name: str
    root: InputBinding
    prepared_context: InputBinding | None = None
    imports: dict[str, InputBinding] = Field(default_factory=dict)
    scope: Literal["root", "closure"] = "root"
    source_derivation: dict[str, Any] | None = None
    license_treatment: str = "local research; redistribution must follow source license"


class ExecutionLock(WireModel):
    """Immutable stage input blueprint; unresolved future fields block only their consumers."""

    schema_version: Literal["exact-explain-execution/1"] = "exact-explain-execution/1"
    design_revision: str
    parent_lock: str | None = None
    audience: Literal["local", "development_demo"] = "local"
    ontologies: list[OntologyBinding] = Field(default_factory=list)
    runtime: dict[str, str] = Field(default_factory=dict)
    runtime_hashes: dict[str, str] = Field(default_factory=dict)
    policy: VisibilityPolicy = Field(default_factory=VisibilityPolicy)
    run: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
    entities: list[dict[str, Any]] = Field(default_factory=list)
    pairs: list[dict[str, Any]] = Field(default_factory=list)
    study_definition: dict[str, Any] | None = None
    profile_prompt: str | None = None
    comparison_prompt: str | None = None
    implementations: dict[str, str] = Field(
        default_factory=lambda: {
            "context-index": "context/2",
            "run-import": "decision-adapter/3",
            "profiles": "grounding-excerpts-and-comparisons/4",
            "comparisons": "grounding-excerpts-and-comparisons/4",
            "portable-export": "inspection-bundle/3",
            "study-export": "study-publication/1",
        }
    )

    @model_validator(mode="after")
    def validate_bindings(self) -> ExecutionLock:
        """Reject ambiguous input names and incomplete bindings before their stage executes."""
        names = [o.name for o in self.ontologies]
        if len(names) != len(set(names)):
            raise ValueError("Ontology binding names must be unique")
        if self.run is not None:
            allowed = {
                "binding",
                "files",
                "source_ontology",
                "target_ontology",
                "run_id",
                "source_root_sha256",
                "target_root_sha256",
            }
            if set(self.run) - allowed or not (allowed - {"run_id"}).issubset(self.run):
                raise ValueError(
                    "Run binding requires complete input hashes and artifact inventory"
                )
            InputBinding.model_validate(self.run["binding"])
            if not self.run["files"]:
                raise ValueError("Run binding must pin every consumed file")
            for entry in self.run["files"]:
                InputBinding.model_validate(entry)
            roots = {o.name: o.root.sha256 for o in self.ontologies}
            for side in ("source", "target"):
                name = self.run[side + "_ontology"]
                if roots.get(name) != self.run[side + "_root_sha256"]:
                    raise ValueError("Run and context ontology input hashes disagree")
        return self


def _without_locators(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_locators(v) for k, v in value.items() if k not in {"path", "run_dir"}}
    if isinstance(value, list):
        return [_without_locators(v) for v in value]
    return value


def stage_identity(lock: ExecutionLock, stage: str) -> str:
    """Invalidate only genuine semantic dependencies, allowing safe relocation and text repair."""
    dependencies = {}
    for name in STAGE_DEPENDENCIES[stage]:
        dependencies[name] = (
            stage_identity(lock, name)
            if name in STAGE_DEPENDENCIES
            else _without_locators(lock.model_dump().get(name))
        )
    if stage == "context-index":
        dependencies["runtime_hashes"] = lock.runtime_hashes
    if stage in {"profiles", "comparisons"}:
        from .generation import COMPARISON_PROMPT, PROFILE_PROMPT, ExplanationOutput

        prompt = (
            (lock.profile_prompt or PROFILE_PROMPT)
            if stage == "profiles"
            else (lock.comparison_prompt or COMPARISON_PROMPT)
        )
        dependencies["prompt"] = canonical_hash(prompt)
        dependencies["output_schema"] = canonical_hash(ExplanationOutput.model_json_schema())
    if stage == "portable-export":
        dependencies["audience"] = lock.audience
        for optional, binding in (
            ("run-import", lock.run),
            ("profiles", lock.profile),
            ("comparisons", lock.pairs),
        ):
            if binding:
                dependencies[optional] = stage_identity(lock, optional)
    if stage == "comparisons":
        dependencies["profile"] = lock.profile
    return canonical_hash(
        {
            "stage": stage,
            "schema": lock.schema_version,
            "implementation": lock.implementations.get(stage, "verify/1"),
            "dependencies": dependencies,
            "policy": (
                lock.policy.model_dump()
                if stage not in {"acquire-verify", "context-index", "run-import"}
                else None
            ),
        }
    )


def stage_readiness(lock: ExecutionLock) -> dict[str, Any]:
    """Report ready and unresolved stages without falsely blocking independent context work."""
    result: dict[str, Any] = {}
    values = lock.model_dump()
    for stage, dependencies in STAGE_DEPENDENCIES.items():
        missing = [
            d
            for d in dependencies
            if (result[d]["status"] != "ready" if d in result else not values.get(d))
        ]
        result[stage] = {
            "status": "blocked_input" if missing else "ready",
            "missing_bindings": missing,
            "identity": stage_identity(lock, stage),
        }
    return result


def distribution_hash(package: str) -> str:
    """Pin the installed distribution RECORD, which inventories every wheel member."""
    distribution = importlib.metadata.distribution(package)
    records = [
        entry for entry in distribution.files or [] if str(entry).endswith(".dist-info/RECORD")
    ]
    if len(records) != 1:
        raise ValueError("Installed distribution lacks a unique RECORD: " + package)
    return file_hash(Path(str(distribution.locate_file(records[0]))))


def bind_execution_lock(template: dict[str, Any], *, input_root: Path) -> ExecutionLock:
    """Hash readable local inputs and installed runtimes without parsing or model calls.

    The JSON template has ExecutionLock fields. File bindings may contain only
    path; provided digests and sizes are checked, never silently replaced.
    Resolved absolute locators permit the resulting lock to be written elsewhere.
    """
    import copy

    value = copy.deepcopy(template)

    def bind(entry: dict[str, Any]) -> dict[str, Any]:
        path = Path(entry["path"]).expanduser()
        path = (path if path.is_absolute() else input_root / path).resolve()
        measured = {"path": str(path), "sha256": file_hash(path), "size": path.stat().st_size}
        if any(key in entry and entry[key] != measured[key] for key in ("sha256", "size")):
            raise DomainError("unverified_input", "Input differs from template pin", 409)
        return InputBinding.model_validate(measured).model_dump()

    for ontology in value.get("ontologies", []):
        ontology["root"] = bind(ontology["root"])
        if ontology.get("prepared_context"):
            ontology["prepared_context"] = bind(ontology["prepared_context"])
        ontology["imports"] = {
            iri: bind(entry) for iri, entry in ontology.get("imports", {}).items()
        }
    runtime = value.setdefault("runtime", {})
    hashes = value.setdefault("runtime_hashes", {})
    for package in set(runtime) | {"pyowl-core", "pyowl2vec-star-projector"}:
        installed, digest = importlib.metadata.version(package), distribution_hash(package)
        if (
            package in runtime
            and runtime[package] != installed
            or package in hashes
            and hashes[package] != digest
        ):
            raise DomainError(
                "runtime_mismatch", "Installed runtime differs from template pin", 409
            )
        runtime[package], hashes[package] = installed, digest
    run = value.get("run")
    if run:
        run["binding"] = bind(run["binding"])
        if "files" not in run:
            directory = Path(run["binding"]["path"]).parent
            run["files"] = [
                {"path": str(path)} for path in sorted(directory.rglob("*")) if path.is_file()
            ]
        run["files"] = [bind(entry) for entry in run["files"]]
        roots = {o["name"]: o["root"]["sha256"] for o in value["ontologies"]}
        for side in ("source", "target"):
            run.setdefault(side + "_root_sha256", roots[run[side + "_ontology"]])
    return ExecutionLock.model_validate(value)


def read_lock(path: Path) -> ExecutionLock:
    """Load strict runtime input bindings; design-only blueprints remain non-executable."""
    return ExecutionLock.model_validate_json(path.read_bytes())


def reuse_plan(old: ExecutionLock, new: ExecutionLock, *, cause: str) -> dict[str, Any]:
    """Describe selective rebuilds before repair; repository revision alone never invalidates data."""
    rows = [
        {
            "stage": stage,
            "old_identity": stage_identity(old, stage),
            "new_identity": stage_identity(new, stage),
            "action": (
                "reuse" if stage_identity(old, stage) == stage_identity(new, stage) else "rebuild"
            ),
        }
        for stage in STAGE_DEPENDENCIES
    ]
    return {
        "schema_version": "exact-explain-reuse/1",
        "cause": cause,
        "stages": rows,
        "comparability": "Changed design or participant exposure requires separate validity review; unchanged matcher bytes remain reusable.",
        "expected_cost": "Only rebuilt generation records may dispatch paid requests; provider ledger retains actual usage.",
    }


class Preparation:
    """Single-writer stage publication with verified inventory reuse and STOP checkpoints."""

    def __init__(
        self,
        lock: ExecutionLock,
        output_root: Path,
        *,
        input_root: Path,
        resume_from: Path | None = None,
    ):
        self.lock, self.root, self.input_root = lock, Path(output_root), Path(input_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.resume_from = Path(resume_from) if resume_from else None
        self.outputs: dict[str, Path] = {}

    def _inventory(self, path: Path) -> list[dict[str, Any]]:
        return [
            Artifact(
                path=str(p.relative_to(path)), sha256=file_hash(p), size=p.stat().st_size
            ).model_dump()
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.name != "stage.json"
        ]

    def _verified(self, path: Path, identity: str) -> bool:
        receipt = path / "stage.json"
        if not receipt.exists():
            return False
        manifest = json.loads(receipt.read_bytes())
        if manifest.get("identity") != identity or manifest.get("state") != "completed":
            return False
        for artifact in manifest["outputs"]:
            file = relative_path(path, artifact["path"])
            if (
                not file.is_file()
                or file.stat().st_size != artifact["size"]
                or file_hash(file) != artifact["sha256"]
            ):
                raise DomainError(
                    "corrupt_stage", "Published preparation artifact failed verification", 409
                )
        return True

    def run(self, stages: list[str] | None = None) -> dict[str, Any]:
        """Run explicitly selected ready stages; never launch matching as an implicit dependency."""
        import fcntl

        selected = stages or [
            s for s, r in stage_readiness(self.lock).items() if r["status"] == "ready"
        ]
        if any(s not in STAGE_DEPENDENCIES for s in selected):
            raise ValueError("Unknown preparation stage")
        with (self.root / ".writer").open("a") as owner:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DomainError(
                    "preparation_busy", "Preparation has another active writer", 409, True
                ) from exc
            if (
                self.resume_from
                and (self.resume_from / "generations").is_dir()
                and not (self.root / "generations").exists()
            ):
                with (self.resume_from / ".writer").open("a") as prior_owner:
                    try:
                        fcntl.flock(prior_owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise DomainError(
                            "source_busy",
                            "Stop the prior preparation before copying its request ledger",
                            409,
                        ) from exc
                    shutil.copytree(self.resume_from / "generations", self.root / "generations")
            atomic_json(self.root / "locks" / (canonical_hash(self.lock)[7:] + ".json"), self.lock)
            for stage in STAGE_DEPENDENCIES:
                identity = stage_identity(self.lock, stage)
                destination = self.root / "stages" / identity[7:]
                if self._verified(destination, identity):
                    self.outputs[stage] = destination
                    continue
                if stage not in selected:
                    continue
                if (self.root / "STOP").exists():
                    break
                readiness = stage_readiness(self.lock)[stage]
                if readiness["status"] != "ready":
                    raise DomainError(
                        "blocked_input", "Stage has unresolved required bindings: " + stage, 503
                    )
                old = self.resume_from / "stages" / identity[7:] if self.resume_from else None
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging = self.root / ".work" / identity[7:]
                staging.parent.mkdir(exist_ok=True)
                if old and self._verified(old, identity):
                    if staging.exists():
                        shutil.rmtree(staging)
                    shutil.copytree(old, staging)
                    os.replace(staging, destination)
                    self.outputs[stage] = destination
                    continue
                staging.mkdir(exist_ok=True)
                start = time.monotonic()
                self._execute(stage, staging)
                atomic_json(
                    staging / "stage.json",
                    {
                        "schema_version": "exact-explain-stage/1",
                        "stage": stage,
                        "identity": identity,
                        "state": "completed",
                        "outputs": self._inventory(staging),
                        "wall_seconds": time.monotonic() - start,
                        "verification": "content_hashes",
                        "implementation": self.lock.implementations.get(stage),
                        "design_revision": self.lock.design_revision,
                    },
                )
                os.replace(staging, destination)
                self.outputs[stage] = destination
            report = {
                "lock_hash": canonical_hash(self.lock),
                "outputs": {s: str(p.relative_to(self.root)) for s, p in self.outputs.items()},
                "stopped": (self.root / "STOP").exists(),
                "readiness": stage_readiness(self.lock),
            }
            atomic_json(self.root / "preparation.json", report)
            return report

    def _contexts(self) -> dict[str, Any]:
        from .context import OntologyContext

        base = self.outputs.get("context-index")
        if base is None:
            raise DomainError(
                "dependency_unavailable", "Prepare context-index before its consumers", 503
            )
        return {
            name: OntologyContext(base / locator)
            for name, locator in json.loads((base / "contexts.json").read_bytes()).items()
        }

    def _execute(self, stage: str, destination: Path) -> None:
        if stage == "acquire-verify":
            verified = {}
            for ontology in self.lock.ontologies:
                verified[ontology.name] = str(ontology.root.verify(self.input_root))
                for binding in ontology.imports.values():
                    binding.verify(self.input_root)
            atomic_json(destination / "verified.json", {"verified": verified})
        elif stage == "context-index":
            from .context import prepare_context

            for package, expected in self.lock.runtime.items():
                if importlib.metadata.version(package) != expected:
                    raise DomainError(
                        "runtime_mismatch", "Installed ontology runtime differs from lock", 409
                    )
            for package, expected in self.lock.runtime_hashes.items():
                if distribution_hash(package) != expected:
                    raise DomainError(
                        "runtime_mismatch", "Installed distribution bytes differ from lock", 409
                    )
            context_locators: dict[str, str] = {}
            for index, ontology in enumerate(self.lock.ontologies):
                root = ontology.root.verify(self.input_root)
                imports = {
                    iri: str(binding.verify(self.input_root))
                    for iri, binding in ontology.imports.items()
                }
                locator = f"ontology-{index}"
                if ontology.prepared_context:
                    from .context import OntologyContext

                    manifest_path = ontology.prepared_context.verify(self.input_root)
                    prepared = OntologyContext(manifest_path.parent)
                    if prepared.manifest.get("policy_filter"):
                        raise DomainError(
                            "incompatible_context",
                            "Prepared ontology context must retain the complete raw scope",
                            409,
                        )
                    identity = prepared.manifest["identity"]
                    expected_imports = {
                        iri: binding.sha256 for iri, binding in ontology.imports.items()
                    }
                    actual_imports = {
                        edge["import_iri"]: edge["source_sha256"]
                        for edge in identity["imports"]
                        if edge["status"] == "resolved"
                    }
                    if (
                        identity["root_sha256"] != ontology.root.sha256
                        or identity["scope"] != ontology.scope
                        or identity["parser"]["version"] != self.lock.runtime.get("pyowl-core")
                        or (ontology.scope == "closure" and actual_imports != expected_imports)
                        or prepared.manifest.get("source_derivation") != ontology.source_derivation
                    ):
                        raise DomainError(
                            "incompatible_context",
                            "Prepared context differs from execution inputs",
                            409,
                        )
                    target = destination / locator
                    target.mkdir(exist_ok=True)
                    for name in ("manifest.json", "context.sqlite"):
                        shutil.copyfile(manifest_path.parent / name, target / name)
                    context_locators[ontology.name] = locator
                    continue
                # Context builders publish atomically and retain incomplete checkpoints.
                prepare_context(
                    root,
                    destination / locator,
                    imports=imports,
                    scope=ontology.scope,
                    expected_hash=ontology.root.sha256,
                    source_derivation=ontology.source_derivation,
                )
                context_locators[ontology.name] = locator
            atomic_json(destination / "contexts.json", context_locators)
        elif stage == "run-import":
            from .decisions import import_run

            run = dict(self.lock.run or {})
            binding = InputBinding.model_validate(run["binding"])
            input_path = binding.verify(self.input_root)
            expected_artifacts = {str(input_path.resolve()): binding.sha256}
            for record in run["files"]:
                file_binding = InputBinding.model_validate(record)
                path = file_binding.verify(self.input_root)
                expected_artifacts[str(path.resolve())] = file_binding.sha256
            contexts = self._contexts()
            source = contexts[run["source_ontology"]]
            target = contexts[run["target_ontology"]]
            context_by_id = {c.ontology_version_id: c for c in contexts.values()}
            imported = destination / "run"
            if imported.exists():
                shutil.rmtree(imported)
            import_run(
                input_path.parent,
                imported,
                source_ontology_version_id=source.ontology_version_id,
                target_ontology_version_id=target.ontology_version_id,
                run_id=run.get("run_id"),
                source_root_sha256=run["source_root_sha256"],
                target_root_sha256=run["target_root_sha256"],
                expected_artifacts=expected_artifacts,
                evidence_resolver=lambda item, entity: context_by_id[
                    entity["ontology_version_id"]
                ].resolve_feature(item, entity),
            )
        elif stage in {"profiles", "comparisons"}:
            from .generation import (
                ExplanationJobs,
                GenerationProfile,
                comparison_packet,
                entity_packet,
            )

            contexts = self._contexts()
            profile = GenerationProfile.model_validate(self.lock.profile)
            jobs = ExplanationJobs(self.root / "generations")
            profiles_path = self.outputs.get("profiles", destination)
            profile_rows = (
                json.loads((profiles_path / "profiles.json").read_bytes())
                if (profiles_path / "profiles.json").exists()
                else {}
            )
            packets = {}
            for entry in self.lock.entities:
                context = contexts[entry["ontology"]]
                entity = EntityRef(
                    ontology_version_id=context.ontology_version_id,
                    iri=entry["iri"],
                    kind=entry.get("kind", "class"),
                )
                facts: list[dict[str, Any]] = []
                for category in ("labels", "definitions", "synonyms", "hierarchy", "restrictions"):
                    page = context.facts(
                        entity, category=category, limit=20, policy=self.lock.policy
                    )
                    facts.extend({**f, "category": category} for f in page["items"])
                packet = entity_packet(
                    entity,
                    facts,
                    context_hash=context.ontology_version_id,
                    policy=self.lock.policy,
                    missingness=(
                        []
                        if any(f["category"] == "definitions" for f in facts)
                        else ["No definition is available in the selected context scope."]
                    ),
                )
                key = entry.get("id") or canonical_hash(entity)
                packets[key] = packet
                if stage == "profiles":
                    profile_rows[key] = jobs.generate(
                        packet,
                        profile,
                        prompt=self.lock.profile_prompt,
                        stopped=lambda: (self.root / "STOP").exists(),
                    )
                    atomic_json(destination / "profiles.json", profile_rows)
            if stage == "comparisons":
                comparisons = {}
                for pair in self.lock.pairs:
                    source, target = pair["source"], pair["target"]
                    packet = comparison_packet(
                        packets[source],
                        packets[target],
                        [profile_rows[source], profile_rows[target]],
                    )
                    result = jobs.generate(
                        packet,
                        profile,
                        prompt=self.lock.comparison_prompt,
                        stopped=lambda: (self.root / "STOP").exists(),
                    )
                    comparisons[result["explanation_id"]] = result
                    atomic_json(destination / "comparisons.json", comparisons)
        elif stage == "portable-export":
            self._export(destination)
        elif stage == "study-export":
            from .study.models import StudyDefinition

            definition = StudyDefinition.model_validate(self.lock.study_definition)
            from .context_resources import validate_ontology_resource
            from .study.resources import validate_explanation_resource

            for asset in definition.assets:
                source = relative_path(self.input_root, asset.path)
                if (
                    source.stat().st_size != asset.size_bytes
                    or file_hash(source).removeprefix("sha256:") != asset.sha256
                ):
                    raise DomainError(
                        "unverified_input", "Study asset differs from its frozen binding", 409
                    )
                target = relative_path(destination, asset.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                if asset.kind == "ontology":
                    assert asset.admission_receipt_path is not None
                    receipt = relative_path(self.input_root, asset.admission_receipt_path)
                    if file_hash(receipt).removeprefix("sha256:") != asset.admission_receipt_sha256:
                        raise DomainError(
                            "unverified_input",
                            "Ontology admission receipt differs from binding",
                            409,
                        )
                    validate_ontology_resource(
                        source, receipt, policy_hash=definition.visibility_policy.policy_hash
                    )
                    receipt_target = relative_path(destination, asset.admission_receipt_path)
                    receipt_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(receipt, receipt_target)
                else:
                    validate_explanation_resource(source.read_bytes(), definition.model_dump())
                shutil.copyfile(source, target)
            atomic_json(destination / "study-definition.json", definition)

    def _export(self, destination: Path) -> None:
        from .context_export import export_policy_context

        contexts = self._contexts()
        ontologies, runs, explanations = {}, {}, {}
        filtered_contexts = {}
        for number, context in enumerate(contexts.values()):
            if not self.lock.policy.allows_ontology(context.ontology_version_id):
                continue
            locator = f"ontologies/{number}"
            filtered = export_policy_context(context, destination / locator, self.lock.policy)
            filtered_contexts[context.ontology_version_id] = filtered
            ontologies[context.ontology_version_id] = locator
        if "run-import" in self.outputs:
            from .decisions import DecisionStore
            from .decisions_export import export_policy_run

            source = DecisionStore(self.outputs["run-import"] / "run")
            manifest = source.manifest()
            if all(
                manifest[side + "_ontology_version_id"] in filtered_contexts
                for side in ("source", "target")
            ):
                exported = export_policy_run(
                    source,
                    destination / "runs" / "run",
                    contexts=filtered_contexts,
                    policy=self.lock.policy,
                )
                runs[exported.manifest()["run_id"]] = "runs/run"
        for stage, name in (("profiles", "profiles.json"), ("comparisons", "comparisons.json")):
            if stage not in self.outputs:
                continue
            for result in json.loads((self.outputs[stage] / name).read_bytes()).values():
                locator = "explanations/" + result["explanation_id"][7:] + ".json"
                atomic_json(destination / locator, result)
                explanations[result["explanation_id"]] = locator
        publish_bundle(
            destination,
            ontologies=ontologies,
            runs=runs,
            explanations=explanations,
            policy=self.lock.policy,
            audience=self.lock.audience,
            capabilities={
                "context": "available",
                "decisions": "available" if runs else "not_exported",
                "generated_explanations": "available" if explanations else "not_requested",
            },
            license_treatment={o.name: o.license_treatment for o in self.lock.ontologies},
        )
