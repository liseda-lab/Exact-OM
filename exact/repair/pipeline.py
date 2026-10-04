"""Opt-in neural proposal/value handoff to the exact finite-pool repair kernel.

Direct typed grammars generate replacements from observed finite symbol menus.
Enumeration remains a matched small-language control. Proposal, benefit and edit
cost stay separate, and all coefficients are frozen before exact selection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from time import monotonic
from typing import Any, Iterable

import pyowl_core as owl

from .graph import (
    EffectivePreparation,
    GraphExplanation,
    GraphNode,
    ObservableGraph,
    feature_vector,
    structural_id,
    validate_admitted_supports,
)
from .records import (
    FrozenMapping,
    GenerationReportV3,
    ObjectiveV2,
    ObjectiveV3,
    ProposalRecordV3,
    RepairInputV2,
    ReplacementCandidateV2,
    ReplacementCandidateV3,
    canonical_hash,
    canonical_json,
    make_objective,
    replace_inventory,
)


@dataclass(frozen=True)
class FrozenNeuralRound:
    """Reviewable immutable solver input and proposal/model coverage provenance."""

    problem: RepairInputV2
    objective: ObjectiveV2
    graph_hash: str
    model_hash: str
    proposal_reports: tuple[GenerationReportV3, ...]
    risk_scorer: Any = None


@dataclass(frozen=True)
class FrozenPlanRisk:
    """Hash-bound CPU snapshot and inert context for complete-plan check ordering."""

    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    graph: ObservableGraph
    objects: tuple[Any, ...]
    supports: tuple[Any, ...]
    model_hash: str
    graph_hash: str
    pool_hash: str

    def __post_init__(self) -> None:
        from pathlib import Path

        from .records import RevisionObjectV2

        if (
            not isinstance(self.checkpoint_path, str)
            or not Path(self.checkpoint_path).is_absolute()
        ):
            raise ValueError("risk model artifact requires an absolute path")
        if type(self.checkpoint_bytes) is not int or self.checkpoint_bytes < 1:
            raise ValueError("risk model artifact size must be a positive integer")
        for value in (self.checkpoint_sha256, self.model_hash, self.graph_hash, self.pool_hash):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("risk descriptor requires canonical SHA256 identities")
        if not isinstance(self.graph, ObservableGraph) or any(
            not isinstance(n, GraphNode) for n in self.graph.nodes
        ):
            raise ValueError("risk descriptor requires an observable graph")
        if not isinstance(self.objects, tuple) or any(
            not isinstance(obj, RevisionObjectV2) for obj in self.objects
        ):
            raise ValueError("risk descriptor requires frozen revision objects")
        if not isinstance(self.supports, tuple) or any(
            not isinstance(support, GraphExplanation) for support in self.supports
        ):
            raise ValueError("risk descriptor requires frozen graph explanations")
        validate_admitted_supports(self.graph)
        if self.supports != self.graph.admitted_supports:
            raise ValueError("risk supports do not match graph admission")
        if self.graph_hash != canonical_hash(self.graph) or self.pool_hash != canonical_hash(
            self.objects
        ):
            raise ValueError("risk descriptor graph or pool identity mismatch")

    @property
    def risk_identity(self) -> str:
        """Keep the existing ledger scheduling identity stable across JSON replay."""
        return canonical_hash(
            (
                "admitted-plan-risk/v3.1",
                self.model_hash,
                self.graph_hash,
                self.pool_hash,
                self.graph.admission_policy,
                self.supports,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize only explicit owned dataclasses and canonical shared OWL values."""
        import json

        schema = "exact-repair/frozen-plan-risk/v3"
        return {
            "schema": schema,
            "hash": canonical_hash((schema, self)),
            "record": json.loads(canonical_json(self)),
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any], *, problem: RepairInputV2, objective: ObjectiveV2
    ) -> FrozenPlanRisk:
        """Strict JSON replay with graph, pool, model and artifact integrity checks.

        Call inside a supervised preparation worker: snapshot verification and
        weights-only model reconstruction are charged to that worker's budget.
        No import path or pickle constructor is selected by the JSON document.
        """
        from .records import ReplacementCandidateV2, RevisionObjectV2, RevisionObjectV3

        schema = "exact-repair/frozen-plan-risk/v3"
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema", "hash", "record"}
            or payload["schema"] != schema
        ):
            raise ValueError("invalid frozen risk descriptor envelope")
        constructors = {
            t.__name__: t
            for t in (
                cls,
                ObservableGraph,
                GraphNode,
                GraphExplanation,
                RevisionObjectV2,
                RevisionObjectV3,
                ReplacementCandidateV2,
                ReplacementCandidateV3,
            )
        }

        def decode(value: Any) -> Any:
            if isinstance(value, list):
                return tuple(decode(v) for v in value)
            if isinstance(value, dict):
                if "$owl" in value:
                    if set(value) != {"$owl"} or not isinstance(value["$owl"], str):
                        raise ValueError("invalid canonical OWL value in risk descriptor")
                    return owl.decode_canonical(bytes.fromhex(value["$owl"]))
                if "$record" in value:
                    constructor = constructors.get(value["$record"])
                    if constructor is None or set(value) != {
                        "$record",
                        *(f.name for f in fields(constructor)),
                    }:
                        raise ValueError("unknown or malformed risk descriptor record")
                    return constructor(**{k: decode(v) for k, v in value.items() if k != "$record"})
                return {k: decode(v) for k, v in value.items()}
            if value is None or type(value) in (str, int, float, bool):
                return value
            raise ValueError("non-JSON value in frozen risk descriptor")

        result = decode(payload["record"])
        if not isinstance(result, cls) or canonical_hash((schema, result)) != payload["hash"]:
            raise ValueError("frozen risk descriptor content hash mismatch")
        if (
            result.pool_hash != canonical_hash(problem.objects)
            or result.graph_hash != problem.graph_identity
            or not isinstance(objective, ObjectiveV3)
            or result.pool_hash != objective.pool_hash
            or result.model_hash != objective.model_hash
        ):
            raise ValueError("frozen risk descriptor does not match input/objective bindings")
        # Always re-read the artifact on replay, even if this process previously
        # used a cached model. Missing or changed artifacts must never remove risk.
        _read_risk_model(
            result.checkpoint_path,
            result.checkpoint_sha256,
            result.checkpoint_bytes,
            result.model_hash,
        )
        return result

    def __call__(self, assignment: tuple[int, ...]) -> float:
        import torch

        model, memory = _load_risk_snapshot(
            self.checkpoint_path,
            self.checkpoint_sha256,
            self.checkpoint_bytes,
            self.graph,
            self.model_hash,
        )
        with torch.no_grad():
            return float(
                model.plan_risk_logit(self.objects, memory, assignment, supports=self.supports)
            )


def _read_risk_model(path: str, digest: str, size: int, expected_model_hash: str) -> Any:
    """Read only a size/checksum-bound weights-only snapshot, then verify its model."""
    from pathlib import Path

    import torch

    from .model import RepairModel

    torch.set_num_threads(1)
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise ValueError(f"risk model artifact is missing: {path}")
    if checkpoint.stat().st_size != size:
        raise ValueError("risk model artifact size mismatch")
    with checkpoint.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
            raise ValueError("risk model artifact integrity mismatch")
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    if payload.get("model_schema") != "exact-repair/model/v3":
        raise ValueError("risk model artifact requires model schema v3")
    model = RepairModel(payload["metadata"], **payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if model_digest(model) != expected_model_hash:
        raise ValueError("risk model artifact model identity mismatch")
    return model


@lru_cache(maxsize=2)
def _load_risk_snapshot(
    path: str, digest: str, size: int, graph: ObservableGraph, model_hash: str
) -> tuple[Any, Any]:
    import torch

    model = _read_risk_model(path, digest, size, model_hash)
    with torch.no_grad():
        return model, model.encode(graph)


def _publish_risk_snapshot(
    model: Any, cache_directory: str | None, max_bytes: int
) -> tuple[str, str, int]:
    """Publish a hash-bound immutable model artifact, never large worker-return weights."""
    import os
    import tempfile
    from pathlib import Path

    import torch

    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("risk snapshot admission cap must be a positive integer")
    directory = (
        Path(
            cache_directory
            or os.environ.get(
                "EXACT_REPAIR_CIRCUIT_CACHE",
                str(Path(tempfile.gettempdir()) / f"exact-repair-circuits-v3-{os.getuid()}"),
            )
        )
        / "risk"
    )
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".risk-", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(
                {
                    "metadata": model.metadata,
                    "config": model.config,
                    "model_schema": "exact-repair/model/v3",
                    "state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        size = temporary.stat().st_size
        if size > max_bytes:
            raise ValueError("risk model snapshot exceeds its declared artifact byte limit")
        with temporary.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        path = directory / (digest + ".pt")
        # Identical racing writers publish identical immutable content atomically.
        os.replace(temporary, path)
        temporary = None
        return str(path.resolve()), digest, size
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def proposal_distribution(
    model: Any,
    memory: Any,
    obj: Any,
    *,
    mixtures: int = 4,
    profile: tuple[tuple[str, float], ...] = (),
    max_depth: int = 2,
    max_constructors: int = 2,
    constraint_identity: str = "",
    encoding: Any = None,
    compile_seconds: float | None = None,
    max_circuit_nodes: int = 100000,
    uniform: bool = False,
    conditioned: bool = True,
    factored: bool = True,
    selected_context: Any = None,
    compiler_cache_directory: str | None = None,
    vtree_type: str = "balanced",
    circuit_limits: dict[str, Any] | None = None,
) -> Any:
    """Connect shared slot/menu embeddings and precomputed logits to exact SDD WMC."""
    import torch

    from .candidates import deduplicate_candidates, normalise_axioms
    from .circuit import ConditionedMixture, compile_encoding, encode_candidates

    if circuit_limits and (not conditioned or not factored):
        raise ValueError("declared family circuit limits require conditioned family compilation")
    obj = replace(obj, candidates=deduplicate_candidates(obj.candidates))
    encoding = encoding or encode_candidates(
        obj.candidates,
        max_depth=max_depth,
        max_constructors=max_constructors,
        constraint_identity=constraint_identity,
    )
    compiled = None
    if not conditioned:
        if not hasattr(encoding, "accepts"):
            raise ValueError("Unconstrained generation requires a direct grammar validator")
    elif factored and hasattr(encoding, "decode"):
        from .grammar import compile_families

        compiled = compile_families(
            encoding,
            seconds=compile_seconds or 20.0,
            max_nodes=max_circuit_nodes,
            cache_directory=compiler_cache_directory,
            vtree_type=vtree_type,
            circuit_limits=circuit_limits,
        )
    elif compile_seconds is not None:
        from .compilation import compile_bounded

        compiled = compile_bounded(
            encoding,
            seconds=compile_seconds,
            max_nodes=max_circuit_nodes,
            cache_directory=compiler_cache_directory,
            vtree_type=vtree_type,
        )
    elif hasattr(encoding, "decode"):
        from .grammar import compile_grammar

        compiled = compile_grammar(encoding, max_nodes=max_circuit_nodes)
    else:
        compiled = compile_encoding(encoding)
    if (
        compiled is not None
        and not hasattr(compiled, "families")
        and compiled.node_count > max_circuit_nodes
    ):
        raise ValueError("Compiled circuit exceeds the declared node budget")
    from .circuit import FactoredConditionedMixture

    distribution_type = (
        FactoredConditionedMixture if hasattr(compiled, "families") else ConditionedMixture
    )
    if uniform:
        if compiled is None:
            raise ValueError("Uniform constrained sampling requires a compiled circuit")
        return distribution_type(
            compiled, model.empty_bundle.new_zeros((1, encoding.variable_count))
        )
    bundles = {candidate.candidate_id: candidate for candidate in obj.candidates}
    retained = {
        owl.structural_hexdigest(axiom): axiom
        for axiom in normalise_axioms(
            (*obj.original_axioms, *(a for c in obj.candidates for a in c.axioms))
        )
    }
    context = model.object_context(obj.object_id, memory)
    choices = []
    for field, categories in encoding.fields:
        for category in categories:
            if field == "bundle":
                embedding = model.candidate_embedding(bundles[category], memory)
            elif field == "template" and category.startswith("fixed:"):
                template = next(t for t in encoding.templates if t.name == category)
                embedding = model.candidate_embedding(template.fixed, memory)
            elif field.startswith("retained:"):
                embedding = model.encode_structure(retained[field.split(":", 1)[1]], memory)
                embedding = embedding + model._role(f"retained:{category}")
            elif field.endswith(":class") and category != "unused":
                embedding = memory.rows[structural_id(owl.Class(owl.IRI(category)))]
            elif field.endswith(":property") and category != "unused":
                embedding = memory.rows[structural_id(owl.ObjectProperty(owl.IRI(category)))]
            else:
                embedding = model._role(f"choice:{category}")
            role = field.split(":", 1)[0] if field.startswith("retained:") else field
            choices.append(model.argument_head(torch.cat((model._role(role), embedding))))
    profile_vector = (
        context.new_tensor(
            feature_vector(GraphNode("profile", "cost_profile", profile), model.feature_dim)
        )
        if profile
        else None
    )
    weights, logits = model.proposal_logits(
        context,
        torch.stack(choices),
        mixtures=mixtures,
        profile_features=profile_vector,
        **({"selected_context": selected_context} if selected_context is not None else {}),
    )
    if compiled is None:
        from .proposals import UnconditionedMixture

        return UnconditionedMixture(encoding, logits, weights)
    return distribution_type(compiled, logits, weights)


def model_digest(model: Any) -> str:
    """Hash architecture identity and exact parameter bytes before an evaluated solve."""
    import torch

    digest = hashlib.sha256(canonical_hash((model.metadata, model.config)).encode())
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(canonical_hash((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def freeze_neural_round(
    problem: RepairInputV2,
    model: Any,
    *,
    graph: ObservableGraph | None = None,
    retrieved_symbols: Iterable[Any] = (),
    draws_per_object: int = 32,
    candidate_cap: int = 64,
    mixtures: int = 4,
    seed: int = 13,
    profile: tuple[tuple[str, float], ...] = (),
    interaction_pairs: Iterable[tuple[int, int]] | None = None,
    max_depth: int = 2,
    max_constructors: int = 2,
    max_circuit_nodes: int = 100000,
    compile_seconds: float = 20.0,
    proposal_arm: str = "grammar_mixture",
    max_graph_nodes: int = 4096,
    max_graph_edges: int = 32768,
    max_explanations: int = 64,
    max_text_tokens: int = 128,
    retrieval_config: Any = None,
    max_enumerated_expressions: int = 10000,
    pair_factor_limit_per_object: int = 16,
    pair_max_pairs: int | None = None,
    pair_max_factors: int | None = None,
    quantization_scale: int = 1000,
    final_candidate_removals: Any = None,
    selected_other_assignment: tuple[int | None, ...] | None = None,
    factored: bool = True,
    compiler_cache_directory: str | None = None,
    vtree_type: str = "balanced",
    contextual_filtering: bool = True,
    max_context_checks: int = 128,
    representative_max_checks: int = 100000,
    omitted_generation_symbols: tuple[str, ...] = (),
    enabled_actions: tuple[str, ...] | None = None,
    preserved_candidates: Any = None,
    circuit_limits: dict[str, Any] | None = None,
    risk_snapshot_max_bytes: int = 268435456,
) -> FrozenNeuralRound:
    """Generate a finite pool and freeze its value/cost objective.

    Each native compilation has an external deadline. Use
    ``bounded_freeze_neural_round`` to bound retrieval, graph/model work and the
    whole proposal stage as well. This local form is useful inside supervised
    training/campaign workers. No model state changes during inference.
    """
    import torch

    from .candidates import (
        budget_candidates,
        deduplicate_candidates,
        materialize_retrieved_endpoints,
    )
    from .retrieval import retrieve_vocabulary

    arms = {
        "bounded_enumeration",
        "grammar_uniform",
        "grammar_product",
        "grammar_mixture",
        "rejection",
    }
    if proposal_arm not in arms:
        raise ValueError(f"unknown proposal arm: {proposal_arm}")
    if draws_per_object < 0 or candidate_cap < 1 or max_circuit_nodes < 1:
        raise ValueError("Invalid finite neural round budget")
    started = monotonic()
    problem = replace(
        problem,
        objects=tuple(
            replace(obj, candidates=deduplicate_candidates(obj.candidates))
            for obj in problem.objects
        ),
    )
    retrieval = retrieve_vocabulary(
        problem, **({"config": retrieval_config} if retrieval_config is not None else {})
    )
    selected_ids = None
    if selected_other_assignment is not None:
        if len(selected_other_assignment) != len(problem.objects):
            raise ValueError("proposal context assignment width differs from the object inventory")
        if any(
            choice is not None
            and (type(choice) is not int or not 0 <= choice < len(obj.candidates))
            for obj, choice in zip(problem.objects, selected_other_assignment)
        ):
            raise ValueError("proposal context contains an invalid selected action")
        selected_ids = tuple(
            None if choice is None else obj.candidates[choice].candidate_id
            for obj, choice in zip(problem.objects, selected_other_assignment)
        )
    omitted = frozenset(omitted_generation_symbols)
    if omitted:
        menus_without = []
        for menu in retrieval.menus:
            values: dict[str, Any] = {
                name: tuple(
                    entity for entity in getattr(menu, name) if str(entity.iri.value) not in omitted
                )
                for name in (
                    "source_classes",
                    "target_classes",
                    "source_properties",
                    "target_properties",
                )
            }
            menus_without.append(
                replace(
                    menu,
                    **values,
                    endpoint_alternatives=tuple(
                        (side, entity)
                        for side, entity in menu.endpoint_alternatives
                        if str(entity.iri.value) not in omitted
                    ),
                )
            )
        # This is an output-vocabulary intervention. The asserted observable
        # ontology/evidence remains visible; only the declared generator menu
        # and endpoint producers lose the omitted symbols.
        retrieval = replace(retrieval, menus=tuple(menus_without))
        objects_without = []
        for obj in problem.objects:
            original_symbols = {
                str(entity.iri.value)
                for axiom in obj.original_axioms
                for entity in owl.signature(axiom)
            }
            candidates = tuple(
                candidate
                for candidate in obj.candidates
                if not (
                    {
                        str(entity.iri.value)
                        for axiom in (*candidate.axioms, *candidate.active_expressions)
                        for entity in owl.signature(axiom)
                    }
                    - original_symbols
                )
                & omitted
            )
            objects_without.append(replace(obj, candidates=candidates))
        problem = replace_inventory(problem, tuple(objects_without))
    problem = materialize_retrieved_endpoints(problem, retrieval)
    if selected_ids is not None:
        selected_other_assignment = tuple(
            (
                None
                if identifier is None
                else next(
                    i
                    for i, candidate in enumerate(obj.candidates)
                    if candidate.candidate_id == identifier
                )
            )
            for obj, identifier in zip(problem.objects, selected_ids)
        )
    removals = {key: frozenset(value) for key, value in (final_candidate_removals or {}).items()}
    if set(removals) - {obj.object_id for obj in problem.objects}:
        raise ValueError("final candidate removal names an unknown object")
    if any(
        not isinstance(value, str) or not value for values in removals.values() for value in values
    ):
        raise ValueError("final removal identities must be nonempty canonical candidate IDs")
    menus = {menu.object_id: menu for menu in retrieval.menus}
    retrieved_symbols = tuple(retrieved_symbols) + retrieval.symbols
    from .retrieval import RetrievalConfig

    preparation = EffectivePreparation(
        max_graph_nodes,
        max_graph_edges,
        max_explanations,
        max_text_tokens,
        pair_factor_limit_per_object,
        pair_max_pairs,
        pair_max_factors,
        retrieval_config or RetrievalConfig(),
        getattr(model, "revision", "v2"),
    )
    if (
        graph is not None
        and getattr(model, "revision", "v2") == "v3"
        and not graph.preparation_identity
    ):
        raise ValueError("Rebuild unqualified graph with EffectivePreparation before v3 freezing")
    if (
        graph is not None
        and graph.preparation_identity
        and graph.preparation_identity != preparation.content_hash
    ):
        raise ValueError("Supplied graph uses incompatible effective preparation settings")
    graph = graph or preparation.graph(problem, retrieval, retrieved_symbols=retrieved_symbols)
    graph_seconds = monotonic() - started
    prior_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            before = monotonic()
            memory = model.encode(graph)
            model_seconds = monotonic() - before
            objects, reports = [], []
            for index, obj in enumerate(problem.objects):
                object_started = monotonic()
                context_assignment = (
                    tuple(
                        choice if position < index else None
                        for position, choice in enumerate(selected_other_assignment)
                    )
                    if selected_other_assignment is not None
                    else None
                )
                from .grammar import mapping_grammar

                menu = menus[obj.object_id]
                encoding = mapping_grammar(
                    obj,
                    menu.classes,
                    menu.properties,
                    max_depth=max_depth,
                    max_constructors=max_constructors,
                    fixed_axioms=problem.fixed_axioms,
                    omitted_generation_symbols=omitted,
                    enabled_actions=enabled_actions,
                    source_classes=menu.source_classes,
                    target_classes=menu.target_classes,
                    source_properties=menu.source_properties,
                    target_properties=menu.target_properties,
                    constraint_identity=canonical_hash((problem.policy, menu)),
                )
                context_started = monotonic()
                if contextual_filtering:
                    from .grammar import with_immutable_context

                    encoding = with_immutable_context(
                        encoding,
                        problem.fixed_axioms,
                        problem.policy,
                        max_checks=max_context_checks,
                    )
                context_seconds = monotonic() - context_started
                before = monotonic()
                distribution = None
                setup_seconds = sampling_seconds = 0.0
                compilation_cache_hit = False
                samples: tuple[Any, ...] = ()
                rejected = 0
                enumerated = 0
                enumeration = None
                if proposal_arm == "bounded_enumeration":
                    from .proposals import enumerate_grammar

                    enumeration = enumerate_grammar(
                        encoding,
                        max_expressions=max_enumerated_expressions,
                        deadline=monotonic() + compile_seconds,
                    )
                    sampled_candidates = enumeration.candidates
                    enumerated = len(sampled_candidates)
                    setup_seconds = monotonic() - before
                    from .grammar import protected_representatives

                    controls, protected_reports = protected_representatives(
                        encoding,
                        max_checks=representative_max_checks,
                        enumeration=enumeration,
                    )
                    context = model.object_context(obj.object_id, memory)
                    ranked = {
                        c.candidate_id: float(model.candidate_value(c, memory, context)[0])
                        for c in sampled_candidates
                    }
                else:
                    from .compilation import compilation_cache_info

                    cache_before = compilation_cache_info()
                    distribution = proposal_distribution(
                        model,
                        memory,
                        obj,
                        mixtures=1 if proposal_arm == "grammar_product" else mixtures,
                        profile=profile,
                        max_depth=max_depth,
                        max_constructors=max_constructors,
                        encoding=encoding,
                        compile_seconds=compile_seconds,
                        max_circuit_nodes=max_circuit_nodes,
                        uniform=proposal_arm == "grammar_uniform",
                        conditioned=proposal_arm != "rejection",
                        factored=factored,
                        compiler_cache_directory=compiler_cache_directory,
                        vtree_type=vtree_type,
                        circuit_limits=circuit_limits,
                        selected_context=(
                            model.plan_context(
                                problem.objects,
                                memory,
                                context_assignment,
                                target_object_id=obj.object_id,
                            )
                            if context_assignment is not None
                            else None
                        ),
                    )
                    setup_seconds = monotonic() - before
                    compilation_cache_hit = compilation_cache_info()["hits"] > cache_before["hits"]
                    sampling_started = monotonic()
                    if proposal_arm == "rejection":
                        samples, rejected = _rejection_samples(
                            distribution, draws_per_object, seed + index
                        )
                    else:
                        samples = distribution.sample(draws_per_object, seed=seed + index)
                    sampling_seconds = monotonic() - sampling_started
                    sampled_candidates = tuple(
                        distribution.candidate(s.assignment) for s in samples
                    )
                    from .grammar import protected_representatives

                    controls, protected_reports = protected_representatives(
                        encoding,
                        getattr(distribution, "circuit", None),
                        max_checks=representative_max_checks,
                    )
                    ranked = {
                        c.candidate_id: max(-1e30, float(distribution.candidate_log_probability(c)))
                        for c in deduplicate_candidates((*controls, *sampled_candidates))
                    }
                mandatory = {c.candidate_id for c in controls}
                offered = deduplicate_candidates((*sampled_candidates, *controls))
                removed = removals.get(obj.object_id, frozenset())
                preserved = tuple((preserved_candidates or {}).get(obj.object_id, ()))
                if any(c.object_id != obj.object_id for c in preserved):
                    raise ValueError("preserved expansion candidate belongs to another object")
                original_symbols = {
                    str(e.iri.value) for ax in obj.original_axioms for e in owl.signature(ax)
                }
                for candidate in preserved:
                    if candidate.candidate_id in removed:
                        continue
                    symbols = {
                        str(e.iri.value)
                        for ax in (*candidate.axioms, *candidate.active_expressions)
                        for e in owl.signature(ax)
                    }
                    if (symbols - original_symbols) & omitted or not encoding.candidate_assignments(
                        candidate
                    ):
                        raise ValueError(
                            "preserved candidate violates the effective nested generation language"
                        )
                offered = deduplicate_candidates((*offered, *preserved))
                if any("keep" in c.action_tags and c.candidate_id in removed for c in offered):
                    raise ValueError(
                        "evaluator removal cannot remove the unchanged mandatory state"
                    )
                removed_present = tuple(
                    sorted(c.candidate_id for c in offered if c.candidate_id in removed)
                )
                offered = tuple(c for c in offered if c.candidate_id not in removed)
                mandatory.difference_update(removed)
                preserved_ids = {c.candidate_id for c in preserved} - removed
                selected = budget_candidates(
                    offered, candidate_cap, scores=ranked, mandatory_ids=mandatory | preserved_ids
                )
                if any(
                    c.candidate_id in removed or not encoding.candidate_assignments(c)
                    or ({str(e.iri.value)
                         for ax in (*c.axioms, *c.active_expressions)
                         for e in owl.signature(ax)} - original_symbols) & omitted
                    for c in selected
                ):
                    raise AssertionError("frozen candidate pool violates its effective declaration")
                retained_ids = {c.candidate_id for c in selected}
                protected_reports = tuple(
                    (
                        {
                            **row,
                            "status": "removed_by_intervention",
                            "detail": "predeclared final-pool intervention",
                        }
                        if row["candidate_id"] in removed
                        else row
                    )
                    for row in protected_reports
                )
                if any(
                    row["status"] == "retained" and row["candidate_id"] not in retained_ids
                    for row in protected_reports
                ):
                    raise AssertionError("protected family representative was evicted")
                objects.append(replace(obj, candidates=selected))
                sampled_ids = {s.candidate_id for s in samples}
                reports.append(
                    {
                        "schema": "exact-repair/generation-report/v3",
                        "object_id": obj.object_id,
                        "generation_identity": canonical_hash(
                            (
                                "protected-generation/current-witnesses-v1",
                                encoding.content_hash,
                                enumeration.content_hash if enumeration is not None else None,
                                representative_max_checks,
                                tuple(sorted(removed)),
                                tuple(sorted(omitted)),
                            )
                        ),
                        "language_hash": encoding.content_hash,
                        "protected_family_reports": protected_reports,
                        "representative_max_checks": representative_max_checks,
                        "generation_status": (
                            "ERROR"
                            if any(
                                f.status == "worker_error"
                                for f in getattr(
                                    getattr(distribution, "circuit", None), "families", ()
                                )
                            )
                            else (
                                "PARTIAL_RESOURCE_LIMIT"
                                if any(
                                    row["status"]
                                    not in {"retained", "empty_language", "removed_by_intervention"}
                                    for row in protected_reports
                                )
                                or (
                                    distribution is not None
                                    and not getattr(
                                        (
                                            distribution.circuit
                                            if proposal_arm != "rejection"
                                            else None
                                        ),
                                        "complete",
                                        True,
                                    )
                                )
                                else (
                                    "SAMPLED"
                                    if distribution is not None
                                    else "COMPLETE_DECLARED_ENUMERATION"
                                )
                            )
                        ),
                        "final_candidate_removals": tuple(sorted(removed)),
                        "omitted_generation_symbols": tuple(sorted(omitted)),
                        "enabled_actions": enabled_actions,
                        "max_nodes_per_family": max_circuit_nodes,
                        "declared_circuit_limits": circuit_limits or {},
                        "preserved_candidate_ids": tuple(sorted(preserved_ids)),
                        "removed_present": removed_present,
                        "proposal_context_hash": canonical_hash(context_assignment),
                        "proposal_context_order": "frozen_inventory_prefix",
                        "proposal_context_object_ids": tuple(
                            o.object_id for o in problem.objects[:index]
                        ),
                        "family_statuses": tuple(
                            (f.name, f.status, f.detail)
                            for f in getattr(getattr(distribution, "circuit", None), "families", ())
                        ),
                        "compiler_telemetry": getattr(
                            getattr(distribution, "circuit", None), "telemetry", ()
                        ),
                        "distribution_scope": (
                            "declared_language"
                            if getattr(getattr(distribution, "circuit", None), "complete", True)
                            else "completed_families_only_unknown_omitted_mass"
                        ),
                        "arm": proposal_arm,
                        "seed": seed + index,
                        "mixtures": (
                            int(distribution.literal_logits.shape[0])
                            if distribution is not None
                            else 0
                        ),
                        "max_depth": max_depth,
                        "max_constructors": max_constructors,
                        "samples": tuple(
                            ProposalRecordV3(
                                {
                                    "schema": "exact-repair/proposal/v3",
                                    "object_id": obj.object_id,
                                    "candidate_id": sample.candidate_id,
                                    "candidate": ReplacementCandidateV3(
                                        **{
                                            field.name: getattr(candidate, field.name)
                                            for field in fields(candidate)
                                        }
                                    ),
                                    "assignment": sample.assignment,
                                    "component": sample.component,
                                    "log_probability": sample.log_probability,
                                    "grammar_hash": encoding.content_hash,
                                    "circuit_hash": (
                                        distribution.circuit.cache_key
                                        if distribution is not None and proposal_arm != "rejection"
                                        else None
                                    ),
                                    "retrieval_hash": canonical_hash(retrieval),
                                    "context_hash": canonical_hash(context_assignment),
                                    "seed": seed + index,
                                    "family": encoding.choices(sample.assignment)["template"],
                                    "probability_semantics": (
                                        "unconditioned_bundle_mass"
                                        if proposal_arm == "rejection"
                                        else "conditioned_bundle_probability"
                                    ),
                                    "distribution_scope": (
                                        "declared_language"
                                        if getattr(
                                            getattr(distribution, "circuit", None), "complete", True
                                        )
                                        else "completed_families_only_unknown_omitted_mass"
                                    ),
                                }
                            )
                            for sample, candidate in zip(samples, sampled_candidates)
                        ),
                        "attempted_draws": draws_per_object if distribution is not None else 0,
                        "enumerated_candidates": enumerated,
                        "probability_semantics": (
                            "unconditioned_bundle_mass"
                            if proposal_arm == "rejection"
                            else (
                                "conditioned_bundle_probability"
                                if distribution is not None
                                else "not_applicable"
                            )
                        ),
                        "valid_draws": len(samples),
                        "rejected_draws": rejected,
                        "unique_draws": len(sampled_ids),
                        "duplicate_draws": len(samples) - len(sampled_ids),
                        "retained": len(selected),
                        "mandatory": len(mandatory),
                        "grammar_hash": encoding.content_hash,
                        "contextual_proof_hashes": tuple(
                            p.content_hash for p in encoding.context_proofs
                        ),
                        "contextual_checks": encoding.contextual_checks,
                        "contextual_truncated": encoding.contextual_truncated,
                        "contextual_scope": "immutable named/one-existential active-unsatisfiability proofs",
                        "circuit_hash": (
                            distribution.circuit.cache_key
                            if distribution is not None and proposal_arm != "rejection"
                            else None
                        ),
                        "circuit_nodes": (
                            distribution.circuit.node_count
                            if distribution is not None and proposal_arm != "rejection"
                            else 0
                        ),
                        "boolean_variables": encoding.variable_count,
                        "compilation_seconds": (
                            distribution.circuit.compilation_seconds
                            if distribution is not None and proposal_arm != "rejection"
                            else 0.0
                        ),
                        "circuit_build_seconds": (
                            distribution.circuit.compilation_seconds
                            if distribution is not None and proposal_arm != "rejection"
                            else 0.0
                        ),
                        "compilation_cache_hit": compilation_cache_hit,
                        "proposal_setup_seconds": setup_seconds,
                        "sampling_seconds": sampling_seconds,
                        "proposal_seconds": monotonic() - object_started,
                        "context_proof_seconds": context_seconds,
                        "graph_seconds": graph_seconds,
                        "model_seconds": model_seconds,
                        "retrieval_hash": canonical_hash(retrieval),
                        "omitted_graph_nodes": graph.omitted_nodes,
                        "omitted_supports": graph.omitted_supports,
                    }
                )
            graph_hash, model_hash = canonical_hash(graph), model_digest(model)
            draft_reports = tuple(FrozenMapping(report) for report in reports)
            frozen_problem = replace_inventory(
                problem,
                tuple(objects),
                candidate_coverage=(
                    "bounded_enumerated_selected"
                    if proposal_arm == "bounded_enumeration"
                    else "bounded_sampled"
                ),
                model_status=f"{model.encoder}:{model_hash}; uncertainty uncalibrated",
                graph_identity=graph_hash,
                proposal_provenance=draft_reports,
            )
            selection = None
            if interaction_pairs is None:
                if model.pair_head is not None:
                    selection = preparation.pairs(frozen_problem, graph)
                    interaction_pairs = selection.pairs
                else:
                    interaction_pairs = ()
            interaction_pairs = tuple(interaction_pairs)
            pair_hash = (
                selection.content_hash
                if selection is not None
                else canonical_hash(("explicit-pairs/v3", interaction_pairs))
            )
            proposal_reports = tuple(
                GenerationReportV3(
                    {
                        **dict(report),
                        "pair_selection_hash": pair_hash,
                        "graph_hash": graph_hash,
                        "model_hash": model_hash,
                        "preparation_identity": preparation.content_hash,
                        "support_admission": graph.admission_policy,
                        "support_omissions": graph.support_omissions,
                        "pair_selection_omissions": selection.omissions if selection else (),
                    }
                )
                for report in draft_reports
            )
            frozen_problem = replace(frozen_problem, proposal_provenance=proposal_reports)
            unary, pairs = model.score_inventory(
                objects, memory, interaction_pairs=interaction_pairs
            )
            objective = make_objective(
                tuple(objects),
                tuple(tuple(float(value) for value in row) for row in unary),
                profile=profile,
                pairs=tuple((*key, float(value)) for key, value in sorted(pairs.items())),
                scale=quantization_scale,
            )
            pool_hash = canonical_hash(frozen_problem.objects)
            objective = ObjectiveV3(
                **{
                    **{field.name: getattr(objective, field.name) for field in fields(objective)},
                    "pool_hash": pool_hash,
                    "pair_selection_hash": pair_hash,
                    "model_hash": model_hash,
                    "target_basis": f"symbolic-semantic-vector/{getattr(model, 'revision', 'v2')}",
                }
            )
            risk = None
            if getattr(model, "revision", "v2") == "v3" and getattr(
                model, "plan_risk_enabled", True
            ):
                path, digest, size = _publish_risk_snapshot(
                    model, compiler_cache_directory, risk_snapshot_max_bytes
                )
                risk = FrozenPlanRisk(
                    path,
                    digest,
                    size,
                    graph,
                    frozen_problem.objects,
                    graph.admitted_supports,
                    model_hash,
                    graph_hash,
                    pool_hash,
                )
            return FrozenNeuralRound(
                frozen_problem, objective, graph_hash, model_hash, proposal_reports, risk
            )
    finally:
        model.train(prior_training)


def _rejection_samples(distribution: Any, count: int, seed: int) -> tuple[tuple[Any, ...], int]:
    """Unconditioned Bernoulli-mixture draws; reject invalid assignments with a finite cap."""
    import torch

    from .circuit import ProposalSample

    generator = torch.Generator(device=distribution.literal_logits.device).manual_seed(seed)
    samples = []
    for _ in range(count):
        component = int(torch.multinomial(distribution.log_mixture.exp(), 1, generator=generator))
        bits = tuple(
            bool(v)
            for v in (
                torch.rand(
                    distribution.literal_logits.shape[1],
                    device=distribution.literal_logits.device,
                    generator=generator,
                )
                < distribution.literal_logits[component].sigmoid()
            ).tolist()
        )
        probability = distribution.log_probability(bits)
        if not torch.isfinite(probability):
            continue
        candidate = distribution.candidate(bits)
        samples.append(
            ProposalSample(
                candidate.candidate_id,
                bits,
                component,
                float(distribution.candidate_log_probability(candidate)),
            )
        )
    return tuple(samples), count - len(samples)


def _freeze_worker(
    problem: RepairInputV2, checkpoint: bytes, options: dict[str, Any]
) -> FrozenNeuralRound:
    import io

    import torch

    from .model import RepairModel

    payload = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
    schema = payload.get("model_schema", "exact-repair/model/v2")
    revision = payload.get("config", {}).get("revision", "v2")
    if revision not in {"v2", "v3"} or schema != f"exact-repair/model/{revision}":
        raise ValueError("Repair checkpoint schema/config mismatch; use explicit migration")
    torch.set_num_threads(1)
    model = RepairModel(payload["metadata"], **payload["config"])
    model.load_state_dict(payload["state_dict"])
    return freeze_neural_round(problem, model, **options)


def bounded_freeze_neural_round(
    problem: RepairInputV2,
    model: Any,
    *,
    seconds: float | None = None,
    memory_mb: float | None = None,
    cpu_seconds: float | None = None,
    **options: Any,
) -> Any:
    """Return complete/timeout/error without publishing a partially frozen objective.

    Startup and model capture run in the independently supervised domain.
    Matching is never invoked; source records and model parameters remain in the parent.
    """
    from .workers import CallResult, bounded_call

    # CUDA IPC and native tensor serialization must never run in a fork broker.
    # The owning supervised training process publishes the artifact instead.
    if any(value.device.type != "cpu" for value in (*model.parameters(), *model.buffers())):
        return CallResult(
            "unsupported",
            detail=(
                "live accelerator model transport requires an immutable checkpoint artifact; "
                "use bounded_freeze_checkpoint after saving inside the supervised training worker"
            ),
        )

    limit = problem.budgets.total_seconds if seconds is None else seconds
    worker_options: dict[str, Any] = {"cpu_seconds": cpu_seconds} if cpu_seconds is not None else {}
    return bounded_call(
        _freeze_model_worker,
        problem,
        model,
        options,
        timeout=limit,
        memory_mb=memory_mb,
        **worker_options,
    )


def _freeze_model_worker(
    problem: RepairInputV2, model: Any, options: dict[str, Any]
) -> FrozenNeuralRound:
    """Capture model bytes inside the independently supervised preparation worker."""
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(
        {
            "metadata": model.metadata,
            "config": model.config,
            "model_schema": f"exact-repair/model/{getattr(model, 'revision', 'v2')}",
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        buffer,
    )
    return _freeze_worker(problem, buffer.getvalue(), options)


def freeze_checkpoint(
    problem: RepairInputV2,
    checkpoint_path: str,
    *,
    checkpoint_sha256: str | None = None,
    **options: Any,
) -> FrozenNeuralRound:
    """Load a frozen XR-2 checkpoint in a supervised worker and generate its pool."""
    from pathlib import Path

    data = Path(checkpoint_path).read_bytes()
    if checkpoint_sha256 is not None and hashlib.sha256(data).hexdigest() != checkpoint_sha256:
        raise ValueError("frozen model checkpoint integrity mismatch")
    return _freeze_worker(problem, data, options)


def bounded_freeze_checkpoint(
    problem: RepairInputV2,
    checkpoint_path: str,
    *,
    seconds: float | None = None,
    memory_mb: float | None = None,
    cpu_seconds: float | None = None,
    checkpoint_sha256: str | None = None,
    **options: Any,
) -> Any:
    """Supervise immutable model artifact transfer/loading without live device state.

    The caller publishes GPU-owned weights within the already supervised training
    process. This boundary receives only a path and optional expected content hash;
    every read, verification, tensor reconstruction and inference occurs in a clean
    independently bounded worker.
    """
    from .workers import bounded_call

    limit = problem.budgets.total_seconds if seconds is None else seconds
    worker_options: dict[str, Any] = {"cpu_seconds": cpu_seconds} if cpu_seconds is not None else {}
    return bounded_call(
        freeze_checkpoint,
        problem,
        str(checkpoint_path),
        timeout=limit,
        memory_mb=memory_mb,
        checkpoint_sha256=checkpoint_sha256,
        **worker_options,
        **options,
    )


def freeze_progressive_rounds(
    problem: RepairInputV2, model: Any, schedule: Iterable[dict[str, Any]], **shared_options: Any
) -> tuple[FrozenNeuralRound, ...]:
    """Execute a predeclared finite expansion schedule with independently frozen objectives.

    Every successful round preserves its previous canonical candidates. A later
    failure propagates; callers can retain the returned/previous saved epoch and
    its proof, but cannot reuse its upper bound for the changed pool.
    """
    stages = tuple(dict(options) for options in schedule)
    if not stages or len(stages) > 32:
        raise ValueError("progressive coverage requires one to 32 declared stages")
    from .candidates import MAPPING_ACTIONS, ONTOLOGY_ACTIONS
    from .retrieval import retrieve_vocabulary

    default_actions = set().union(
        *(MAPPING_ACTIONS if obj.kind == "mapping" else ONTOLOGY_ACTIONS for obj in problem.objects)
    )
    prior_limits = None
    prior_language = None
    for options in stages:
        effective = {
            "max_depth": 2,
            "max_constructors": 2,
            "candidate_cap": 64,
            "draws_per_object": 32,
            **shared_options,
            **options,
        }
        limits = tuple(
            effective[key]
            for key in ("max_depth", "max_constructors", "candidate_cap", "draws_per_object")
        )
        if prior_limits is not None and any(new < old for new, old in zip(limits, prior_limits)):
            raise ValueError("progressive coverage bounds must be nondecreasing")
        config = effective.get("retrieval_config")
        retrieved = retrieve_vocabulary(
            problem, **({"config": config} if config is not None else {})
        )
        vocabulary = {
            (menu.object_id, field): {canonical_hash(entity) for entity in getattr(menu, field)}
            for menu in retrieved.menus
            for field in (
                "source_classes",
                "target_classes",
                "source_properties",
                "target_properties",
                "endpoint_alternatives",
            )
        }
        actions = effective.get("enabled_actions")
        removals = effective.get("final_candidate_removals") or {}
        language: dict[str, Any] = dict(
            actions=default_actions if actions is None else set(actions) | {"keep"},
            omitted=set(effective.get("omitted_generation_symbols", ())),
            removed={(obj, candidate) for obj, ids in removals.items() for candidate in ids},
            context=canonical_hash(
                (
                    effective.get("contextual_filtering", True),
                    effective.get("max_context_checks", 128),
                    problem.fixed_axioms,
                    problem.policy,
                )
            ),
            vocabulary=vocabulary,
        )
        if prior_language is not None:
            if (
                not prior_language["actions"] <= language["actions"]
                or not language["omitted"] <= prior_language["omitted"]
                or not language["removed"] <= prior_language["removed"]
                or language["context"] != prior_language["context"]
                or any(
                    not symbols <= language["vocabulary"].get(key, set())
                    for key, symbols in prior_language["vocabulary"].items()
                )
            ):
                raise ValueError(
                    "progressive schedule must use nested actions/vocabulary/filters and unchanged context"
                )
        prior_language = language
        prior_limits = limits
    previous: dict[str, tuple[ReplacementCandidateV2, ...]] = {}
    rounds: list[FrozenNeuralRound] = []
    schedule_hash = canonical_hash((stages, shared_options))
    for index, options in enumerate(stages):
        try:
            frozen = freeze_neural_round(
                problem, model, **{**shared_options, **options, "preserved_candidates": previous}
            )
        except Exception as error:
            # Preserve independently completed epochs for a resumable caller;
            # none of their old objective bounds applies to the failed expansion.
            setattr(error, "completed_rounds", tuple(rounds))
            setattr(error, "failed_expansion_index", index)
            raise
        provenance = tuple(
            GenerationReportV3(
                {
                    **dict(row),
                    "expansion_schedule_hash": schedule_hash,
                    "expansion_index": index,
                    "previous_pool_hash": rounds[-1].problem.content_hash if rounds else None,
                }
            )
            for row in frozen.proposal_reports
        )
        frozen = replace(
            frozen,
            problem=replace(frozen.problem, proposal_provenance=provenance),
            proposal_reports=provenance,
        )
        rounds.append(frozen)
        previous = {obj.object_id: obj.candidates for obj in frozen.problem.objects}
    return tuple(rounds)


def repair_neural_round(
    frozen: FrozenNeuralRound,
    *,
    shortlist_size: int = 1,
    utility_window: int = 0,
    shortlist_seconds: float | None = None,
    **options: Any,
) -> Any:
    """Pass a frozen objective and separate captured risk scorer to the exact kernel.

    Shortlist size one retains the baseline selection order. The study/run
    manifest must explicitly supply a larger size to enable risk scheduling.
    """
    from .kernel import repair

    if not isinstance(
        frozen.objective, ObjectiveV3
    ) or frozen.objective.pool_hash != canonical_hash(frozen.problem.objects):
        raise ValueError("frozen neural objective no longer matches its candidate pool")
    risk_options = {}
    if frozen.risk_scorer is not None:
        risk_options = {
            "risk_order": frozen.risk_scorer,
            "risk_identity": frozen.risk_scorer.risk_identity,
        }
    return repair(
        frozen.problem,
        frozen.objective,
        shortlist_size=shortlist_size,
        utility_window=utility_window,
        shortlist_seconds=shortlist_seconds,
        **risk_options,
        **options,
    )
