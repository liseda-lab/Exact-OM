"""Opt-in neural proposal/value handoff to the exact finite-pool repair kernel.

Direct typed grammars generate replacements from observed finite symbol menus.
Enumeration remains a matched small-language control. Proposal, benefit and edit
cost stay separate, and all coefficients are frozen before exact selection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from time import monotonic
from typing import Any, Iterable

import pyowl_core as owl

from .graph import (
    GraphNode,
    ObservableGraph,
    build_observable_graph,
    feature_vector,
    structural_id,
)
from .records import (
    FrozenMapping,
    ObjectiveV2,
    RepairInputV2,
    canonical_hash,
    make_objective,
)


@dataclass(frozen=True)
class FrozenNeuralRound:
    """Reviewable immutable solver input and proposal/model coverage provenance."""

    problem: RepairInputV2
    objective: ObjectiveV2
    graph_hash: str
    model_hash: str
    proposal_reports: tuple[FrozenMapping, ...]


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
) -> Any:
    """Connect shared slot/menu embeddings and precomputed logits to exact SDD WMC."""
    import torch

    from .candidates import deduplicate_candidates, normalise_axioms
    from .circuit import ConditionedMixture, compile_encoding, encode_candidates

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
    elif compile_seconds is not None:
        from .compilation import compile_bounded

        compiled = compile_bounded(encoding, seconds=compile_seconds, max_nodes=max_circuit_nodes)
    elif hasattr(encoding, "decode"):
        from .grammar import compile_grammar

        compiled = compile_grammar(encoding, max_nodes=max_circuit_nodes)
    else:
        compiled = compile_encoding(encoding)
    if compiled is not None and compiled.node_count > max_circuit_nodes:
        raise ValueError("Compiled circuit exceeds the declared node budget")
    if uniform:
        if compiled is None:
            raise ValueError("Uniform constrained sampling requires a compiled circuit")
        return ConditionedMixture(
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
        context, torch.stack(choices), mixtures=mixtures, profile_features=profile_vector
    )
    if compiled is None:
        from .proposals import UnconditionedMixture

        return UnconditionedMixture(encoding, logits, weights)
    return ConditionedMixture(compiled, logits, weights)


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
    quantization_scale: int = 1000,
) -> FrozenNeuralRound:
    """Generate a finite pool and freeze its value/cost objective.

    Each native compilation has an external deadline. Use
    ``bounded_freeze_neural_round`` to bound retrieval, graph/model work and the
    whole proposal stage as well. This local form is useful inside supervised
    training/campaign workers. No model state changes during inference.
    """
    import torch

    from .candidates import budget_candidates, deduplicate_candidates
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
    menus = {menu.object_id: menu for menu in retrieval.menus}
    retrieved_symbols = tuple(retrieved_symbols) + retrieval.symbols
    graph = graph or build_observable_graph(
        problem.objects,
        fixed_axioms=problem.fixed_axioms,
        source_axioms=problem.source_axioms,
        target_axioms=problem.target_axioms,
        evidence=retrieval.graph_evidence(problem.evidence),
        retrieved_symbols=retrieved_symbols,
        explanations=retrieval.explanations,
        max_nodes=max_graph_nodes,
        max_edges=max_graph_edges,
        max_explanations=max_explanations,
        max_text_tokens=max_text_tokens,
    )
    if interaction_pairs is None:
        from .graph import observable_interaction_pairs

        interaction_pairs = (
            observable_interaction_pairs(
                problem,
                per_object_limit=pair_factor_limit_per_object,
                explanations=retrieval.explanations,
            )
            if model.pair_head is not None
            else ()
        )
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
                from .grammar import mapping_grammar

                menu = menus[obj.object_id]
                encoding = mapping_grammar(
                    obj,
                    menu.classes,
                    menu.properties,
                    max_depth=max_depth,
                    max_constructors=max_constructors,
                    fixed_axioms=problem.fixed_axioms,
                    source_classes=menu.source_classes,
                    target_classes=menu.target_classes,
                    source_properties=menu.source_properties,
                    target_properties=menu.target_properties,
                    constraint_identity=canonical_hash((problem.policy, menu)),
                )
                before = monotonic()
                distribution = None
                setup_seconds = sampling_seconds = 0.0
                compilation_cache_hit = False
                samples: tuple[Any, ...] = ()
                rejected = 0
                enumerated = 0
                if proposal_arm == "bounded_enumeration":
                    from .proposals import enumerate_grammar_candidates

                    sampled_candidates = enumerate_grammar_candidates(
                        encoding,
                        max_expressions=max_enumerated_expressions,
                        deadline=monotonic() + compile_seconds,
                    )
                    enumerated = len(sampled_candidates)
                    setup_seconds = monotonic() - before
                    controls = encoding.representatives()
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
                    controls = encoding.representatives()
                    ranked = {
                        c.candidate_id: float(distribution.candidate_log_probability(c))
                        for c in deduplicate_candidates((*controls, *sampled_candidates))
                    }
                elementary = {"keep", "delete", "retain_subsumption", "replace_endpoint"}
                mandatory = {c.candidate_id for c in controls if elementary & set(c.action_tags)}
                for family in sorted({t for c in controls for t in c.action_tags}):
                    members = [c for c in controls if family in c.action_tags]
                    if not any(c.candidate_id in mandatory for c in members):
                        mandatory.add(
                            max(
                                members, key=lambda c: (ranked[c.candidate_id], c.candidate_id)
                            ).candidate_id
                        )
                offered = deduplicate_candidates(
                    (*sampled_candidates, *(c for c in controls if c.candidate_id in mandatory))
                )
                selected = budget_candidates(offered, candidate_cap, scores=ranked)
                objects.append(replace(obj, candidates=selected))
                sampled_ids = {s.candidate_id for s in samples}
                reports.append(
                    {
                        "object_id": obj.object_id,
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
                            {
                                "candidate_id": sample.candidate_id,
                                "assignment": sample.assignment,
                                "component": sample.component,
                                "log_probability": sample.log_probability,
                            }
                            for sample in samples
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
                        "proposal_seconds": monotonic() - before,
                        "graph_seconds": graph_seconds,
                        "model_seconds": model_seconds,
                        "retrieval_hash": canonical_hash(retrieval),
                        "omitted_graph_nodes": graph.omitted_nodes,
                        "omitted_supports": graph.omitted_supports,
                    }
                )
            graph_hash, model_hash = canonical_hash(graph), model_digest(model)
            proposal_reports = tuple(FrozenMapping(report) for report in reports)
            frozen_problem = replace(
                problem,
                objects=tuple(objects),
                candidate_coverage=(
                    "bounded_enumerated_selected"
                    if proposal_arm == "bounded_enumeration"
                    else "bounded_sampled"
                ),
                model_status=f"{model.encoder}:{model_hash}; uncertainty uncalibrated",
                graph_identity=graph_hash,
                proposal_provenance=proposal_reports,
            )
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
            return FrozenNeuralRound(
                frozen_problem, objective, graph_hash, model_hash, proposal_reports
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
    if payload.get("model_schema", "exact-repair/model/v2") != "exact-repair/model/v2":
        raise ValueError(
            "Unsupported repair checkpoint schema; legacy models require explicit migration"
        )
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
    **options: Any,
) -> Any:
    """Return complete/timeout/error without publishing a partially frozen objective.

    Transport CPU tensor bytes instead of live CUDA state or native circuit handles.
    Matching is never invoked; source records and model parameters remain in the parent.
    """
    import io

    import torch

    from .workers import bounded_call

    started = monotonic()
    limit = problem.budgets.total_seconds if seconds is None else seconds
    buffer = io.BytesIO()
    torch.save(
        {
            "metadata": model.metadata,
            "config": model.config,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        buffer,
    )
    return bounded_call(
        _freeze_worker, problem, buffer.getvalue(), options, timeout=limit - (monotonic() - started)
    )


def freeze_checkpoint(
    problem: RepairInputV2, checkpoint_path: str, **options: Any
) -> FrozenNeuralRound:
    """Load a frozen XR-2 checkpoint in a supervised worker and generate its pool."""
    from pathlib import Path

    return _freeze_worker(problem, Path(checkpoint_path).read_bytes(), options)
