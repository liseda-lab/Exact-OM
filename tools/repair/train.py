"""Opt-in, bounded generated-case training smoke; never launches a benchmark campaign."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from exact.repair.graph import build_observable_graph
from exact.repair.learning import (
    OwlTeacherOracle,
    RepairLabel,
    TeacherCache,
    benefit_losses,
    enumerate_teacher,
    evaluate_teacher,
    proposal_loss,
)
from exact.repair.records import canonical_hash
from exact.repair.workers import bounded_call
from tools.repair.corpus import GeneratedCase, generate_corpus

DEFAULT_PROFILE = (
    ("delete", 0.1),
    ("edit", 0.05),
    ("ontology_edit", 0.02),
    ("human_authored_ontology_edit", 0.03),
)


def _assignment_label(
    case: GeneratedCase, assignment: tuple[int, ...], profile: tuple
) -> RepairLabel:
    from exact.repair.kernel import materialize
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms

    axioms, active = materialize(case.problem, assignment)
    verifier = OwlVerifier("hermit", backend="python")
    snapshot = snapshot_from_axioms(axioms)
    report = verifier.check_theory(
        snapshot,
        case.problem.policy.monitored_classes,
        required=case.problem.policy.required,
        prohibited=case.problem.policy.prohibited,
        activated=active,
    )
    cost_weights = dict(profile)
    cost = sum(
        cost_weights.get(key, 0.0) * value
        for obj, choice in zip(case.problem.objects, assignment)
        for key, value in obj.candidates[choice].cost_features
    )
    if report.logical_status != "VERIFIED_FEASIBLE":
        feasible = False if report.logical_status == "VERIFIED_INFEASIBLE" else None
        return RepairLabel(assignment, feasible, None, cost)
    semantic = evaluate_teacher(OwlTeacherOracle(verifier, snapshot, case.probes), case.probes)
    return RepairLabel(assignment, True, semantic.benefit, cost, semantic.outcomes)


def _verify_intended(case: GeneratedCase) -> bool:
    """Verify the evaluator-only intended parent even when its candidate is withheld."""
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms

    verifier = OwlVerifier("hermit", backend="python")
    snapshot = snapshot_from_axioms(case.intended_theory)
    report = verifier.check_theory(
        snapshot, case.problem.policy.monitored_classes, activated=case.intended_active
    )
    return (
        report.logical_status == "VERIFIED_FEASIBLE"
        and evaluate_teacher(
            OwlTeacherOracle(verifier, snapshot, case.probes), case.probes
        ).complete
    )


def label_case(
    case: GeneratedCase,
    *,
    max_assignments: int = 256,
    deadline_seconds: float = 30.0,
    call_seconds: float = 5.0,
    profile: tuple = DEFAULT_PROFILE,
) -> TeacherCache:
    """Verify the intended clean parent, then enumerate whole-case labels under deadlines."""
    from importlib.metadata import version

    started = time.monotonic()
    intended = bounded_call(_verify_intended, case, timeout=min(call_seconds, deadline_seconds))
    if intended.status != "complete" or intended.value is not True:
        raise ValueError(
            "Generated intended parent has not been verified feasible and query-complete"
        )

    def label(assignment: tuple[int, ...]) -> RepairLabel:
        remaining = deadline_seconds - (time.monotonic() - started)
        result = bounded_call(
            _assignment_label,
            case,
            assignment,
            profile,
            timeout=min(call_seconds, max(0.0, remaining)),
        )
        if result.status == "complete":
            return cast(RepairLabel, result.value)
        return RepairLabel(assignment, None, None, 0.0)

    return enumerate_teacher(
        tuple(len(obj.candidates) for obj in case.problem.objects),
        label,
        hashes={
            "input": case.problem.content_hash,
            "patch": canonical_hash(case.problem.objects),
            "policy": case.problem.policy.content_hash,
            "query": canonical_hash(case.probes),
            "inventory": canonical_hash(tuple(obj.candidates for obj in case.problem.objects)),
            "backend": canonical_hash(("pyhermit", version("pyhermit"), "python")),
            "profile": canonical_hash(profile),
        },
        max_assignments=max_assignments,
        deadline_seconds=max(0.001, deadline_seconds - (time.monotonic() - started)),
    )


def train_cases(
    training: Sequence[tuple[GeneratedCase, TeacherCache]],
    development: Sequence[tuple[GeneratedCase, TeacherCache]],
    *,
    encoder: str = "hgt",
    epochs: int = 2,
    seed: int = 13,
    hidden_dim: int = 32,
    pairwise: bool = False,
    learning_rate: float = 0.001,
    profile: tuple = DEFAULT_PROFILE,
    arm: str = "generated_only",
    warm_start_weights: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train joint benefits/proposals and select checkpoints using development loss only.

    Cases without complete finite distributions still contribute masked benefit
    regression/ranking. No test case can enter optimization or checkpoint selection.
    """
    import torch

    from exact.repair.model import RepairModel, repair_benefits
    from exact.repair.pipeline import model_digest, proposal_distribution

    if not training or not development or epochs < 1:
        raise ValueError("Training requires train/development cases and a positive epoch count")
    if any(case.split != "train" for case, _ in training) or any(
        case.split != "development" for case, _ in development
    ):
        raise ValueError("Training/checkpoint selection must respect saved clean-parent splits")
    if {case.structural_parent for case, _ in training} & {
        case.structural_parent for case, _ in development
    }:
        raise ValueError("A structural parent cannot cross training and development")
    if arm not in {"generated_only", "training_side_adaptation"}:
        raise ValueError("Unknown training arm")
    expected_origin = "generated" if arm == "generated_only" else "training_side_real"
    if any(case.origin != expected_origin for case, _ in [*training, *development]):
        raise ValueError("Training provenance must match the separately declared arm")
    if arm == "training_side_adaptation" and warm_start_weights is None:
        raise ValueError("Adaptation requires pretrained model weights only")
    for case, cache in [*training, *development]:
        expected = {
            "input": case.problem.content_hash,
            "policy": case.problem.policy.content_hash,
            "query": canonical_hash(case.probes),
            "inventory": canonical_hash(tuple(obj.candidates for obj in case.problem.objects)),
            "profile": canonical_hash(profile),
        }
        if any(dict(cache.hashes).get(key) != digest for key, digest in expected.items()):
            raise ValueError("Teacher cache dependencies/profile do not match this training case")
        if cache.candidate_counts != tuple(len(obj.candidates) for obj in case.problem.objects):
            raise ValueError("Teacher cache candidate counts do not match")
    torch.manual_seed(seed)
    all_cases = [*training, *development]
    graphs = {
        case.case_id: build_observable_graph(
            case.problem.objects,
            fixed_axioms=case.problem.fixed_axioms,
            evidence=dict(case.problem.evidence),
        )
        for case, _ in all_cases
    }
    node_types = {kind for graph in graphs.values() for kind in graph.metadata[0]}
    edge_types = {edge for graph in graphs.values() for edge in graph.metadata[1]}
    model = RepairModel(
        (node_types, edge_types),
        hidden_dim=hidden_dim,
        heads=4,
        layers=1,
        encoder=encoder,
        pairwise=pairwise,
    )
    if warm_start_weights is not None:
        model.load_state_dict(dict(warm_start_weights), strict=True)
    warm_start_hash = model_digest(model) if warm_start_weights is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    best_loss, best_state = float("inf"), None

    def case_loss(case: GeneratedCase, cache: TeacherCache) -> Any:
        if not cache.labels:
            return model.empty_bundle.sum() * 0.0
        graph = graphs[case.case_id]
        memory = model.encode(graph)
        # Shared original syntax is observable; no teacher support determines pairs.
        shared = (
            tuple(
                (i, j)
                for i in range(len(case.problem.objects))
                for j in range(i + 1, len(case.problem.objects))
                if set(case.problem.objects[i].original_axioms)
                & set(case.problem.objects[j].original_axioms)
            )
            if pairwise
            else ()
        )
        unary, pairs = model.score_inventory(case.problem.objects, memory, interaction_pairs=shared)
        predictions = repair_benefits((row.assignment for row in cache.labels), unary, pairs)
        losses = benefit_losses(predictions, cache.labels, seed=seed)
        loss = losses["value"] + 0.2 * losses["rank"]
        if cache.complete and any(label.usable for label in cache.labels):
            probabilities = []
            for obj in case.problem.objects:
                distribution = proposal_distribution(model, memory, obj, profile=profile)
                by_id = dict(
                    zip(
                        distribution.circuit.encoding.candidate_ids,
                        distribution.circuit.encoding.assignments,
                    )
                )
                probabilities.append(
                    torch.stack(
                        [
                            distribution.log_probability(by_id[candidate.candidate_id])
                            for candidate in obj.candidates
                        ]
                    )
                )
            loss = loss + proposal_loss(probabilities, cache)
        return loss

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for case, cache in training:
            optimizer.zero_grad(set_to_none=True)
            loss = case_loss(case, cache)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach())
        model.eval()
        with torch.no_grad():
            dev_loss = sum(float(case_loss(case, cache)) for case, cache in development) / len(
                development
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(training),
                "development_loss": dev_loss,
            }
        )
        if dev_loss < best_loss:
            best_loss, best_state = dev_loss, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError("No finite development checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "schema": "exact-repair/training-smoke/v2",
        "encoder": encoder,
        "seed": seed,
        "epochs": epochs,
        "history": history,
        "checkpoint_criterion": "development_joint_loss",
        "model_hash": model_digest(model),
        "splits": {
            case.case_id: {"parent": case.structural_parent, "split": case.split}
            for case, _ in all_cases
        },
        "coverage": {case.case_id: cache.coverage for case, cache in all_cases},
        "arm": arm,
        "warm_start_hash": warm_start_hash,
        "scope": "conformance; no model-quality or transfer claim",
    }


def main() -> int:
    """Run at most three generated cases and two epochs, saving complete provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("hgt", "rgcn", "none"), default="hgt")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    cases = generate_corpus(parents_per_family=3, siblings_per_parent=1, seed=args.seed)
    training = [case for case in cases if case.split == "train"][:2]
    development = [case for case in cases if case.split == "development"][:1]
    if not training or not development:
        raise ValueError("This seed produced no small train/development selection")
    labelled_train = [(case, label_case(case)) for case in training]
    labelled_dev = [(case, label_case(case)) for case in development]
    model, report = train_cases(labelled_train, labelled_dev, encoder=args.encoder, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "teacher-cache.json").write_text(
        json.dumps(
            {case.case_id: asdict(cache) for case, cache in [*labelled_train, *labelled_dev]},
            indent=2,
        )
        + "\n"
    )
    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": model.metadata,
            "encoder": model.encoder,
            "hidden_dim": model.hidden_dim,
            "feature_dim": model.feature_dim,
            "layers": 1,
            "heads": 4,
        },
        args.output / "model.pt",
    )
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
