"""Small deterministic structural parents for conformance and generated pretraining.

Splits are assigned to clean structural parents before namespaces, noisy scores,
provenance and corrupted observations are generated. Teacher queries/intended
assignments stay in this evaluator record, outside RepairInputV2 observations.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import pyowl_core as owl

from exact.repair.candidates import make_candidate, replacement_cost_features
from exact.repair.learning import TeacherProbe, grouped_split
from exact.repair.records import (
    PolicyV2,
    RepairInputV2,
    ReplacementCandidateV2,
    RevisionObjectV2,
)

FAMILIES = (
    "papers",
    "overlap",
    "disjointness",
    "conjunct",
    "participant",
    "range",
    "domain",
    "filler",
    "interactions",
    "interaction_complementary",
    "mixed",
)

# Protocol names identify held-out templates even when they share a construction.
MECHANISMS = {
    "directional_strengthening": "papers",
    "subclass_specialisation_source": "papers",
    "subclass_specialisation_target": "papers",
    "necessary_condition_source": "papers",
    "necessary_condition_target": "papers",
    "endpoint_confusion": "papers",
    "disjointness_revision": "disjointness",
    "superclass_conjunct_removal": "conjunct",
    "ontology_subclass_specialisation": "participant",
    "superclass_generalisation": "participant",
    "domain_generalisation": "domain",
    "range_generalisation": "range",
    "existential_filler_generalisation": "filler",
    "interaction_redundant": "interactions",
}


@dataclass(frozen=True)
class GeneratedCase:
    """Evaluator-only supervision paired with a separate observable repair input."""

    case_id: str
    structural_parent: str
    family: str
    split: str
    problem: RepairInputV2
    probes: tuple[TeacherProbe, ...]
    intended_assignment: tuple[int, ...]
    generation_seed: int
    mirrored: bool
    origin: str = "generated"
    control: str = "corrupted"
    intended_theory: tuple[Any, ...] = ()
    intended_active: tuple[Any, ...] = ()
    variation: tuple[tuple[str, Any], ...] = ()
    ambiguity_group: str = ""


def generate_corpus(
    *,
    parents_per_family: int = 3,
    siblings_per_parent: int = 2,
    seed: int = 13,
    families: Iterable[str] = FAMILIES,
    heldout_families: Iterable[str] = (),
    coherent_controls: bool = False,
    missing_candidate_controls: bool = False,
    ambiguity_controls: bool = False,
    split_counts: Mapping[str, int] | None = None,
    score_noise: float = 0.1,
    feature_dropout: float = 0.0,
    misleading_label_fraction: float = 0.25,
) -> tuple[GeneratedCase, ...]:
    """Generate grouped siblings with connected core, evidence and composition variation.

    ``split_counts`` assigns exact parent counts before any corruption; otherwise
    stable hash splits are used. Template/composition holdouts always take priority.
    """
    families = tuple(families)
    if (
        parents_per_family < 1
        or siblings_per_parent < 1
        or not set(families) <= set(FAMILIES) | set(MECHANISMS)
    ):
        raise ValueError("Positive counts and known structural families are required")
    if score_noise < 0 or not 0 <= feature_dropout <= 1 or not 0 <= misleading_label_fraction <= 1:
        raise ValueError("Noise must be nonnegative and dropout must be in [0, 1]")
    if split_counts is not None:
        if set(split_counts) != {"train", "development", "test"} or any(
            type(count) is not int or count < 0 for count in split_counts.values()
        ):
            raise ValueError("Split counts require nonnegative train/development/test counts")
        parents_per_family = sum(split_counts.values())
        if not parents_per_family:
            raise ValueError("At least one structural parent is required")
    parents = {
        f"{family}:path-{depth}": family
        for family in families
        for depth in range(1, parents_per_family + 1)
    }
    splits = grouped_split(parents, seed=seed, heldout_families=heldout_families)
    if split_counts is not None:
        for family in families:
            ordered = sorted(
                (parent for parent, value in parents.items() if value == family),
                key=lambda parent: hashlib.sha256(f"{seed}:{parent}".encode()).digest(),
            )
            assignments = [split for split, count in split_counts.items() for _ in range(count)]
            for parent, split in zip(ordered, assignments):
                splits[parent] = "test" if family in heldout_families else split
    cases = tuple(
        _case(
            parent,
            family,
            splits[parent],
            depth,
            sibling,
            seed,
            score_noise,
            feature_dropout,
            misleading_label_fraction,
        )
        for parent, family in sorted(parents.items())
        for depth in (int(parent.rsplit("-", 1)[1]),)
        for sibling in range(siblings_per_parent)
    )
    controls = []
    for case in cases:
        if coherent_controls:
            controls.append(coherent_control(case))
        if missing_candidate_controls:
            objects = list(case.problem.objects)
            obj, choice = objects[0], case.intended_assignment[0]
            objects[0] = replace(
                obj, candidates=tuple(c for i, c in enumerate(obj.candidates) if i != choice)
            )
            controls.append(
                replace(
                    case,
                    case_id=case.case_id + ":missing-candidate",
                    control="missing_candidate",
                    problem=replace(case.problem, objects=tuple(objects)),
                    intended_assignment=tuple(-1 for _ in objects),
                )
            )
        if ambiguity_controls:
            # The same observation has two explicitly unresolved intentions. No
            # single target is selected or silently averaged in training.
            controls.append(
                replace(
                    case,
                    case_id=case.case_id + ":ambiguous",
                    control="ambiguous",
                    ambiguity_group=case.case_id,
                    intended_assignment=(),
                )
            )
    return cases + tuple(controls)


def coherent_control(case: GeneratedCase, *, case_id: str | None = None) -> GeneratedCase:
    """Use the verified intended input as observation, including source-side views."""
    objects = []
    sides = [set(case.problem.source_axioms), set(case.problem.target_axioms)]
    for obj, choice in zip(case.problem.objects, case.intended_assignment):
        chosen = obj.candidates[choice]
        keep = make_candidate(
            obj.object_id, chosen.axioms, ("keep",), active_expressions=chosen.active_expressions
        )
        alternatives = tuple(
            replace(
                c,
                cost_features=replacement_cost_features(
                    chosen.axioms,
                    c.axioms,
                    kind=obj.kind,
                    authorship=obj.authorship,
                    active_expressions=c.active_expressions,
                ),
            )
            for index, c in enumerate(obj.candidates)
            if index not in {0, choice}
            and (c.axioms, c.active_expressions) != (chosen.axioms, chosen.active_expressions)
        )
        objects.append(
            replace(obj, original_axioms=chosen.axioms, candidates=(keep, *alternatives))
        )
        if obj.kind == "ontology_axiom":
            for axioms in sides:
                if set(obj.original_axioms) <= axioms:
                    axioms.difference_update(obj.original_axioms)
                    axioms.update(chosen.axioms)
    return replace(
        case,
        case_id=case_id or case.case_id + ":coherent",
        control="coherent",
        problem=replace(
            case.problem,
            objects=tuple(objects),
            source_axioms=tuple(sorted(sides[0], key=owl.structural_hexdigest)),
            target_axioms=tuple(sorted(sides[1], key=owl.structural_hexdigest)),
        ),
        intended_assignment=tuple(0 for _ in objects),
    )


def _case(
    parent: str,
    family: str,
    split: str,
    depth: int,
    sibling: int,
    seed: int,
    score_noise: float = 0.1,
    feature_dropout: float = 0.0,
    misleading_label_fraction: float = 0.25,
) -> GeneratedCase:
    namespace = hashlib.sha256(f"{seed}:{parent}:{sibling}".encode()).hexdigest()[:16]
    rng = random.Random(f"observed-evidence:{seed}:{parent}:{sibling}")
    mirrored = bool(sibling % 2) ^ family.endswith("_target")
    mechanism = MECHANISMS.get(family, family)
    if mechanism == "mixed":
        return _mixed_case(
            parent,
            split,
            depth,
            sibling,
            seed,
            score_noise,
            feature_dropout,
            misleading_label_fraction,
        )
    prefix = f"urn:exact:generated:{namespace}:"

    def cls(name: str) -> Any:
        if mirrored:
            if name.endswith("_s"):
                name = name[:-2] + "_t"
            elif name.endswith("_t"):
                name = name[:-2] + "_s"
        return owl.Class(owl.IRI(prefix + name))

    def prop(name: str) -> Any:
        return owl.ObjectProperty(owl.IRI(prefix + name))

    sub = owl.SubClassOf

    def conjunction(*values):
        return owl.ObjectIntersectionOf(values)

    def disjoint(*values):
        return owl.DisjointClasses(values)

    def equivalent(a, b):
        return (sub(a, b), sub(b, a))

    objects: list[RevisionObjectV2] = []
    fixed: list[Any] = []
    probes: list[TeacherProbe] = []
    intended: list[int] = []

    def add_object(
        original: Iterable[Any],
        alternatives: Iterable[tuple[str, tuple[Any, ...], tuple[Any, ...]]],
        *,
        ontology: bool = False,
    ) -> None:
        index = len(objects)
        object_id = f"object-{index}"
        original = tuple(original)
        authored = "human" if (depth + sibling + index) % 2 else "generated"
        candidates: list[ReplacementCandidateV2] = [
            make_candidate(object_id, original, ("keep",)),
            make_candidate(
                object_id,
                (),
                ("delete",),
                cost_features=replacement_cost_features(
                    original,
                    (),
                    kind="ontology_axiom" if ontology else "mapping",
                    authorship=authored,
                ),
            ),
        ]
        for tag, axioms, active in alternatives:
            costs = replacement_cost_features(
                original,
                axioms,
                kind="ontology_axiom" if ontology else "mapping",
                authorship=authored,
                active_expressions=active,
            )
            candidates.append(
                make_candidate(
                    object_id, axioms, (tag,), active_expressions=active, cost_features=costs
                )
            )
        objects.append(
            RevisionObjectV2(
                object_id,
                "ontology_axiom" if ontology else "mapping",
                original,
                tuple(candidates),
                occurrence_id=f"occurrence-{index}" if ontology else "",
                authorship=authored,
                source="target" if mirrored else "source",
                source_entity=original[0].sub_class if not ontology else None,
                target_entity=original[0].super_class if not ontology else None,
            )
        )
        intended.append(len(candidates) - 1)

    if mechanism == "papers":
        paper, rejected, accepted, target_rejected, target_paper, submission, acceptance = map(
            cls,
            (
                "Paper_s",
                "Rejected_s",
                "Accepted_t",
                "Rejected_t",
                "Paper_t",
                "Submission_s",
                "Acceptance_s",
            ),
        )
        expression = acceptance
        for level in range(depth % 2):
            expression = owl.ObjectSomeValuesFrom(prop(f"hasDecision{level}"), expression)
        active = conjunction(paper, expression)
        fixed.extend(
            (
                sub(rejected, paper),
                sub(submission, active),
                disjoint(rejected, expression),
                sub(accepted, target_paper),
                sub(target_rejected, target_paper),
                disjoint(accepted, target_rejected),
                *equivalent(rejected, target_rejected),
            )
        )
        specialised = (sub(active, accepted), sub(accepted, paper))
        necessary = sub(accepted, expression)
        add_object(
            equivalent(paper, accepted),
            (
                ("retain_subsumption", (sub(accepted, paper),), ()),
                ("replace_endpoint", equivalent(paper, target_paper), ()),
                ("specialise_subclass", specialised, (active,)),
                ("add_necessary_condition", (sub(accepted, paper), necessary), ()),
                ("complex_equivalence", (*specialised, necessary), (active,)),
            ),
        )
        selected_tag = {
            "directional_strengthening": "retain_subsumption",
            "subclass_specialisation_source": "specialise_subclass",
            "subclass_specialisation_target": "specialise_subclass",
            "necessary_condition_source": "add_necessary_condition",
            "necessary_condition_target": "add_necessary_condition",
            "endpoint_confusion": "replace_endpoint",
        }.get(family)
        if selected_tag is not None:
            intended[-1] = next(
                i
                for i, candidate in enumerate(objects[-1].candidates)
                if selected_tag in candidate.action_tags
            )
        for index, axiom in enumerate(
            (
                sub(submission, accepted),
                sub(accepted, paper),
                necessary,
                sub(rejected, target_rejected),
                disjoint(accepted, target_rejected),
            )
        ):
            query_indices = {
                "directional_strengthening": {1, 3, 4},
                "subclass_specialisation_source": {0, 1, 3, 4},
                "subclass_specialisation_target": {0, 1, 3, 4},
                "necessary_condition_source": {1, 2, 3, 4},
                "necessary_condition_target": {1, 2, 3, 4},
                "endpoint_confusion": {3, 4},
            }.get(family, set(range(5)))
            if index in query_indices:
                probes.append(TeacherProbe(f"paper-{index}", axiom, "paper"))
        if family == "endpoint_confusion":
            probes.extend(
                (
                    TeacherProbe("endpoint-forward", sub(paper, target_paper), "paper"),
                    TeacherProbe("endpoint-reverse", sub(target_paper, paper), "paper"),
                )
            )
    elif mechanism in {"overlap", "disjointness", "conjunct"}:
        author, reviewer, overlap, target_author, target_reviewer = map(
            cls, ("Author_s", "Reviewer_s", "Overlap_s", "Author_t", "Reviewer_t")
        )
        source_axiom = sub(overlap, conjunction(author, reviewer))
        target_axiom = disjoint(target_author, target_reviewer)
        if mechanism == "overlap":
            fixed.extend((source_axiom, target_axiom))
            add_object(
                equivalent(author, target_author),
                (("retain_subsumption", (sub(target_author, author),), ()),),
            )
            add_object(
                equivalent(reviewer, target_reviewer),
                (("retain_subsumption", (sub(target_reviewer, reviewer),), ()),),
            )
        elif mechanism == "disjointness":
            fixed.extend(
                (
                    source_axiom,
                    *equivalent(author, target_author),
                    *equivalent(reviewer, target_reviewer),
                )
            )
            # N-ary disjointness has a retained independent commitment after one pair edit.
            unrelated = cls("Unrelated_t")
            original = disjoint(target_author, target_reviewer, unrelated)
            retained = (disjoint(target_author, unrelated), disjoint(target_reviewer, unrelated))
            add_object((original,), (("remove_disjointness", retained, ()),), ontology=True)
            probes.append(TeacherProbe("retained_disjointness", retained[0], "roles"))
        else:
            fixed.extend(
                (
                    target_axiom,
                    *equivalent(author, target_author),
                    *equivalent(reviewer, target_reviewer),
                )
            )
            add_object(
                (source_axiom,),
                (("remove_superclass_conjunct", (sub(overlap, author),), ()),),
                ontology=True,
            )
        probes.extend(
            (
                TeacherProbe("author", sub(target_author, author), "roles"),
                TeacherProbe("reviewer", sub(target_reviewer, reviewer), "roles"),
            )
        )
    elif mechanism == "participant":
        participant, speaker, listener, presenting, person, target_listener, target_speaker = map(
            cls,
            (
                "Participant_s",
                "Speaker_s",
                "Listener_s",
                "Presenting_s",
                "Person_s",
                "Listener_t",
                "Speaker_t",
            ),
        )
        fixed.extend(
            (
                sub(listener, participant),
                sub(presenting, participant),
                disjoint(listener, presenting),
                sub(speaker, person),
                *equivalent(listener, target_listener),
                *equivalent(speaker, target_speaker),
                disjoint(target_listener, target_speaker),
            )
        )
        active = conjunction(participant, presenting)
        add_object(
            (sub(participant, speaker),),
            (
                ("generalise_superclass", (sub(participant, person),), ()),
                ("specialise_ontology_subclass", (sub(active, speaker),), (active,)),
            ),
            ontology=True,
        )
        if family == "superclass_generalisation":
            intended[-1] = 2
        probes.append(
            TeacherProbe(
                "presenter",
                (
                    sub(participant, person)
                    if family == "superclass_generalisation"
                    else sub(presenting, speaker)
                ),
                "typing",
            )
        )
    elif mechanism in {"domain", "range", "filler"}:
        witness, software, person, agent, target_software, target_person, review = map(
            cls,
            (
                "Witness_s",
                "Software_s",
                "Person_s",
                "Agent_s",
                "Software_t",
                "Person_t",
                "Review_s",
            ),
        )
        relation = prop("writes" if mechanism == "domain" else "writtenBy")
        restriction = owl.ObjectSomeValuesFrom(
            relation, review if mechanism == "domain" else software
        )
        asserted = sub(
            witness, conjunction(software, restriction) if mechanism == "domain" else restriction
        )
        constraint = owl.ObjectPropertyDomain if mechanism == "domain" else owl.ObjectPropertyRange
        fixed.extend(
            (
                sub(person, agent),
                sub(software, agent),
                *equivalent(person, target_person),
                *equivalent(software, target_software),
                disjoint(target_person, target_software),
                asserted,
            )
        )
        if mechanism == "filler":
            fixed.remove(asserted)
            fixed.append(constraint(relation, person))
            replacement = sub(witness, owl.ObjectSomeValuesFrom(relation, agent))
            add_object(
                (asserted,), (("generalise_existential_filler", (replacement,), ()),), ontology=True
            )
            probes.append(TeacherProbe("broad_filler", replacement, "typing"))
        else:
            add_object(
                (constraint(relation, person),),
                ((f"generalise_{mechanism}", (constraint(relation, agent),), ()),),
                ontology=True,
            )
        if mechanism != "filler":
            probes.append(TeacherProbe("witness", asserted, "typing"))
    else:
        a, b, c, d, e = map(cls, "ABCDE")
        fixed.extend((sub(c, a), disjoint(a, b), owl.Declaration(d), owl.Declaration(e)))
        if mechanism == "interaction_complementary":
            fixed.append(sub(d, a))
        for index in range(2):
            start_class = d if mechanism == "interaction_complementary" and index == 1 else c
            end_class = d if mechanism == "interaction_complementary" and index == 0 else e
            add_object(
                (sub(start_class, b),), (("replace_endpoint", (sub(start_class, end_class),), ()),)
            )
        probes.append(TeacherProbe("shared_benefit", sub(c, e), "shared"))
    # Replace core inclusions by paths and redundant branches. Every inserted
    # path participates in the original conflict or intended consequence; the
    # parent distinction is not just a renamed fixture plus disconnected noise.
    enriched: list[Any] = []
    path_length = depth
    branch_count = 1 + (depth - 1) % 3
    overlap_count = (depth - 1) % 2
    generalisation_pairs = {
        (before, after)
        for obj in objects
        for candidate in obj.candidates
        if any(tag.startswith("generalise_") for tag in candidate.action_tags)
        for original in obj.original_axioms
        for emitted in candidate.axioms
        for before in owl.signature(original)
        for after in owl.signature(emitted)
        if isinstance(before, owl.Class) and isinstance(after, owl.Class)
    }
    for edge, axiom in enumerate(fixed):
        if not isinstance(axiom, owl.SubClassOf) or not isinstance(axiom.sub_class, owl.Class):
            enriched.append(axiom)
            continue
        if (axiom.sub_class, axiom.super_class) in generalisation_pairs:
            # Certified generalisation uses this exact fixed asserted premise.
            enriched.append(axiom)
        branches = branch_count if edge % 2 == 0 else 1
        named_sides = {
            side
            for entity in owl.signature(axiom)
            if isinstance(entity, owl.Class)
            for side in ("s", "t")
            if entity.iri.value.endswith("_" + side)
        }
        side_suffix = "_" + next(iter(named_sides)) if len(named_sides) == 1 else ""
        if mirrored and side_suffix:
            side_suffix = "_t" if side_suffix == "_s" else "_s"
        shared_tail = (
            cls(f"Core{edge}_join{side_suffix}") if overlap_count and branches > 1 else None
        )
        for branch in range(branches):
            previous = axiom.sub_class
            for step in range(path_length - 1):
                node = cls(f"Core{edge}_branch{branch}_step{step}{side_suffix}")
                enriched.append(sub(previous, node))
                previous = node
            if shared_tail is not None:
                enriched.append(sub(previous, shared_tail))
                previous = shared_tail
            enriched.append(sub(previous, axiom.super_class))
    fixed = list(dict.fromkeys(enriched))
    # A separate nuisance axis remains explicit instead of defining the parent.
    for index in range((depth - 1) % 3):
        fixed.append(sub(cls(f"Noise{index}"), cls(f"Noise{index + 1}")))
    signature = {
        node
        for axiom in (*fixed, *(a for obj in objects for a in obj.original_axioms))
        for node in owl.walk(axiom)
        if isinstance(node, owl.Class) and node != owl.OWL_NOTHING
    }
    evidence_rows = []
    for obj in objects:
        # Label overlap drives scores; neither correctness nor the intended index
        # is an input to this simulator. Calibration/noise/dropout vary separately.
        tokens = ["topic", f"entity{rng.randrange(4)}", f"role{rng.randrange(3)}"]
        other = tokens[: rng.randrange(1, 4)] + [f"entity{rng.randrange(4)}"]
        if rng.random() < misleading_label_fraction:
            other = [f"unrelated{rng.randrange(8)}", f"misleading{rng.randrange(8)}"]
        lexical = len(set(tokens) & set(other)) / len(set(tokens) | set(other))
        calibration = (0.7, 1.0, 1.3)[(depth - 1) % 3]
        score = min(1.0, max(0.0, calibration * lexical + rng.gauss(0.0, score_noise)))
        channels = {"lexical": lexical, "structural": rng.random(), "semantic": rng.random()}
        evidence_rows.append(
            (
                obj.object_id,
                {
                    "score": score,
                    "matcher": "label-jaccard-plus-noise",
                    "source_text": " ".join(tokens),
                    "target_text": " ".join(other),
                    "description": "observed topic and role vocabulary",
                    "channels": {
                        key: value
                        for key, value in channels.items()
                        if rng.random() >= feature_dropout
                    },
                    "provenance": obj.authorship,
                },
            )
        )
    evidence = tuple(evidence_rows)
    partitioned: dict[str, list[Any]] = {"s": [], "t": []}
    for axiom in (
        *fixed,
        *(a for obj in objects if obj.kind == "ontology_axiom" for a in obj.original_axioms),
    ):
        sides = {
            side
            for node in owl.signature(axiom)
            if isinstance(node, owl.Class)
            for side in ("s", "t")
            if node.iri.value.endswith("_" + side)
        }
        if len(sides) == 1:
            partitioned[next(iter(sides))].append(axiom)
    problem = RepairInputV2(
        tuple(fixed),
        tuple(objects),
        PolicyV2(tuple(sorted(signature, key=owl.structural_hexdigest))),
        matcher_identity="synthetic-noisy",
        evidence=evidence,
        source_axioms=tuple(partitioned["s"]),
        target_axioms=tuple(partitioned["t"]),
    )
    return GeneratedCase(
        f"{parent}:sibling-{sibling}",
        parent,
        family,
        split,
        problem,
        tuple(probes),
        tuple(intended),
        seed,
        mirrored,
        intended_theory=tuple(fixed)
        + tuple(a for obj, choice in zip(objects, intended) for a in obj.candidates[choice].axioms),
        intended_active=tuple(
            e
            for obj, choice in zip(objects, intended)
            for e in obj.candidates[choice].active_expressions
        ),
        variation=(
            ("path_length", path_length),
            ("branch_count", branch_count),
            ("explanation_overlap", overlap_count),
            ("expression_depth", 1 + depth % 2),
            ("score_noise", score_noise),
            ("feature_dropout", feature_dropout),
            ("misleading_label_fraction", misleading_label_fraction),
        ),
    )


def _mixed_case(
    parent,
    split,
    depth,
    sibling,
    seed,
    score_noise,
    feature_dropout,
    misleading_label_fraction=0.25,
) -> GeneratedCase:
    """Compose two mechanisms with a shared, satisfiable interface class."""
    left = _case(
        parent + ":left",
        "papers",
        split,
        depth,
        sibling,
        seed,
        score_noise,
        feature_dropout,
        misleading_label_fraction,
    )
    right = _case(
        parent + ":right",
        "range",
        split,
        depth,
        sibling,
        seed,
        score_noise,
        feature_dropout,
        misleading_label_fraction,
    )
    objects: list[RevisionObjectV2] = []
    evidence: list[tuple[str, Any]] = []
    for part_index, part in enumerate((left, right)):
        for obj in part.problem.objects:
            object_id = f"part{part_index}:{obj.object_id}"
            objects.append(
                replace(
                    obj,
                    object_id=object_id,
                    occurrence_id=(
                        f"part{part_index}:{obj.occurrence_id}" if obj.occurrence_id else ""
                    ),
                    candidates=tuple(replace(c, object_id=object_id) for c in obj.candidates),
                )
            )
        evidence.extend((f"part{part_index}:{key}", value) for key, value in part.problem.evidence)
    shared = owl.Class(owl.IRI("urn:exact:mixed:" + hashlib.sha256(parent.encode()).hexdigest()))
    interface = tuple(
        owl.SubClassOf(part.problem.policy.monitored_classes[0], shared) for part in (left, right)
    )
    fixed = (*left.problem.fixed_axioms, *right.problem.fixed_axioms, *interface)
    problem = RepairInputV2(
        fixed,
        tuple(objects),
        PolicyV2(
            (
                *left.problem.policy.monitored_classes,
                *right.problem.policy.monitored_classes,
                shared,
            )
        ),
        matcher_identity="label-jaccard-plus-noise",
        evidence=tuple(evidence),
        source_axioms=left.problem.source_axioms + right.problem.source_axioms,
        target_axioms=left.problem.target_axioms + right.problem.target_axioms,
    )
    probes = tuple(
        replace(probe, probe_id=f"part{index}:{probe.probe_id}")
        for index, part in enumerate((left, right))
        for probe in part.probes
    )
    return GeneratedCase(
        f"{parent}:sibling-{sibling}",
        parent,
        "mixed",
        split,
        problem,
        probes,
        left.intended_assignment + right.intended_assignment,
        seed,
        left.mirrored,
        intended_theory=(*left.intended_theory, *right.intended_theory, *interface),
        intended_active=left.intended_active + right.intended_active,
        variation=left.variation + (("composition", "papers+range"),),
    )
