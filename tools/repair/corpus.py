"""Small deterministic structural parents for conformance and generated pretraining.

Splits are assigned to clean structural parents before namespaces, noisy scores,
provenance and corrupted observations are generated. Teacher queries/intended
assignments stay in this evaluator record, outside RepairInputV2 observations.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from typing import Any, Iterable

import pyowl_core as owl

from exact.repair.candidates import make_candidate
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
)


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


def generate_corpus(
    *,
    parents_per_family: int = 3,
    siblings_per_parent: int = 2,
    seed: int = 13,
    families: Iterable[str] = FAMILIES,
    heldout_families: Iterable[str] = (),
    coherent_controls: bool = False,
    missing_candidate_controls: bool = False,
) -> tuple[GeneratedCase, ...]:
    """Generate finite sibling groups, varying hierarchy length and ontology orientation."""
    families = tuple(families)
    if parents_per_family < 1 or siblings_per_parent < 1 or not set(families) <= set(FAMILIES):
        raise ValueError("Positive counts and known structural families are required")
    parents = {
        f"{family}:path-{depth}": family
        for family in families
        for depth in range(1, parents_per_family + 1)
    }
    splits = grouped_split(parents, seed=seed, heldout_families=heldout_families)
    cases = tuple(
        _case(parent, family, splits[parent], depth, sibling, seed)
        for parent, family in sorted(parents.items())
        for depth in (int(parent.rsplit("-", 1)[1]),)
        for sibling in range(siblings_per_parent)
    )
    controls = []
    for case in cases:
        if coherent_controls:
            objects = []
            for obj, choice in zip(case.problem.objects, case.intended_assignment):
                chosen = obj.candidates[choice]
                keep = make_candidate(obj.object_id, chosen.axioms, ("keep",))
                alternatives = tuple(
                    c
                    for index, c in enumerate(obj.candidates)
                    if index not in {0, choice} and c.axioms != chosen.axioms
                )
                objects.append(
                    replace(obj, original_axioms=chosen.axioms, candidates=(keep, *alternatives))
                )
            controls.append(
                replace(
                    case,
                    case_id=case.case_id + ":coherent",
                    control="coherent",
                    problem=replace(case.problem, objects=tuple(objects)),
                    intended_assignment=tuple(0 for _ in objects),
                )
            )
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
    return cases + tuple(controls)


def _case(
    parent: str, family: str, split: str, depth: int, sibling: int, seed: int
) -> GeneratedCase:
    namespace = hashlib.sha256(f"{seed}:{parent}:{sibling}".encode()).hexdigest()[:16]
    rng = random.Random(f"observed-evidence:{seed}:{parent}:{sibling}")
    mirrored = bool(sibling % 2)
    prefix = f"urn:exact:generated:{namespace}:"

    def cls(name: str) -> Any:
        if mirrored:
            name = name.replace("_s", "_mirror").replace("_t", "_s").replace("_mirror", "_t")
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
                cost_features=(
                    (
                        ("delete", 1.0),
                        ("ontology_edit", 1.0),
                        ("human_authored_ontology_edit", float(authored == "human")),
                    )
                    if ontology
                    else (("delete", 1.0),)
                ),
            ),
        ]
        for tag, axioms, active in alternatives:
            costs = [("edit", 1.0)]
            if ontology:
                costs.extend(
                    (
                        ("ontology_edit", 1.0),
                        ("human_authored_ontology_edit", float(authored == "human")),
                    )
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

    if family == "papers":
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
        expression = owl.ObjectSomeValuesFrom(prop("hasDecision"), acceptance)
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
        for index, axiom in enumerate(
            (
                sub(submission, accepted),
                sub(accepted, paper),
                necessary,
                sub(rejected, target_rejected),
                disjoint(accepted, target_rejected),
            )
        ):
            probes.append(TeacherProbe(f"paper-{index}", axiom, "paper"))
    elif family in {"overlap", "disjointness", "conjunct"}:
        author, reviewer, overlap, target_author, target_reviewer = map(
            cls, ("Author_s", "Reviewer_s", "Overlap_s", "Author_t", "Reviewer_t")
        )
        source_axiom = sub(overlap, conjunction(author, reviewer))
        target_axiom = disjoint(target_author, target_reviewer)
        if family == "overlap":
            fixed.extend((source_axiom, target_axiom))
            add_object(
                equivalent(author, target_author),
                (("retain_subsumption", (sub(target_author, author),), ()),),
            )
            add_object(
                equivalent(reviewer, target_reviewer),
                (("retain_subsumption", (sub(target_reviewer, reviewer),), ()),),
            )
        elif family == "disjointness":
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
    elif family == "participant":
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
        probes.append(TeacherProbe("presenter", sub(presenting, speaker), "typing"))
    elif family in {"domain", "range", "filler"}:
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
        relation = prop("writes" if family == "domain" else "writtenBy")
        restriction = owl.ObjectSomeValuesFrom(relation, review if family == "domain" else software)
        asserted = sub(
            witness, conjunction(software, restriction) if family == "domain" else restriction
        )
        constraint = owl.ObjectPropertyDomain if family == "domain" else owl.ObjectPropertyRange
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
        if family == "filler":
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
                ((f"generalise_{family}", (constraint(relation, agent),), ()),),
                ontology=True,
            )
        if family != "filler":
            probes.append(TeacherProbe("witness", asserted, "typing"))
    else:
        a, b, c, d, e = map(cls, "ABCDE")
        fixed.extend((sub(c, a), disjoint(a, b), owl.Declaration(d), owl.Declaration(e)))
        for _ in range(2):
            add_object((sub(c, b),), (("replace_endpoint", (sub(d, e),), ()),))
        probes.append(TeacherProbe("shared_benefit", sub(d, e), "shared"))
    # Distinct clean parent structures use different asserted hierarchy lengths;
    # renamed/mirrored siblings always retain the same parent/split identity.
    for index in range(depth):
        fixed.append(sub(cls(f"Context{index}"), cls(f"Context{index + 1}")))
    signature = {
        node
        for axiom in (*fixed, *(a for obj in objects for a in obj.original_axioms))
        for node in owl.walk(axiom)
        if isinstance(node, owl.Class) and node != owl.OWL_NOTHING
    }
    evidence = tuple(
        (
            obj.object_id,
            {
                "score": rng.random(),
                "matcher": "synthetic-noisy",
                "channels": {
                    "lexical": rng.random(),
                    "structural": rng.random(),
                    "semantic": rng.random(),
                },
                "provenance": obj.authorship,
            },
        )
        for obj in objects
    )
    problem = RepairInputV2(
        tuple(fixed),
        tuple(objects),
        PolicyV2(tuple(sorted(signature, key=owl.structural_hexdigest))),
        matcher_identity="synthetic-noisy",
        evidence=evidence,
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
    )
