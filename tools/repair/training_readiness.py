"""Fail closed on sparse weak supervision before either paired primary fit."""

from collections import Counter, defaultdict

from exact.repair.records import canonical_hash
from exact.repair.semantic_fidelity import read_fidelity_training_artifact
from tools.repair.shared_release import authenticate, bound

SCHEMA = "exact-repair/training-data-readiness/v1"
FAMILIES = (
    "conjunct",
    "disjointness",
    "domain",
    "filler",
    "overlap",
    "papers",
    "participant",
    "range",
)
# Spec 16 targets independent parents, rather than a large count of correlated
# comparisons. Freeze this campaign's acceptance rule before collecting more.
REQUIREMENTS = dict(
    families=list(FAMILIES),
    minimum_parents_per_family=4,
    controls=["coherent", "corrupted"],
    minimum_non_tie_parents=4,
)


def assess(declarations, labels):
    rows = {r["case_id"]: r for r in declarations}
    if len(rows) != len(declarations) or any(r["split"] != "train" for r in rows.values()):
        raise ValueError("Readiness requires unique frozen TRAIN declarations")
    parents, controls, non_tie = defaultdict(set), defaultdict(set), set()
    counts, seen = Counter(), set()
    for case_id, values in labels.items():
        if case_id not in rows:
            raise ValueError("Weak labels outside frozen TRAIN membership")
        declaration = rows[case_id]
        for packet, aggregate in values:
            if (
                packet.case_id != case_id
                or packet.split != "train"
                or packet.parent_group_id != declaration["structural_parent"]
                or not packet.eligible
                or not aggregate.global_target_eligible
            ):
                raise ValueError("Unqualified or misbound weak supervision")
            key = (case_id, tuple(sorted((packet.plan_a.plan_id, packet.plan_b.plan_id))))
            if key in seen:
                raise ValueError("Duplicate plan comparison cannot increase readiness")
            seen.add(key)
            family, parent = declaration["family"], declaration["structural_parent"]
            parents[family].add(parent)
            controls[family].add(declaration["control"])
            counts[family] += 1
            if aggregate.decision in {"A", "B"}:
                non_tie.add(parent)
    failures = []
    for family in FAMILIES:
        if len(parents[family]) < REQUIREMENTS["minimum_parents_per_family"]:
            failures.append(f"{family}: fewer than four independent labelled TRAIN parents")
        if controls[family] != set(REQUIREMENTS["controls"]):
            failures.append(f"{family}: missing coherent/corrupted weak-label coverage")
    if len(non_tie) < REQUIREMENTS["minimum_non_tie_parents"]:
        failures.append("Fewer than four independent parents with a valid non-tie contrast")
    return dict(
        requirements=REQUIREMENTS,
        ready=not failures,
        failures=failures,
        valid_unique_comparisons=len(seen),
        independent_non_tie_parents=len(non_tie),
        families={
            f: dict(
                comparisons=counts[f],
                independent_parents=len(parents[f]),
                controls=sorted(controls[f]),
            )
            for f in FAMILIES
        },
    )


def require_ready(reference, labels_reference, cases):
    if not reference:
        raise ValueError("Primary training requires a measured data-readiness receipt")
    receipt = bound(reference)
    declarations = [
        dict(
            case_id=c.case_id,
            split=c.split,
            family=c.family,
            control=c.control,
            structural_parent=c.structural_parent,
        )
        for c in cases
        if c.split == "train"
    ]
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("labels") != labels_reference
        or receipt.get("train_declarations_hash") != canonical_hash(declarations)
    ):
        raise ValueError("Training-readiness data dependencies changed")
    measured = assess(
        declarations, read_fidelity_training_artifact(authenticate(labels_reference), "train")
    )
    if receipt.get("measurement") != measured or not measured["ready"]:
        raise ValueError("Primary training blocked: inadequate validated TRAIN coverage")
    return measured
