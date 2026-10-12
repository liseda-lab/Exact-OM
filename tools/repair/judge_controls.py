"""Independent authored controls for the frozen four-model judge pilot.

No study parent, label, checkpoint, DEV repair or held-out TEST case is read.
Gold is a consequence of the fictional definitions, not another LLM's answer.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pyowl_core as owl

from exact.repair.candidates import make_candidate, mapping_candidates
from exact.repair.kernel import materialize, verify_assignment
from exact.repair.learning import OwlTeacherOracle, TeacherProbe
from exact.repair.owl import OwlVerifier, snapshot_from_axioms
from exact.repair.records import (
    PolicyV2,
    RepairInputV2,
    RevisionObjectV2,
    canonical_hash,
    freeze_public_policy,
    promote_input_v3,
)
from exact.repair.semantic_fidelity import (
    SemanticConsequenceReportV3,
    consequence_basis_from_probes,
    semantic_plan_from_verification,
)
from tools.repair.grounded_supervision import render
from tools.repair.historical_regression import binding
from tools.repair.judge_pilot import CRITERIA
from tools.repair.shared_release import immutable

NAMESPACE = "isolated_judge_qualification_20261012"


def controls(directory, *, background_count=1600, names=None):
    """Build full theories and native consequence/non-vacuity evidence offline."""
    directory = Path(directory)
    result = []
    for name in names or ("direction", "meaning_loss", "range_context", "tie", "insufficient"):
        scenario = {
            "unsupported_qualifier": "c901",
            "ontology_collateral": "c902",
            "partial_evidence": "c903",
        }.get(name, name)
        iri = "urn:" + NAMESPACE + ":" + scenario + ":"
        cls = lambda label: owl.Class(owl.IRI(iri + label))
        a, b = (
            (cls("ConceptX"), cls("ConceptY"))
            if name == "insufficient"
            else (
                (cls("ResearchPaper"), cls("ResearchArticle"))
                if name == "meaning_loss"
                else (cls("AcceptedSubmission"), cls("Submission"))
            )
        )
        candidates = mapping_candidates("mapping", a, b, "=")
        obj = RevisionObjectV2(
            "mapping", "mapping", candidates[0].axioms, candidates, source_entity=a, target_entity=b
        )
        forward = next(i for i, c in enumerate(candidates) if c.axioms == (owl.SubClassOf(a, b),))
        delete = next(i for i, c in enumerate(candidates) if not c.axioms)
        assignments = [(forward,), (0,)]
        fixed: tuple[owl.AxiomNode, ...] = ()
        monitored: tuple[owl.Class, ...] = (a, b)
        probes = (
            TeacherProbe("forward", owl.SubClassOf(a, b), "meaning"),
            TeacherProbe("reverse", owl.SubClassOf(b, a), "meaning"),
        )
        definition = (
            "AcceptedSubmission consists exactly of accepted submissions. Submission "
            "includes accepted and rejected submissions. Rejected submissions exist. "
            "The supported focal relation is AcceptedSubmission SubClassOf Submission; "
            "its reverse would wrongly classify rejected submissions as accepted."
        )
        preference = "A"
        allowed = {k: dict(a=["full"], b=["full"]) for k in CRITERIA}
        allowed["assertion_fidelity"]["b"] = ["limited", "mixed", "mostly"]
        # The unsupported reverse direction is indisputable. Whether to count
        # that same commitment again as retention/collateral damage is a rubric
        # judgment: do not reject a defensible reading as a factual error.
        allowed["meaning_retention"]["b"] = ["mixed", "mostly", "full"]
        allowed["collateral_fidelity"]["b"] = ["limited", "mixed", "mostly", "full"]
        if name == "meaning_loss":
            definition = (
                "ResearchPaper and ResearchArticle are exact synonyms in this fictional "
                "catalogue: the same set of original research manuscripts. Both directions "
                "are independently supported focal meaning. No other knowledge is in scope."
            )
            assignments, preference = [(delete,), (0,)], "B"
            allowed = {k: dict(a=["full"], b=["full"]) for k in CRITERIA}
            allowed["meaning_retention"]["a"] = ["none"]
        elif name == "range_context":
            paper, person, agent = cls("Paper"), cls("Person"), cls("Agent")
            prop = owl.ObjectProperty(owl.IRI(iri + "authoredBy"))
            axiom = owl.ObjectPropertyRange(prop, person)
            candidates = (
                make_candidate("range", (axiom,), ("keep",)),
                make_candidate("range", (), ("delete",)),
            )
            obj = RevisionObjectV2(
                "range",
                "ontology_axiom",
                (axiom,),
                candidates,
                occurrence_id="authored-catalogue/range",
            )
            fixed = (
                owl.SubClassOf(person, agent),
                owl.ObjectPropertyDomain(prop, paper),
                *(
                    owl.Declaration(cls(f"BackgroundConcept{i:04d}"))
                    for i in range(background_count)
                ),
            )
            monitored, assignments = (paper, person, agent), [(0,), (1,)]
            probes = (
                TeacherProbe("intended_range", axiom, "meaning"),
                TeacherProbe("intended_domain", owl.ObjectPropertyDomain(prop, paper), "meaning"),
            )
            definition = (
                "In this fictional catalogue, authoredBy relates only papers to human "
                "persons. Person is a kind of Agent. Retain both Paper domain and Person "
                "range as focal meaning. BackgroundConcept declarations are isolated names: "
                "they impose no relationship or extra semantic commitment, but all of them "
                "are supplied as fixed context. No text or axiom is omitted."
            )
            allowed = {k: dict(a=["full"], b=["full"]) for k in CRITERIA}
            allowed["meaning_retention"]["b"] = ["limited", "mixed", "mostly"]
        elif name == "tie":
            assignments, preference = [(forward,), (forward,)], "tie"
            allowed = {k: dict(a=["full"], b=["full"]) for k in CRITERIA}
        elif name == "insufficient":
            definition = (
                "ConceptX and ConceptY are opaque catalogue identifiers. No definitions, "
                "examples, intended relationship, or independent domain evidence is available. "
                "The observed mapping is untrusted; logical feasibility is not meaning evidence."
            )
            preference = "abstain"
            allowed = {k: dict(a=["unknown"], b=["unknown"]) for k in CRITERIA}
        elif name == "unsupported_qualifier":
            a, b, english = cls("AccreditedResearcher"), cls("Researcher"), cls("English")
            speaks = owl.ObjectProperty(owl.IRI(iri + "speaks"))
            fixed = tuple(owl.Declaration(x) for x in (a, b, english, speaks))
            qualifier = owl.ObjectSomeValuesFrom(speaks, english)
            good = owl.SubClassOf(a, b)
            bad = owl.SubClassOf(a, owl.ObjectIntersectionOf(owl.CanonicalSet((b, qualifier))))
            candidates = (
                make_candidate("mapping", (good,), ("keep",)),
                make_candidate("mapping", (bad,), ("qualify",)),
            )
            obj = RevisionObjectV2(
                "mapping", "mapping", (good,), candidates, source_entity=a, target_entity=b
            )
            assignments, monitored = [(0,), (1,)], (a, b, english)
            probes = (
                TeacherProbe("focal", good, "meaning"),
                TeacherProbe("qualifier", owl.SubClassOf(a, qualifier), "meaning"),
            )
            definition = (
                "AccreditedResearcher is a kind of Researcher in this fictional registry. "
                "Accreditation imposes no language requirement. Some accredited researchers "
                "speak only Mandarin and do not speak English. speaks relates a person to "
                "a language they speak, and English denotes the English language. The focal "
                "meaning is the researcher classification. No other knowledge is in scope."
            )
            allowed = {k: dict(a=["full"], b=["full"]) for k in CRITERIA}
            allowed["assertion_fidelity"]["b"] = ["limited", "mixed", "mostly"]
            allowed["collateral_fidelity"]["b"] = ["limited", "mixed", "mostly", "full"]
        elif name == "ontology_collateral":
            valve, sensor = cls("SafetyValve"), cls("Sensor")
            focal = owl.SubClassOf(a, b)
            other = owl.DisjointClasses(owl.CanonicalSet((valve, sensor)))
            candidates = (
                make_candidate("separation", (other,), ("keep",)),
                make_candidate("separation", (), ("delete",)),
            )
            obj = RevisionObjectV2(
                "separation",
                "ontology_axiom",
                (other,),
                candidates,
                occurrence_id="authored/separation",
            )
            fixed = (focal, owl.Declaration(valve), owl.Declaration(sensor))
            monitored, assignments = (a, b, valve, sensor), [(0,), (1,)]
            probes = (
                TeacherProbe("focal", focal, "meaning"),
                TeacherProbe("separation", other, "meaning"),
            )
            definition = (
                "AcceptedSubmission is a kind of Submission; this is the focal classification. "
                "Separately, this fictional equipment catalogue defines SafetyValve and Sensor "
                "as mutually exclusive categories: no device can belong to both. Preserving "
                "that exclusion is the only other knowledge assessed by collateral fidelity. "
                "Both device categories are populated. No additional knowledge is in scope."
            )
            allowed = {k: dict(a=["full"], b=["full"]) for k in CRITERIA}
            allowed["collateral_fidelity"]["b"] = ["none"]
        elif name == "partial_evidence":
            unknown = cls("OpaqueCategory")
            fixed = (owl.SubClassOf(b, unknown),)
            monitored, assignments, preference = (a, b, unknown), [(forward,), (delete,)], "abstain"
            probes = (
                TeacherProbe("focal", owl.SubClassOf(a, b), "meaning"),
                TeacherProbe("collateral", owl.SubClassOf(a, unknown), "meaning"),
            )
            definition = (
                "AcceptedSubmission is a kind of Submission in this fictional registry. "
                "This is the focal relation for meaning retention and assertion fidelity. "
                "The fixed Submission-to-OpaqueCategory assertion is fallible context. "
                "Collateral fidelity must assess the effect on the OpaqueCategory classification, "
                "but no definition, scope note, example or intended status of OpaqueCategory "
                "is available. Logical query results alone cannot establish that meaning."
            )
            allowed = {k: dict(a=["full"], b=["full"]) for k in CRITERIA}
            allowed["meaning_retention"]["b"] = ["none"]
            allowed["collateral_fidelity"] = dict(a=["unknown"], b=["unknown"])
        elif name not in ("direction", "meaning_loss", "range_context", "tie"):
            raise ValueError("Unknown authored control: " + name)
        problem = promote_input_v3(RepairInputV2(fixed, (obj,), PolicyV2(monitored)))
        problem = replace(problem, policy=freeze_public_policy(problem))
        basis = consequence_basis_from_probes(probes)
        plans, native = [], []
        evidence: dict[str, dict[str, Any]] = dict(
            D1=dict(
                kind="authored_definition",
                text=definition,
                source_id=NAMESPACE + "/" + scenario,
                release="v1",
            )
        )
        for assignment in assignments:
            report = verify_assignment(problem, assignment)
            if not report.authorizes:
                raise ValueError("Control failed full native verification: " + name)
            theory, _ = materialize(problem, assignment)
            oracle = OwlTeacherOracle(
                OwlVerifier("hermit", backend="native", timeout_seconds=30),
                snapshot_from_axioms(theory),
                probes,
            )
            outcomes = {}
            for probe in probes:
                value = oracle.entails(probe.axiom)
                nonvac = [oracle.satisfiable(c) for c in probe.conditions()]
                if value is None or not all(x is True for x in nonvac):
                    raise ValueError(
                        "Control query/non-vacuity unavailable: "
                        + name
                        + "/"
                        + probe.probe_id
                        + f" (entailed={value}, nonvacuity={nonvac})"
                    )
                outcomes[probe.probe_id] = dict(
                    status=str(value).lower(), complete=True, nonvacuity="pass"
                )
            consequence = SemanticConsequenceReportV3(
                report.theory_hash,
                report.policy_hash,
                canonical_hash(basis),
                outcomes,
                "hermit-0.2/native/complete-authored-controls",
            )
            plan = semantic_plan_from_verification(
                problem, assignment, report, consequence_basis=basis, consequence_report=consequence
            )
            if not plan.eligible:
                raise ValueError("Incomplete control plan")
            plans.append(
                dict(
                    plan_id=plan.plan_id,
                    content=list(plan.content),
                    query_outcomes=dict(plan.query_outcomes),
                    theory_hash=plan.theory_hash,
                    policy_hash=plan.policy_hash,
                    query_basis_hash=plan.query_basis_hash,
                    verification_status=plan.verification_status,
                )
            )
            native.append(
                dict(
                    verification=report.to_dict(),
                    consequence=consequence.to_dict(),
                    plan=plan.to_dict(),
                    assignment=assignment,
                )
            )
            for probe in probes:
                key = "F" + str(len(evidence))
                evidence[key] = dict(
                    kind="native_query",
                    plan_id=plan.plan_id,
                    query=render((probe.axiom,)),
                    **outcomes[probe.probe_id],
                )
        packet = dict(
            namespace=NAMESPACE,
            study_split=None,
            case=name,
            native_verified=True,
            problem=problem.to_dict(),
            native=native,
            gold=dict(
                preference=preference,
                allowed_anchors=allowed,
                source="Independent explicit fictional construction; frozen before any model call",
            ),
            payload=dict(
                task="Assess declared intended meaning using all evidence and complete theories.",
                evidence=evidence,
                plan_a=plans[0],
                plan_b=plans[1],
                coverage=dict(complete=True, omissions=[]),
            ),
        )
        path = directory / (name + ".json")
        immutable(path, packet)
        result.append(dict(case=name, packet=binding(path)))
    return result
