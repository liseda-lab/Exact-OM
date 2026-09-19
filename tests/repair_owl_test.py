"""Small semantic XR-2 conformance cases; no benchmark experiments."""

from types import SimpleNamespace

import pyowl_core as owl
import pytest

import exact.repair.owl as repair_owl
from exact.repair.owl import OwlVerifier, named_classes, snapshot_from_axioms


def cls(name):
    return owl.Class(owl.IRI(f"urn:repair:test:{name}"))


def prop(name):
    return owl.ObjectProperty(owl.IRI(f"urn:repair:test:{name}"))


def intersection(*expressions):
    return owl.ObjectIntersectionOf(expressions)


def disjoint(*expressions):
    return owl.DisjointClasses(expressions)


def equivalent(left, right):
    return (owl.SubClassOf(left, right), owl.SubClassOf(right, left))


@pytest.fixture(params=["hermit", "elk"])
def verifier(request):
    pytest.importorskip("pyhermit" if request.param == "hermit" else "pyelk")
    return OwlVerifier(request.param, backend="python")


@pytest.fixture
def hermit():
    pytest.importorskip("pyhermit")
    return OwlVerifier("hermit", backend="python")


def test_overlap_all_named_classes_and_alternative_replacements(verifier):
    author, reviewer, overlap, target_author, target_reviewer = map(
        cls, ("Author_s", "Reviewer_s", "AuthorReviewer", "Author_t", "Reviewer_t")
    )
    conjunctive = owl.SubClassOf(overlap, intersection(author, reviewer))
    target_disjoint = disjoint(target_author, target_reviewer)
    author_mapping = equivalent(author, target_author)
    reviewer_mapping = equivalent(reviewer, target_reviewer)
    source = snapshot_from_axioms([conjunctive])
    target = snapshot_from_axioms([target_disjoint])
    baseline = verifier.diagnose_baselines(source, target, (*author_mapping, *reviewer_mapping))
    assert baseline.source.logical_status == baseline.target.logical_status == "VERIFIED_FEASIBLE"
    assert baseline.union.logical_status == "VERIFIED_FEASIBLE"
    assert baseline.alignment.unsatisfiable_classes == (overlap.iri.value,)
    assert baseline.alignment.logical_status == "VERIFIED_INFEASIBLE"
    alternatives = [
        [conjunctive, target_disjoint, *author_mapping],
        [conjunctive, target_disjoint, owl.SubClassOf(target_author, author), *reviewer_mapping],
        [conjunctive, *author_mapping, *reviewer_mapping],
        [owl.SubClassOf(overlap, author), target_disjoint, *author_mapping, *reviewer_mapping],
    ]
    monitored = tuple(
        map(
            lambda value: value.iri.value,
            (author, reviewer, overlap, target_author, target_reviewer),
        )
    )
    for axioms in alternatives:
        report = verifier.check_theory(snapshot_from_axioms(axioms), monitored)
        assert report.logical_status == "VERIFIED_FEASIBLE", report.support.issues
        assert (
            len([item for item in report.obligations if item.kind == "class_satisfiability"]) == 5
        )


def test_accepted_papers_specialisation_and_independent_necessary_condition(verifier):
    paper, rejected, accepted, target_rejected, target_paper, submission, acceptance, invited = map(
        cls,
        (
            "Paper_s",
            "Rejected_s",
            "Accepted_t",
            "Rejected_t",
            "Paper_t",
            "AcceptedSubmission_s",
            "Acceptance",
            "Invited_t",
        ),
    )
    expression = owl.ObjectSomeValuesFrom(prop("hasDecision"), acceptance)
    active = intersection(paper, expression)
    source = [
        owl.SubClassOf(rejected, paper),
        owl.SubClassOf(submission, active),
        owl.SubClassOf(intersection(rejected, expression), owl.OWL_NOTHING),
    ]
    target = [
        owl.SubClassOf(accepted, target_paper),
        owl.SubClassOf(target_rejected, target_paper),
        disjoint(accepted, target_rejected),
    ]
    fixed = [*source, *target, *equivalent(rejected, target_rejected)]
    original = snapshot_from_axioms([*fixed, *equivalent(paper, accepted)])
    monitored = named_classes(original)
    assert verifier.check_theory(original).logical_status == "VERIFIED_INFEASIBLE"
    assert (
        verifier.check_theory(
            snapshot_from_axioms([*fixed, *equivalent(paper, target_paper)]), monitored
        ).logical_status
        == "VERIFIED_FEASIBLE"
    )
    specialised = [owl.SubClassOf(active, accepted), owl.SubClassOf(accepted, paper)]
    necessary = owl.SubClassOf(accepted, expression)
    for replacement in (
        specialised,
        [*specialised, necessary],
        [owl.SubClassOf(accepted, paper), necessary],
    ):
        report = verifier.check_theory(
            snapshot_from_axioms([*fixed, *replacement]),
            monitored,
            activated=(
                [active] if replacement != [owl.SubClassOf(accepted, paper), necessary] else []
            ),
        )
        assert report.logical_status == "VERIFIED_FEASIBLE", report.support.issues
    invited_axioms = [
        owl.SubClassOf(invited, accepted),
        owl.SubClassOf(intersection(invited, expression), owl.OWL_NOTHING),
    ]
    assert (
        verifier.check_theory(
            snapshot_from_axioms([*fixed, *invited_axioms, *specialised]), activated=[active]
        ).logical_status
        == "VERIFIED_FEASIBLE"
    )
    strengthened = verifier.check_theory(
        snapshot_from_axioms([*fixed, *invited_axioms, *specialised, necessary]), activated=[active]
    )
    assert strengthened.unsatisfiable_classes == (invited.iri.value,)


def test_participant_ontology_weakening(verifier):
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
    fixed = [
        owl.SubClassOf(listener, participant),
        owl.SubClassOf(presenting, participant),
        disjoint(listener, presenting),
        owl.SubClassOf(speaker, person),
        *equivalent(listener, target_listener),
        *equivalent(speaker, target_speaker),
        disjoint(target_listener, target_speaker),
    ]
    assert verifier.check_theory(
        snapshot_from_axioms([*fixed, owl.SubClassOf(participant, speaker)])
    ).unsatisfiable_classes == tuple(sorted([listener.iri.value, target_listener.iri.value]))
    active = intersection(participant, presenting)
    assert (
        verifier.check_theory(
            snapshot_from_axioms([*fixed, owl.SubClassOf(active, speaker)]), activated=[active]
        ).logical_status
        == "VERIFIED_FEASIBLE"
    )
    assert (
        verifier.check_theory(
            snapshot_from_axioms([*fixed, owl.SubClassOf(participant, person)])
        ).logical_status
        == "VERIFIED_FEASIBLE"
    )


@pytest.mark.parametrize("domain", [False, True])
def test_range_and_domain_typing_weakening(verifier, domain):
    witness, software, person, agent, target_software, target_person, review = map(
        cls, ("Witness", "Software_s", "Person_s", "Agent_s", "Software_t", "Person_t", "Review")
    )
    relation = prop("writes" if domain else "writtenBy")
    restriction = owl.ObjectSomeValuesFrom(relation, review if domain else software)
    original_subclass = owl.SubClassOf(
        witness, intersection(software, restriction) if domain else restriction
    )
    constraint = owl.ObjectPropertyDomain if domain else owl.ObjectPropertyRange
    fixed = [
        owl.SubClassOf(person, agent),
        owl.SubClassOf(software, agent),
        *equivalent(person, target_person),
        *equivalent(software, target_software),
        disjoint(target_person, target_software),
    ]
    baseline = verifier.check_theory(
        snapshot_from_axioms([*fixed, original_subclass, constraint(relation, person)])
    )
    assert baseline.unsatisfiable_classes == (witness.iri.value,)
    weakened = verifier.check_theory(
        snapshot_from_axioms([*fixed, original_subclass, constraint(relation, agent)])
    )
    assert weakened.logical_status == "VERIFIED_FEASIBLE", weakened.support.issues
    if not domain:
        replacement = owl.SubClassOf(witness, owl.ObjectSomeValuesFrom(relation, agent))
        assert (
            verifier.check_theory(
                snapshot_from_axioms([*fixed, replacement, constraint(relation, person)])
            ).logical_status
            == "VERIFIED_FEASIBLE"
        )


def test_full_signature_catches_witness_outside_mapping_endpoints(verifier):
    a, b, c = map(cls, ("A", "B", "C"))
    r = prop("r")
    snapshot = snapshot_from_axioms(
        [
            owl.SubClassOf(c, owl.ObjectSomeValuesFrom(r, a)),
            owl.SubClassOf(owl.ObjectSomeValuesFrom(r, b), owl.OWL_NOTHING),
            owl.SubClassOf(a, b),
        ]
    )
    assert verifier.check_theory(snapshot).unsatisfiable_classes == (c.iri.value,)


def test_only_activated_whole_antecedent_must_be_satisfiable(verifier):
    source, expression = cls("Source"), cls("Expression")
    snapshot = snapshot_from_axioms([disjoint(source, expression)])
    assert verifier.check_theory(snapshot).logical_status == "VERIFIED_FEASIBLE"
    assert (
        verifier.check_theory(snapshot, activated=[expression]).logical_status
        == "VERIFIED_FEASIBLE"
    )
    checked = verifier.check_theory(snapshot, activated=[intersection(source, expression)])
    assert checked.logical_status == "VERIFIED_INFEASIBLE"
    assert checked.unsatisfiable_classes == ()


@pytest.mark.parametrize("kind", ["universal", "cardinality", "abox"])
def test_expressive_semantics_cannot_be_silently_dropped(hermit, kind):
    a, b, c = map(cls, ("A", "B", "C"))
    r = prop("r")
    if kind == "universal":
        axioms = [
            owl.SubClassOf(a, owl.ObjectSomeValuesFrom(r, b)),
            owl.SubClassOf(a, owl.ObjectAllValuesFrom(r, c)),
            disjoint(b, c),
        ]
    elif kind == "cardinality":
        axioms = [
            owl.SubClassOf(a, owl.ObjectMinCardinality(2, r, b)),
            owl.SubClassOf(a, owl.ObjectMaxCardinality(1, r, b)),
        ]
    else:
        individual = owl.NamedIndividual(owl.IRI("urn:repair:test:individual"))
        axioms = [
            disjoint(a, b),
            owl.ClassAssertion(a, individual),
            owl.ClassAssertion(b, individual),
        ]
    snapshot = snapshot_from_axioms(axioms)
    assert hermit.check_theory(snapshot).logical_status == "VERIFIED_INFEASIBLE"
    if kind != "abox":
        pytest.importorskip("pyelk")
        report = OwlVerifier("elk", backend="python").check_theory(snapshot)
        assert report.logical_status == "UNKNOWN"
        assert report.support.input_supported is False
        assert any("unsupported" in issue for issue in report.support.issues)


def test_unknown_baseline_never_invents_source_exception():
    pytest.importorskip("pyelk")
    a, b = cls("A"), cls("B")
    source = snapshot_from_axioms([owl.SubClassOf(a, owl.ObjectAllValuesFrom(prop("r"), b))])
    report = OwlVerifier("elk", backend="python").diagnose_baselines(
        source, snapshot_from_axioms([]), [], allow_source_exceptions=True
    )
    assert report.source.logical_status == "UNKNOWN"
    assert report.exceptions == report.exception_evidence == ()


def test_only_source_proofs_define_frozen_exceptions(verifier):
    a, b = cls("A"), cls("B")
    source = snapshot_from_axioms([owl.SubClassOf(a, owl.OWL_NOTHING)])
    target = snapshot_from_axioms([owl.SubClassOf(b, a)])
    report = verifier.diagnose_baselines(source, target, [], allow_source_exceptions=True)
    assert report.exceptions == (a.iri.value,)
    assert report.union.unsatisfiable_classes == (b.iri.value,)
    assert report.alignment.logical_status == "VERIFIED_INFEASIBLE"
    assert report.exception_evidence[0][1:3] == ("source", report.source.theory_hash)


def test_required_prohibited_consequences_use_owl_entailment(verifier):
    a, b = cls("A"), cls("B")
    query = owl.SubClassOf(a, b)
    empty = snapshot_from_axioms([owl.Declaration(a), owl.Declaration(b)])
    assert verifier.check_theory(empty, required=[query]).logical_status == "VERIFIED_INFEASIBLE"
    assert verifier.check_theory(empty, prohibited=[query]).logical_status == "VERIFIED_FEASIBLE"
    present = snapshot_from_axioms([query])
    assert (
        verifier.check_theory(present, prohibited=[query]).logical_status == "VERIFIED_INFEASIBLE"
    )


def test_reconstruction_keeps_fixed_duplicate_and_discards_stale_consequences(verifier):
    a, b, c = map(cls, ("A", "B", "C"))
    original = owl.SubClassOf(a, b)
    fixed = [disjoint(b, c), owl.SubClassOf(a, c)]
    selected = [SimpleNamespace(axioms=(), active_expressions=())]
    monitored = (a, b, c)
    assert (
        verifier.check_assignment([*fixed, original], selected, monitored).logical_status
        == "VERIFIED_INFEASIBLE"
    )
    assert (
        verifier.check_assignment(fixed, selected, monitored).logical_status == "VERIFIED_FEASIBLE"
    )


def test_shared_snapshot_handoff_is_identity_preserving(monkeypatch, hermit):
    import pyhermit

    snapshot = snapshot_from_axioms([owl.SubClassOf(cls("A"), cls("B"))])
    actual = pyhermit.Reasoner
    seen = []

    def construct(view, **kwargs):
        seen.append(view)
        return actual(view, **kwargs)

    monkeypatch.setattr(pyhermit, "Reasoner", construct)
    monkeypatch.setattr(owl, "load_snapshot", lambda *_a, **_k: pytest.fail("source reparse"))
    assert hermit.check_theory(snapshot).logical_status == "VERIFIED_FEASIBLE"
    assert seen == [snapshot]


def test_timeout_and_backend_unavailability_stay_unknown(monkeypatch):
    snapshot = snapshot_from_axioms([owl.Declaration(cls("A"))])
    verifier = OwlVerifier()

    def timeout(_snapshot):
        raise TimeoutError("bounded test")

    monkeypatch.setattr(verifier, "_open", timeout)
    report = verifier.check_theory(snapshot)
    assert report.logical_status == "UNKNOWN"
    assert all(not item.complete for item in report.obligations)
    assert report.support.issues[0].startswith("timeout:")

    def missing(_name):
        raise ImportError("optional backend absent")

    monkeypatch.setattr(repair_owl, "import_module", missing)
    report = OwlVerifier().check_theory(snapshot)
    assert report.logical_status == "UNKNOWN"
    assert report.support.issues[0].startswith("backend_unavailable:")
