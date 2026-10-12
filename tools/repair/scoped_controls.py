"""Freeze new vocabulary instantiations and explicit scope before any API call.

These are five unexposed authored packets in existing logical template families,
not five independently sampled domains. Two old controls and two DEV examples
are separately marked as regressions; old gold and evidence stay immutable.
"""

from pathlib import Path

from tools.repair import judge_controls, scoped_judge as judge
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import bound, immutable


def build(previous_cases, directory):
    directory = Path(directory)
    vocabulary = dict(
        AcceptedSubmission="CertifiedDevice",
        Submission="Device",
        Paper="Bridge",
        Person="Inspector",
        Agent="Worker",
        authoredBy="inspectedBy",
        AccreditedResearcher="LicensedPilot",
        Researcher="Pilot",
        English="JetAircraft",
        speaks="operates",
        SafetyValve="DigitalTicket",
        Sensor="PrintedTicket",
        ConceptX="CodeM",
        ConceptY="CodeN",
    )
    definitions = dict(
        direction="CertifiedDevice is a Device that has passed certification. Some devices have never been certified. The focal intended inclusion is CertifiedDevice in Device, with no implication that every Device is certified. No other meaning is in scope.",
        range_context="In this fictional infrastructure register, inspectedBy relates a Bridge to an Inspector. Every Inspector is a Worker. Both the Bridge domain and Inspector range are intended focal commitments of equal importance. All other supplied declarations are isolated names and impose no relationships. No other meaning is intended beyond the Inspector-to-Worker inclusion.",
        unsupported_qualifier="In this fictional aviation register, LicensedPilot is a kind of Pilot. A license does not require operation of jet aircraft. Some licensed pilots operate gliders only and never operate any JetAircraft. operates relates pilots to the aircraft they operate. The intended focal meaning is the pilot classification; no other knowledge is in scope.",
        ontology_collateral="CertifiedDevice is a kind of Device. Separately, a fictional admission ledger defines DigitalTicket and PrintedTicket as mutually exclusive ticket categories. Both categories are populated. Their exclusion is the only other intended knowledge besides the focal device classification.",
        insufficient="CodeM and CodeN are opaque identifiers. No definitions, examples, intended relationship or meanings of their effects are supplied. The observed mapping is fallible and logical feasibility cannot establish meaning.",
    )
    overrides = {}
    for name in ("direction", "unsupported_qualifier"):
        allowed = {k: dict(a=["full"], b=["full"]) for k in judge.chat.CRITERIA}
        allowed["assertion_fidelity"]["b"] = ["none", "limited", "mixed"]
        overrides[name] = allowed
    refs = judge_controls.controls(
        directory / "fresh",
        background_count=16,
        names=tuple(definitions),
        vocabulary=vocabulary,
        namespace="scoped_qualification_20261012",
        definitions=definitions,
        gold_overrides=overrides,
    )
    scopes = dict(
        direction=(
            "CertifiedDevice inclusion in Device.",
            "The retained/introduced mapping between CertifiedDevice and Device, in all asserted directions.",
            "No other commitments are in scope; an unsupported reverse focal direction belongs only to assertion fidelity.",
        ),
        range_context=(
            "Both inspectedBy domain Bridge and range Inspector, of equal importance.",
            "The supplied inspectedBy domain/range assertions; deletion is absence, not a false assertion.",
            "The other Inspector-to-Worker inclusion. Isolated declarations have no additional meaning effects.",
        ),
        unsupported_qualifier=(
            "LicensedPilot inclusion in Pilot.",
            "The mapping's direction, endpoints and all qualifiers, including aircraft-operation restrictions.",
            "No other intended commitments are in scope; the focal qualifier is assessed only under assertion fidelity.",
        ),
        ontology_collateral=(
            "CertifiedDevice inclusion in Device.",
            "The focal CertifiedDevice-to-Device classification.",
            "Preservation or loss of the separately intended DigitalTicket/PrintedTicket mutual exclusion.",
        ),
        insufficient=(
            "Intended relationship of CodeM and CodeN.",
            "Justification of relationships asserted between CodeM and CodeN.",
            "The other conceptual effects of relating CodeM and CodeN; their intended meanings are required.",
        ),
        partial_evidence=(
            "AcceptedSubmission inclusion in Submission.",
            "Only the focal editable AcceptedSubmission-to-Submission mapping; the fixed Submission/OpaqueCategory assertion is assessed under collateral.",
            "The fixed and induced OpaqueCategory commitments; intended meaning of OpaqueCategory is required for both plans, including deletion.",
        ),
        tie=(
            "AcceptedSubmission inclusion in Submission.",
            "The retained/introduced mapping between AcceptedSubmission and Submission.",
            "No other commitments are in scope; no focal omission/restriction is counted twice.",
        ),
        dev_domain=(
            "The intended writes domain Agent_s in D1/D2/D3, with the full supplied fixed formal context.",
            "The selected editable writes-domain assertion including any narrower endpoint; the original Person_s domain is fallible.",
            "Other intended formal/contextual relationships in D1/D2/D3, including distinctions of persons/software. A narrower focal domain is not counted again unless it causes a distinct supported downstream mismatch.",
        ),
        dev_conjunction=(
            "The intended Overlap_s-to-Author_s commitment in D1/D2/D3, with complete fixed formal context.",
            "The selected editable Overlap_s role assertion, including reviewer or author commitments.",
            "Other intended fixed/contextual relationships in D1/D2/D3. Do not count the same focal author omission or focal reviewer assertion a second time.",
        ),
    )
    cases = []
    for row in read(previous_cases):
        if row["id"] in ("partial_evidence", "tie", "dev_domain", "dev_conjunction"):
            cases.append(
                {
                    **row,
                    "fresh": False,
                    "repeats": 2,
                    "scope": dict(zip(judge.chat.CRITERIA, scopes[row["id"]])),
                }
            )
    for ref in refs:
        source = bound(ref["packet"])
        cases.append(
            dict(
                id="fresh_" + source["case"],
                role="authored_control",
                fresh=True,
                freshness="unexposed vocabulary/definition instantiation of an existing logical template",
                repeats=1,
                scope=dict(zip(judge.chat.CRITERIA, scopes[source["case"]])),
                payload=source["payload"],
                gold=source["gold"],
                source=ref["packet"],
            )
        )
    assert len(cases) == 9
    immutable(directory / "cases.json", cases)
    return binding(directory / "cases.json")
