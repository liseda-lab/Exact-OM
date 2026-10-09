"""Frozen TRAIN-side price/label calibration for the approved $35 campaign.

Four controlled semantic constructions, not expert labels or held-out evaluation.
All four teacher candidates see identical packets in both presentation orders.
"""

from __future__ import annotations
import argparse
import dataclasses
from pathlib import Path

import pyowl_core as owl
from exact.repair.api import write_artifact
from exact.repair.candidates import mapping_candidates
from exact.repair.learning import TeacherProbe
from exact.repair.records import (
    RepairInputV2,
    RevisionObjectV2,
    PolicyV2,
    promote_input_v3,
    freeze_public_policy,
    canonical_hash,
    read_record,
)
from exact.repair.semantic_fidelity import SemanticEvidencePacketV3, CRITERIA
from tools.repair.batch import read, sha
from tools.repair.corpus import GeneratedCase
from tools.repair.prepare import case_to_dict
from tools.repair.corrective_semantics import _verified_plan, run as annotate, PHASE_LIMITS
from tools.repair.expanded_corpus import binding, immutable

SHORTLIST = ("deepseek", "qwen", "glm", "sonnet_reference")
MODELS = {
    "deepseek": ("deepseek/deepseek-v4.1-flash", "DeepSeek", 0.30, 1.20),
    "qwen": ("qwen/qwen3.8-flash", "Alibaba", 0.15, 0.47),
    "glm": ("z-ai/glm-5.3-flash", "Z.AI", 0.15, 0.50),
    "sonnet_reference": ("anthropic/claude-sonnet-4.6", "Anthropic", 3.0, 15.0),
    "independent_test": ("google/gemini-2.5-pro", "Google AI Studio", 1.25, 10.0),
}
SETTINGS = (
    (
        "AcceptedSubmission",
        "Submission",
        "AcceptedSubmission denotes every submission accepted after review. Submission denotes all submitted papers, whether accepted or rejected.",
        "A",
    ),
    (
        "InvitedSpeaker",
        "Speaker",
        "InvitedSpeaker denotes speakers personally invited by the programme committee. Speaker denotes all people giving a talk, including invited and contributed speakers.",
        "A",
    ),
    (
        "ResearchPaper",
        "ResearchArticle",
        "ResearchPaper and ResearchArticle are synonymous here: both denote precisely the same set of original research manuscripts.",
        "B",
    ),
    (
        "WorkshopPaper",
        "WorkshopSubmission",
        "WorkshopPaper denotes only accepted workshop manuscripts. WorkshopSubmission denotes all submitted workshop manuscripts, including rejected ones.",
        "A",
    ),
)


def prepare(campaign):
    campaign = Path(campaign)
    directory = campaign / "annotations/calibration"
    profiles = {}
    prices = {}
    for name, (model, provider, input_price, output_price) in MODELS.items():
        profiles[name] = dict(
            backend="openrouter",
            model=model,
            api_base="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            api_key_path="~/.config/openrouter/api_key",
            timeout_secs=90,
            provider=dict(
                only=[provider],
                allow_fallbacks=False,
                max_price=dict(prompt=input_price, completion=output_price),
            ),
        )
        prices[name] = dict(input=input_price, output=output_price)
    rows = []
    parents = {}
    gold = {}
    for i, (left, right, definitions, decision) in enumerate(SETTINGS):
        a, b = [owl.Class(owl.IRI("urn:repair-calibration:" + v)) for v in (left, right)]
        candidates = mapping_candidates("mapping", a, b, "=")
        obj = RevisionObjectV2(
            "mapping", "mapping", candidates[0].axioms, candidates, source_entity=a, target_entity=b
        )
        problem = promote_input_v3(RepairInputV2((), (obj,), PolicyV2((a, b))))
        problem = dataclasses.replace(problem, policy=freeze_public_policy(problem))
        parent = "authored-semantic-calibration-" + str(i)
        case = GeneratedCase(
            parent,
            parent,
            "authored_semantic_calibration",
            "train",
            problem,
            (
                TeacherProbe("forward", owl.SubClassOf(a, b), "meaning"),
                TeacherProbe("reverse", owl.SubClassOf(b, a), "meaning", decision == "B"),
            ),
            (0,),
            20261009,
            False,
            origin="authored_controlled_semantics",
            schema_revision="v3",
        )
        parents[parent] = "train"
        forward = next(
            j
            for j, c in enumerate(problem.objects[0].candidates)
            if c.axioms == (owl.SubClassOf(a, b),)
        )
        plans = [_verified_plan(case_to_dict(case), (j,)) for j in (forward, 0)]
        if any(p is None or not p.eligible for p in plans):
            raise ValueError("Calibration construction failed qualified verification")
        evidence = {
            "D1": dict(
                source_id="controlled-construction-" + str(i),
                release="20261009-v1",
                text=definitions,
                kind="authored_definition",
                symbolic_value=None,
            )
        }
        packet = SemanticEvidencePacketV3(
            parent,
            parent,
            "train",
            "Preserve exactly the declared domain meaning; logical feasibility is already checked. The observed equivalence can be too strong.",
            (left + " EquivalentTo " + right,),
            evidence,
            (),
            *plans,
            "grounded-retained-meaning/20261009-v1",
            {name: 1 / 3 for name in CRITERIA},
            ("forward", "reverse"),
            dict(
                complete=True,
                omissions=[],
                stop_reason="complete_controlled_construction",
                byte_budget=8000,
            ),
        )
        bnd = immutable(directory / "packets" / f"{i}.json", packet.to_dict())
        gold[packet.case_id] = decision
        for profile in SHORTLIST:
            for swapped in (False, True):
                rows.append(
                    dict(
                        id=f"{profile}-{i}-{int(swapped)}",
                        packet=bnd,
                        profile=profile,
                        swapped=swapped,
                    )
                )
    if len(rows) != 32 or len({row["id"] for row in rows}) != 32:
        raise ValueError("Calibration requires exactly32 unique planned observations")
    from tools.repair.corrective_campaign import source_identity

    implementation = source_identity()
    manifest = dict(
        implementation_source=implementation,
        schema="exact-repair/corrective-annotation/v1",
        authorized=True,
        lineage_id="exact-repair-corrective-20261009",
        phase="calibration",
        cost_ceiling_usd=35,
        request_limits=PHASE_LIMITS,
        profile="deepseek",
        profiles=profiles,
        prices_per_million=prices,
        teacher_profile="deepseek",
        test_profile="independent_test",
        selection_model_ids=[MODELS[p][0] for p in SHORTLIST],
        parent_splits=parents,
        ledger_directory=str(campaign / "annotations/ledger"),
        data_permissions="Original authored controlled constructions; user authorized hosted label calibration",
        slots=rows,
        seconds=3600,
        deadline_epoch=1791748800.0,
        rubric_version="grounded-retained-meaning/20261009-v1",
        criterion_weights={name: 1 / 3 for name in CRITERIA},
        calibration_scope="Four controlled TRAIN constructions; no broad equivalence or real-data validation claim",
        gold=gold,
        selection_rule="Require8/8valid grounded judgments,8/8declared semantic decisions,4/4order consistency;lowest frozen maximum cost among eligible cheap models;Sonnet fallback onlyifqualified",
        catalog_sources=[
            "https://openrouter.ai/api/v1/models",
            *["https://openrouter.ai/" + MODELS[p][0] for p in MODELS],
        ],
        price_captured_utc="2026-10-09",
        calibration_gate=None,
    )
    manifest_path = directory / "manifest.json"
    # Source identity can change while authoring an unsubmitted draft. Once
    # batches or hosted reservations exist, the manifest is immutable.
    deployed = (campaign / "batches").exists() or (campaign / "annotations/ledger").exists()
    if deployed:
        immutable(manifest_path, manifest)
    else:
        write_artifact(manifest_path, manifest)
    return directory / "manifest.json"


def summarize(manifest_path, output):
    manifest = read(manifest_path)
    output = Path(output)
    metrics = {}
    for profile in SHORTLIST:
        valid = correct = consistent = 0
        decisions = {}
        for slot in manifest["slots"]:
            if slot["profile"] != profile:
                continue
            path = output / slot["id"] / "labels.json"
            if not path.exists():
                continue
            receipt_path = path.with_name("receipt.json")
            if not receipt_path.exists():
                continue
            receipt = read(receipt_path)
            if receipt.get("status") != "complete":
                continue
            if receipt.get("labels_sha256") != sha(path):
                raise ValueError("Committed calibration labels changed")
            if sha(slot["packet"]["path"]) != slot["packet"]["sha256"]:
                raise ValueError("Calibration packet checksum changed")
            expected = read_record(read(slot["packet"]["path"]))
            comparisons = read(path)["comparisons"]
            if len(comparisons) != 1:
                raise ValueError("Calibration slot requires one committed aggregate")
            row = comparisons[0]
            packet, aggregate = read_record(row["packet"]), read_record(row["comparison"])
            if packet != expected or aggregate.packet_hash != packet.content_hash:
                raise ValueError("Calibration result changed its exact packet")
            if any(
                observation.annotator.get("actual_model") != manifest["profiles"][profile]["model"]
                or observation.presentation.get("swapped") != slot["swapped"]
                for observation in aggregate.observations
            ):
                raise ValueError("Calibration result changed model or presentation identity")
            if not aggregate.global_target_eligible:
                continue
            valid += 1
            correct += aggregate.decision == manifest["gold"][packet.case_id]
            decisions.setdefault(packet.case_id, []).append(aggregate.decision)
        consistent = sum(
            len(values) == 2 and values[0] == values[1] for values in decisions.values()
        )
        metrics[profile] = dict(
            valid=valid,
            correct=correct,
            order_consistent=consistent,
            eligible=(valid, correct, consistent) == (8, 8, 4),
        )
    eligible = [p for p in SHORTLIST[:-1] if metrics[p]["eligible"]]
    if not eligible and metrics["sonnet_reference"]["eligible"]:
        eligible = ["sonnet_reference"]

    def price(name):
        value = manifest["prices_per_million"][name]
        return 8000 * value["input"] + 2000 * value["output"]

    chosen = min(eligible, key=lambda p: (price(p), p)) if eligible else None
    report = dict(
        schema="exact-repair/semantic-calibration-gate/v1",
        status="qualified" if chosen else "not_qualified",
        selected_profile=chosen,
        manifest=binding(manifest_path),
        metrics=metrics,
        selection_model_ids=manifest["selection_model_ids"],
        independent_test_profile="independent_test",
        heldout_judge_called=False,
        limitations=[
            "Small controlled calibration establishes operational suitability, not equivalent intelligence.",
            "Grounded real Conference coverage still required and must be reported separately.",
        ],
    )
    write_artifact(output / "calibration-gate.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.path))
    elif args.action == "run":
        annotate(args.path, args.output)
        summarize(args.path, args.output)
    else:
        summarize(args.path, args.output)


if __name__ == "__main__":
    main()
