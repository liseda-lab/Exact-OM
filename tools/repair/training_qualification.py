"""Bind current qualification fixtures to the original 160-row label limitations."""

import argparse
from pathlib import Path

from exact.repair.api import write_artifact
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.training_audit import attempt_report, requirement_review


def run(plan_path, qualification_path, output):
    plan = read(plan_path)
    if plan.get("schema") != "exact-repair/training-qualification-plan/v1":
        raise ValueError("Unknown qualification plan")
    audit, _, _ = attempt_report(plan["label_audit"])
    if (
        audit["scheduled_rows"] != 160
        or audit["heldout_cases_opened"]
        or audit["supervision_admitted"]
    ):
        raise ValueError("Label audit scope changed")
    previous = [(item, *attempt_report(item)[::2]) for item in plan["prior_qualifications"]]
    qualification = read(qualification_path)
    if qualification["status"] != "complete":
        raise ValueError("Current-source qualification incomplete")
    source = Path(__file__).resolve().parents[2]
    current = (
        dict(run_id=plan["run_id"], report=binding(qualification_path)),
        qualification,
        dict(code=str(source)),
    )
    requirements = requirement_review(plan["requirement_matrices"], [*previous, current], source)
    output = Path(output)
    write_artifact(output / "requirements.json", requirements)
    report = dict(
        schema="exact-repair/training-qualification/v1",
        status="complete",
        plan=binding(plan_path),
        qualification=binding(qualification_path),
        requirement_review=binding(output / "requirements.json"),
        label_audit=plan["label_audit"],
        scheduled_rows=160,
        summaries=audit["summaries"],
        admitted_semantic_supervision=False,
        fitting_started=False,
        heldout_cases_opened=False,
        gates={key: "not_established" for key in ("G0", "G1", "G2")},
        current_receipt_required=True,
        limitations=[
            "Authenticate this report against its nonce-bound completion and output manifest before adoption.",
            "Fixture source compatibility is not research gate passage or live backend/query coverage.",
            "Usable fixed-inventory semantic candidates cover only overlap:14/128train and4/32development.",
            "All160label outcomes and64held-out identities remain required; unknown labels stay unknown.",
            "Full-schema GPU/proposal probes are separately scheduled; generated acquisition and decoded selection remain required.",
            "The18fitting protocols, matched symbolic controls, held-out comparisons and final report remain registered.",
        ],
    )
    write_artifact(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.qualification, args.output)


if __name__ == "__main__":
    main()
