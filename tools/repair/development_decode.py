"""Development-only generated decoding with disposable, untrained checkpoints.

This exercises the selection/evaluator transport path before fitting. Checkpoints
are never eligible models, warm starts or semantic teachers. Original acquisitions
remain immutable; selected generated assignments are separately charged diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from tools.repair import fresh_evaluation as fresh
from tools.repair.batch import read, sha
from tools.repair.expanded_profile import checkpoint
from tools.repair.historical_regression import binding, verify_binding
from tools.repair.training_probe import ARMS

SCHEMA = "exact-repair/development-decode/v1"
ALL_ARMS = (*ARMS, *fresh.CONTROLS)
BUDGET = dict(seconds=300, cpu_seconds=600, memory_mb=8192)


def make_rows(cases):
    rows = []
    for arm in ALL_ARMS:
        for index, case in enumerate(cases):
            row = dict(
                arm_id=arm,
                case_index=index,
                case_id=case["case_id"],
                study_kind="untrained_development_decode",
                **BUDGET,
            )
            row["id"] = canonical_hash(row)
            rows.append(row)
    return rows


def initialize_model(declaration, settings, arm, output):
    """Initialize the declared language without looking at any corpus payload."""
    import torch
    from exact.repair.graph_schema import declared_metadata
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import model_digest
    from tools.repair.train import save_training_state

    if arm not in ARMS:
        raise ValueError("Unknown untrained diagnostic arm")
    torch.set_num_threads(1)
    torch.manual_seed(settings["seed"])
    encoder, head = arm.split("-")
    model = RepairModel(
        declared_metadata(declaration),
        graph_schema=declaration,
        encoder=encoder,
        pairwise=head == "pairwise",
        **settings["model"],
    )
    model.eval()
    output = Path(output)
    if output.exists():
        raise ValueError("Never overwrite a frozen diagnostic checkpoint")
    digest = model_digest(model)
    save_training_state(
        output,
        dict(
            model_schema="exact-repair/model/v3",
            metadata=model.metadata,
            config=model.config,
            state_dict=model.state_dict(),
            diagnostic_only=True,
            fitting_steps=0,
            warm_start_allowed=False,
            seed=settings["seed"],
            model_hash=digest,
        ),
    )
    return dict(
        model=binding(output),
        model_hash=digest,
        parameter_count=sum(p.numel() for p in model.parameters()),
        diagnostic_only=True,
        fitting_steps=0,
        warm_start_allowed=False,
    )


def validate_schedule(schedule):
    # Check the exclusion boundary before any file, particularly evaluator, opens.
    if (
        schedule.get("schema") != SCHEMA
        or schedule.get("split") != "development"
        or schedule.get("expected_cases") != 32
        or schedule.get("scheduled_rows") != 288
        or any(
            schedule.get(k) is not False
            for k in ("heldout_use", "model_fitting", "warm_start", "supervision_admitted")
        )
    ):
        raise ValueError("Development diagnostic boundary changed")
    release = verify_binding(schedule["release"])
    cases = schedule["cases"]
    if (
        cases != release["rows"]
        or len(cases) != 32
        or len({r["case_id"] for r in cases}) != 32
        or any(r["split"] != "development" or r["status"] != "materialized" for r in cases)
        or len({r["structural_parent"] for r in cases}) != 16
    ):
        raise ValueError("Development release, parent grouping or denominator changed")
    for parent in {r["structural_parent"] for r in cases}:
        if Counter(r["control"] for r in cases if r["structural_parent"] == parent) != {
            "coherent": 1,
            "corrupted": 1,
        }:
            raise ValueError("Development parent lost its paired controls")
    if [a["id"] for a in schedule["arms"]] != list(ALL_ARMS):
        raise ValueError("Development arm denominator changed")
    if schedule["rows"] != make_rows(cases):
        raise ValueError("Development row identity or scientific budget changed")
    for key in ("program", "authorization", "corpus_completion", "declaration"):
        verify_binding(schedule[key])
    for arm in schedule["arms"]:
        protocol = verify_binding(arm["protocol"])
        if any(
            protocol["resources"][k] != v
            for k, v in dict(
                case_wall_seconds=300, case_cpu_seconds=600, case_rss_mb=8192, generation_seconds=60
            ).items()
        ):
            raise ValueError("Development scientific budget differs")
        if arm["status"] != "available":
            raise ValueError("Frozen arm is not available")
        if arm["id"] in ARMS:
            if (
                arm.get("kind") != "learned"
                or arm.get("diagnostic_only") is not True
                or arm.get("fitting_steps") != 0
                or arm.get("warm_start_allowed") is not False
            ):
                raise ValueError("Trained or reusable checkpoint entered development diagnostic")
            fresh.pilot.check_binding(arm["model"])
            if protocol["model"].get("graph_schema") != verify_binding(schedule["declaration"]):
                raise ValueError("Diagnostic model graph declaration differs")
        elif arm.get("kind") != "control":
            raise ValueError("Control arm kind differs")
    for case in cases:
        observable = read_record(verify_binding(case["observable"]))
        if observable.content_hash != case["input_hash"]:
            raise ValueError("Development observable identity changed")
    # Evaluator payloads remain unopened here; fresh.evaluate_row opens only after selection.


def evaluate_row(schedule, row, directory):
    return fresh.evaluate_row(schedule, row, directory, record_native_labels=True)


def validate_weights(schedule):
    import torch
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import model_digest
    from exact.repair.graph_schema import declared_metadata

    torch.set_num_threads(1)
    expected_metadata = declared_metadata(verify_binding(schedule["declaration"]))
    rows = []
    for arm in schedule["arms"]:
        if arm["id"] not in ARMS:
            continue
        state = torch.load(
            fresh.pilot.check_binding(arm["model"]), map_location="cpu", weights_only=True
        )
        if (
            state.get("diagnostic_only") is not True
            or state.get("fitting_steps") != 0
            or state.get("warm_start_allowed") is not False
            or state.get("seed") != schedule["model_initialization_seed"]
            or state.get("metadata") != expected_metadata
        ):
            raise ValueError("Diagnostic checkpoint provenance or full schema differs")
        encoder, head = arm["id"].split("-")
        if state["config"]["encoder"] != encoder or state["config"]["pairwise"] != (
            head == "pairwise"
        ):
            raise ValueError("Diagnostic checkpoint belongs to another arm")
        model = RepairModel(state["metadata"], **state["config"])
        model.load_state_dict(state["state_dict"], strict=True)
        if model_digest(model) != arm["model_hash"]:
            raise ValueError("Diagnostic checkpoint digest differs")
        rows.append(
            dict(
                arm=arm["id"],
                model=arm["model"],
                model_hash=arm["model_hash"],
                fitting_steps=0,
                warm_start_allowed=False,
            )
        )
    return rows


def run(schedule_path, output, start, stop):
    from exact.repair.study import runtime_manifest

    schedule = read(schedule_path)
    validate_schedule(schedule)
    if not 0 <= start < stop <= len(schedule["rows"]):
        raise ValueError("Invalid development diagnostic slice")
    runtime = runtime_manifest()
    identity = canonical_hash((sha(schedule_path), sha(__file__), sha(fresh.__file__), runtime))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, statuses = [], Counter()
        for row in schedule["rows"][start:stop]:
            write_artifact(output / "progress.json", dict(recorded=len(rows), row=row["id"]))
            saved = fresh.one_row(schedule, row, output, identity, evaluator=evaluate_row)
            fresh.raise_on_software_failure(saved)
            status = (
                verify_binding(saved["result"])["status"]
                if saved.get("result")
                else saved["status"]
            )
            statuses[status] += 1
            rows.append(
                dict(
                    **binding(output / "rows" / (row["id"] + ".json")),
                    row_id=row["id"],
                    status=status,
                )
            )
        return checkpoint(
            output / "report.json",
            identity,
            schema=SCHEMA,
            status="complete",
            schedule=binding(schedule_path),
            slice=[start, stop],
            scheduled=stop - start,
            recorded=len(rows),
            rows=rows,
            outcomes=dict(statuses),
            runtime=runtime,
            study_complete=False,
            gates_passed=False,
            model_fitting=False,
            heldout_cases_opened=False,
            supervision_admitted=False,
            checkpoint_selection="unavailable_untrained_diagnostic_models",
            intended_parent_qualification="original acquisition limitations retained",
            semantic_scope="selected-only original query basis; native receipts retained; not admitted labels",
            followup="xr21-expanded-training-001",
            api_spend_usd=0,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        schedule = read(args.schedule)
        validate_schedule(schedule)
        models = validate_weights(schedule)
        write_artifact(
            args.output / "report.json",
            dict(
                schema=SCHEMA,
                status="complete",
                schedule=binding(args.schedule),
                scheduled_rows=288,
                model_fitting=False,
                heldout_cases_opened=False,
                gates_passed=False,
                scientific_rows_executed=0,
                initialized_models=models,
            ),
        )
    else:
        run(args.schedule, args.output, args.start, args.stop)


if __name__ == "__main__":
    main()
