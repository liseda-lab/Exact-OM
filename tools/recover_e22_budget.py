"""Migrate only independently verified E22 controls after supervision-key repair."""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

from tools.prepared_batch import binding, read, verified, write

OLD_LINE = "            supervision=cell.resolved_supervision,\n"
NEW_LINE = "            resolved_supervision=cell.resolved_supervision,\n"


def migrate_execution_measurement(
    data: bytes, *, source_artifact_id: str, target_artifact_id: str, source_sha256: str
) -> bytes:
    """Rebind verified execution cost without changing the original observation."""
    if hashlib.sha256(data).hexdigest() != source_sha256:
        raise ValueError("Original execution measurement checksum differs")
    measurement = json.loads(data)
    if (
        not isinstance(measurement, dict)
        or measurement.get("schema_version") != 1
        or measurement.get("artifact_id") != source_artifact_id
        or "identity_migration" in measurement
    ):
        raise ValueError("Original execution measurement does not match its extraction")
    measurement.update(
        artifact_id=target_artifact_id,
        identity_migration={
            "reason": "E22-supervision-key-v1",
            "source_artifact_id": source_artifact_id,
            "source_measurement_sha256": source_sha256,
        },
    )
    return (json.dumps(measurement, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def verify_identity_transition(old, new, *, old_implementation, stage, parent=None):
    """Accept only the exact key repair, never changes to numerical parameters."""
    expected = copy.deepcopy(new)
    expected.pop("artifact_id")
    if stage == "extraction":
        expected["implementation"] = old_implementation
        expected["parameters"]["supervision"] = expected["parameters"].pop("resolved_supervision")
    if parent is not None:
        expected["parents"] = [parent]
    retained = copy.deepcopy(old)
    retained.pop("artifact_id")
    if retained != expected:
        raise ValueError("Historical control identity differs beyond the supervision-key repair")


def import_verified_controls(recipe, campaign, runtime, code, *, verify_only=False):
    """Account for actual migration work, retaining prior execution charges once."""
    if verify_only:
        return _import_verified_controls(recipe, campaign, runtime, code, verify_only=True)
    from exact.experiments.budget import BudgetLedger
    from tools.finalize_prepared_selection import charge_failed_finalization

    state = read(runtime / "budget.json")
    ledger = BudgetLedger(runtime / "budget.json", state["limits"])
    charge_failed_finalization(recipe, ledger, state)
    work_id = "preparation/E22/supervision-key-migration/" + uuid.uuid4().hex
    ledger.admit(work_id, group="reserve", seconds=0, forecast_known=False)
    started, status = time.time(), "failed"
    try:
        result = _import_verified_controls(recipe, campaign, runtime, code)
        status = "complete"
        return result
    finally:
        ledger.finish(
            work_id,
            start=started,
            end=time.time(),
            status=status,
            requests=0,
            tokens=0,
            actual_usd=0,
        )


def _import_verified_controls(recipe, campaign, runtime, code, *, verify_only=False):
    """Verify both original controls, then import without fits or duplicated charges."""
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments import harness
    from exact.experiments.campaign import (
        external_selection_result,
        load_campaign,
        materialize_campaign,
    )
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import CellRecovery, _code_identity, _hash

    settings = recipe["e22_recovery"]
    verified(settings["reuse_plan"])
    old_code = Path(settings["source_code"])
    old_runtime = verified(settings["runtime_source"])
    old_text = old_runtime.read_text()
    if (
        old_runtime != old_code / "exact/experiments/runtime.py"
        or old_text.count(OLD_LINE) != 1
        or (code / "exact/experiments/runtime.py").read_text()
        != old_text.replace(OLD_LINE, NEW_LINE)
    ):
        raise ValueError("Recovery adapter changed beyond the reviewed key repair")
    old_impl = _code_identity(old_code, evaluation=False)
    current_impl = _code_identity(code, evaluation=False)
    before, after = dict(old_impl["files"]), dict(current_impl["files"])
    before.pop("exact/experiments/runtime.py")
    after.pop("exact/experiments/runtime.py")
    if before != after or _code_identity(old_code, evaluation=True) != _code_identity(
        code, evaluation=True
    ):
        raise ValueError("Numerical implementation changed outside the recovery-key repair")
    if recipe["scientific_step"] != "E22":
        raise ValueError("This migration is restricted to the declared E22 controls")
    lock, _ = load_campaign(campaign)
    suite = materialize_campaign(campaign, runtime / "declarations/screen", stage="screen")
    source = next(s for s in suite.sources if s.config.experiment_id == "E22")
    experiments = {}
    for identifier in source.config.depends_on:
        step = next(s for s in lock.steps if s.id == identifier)
        if step.external_selection is None:
            raise ValueError("Control inheritance requires a verified historical selection")
        historical = external_selection_result(lock, step, campaign.parent)
        producer = next(s for s in suite.sources if s.config.experiment_id == identifier)
        experiments[identifier] = harness._bind_external_selection(producer, suite, historical)
    cells = harness.build_cells(
        suite,
        source,
        stage="screen",
        output_root=runtime.parent,
        inherited_overlay=harness.inherited_selection_overlay(
            {"experiments": experiments}, source.config.depends_on
        ),
    )
    if {c.arm_id for c in cells} != {
        "budget_25",
        "budget_100",
        "budget_400",
        "active_100",
        "label_free",
    } or len(cells) != 5:
        raise ValueError("E22 recovery must retain all five original arms")
    requests = {read(verified(item))["arm_id"]: item for item in settings["controls"]}
    if set(requests) != {"budget_25", "label_free"} or len(settings["controls"]) != 2:
        raise ValueError("Only the two independently executed controls are compatible")
    old_store = ArtifactStore(Path(settings["source_runtime"]))
    store = ArtifactStore(runtime)
    imports, identities = [], {}
    for cell in cells:
        expected = replace(cell, recovery={"root": str(runtime), "reuse_plan_only": True})
        recovery = CellRecovery(
            expected, harness._provenance_payload(expected, suite, workdir=code), code
        )
        identities[cell.arm_id] = recovery.identities["extraction"]["artifact_id"]
        if cell.arm_id not in requests:
            continue
        path = verified(requests[cell.arm_id])
        report = read(path)
        required = {
            "experiment_id": "E22",
            "arm_id": cell.arm_id,
            "task_id": cell.task_id,
            "seed": cell.seed,
            "stage": "screen",
            "status": "complete",
            "return_code": 0,
            "extraction_complete": True,
            "generate_rationales": False,
            "split_role": cell.split_role,
            "reference_role": cell.reference_role,
            "source_cap": cell.source_cap,
            "arm_role": cell.arm_role,
            "resolved_config_hash": cell.config_hash,
        }
        if any(report.get(k) != v for k, v in required.items()):
            raise ValueError("Original control role, population or configuration differs")
        if "extraction" in report["recovery"]["reused_stages"]:
            raise ValueError("A copied treatment cannot establish compatible fitted artifacts")
        if load_yaml_mapping(path.parent / "_inputs/resolved.config.yaml") != cell.resolved_config:
            raise ValueError("Original control full supervision/configuration differs")
        artifacts = report["recovery"]["artifacts"]
        if set(artifacts) != {"inputs", "extraction", "evaluation"}:
            raise ValueError("Original control has incomplete artifacts")
        saved = {stage: old_store.verify(identifier) for stage, identifier in artifacts.items()}
        for stage, payload in saved.items():
            verify_identity_transition(
                payload["identity"],
                recovery.identities[stage],
                old_implementation=old_impl,
                stage=stage,
                parent=artifacts["extraction"] if stage == "evaluation" else None,
            )
            if stage != "inputs":
                for name, output in payload["outputs"].items():
                    if binding(path.parent / name)["sha256"] != output["sha256"]:
                        raise ValueError("Original exposed control bytes differ: " + name)
        if cell.arm_id == "budget_25":
            units = list((path.parent / "fitting").glob("**/training_units.json"))
            if len(units) != 1:
                raise ValueError("Control training units are ambiguous")
            training = read(units[0])
            if (training["requested_groups"], training["selection"], training["seed"]) != (
                25,
                "passive",
                17,
            ):
                raise ValueError("Control training recipe differs")
        measurement = saved["extraction"]["outputs"].get("stats/execution_measurement.json")
        if measurement is None:
            raise ValueError("Original control lacks its execution measurement")
        migrated_measurement = migrate_execution_measurement(
            old_store._blob(measurement["sha256"]).read_bytes(),
            source_artifact_id=artifacts["extraction"],
            target_artifact_id=recovery.identities["extraction"]["artifact_id"],
            source_sha256=measurement["sha256"],
        )
        imports.append((cell, recovery, report, path, saved, migrated_measurement))
    if len(set(identities.values())) != 5:
        raise ValueError("Distinct E22 supervision recipes still share prediction identity")
    records = []
    for cell, recovery, report, path, saved, migrated_measurement in imports:
        migrated = {k: v["artifact_id"] for k, v in recovery.identities.items()}
        records.append(
            {
                "arm": cell.arm_id,
                "source_manifest": binding(path),
                "original_artifacts": report["recovery"]["artifacts"],
                "migrated_artifacts": migrated,
            }
        )
        if verify_only:
            continue
        for stage in ("inputs", "extraction", "evaluation"):
            outputs: dict[str, Path | bytes] = {
                name: old_store._blob(output["sha256"])
                for name, output in saved[stage]["outputs"].items()
            }
            if stage == "extraction":
                outputs["stats/execution_measurement.json"] = migrated_measurement
            store.publish(
                recovery.identities[stage],
                outputs,
            )
        index = runtime / "recovery/cells" / (_hash([cell.suite_id, cell.cell_id]) + ".json")
        if not index.exists():
            write(
                index,
                {
                    "artifacts": migrated,
                    "attempt_id": report["recovery"]["attempt_id"],
                    "output_dir": str(path.parent),
                    "migration": "E22-supervision-key-v1",
                    "source_manifest": binding(path),
                },
                immutable=True,
            )
    receipt = {
        "schema_version": 1,
        "controls": records,
        "extraction_identities": identities,
        "new_scientific_executions": 0,
        "original_charges_retained": True,
        "verify_only": verify_only,
    }
    write(
        runtime / ("control-verification.json" if verify_only else "control-imports.json"),
        receipt,
        immutable=True,
    )
    return receipt
