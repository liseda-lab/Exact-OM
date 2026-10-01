"""Reuse completed E22 treatments after repairing the two control measurements."""

from dataclasses import replace
from pathlib import Path

from tools.prepared_batch import binding, read, verified, write

TREATMENTS = {"budget_100", "budget_400", "active_100"}
STAGES = ("inputs", "extraction", "evaluation")


def verify_completed_cell(cell, recovery, path, store, *, pool_fingerprint):
    """Check numerical identity, independent execution, and every exposed byte."""
    from exact.core.entities.configs.yaml_io import load_yaml_mapping

    report = read(path)
    required = dict(
        experiment_id="E22",
        arm_id=cell.arm_id,
        task_id=cell.task_id,
        seed=cell.seed,
        stage="screen",
        status="complete",
        return_code=0,
        extraction_complete=True,
        generate_rationales=False,
        split_role=cell.split_role,
        reference_role=cell.reference_role,
        source_cap=cell.source_cap,
        arm_role=cell.arm_role,
        resolved_config_hash=cell.config_hash,
        candidate_pool_fingerprint=pool_fingerprint,
    )
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("Completed E22 treatment configuration, role, or pool differs")
    if load_yaml_mapping(path.parent / "_inputs/resolved.config.yaml") != cell.resolved_config:
        raise ValueError("Completed E22 treatment full configuration differs")
    previous = report["recovery"]
    if "extraction" in previous["reused_stages"]:
        raise ValueError("Completed E22 treatment was not independently executed")
    expected = {stage: recovery.identities[stage]["artifact_id"] for stage in STAGES}
    if previous["artifacts"] != expected:
        raise ValueError("Completed E22 treatment artifact identity differs")
    pool = report["candidate_pool_manifest_provenance"]
    pool_path = verified({key: pool[key] for key in ("path", "sha256")})
    if pool_path.parent != path.parent / "dataset":
        raise ValueError("Completed E22 treatment pool provenance is outside its output")
    if read(pool_path)["fingerprint"] != pool_fingerprint:
        raise ValueError("Completed E22 treatment pool fingerprint differs")
    saved = {}
    for stage in STAGES:
        payload = store.verify(expected[stage])
        if payload["identity"] != recovery.identities[stage]:
            raise ValueError("Completed E22 treatment stored identity differs")
        if stage != "inputs":
            for name, output in payload["outputs"].items():
                if binding(path.parent / name)["sha256"] != output["sha256"]:
                    raise ValueError("Completed E22 treatment exposed bytes differ: " + name)
        saved[stage] = payload
    measurement = saved["extraction"]["outputs"]["stats/execution_measurement.json"]
    if read(store._blob(measurement["sha256"]))["artifact_id"] != expected["extraction"]:
        raise ValueError("Completed E22 treatment execution measurement identity differs")
    return report, saved


def _terminal_source(settings):
    root = Path(settings["source_runtime"]).parent.parent
    completion = read(root / "completion.json")
    exit_code = int((root / "exit-code").read_text().strip())
    if (
        completion.get("status") != "failed"
        or exit_code == 0
        or completion.get("exit_code") != exit_code
    ):
        raise ValueError("E22 source must have matching terminal failure and exit receipts")
    campaign = verified(settings["source_campaign"])
    if campaign != root / "campaign.lock.yaml":
        raise ValueError("E22 source campaign is outside its recorded runtime")
    return binding(root / "completion.json")


def import_completed_treatments(recipe, campaign, runtime, code, *, verify_only=False):
    """Fail closed unless all five arms can be restored without numerical work."""
    from exact.experiments import harness
    from exact.experiments.campaign import (
        external_selection_result,
        load_campaign,
        materialize_campaign,
    )
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import CellRecovery, _code_identity, _hash

    settings = recipe["e22_completed_treatments"]
    if recipe["scientific_step"] != "E22" or settings["parent_run_id"] != recipe["parent_run_id"]:
        raise ValueError("Completed treatment import is restricted to its E22 predecessor")
    terminal = _terminal_source(settings)
    old_code = Path(settings["source_code"])
    if any(
        _code_identity(old_code, evaluation=part) != _code_identity(code, evaluation=part)
        for part in (False, True)
    ):
        raise ValueError("Completed E22 treatment numerical implementation differs")
    controls = read(runtime / "control-imports.json")
    if {row["arm"] for row in controls["controls"]} != {"budget_25", "label_free"}:
        raise ValueError("Both verified E22 controls must be imported first")
    pools = {
        read(verified(row["source_manifest"]))["candidate_pool_fingerprint"]
        for row in controls["controls"]
    }
    if len(pools) != 1 or not next(iter(pools)):
        raise ValueError("Verified E22 controls have unequal candidate pools")
    lock, _ = load_campaign(campaign)
    suite = materialize_campaign(campaign, runtime / "declarations/screen", stage="screen")
    source = next(s for s in suite.sources if s.config.experiment_id == "E22")
    experiments = {}
    for identifier in source.config.depends_on:
        step = next(s for s in lock.steps if s.id == identifier)
        if step.external_selection is None:
            raise ValueError("Completed treatment inheritance needs historical selection")
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
    if len(cells) != 5 or {c.arm_id for c in cells} != TREATMENTS | {"budget_25", "label_free"}:
        raise ValueError("E22 treatment import requires the original five cells")
    old_runtime = Path(settings["source_runtime"])
    old_store, store = ArtifactStore(old_runtime), ArtifactStore(runtime)
    identities, imports = {}, []
    for cell in cells:
        expected = replace(cell, recovery={"root": str(runtime), "reuse_plan_only": True})
        recovery = CellRecovery(
            expected, harness._provenance_payload(expected, suite, workdir=code), code
        )
        identities[cell.arm_id] = recovery.identities["extraction"]["artifact_id"]
        if cell.arm_id not in TREATMENTS:
            for stage in STAGES:
                store.verify(recovery.identities[stage]["artifact_id"])
            continue
        path = old_runtime / cell.output_dir.relative_to(runtime) / "experiment_manifest.json"
        report, saved = verify_completed_cell(
            cell,
            recovery,
            path,
            old_store,
            pool_fingerprint=next(iter(pools)),
        )
        imports.append((cell, recovery, path, report, saved))
    if len(set(identities.values())) != 5 or identities != controls["extraction_identities"]:
        raise ValueError("Completed treatment identities differ from the verified control plan")
    # No artifact/index writes occur until every source and target has been checked.
    verified(terminal)
    records = []
    for cell, recovery, path, report, saved in imports:
        records.append(
            {
                "arm": cell.arm_id,
                "source_manifest": binding(path),
                "artifacts": report["recovery"]["artifacts"],
            }
        )
        if verify_only:
            continue
        for stage in STAGES:
            store.publish(
                recovery.identities[stage],
                {
                    name: old_store._blob(output["sha256"])
                    for name, output in saved[stage]["outputs"].items()
                },
            )
        write(
            runtime / "recovery/cells" / (_hash([cell.suite_id, cell.cell_id]) + ".json"),
            {
                "artifacts": report["recovery"]["artifacts"],
                "attempt_id": report["recovery"]["attempt_id"],
                "output_dir": str(path.parent),
                "migration": "E22-completed-treatments-v1",
                "source_manifest": binding(path),
            },
            immutable=True,
        )
    if not verify_only:
        for identifier in identities.values():
            store.verify(identifier)
    receipt = dict(
        schema_version=1,
        treatments=records,
        extraction_identities=identities,
        source_completion=terminal,
        new_scientific_executions=0,
        original_charges_retained=True,
        verify_only=verify_only,
    )
    write(
        runtime / ("treatment-verification.json" if verify_only else "treatment-imports.json"),
        receipt,
        immutable=True,
    )
    return receipt
