"""Reuse verified raw E22 features and its label-free control after budget enforcement."""

from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

from tools.prepared_batch import binding, read, verified, write

FITTING = "exact/impl/trainer/fitting.py"
POLICY = "exact/experiments/label_policy.py"
SUPERVISED = {"budget_25", "budget_100", "budget_400", "active_100"}
STAGES = ("inputs", "extraction", "evaluation")


def verify_code_change(old_code, code, repair):
    """Permit only checksum-pinned budget fixes and unchanged raw feature generation."""
    from exact.experiments.runtime import _code_identity

    old = _code_identity(old_code, evaluation=False)
    new = _code_identity(code, evaluation=False)
    changes = repair.get("changes", {})
    if repair.get("schema_version") != 1 or set(changes) not in ({FITTING}, {FITTING, POLICY}):
        raise ValueError("Only the explicit E22 label-budget repair may migrate features")
    before, after = dict(old["files"]), dict(new["files"])
    for name, change in changes.items():
        if before.pop(name, None) != change["before"] or after.pop(name, None) != change["after"]:
            raise ValueError("Approved label-budget implementation hashes differ")
    if before != after or _code_identity(old_code, evaluation=True) != _code_identity(
        code, evaluation=True
    ):
        raise ValueError("Implementation changed outside the reviewed label-budget repair")
    start = "        raw = read_table(Path(path))\n"
    end = "        from exact.impl.models.selector.label_budget import select_label_budget\n"
    segments = []
    for root in (old_code, code):
        text = (root / FITTING).read_text()
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError("Raw feature generation boundary is ambiguous")
        segments.append(text.split(start, 1)[1].split(end, 1)[0])
    if segments[0] != segments[1]:
        raise ValueError("Raw training feature generation changed")
    return old


def feature_config(config):
    """Retain every input and scorer setting; remove only downstream selector controls."""
    value = copy.deepcopy(config)
    supervision = value.pop("supervision", {})
    modes = supervision.get("components", {})
    if (
        supervision.get("transfer_artifact")
        or supervision.get("inference_artifact")
        or any(
            mode != "label_free" for name, mode in modes.items() if name not in {"rerank", "accept"}
        )
        or value.get("matching", {}).get("fusion", {}).get("mode") not in {None, "analytic_shipped"}
        or value.get("matching", {}).get("channels", {}).get("graph", {}).get("mode")
        not in {None, "off"}
        or (
            value.get("llm", {}).get("experiment", {}).get("enabled", True)
            and value.get("llm", {}).get("experiment", {}).get("gate", {}).get("mode") != "off"
        )
        or value.get("llm", {}).get("experiment", {}).get("exemplars") not in {None, "off"}
        or value.get("llm", {}).get("experiment", {}).get("distill") not in {None, "off"}
    ):
        raise ValueError("Raw feature migration requires the unchanged unfitted E22 scorer")
    value["training_negative_label_policy"] = supervision.get("negative_label_policy", "unknown")
    value.pop("selector", None)
    value["pipeline"] = [
        item for item in value.get("pipeline", []) if item.get("name") != "CandidateSetSelector"
    ]
    return value


def verify_raw_identity(source, target):
    """A derived count rule acts after raw scores; all feature inputs remain exact."""

    def inputs(identity):
        return {
            name: digest
            for name, digest in identity["inputs"].items()
            if name != "artifact/supervision.auto_policy.artifact"
        }

    if inputs(source) != inputs(target) or any(
        source[key] != target[key] for key in ("dependencies", "role", "entity_kind", "seed")
    ):
        raise ValueError("Raw feature inputs, dependencies, role, kind, or seed differ")


def verify_raw_snapshot(snapshot, source_runtime):
    """Verify complete raw files, never fitted heads, predictions, or temporary shards."""
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.utils.fitted_artifacts import fingerprint

    identity = snapshot.get("training_identity", "")
    if snapshot.get("schema_version") != 1 or re.fullmatch(r"[a-f0-9]{64}", identity) is None:
        raise ValueError("Invalid raw feature snapshot identity")
    config_path = verified(snapshot["source_config"])
    source_cell = config_path.parent.parent
    if (
        config_path.name != "resolved.config.yaml"
        or config_path.parent.name != "_inputs"
        or not source_cell.is_relative_to(source_runtime / "screen/runs")
    ):
        raise ValueError("Raw feature source is outside the recorded development runtime")
    config = load_yaml_mapping(config_path)
    feature_config(config)
    files, names, sources = [], set(), set()
    for item in snapshot.get("files", []):
        name = item["name"]
        if name in names or (
            name != "training_scores.json" and re.fullmatch(r"[a-f0-9]{64}\.json", name) is None
        ):
            raise ValueError("Only unique complete raw feature files may be reused")
        path = verified({key: item[key] for key in ("path", "sha256")})
        if path != source_cell / "fitting" / identity / name:
            raise ValueError("Raw feature file is outside the declared source identity")
        names.add(name)
        if name == "training_scores.json":
            # The 2+ GB full frame is authenticated against its completed CAS artifact,
            # avoiding a second in-memory copy just to validate the migration.
            from exact.experiments.recovery import ArtifactStore

            manifest_path = verified(snapshot["source_manifest"])
            manifest = read(manifest_path)
            if manifest_path.parent != source_cell or manifest.get("status") != "complete":
                raise ValueError("Consolidated raw features need a completed source cell")
            store = ArtifactStore(source_runtime)
            payload = read(store._manifest_path(manifest["recovery"]["artifacts"]["extraction"]))
            relative = path.relative_to(source_cell).as_posix()
            if payload["outputs"][relative]["sha256"] != item["sha256"]:
                raise ValueError("Raw feature snapshot differs from completed extraction bytes")
            with path.open() as stream:
                header = stream.read(160)
            if not re.search(r'"identity"\s*:\s*"' + identity + '"', header):
                raise ValueError("Raw training feature identity differs")
        else:
            shard = read(path)
            members = shard["source_ids"]
            observed = {str(row["Src"]) for row in shard["rows"]}
            pairs = [(str(row["Src"]), str(row["Tgt"])) for row in shard["rows"]]
            if (
                not members
                or len(set(members)) != len(members)
                or observed != set(members)
                or sources & observed
                or len(set(pairs)) != len(pairs)
                or fingerprint(members) != path.stem
            ):
                raise ValueError("Raw feature shard membership or identity differs")
            sources.update(observed)
        files.append((name, path, item["sha256"]))
    if not files or ("training_scores.json" in names and len(names) != 1):
        raise ValueError("Use one consolidated frame or disjoint complete feature shards")
    return config, files


def verify_label_free(cell, recovery, source, old_store, old_impl):
    """Validate the sole fitting-independent control and rebind its measured identity."""
    from exact.core.entities.configs.yaml_io import load_yaml_mapping

    if cell.arm_id != "label_free":
        raise ValueError("Supervised E22 predictions cannot be migrated")
    report = read(source)
    config = load_yaml_mapping(source.parent / "_inputs/resolved.config.yaml")
    modes = config.get("supervision", {}).get("components", {})
    feature_config(config)
    if (
        not modes
        or any(value != "label_free" for value in modes.values())
        or config["supervision"].get("label_budget") is not None
        or (source.parent / "fitting").exists()
    ):
        raise ValueError("The reusable E22 control must not fit any supervised component")
    expected = dict(
        experiment_id="E22",
        arm_id="label_free",
        stage="screen",
        status="complete",
        return_code=0,
        extraction_complete=True,
        generate_rationales=False,
        task_id=cell.task_id,
        seed=cell.seed,
        split_role=cell.split_role,
        reference_role=cell.reference_role,
        source_cap=cell.source_cap,
        arm_role=cell.arm_role,
        resolved_config_hash=cell.config_hash,
    )
    if config != cell.resolved_config or any(
        report.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Label-free control configuration, role, or population differs")
    artifacts = report["recovery"]["artifacts"]
    if set(artifacts) != set(STAGES):
        raise ValueError("Label-free control lacks complete verified artifacts")
    saved = {}
    for stage in STAGES:
        payload = old_store.verify(artifacts[stage])
        expected_identity = copy.deepcopy(recovery.identities[stage])
        expected_identity.pop("artifact_id")
        if stage == "extraction":
            expected_identity["implementation"] = old_impl
        elif stage == "evaluation":
            expected_identity["parents"] = [artifacts["extraction"]]
        actual = dict(payload["identity"])
        actual.pop("artifact_id")
        if actual != expected_identity:
            raise ValueError("Label-free identity differs beyond the approved fitting-only repair")
        if stage != "inputs":
            for name, output in payload["outputs"].items():
                if binding(source.parent / name)["sha256"] != output["sha256"]:
                    raise ValueError("Label-free exposed outputs differ from stored bytes")
        saved[stage] = payload
    measurement = saved["extraction"]["outputs"]["stats/execution_measurement.json"]
    original = read(old_store._blob(measurement["sha256"]))
    if (
        original.get("schema_version") != 1
        or original.get("artifact_id") != artifacts["extraction"]
    ):
        raise ValueError("Label-free execution measurement does not match its predictions")
    migrated = {
        **original,
        "artifact_id": recovery.identities["extraction"]["artifact_id"],
        "identity_migration": {
            "reason": "E22-label-budget-enforcement-v1",
            "source_artifact_id": artifacts["extraction"],
            "source_measurement_sha256": measurement["sha256"],
            "previous": original.get("identity_migration"),
        },
    }
    return report, saved, (json.dumps(migrated, indent=2, sort_keys=True) + "\n").encode()


def _cells(recipe, campaign, runtime):
    from exact.experiments import harness
    from exact.experiments.campaign import (
        external_selection_result,
        load_campaign,
        materialize_campaign,
    )

    lock, _ = load_campaign(campaign)
    suite = materialize_campaign(campaign, runtime / "declarations/screen", stage="screen")
    source = next(s for s in suite.sources if s.config.experiment_id == recipe["scientific_step"])
    experiments, manifests = {}, []
    for identifier in source.config.depends_on:
        step = next(s for s in lock.steps if s.id == identifier)
        if step.external_selection is None:
            raise ValueError("Label recovery requires verified completed producer selections")
        historical = external_selection_result(
            lock, step, campaign.parent, producer_manifests=manifests
        )
        producer = next(s for s in suite.sources if s.config.experiment_id == identifier)
        experiments[identifier] = harness._bind_external_selection(producer, suite, historical)
    inherited = harness.inherited_selection_overlay(
        {"experiments": experiments}, source.config.depends_on
    )
    source, inherited = harness._materialize_campaign_evidence(
        source, suite, [], experiments, inherited, external_manifests=manifests
    )
    cells = harness.build_cells(
        suite, source, stage="screen", output_root=runtime.parent, inherited_overlay=inherited
    )
    expected = (
        SUPERVISED | {"label_free"}
        if recipe["scientific_step"] == "E22"
        else {"fixed_count", "fitted_count"}
    )
    if len(cells) != len(expected) or {cell.arm_id for cell in cells} != expected:
        raise ValueError("E22 label recovery must retain every declared original arm")
    return suite, cells


def prepare_label_recovery(recipe, campaign, runtime, code, *, verify_only=False):
    """Seed raw scores and one label-free control; all supervised outputs run normally."""
    from exact.experiments import harness
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import CellRecovery, _hash

    if recipe["scientific_step"] not in {"E22", "E22-policy"}:
        raise ValueError("Label repair applies only to E22 and its policy follow-up")
    settings = recipe["e22_label_repair"]
    source_runtime = Path(settings["source_runtime"])
    old_impl = verify_code_change(
        Path(settings["source_code"]), code, read(verified(settings["repair_record"]))
    )
    snapshot = read(verified(settings["raw_features"]))
    config, files = verify_raw_snapshot(snapshot, source_runtime)
    runtime_binding = verified(snapshot["source_runtime_binding"])
    if (
        runtime_binding
        != Path(snapshot["source_config"]["path"]).parent.parent / "recovery-runtime.json"
    ):
        raise ValueError("Raw feature runtime binding is outside its source cell")
    source_identity = read(runtime_binding)["identity"]
    if source_identity["implementation"] != old_impl:
        raise ValueError("Raw features were produced by a different implementation")
    suite, cells = _cells(recipe, campaign, runtime)
    store, old_store = ArtifactStore(runtime), ArtifactStore(source_runtime)
    imports, targets, identities = [], [], {}
    for cell in cells:
        recovery = CellRecovery(
            replace(cell, recovery={"root": str(runtime), "reuse_plan_only": True}),
            harness._provenance_payload(cell, suite, workdir=code),
            code,
        )
        identities[cell.arm_id] = recovery.identities["extraction"]["artifact_id"]
        if cell.arm_id == "label_free":
            source = verified(settings["label_free"])
            if not source.is_relative_to(source_runtime / "screen/runs/E22/label_free"):
                raise ValueError("Label-free control is outside the recorded source runtime")
            report, saved, measurement = verify_label_free(
                cell, recovery, source, old_store, old_impl
            )
            imports.append((cell, recovery, source, report, saved, measurement))
        else:
            identity = recovery.identities["extraction"]
            verify_raw_identity(source_identity, identity)
            if feature_config(config) != feature_config(cell.resolved_config):
                raise ValueError(
                    "Raw training features differ in scorer settings or input bindings"
                )
            for name, path, digest in files:
                target = cell.output_dir / "fitting" / snapshot["training_identity"] / name
                if target.exists() and binding(target)["sha256"] != digest:
                    raise ValueError("Destination raw feature file already differs")
                targets.append((path, target, digest))
    if len(set(identities.values())) != len(cells):
        raise ValueError("Distinct label treatments share prediction identity")
    if not verify_only:
        for path, target, digest in targets:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".migration-partial")
                shutil.copyfile(path, temporary)
                if binding(temporary)["sha256"] != digest:
                    raise ValueError("Copied raw feature checksum differs")
                temporary.replace(target)
        for cell, recovery, source, report, saved, measurement in imports:
            for stage in STAGES:
                outputs = {
                    name: old_store._blob(item["sha256"])
                    for name, item in saved[stage]["outputs"].items()
                }
                if stage == "extraction":
                    outputs["stats/execution_measurement.json"] = measurement
                store.publish(recovery.identities[stage], outputs)
            write(
                runtime / "recovery/cells" / (_hash([cell.suite_id, cell.cell_id]) + ".json"),
                {
                    "artifacts": {
                        stage: recovery.identities[stage]["artifact_id"] for stage in STAGES
                    },
                    "attempt_id": report["recovery"]["attempt_id"],
                    "output_dir": str(source.parent),
                    "migration": "E22-label-budget-enforcement-v1",
                    "source_manifest": binding(source),
                },
                immutable=True,
            )
    receipt = {
        "schema_version": 1,
        "repair_record": settings["repair_record"],
        "raw_features": settings["raw_features"],
        "seeded_files": len(targets),
        "controls": [cell.arm_id for cell, *_ in imports],
        "extraction_identities": identities,
        "supervised_predictions_imported": 0,
        "original_charges_retained": True,
        "verify_only": verify_only,
    }
    write(
        runtime
        / ("label-repair-verification.json" if verify_only else "label-repair-imports.json"),
        receipt,
        immutable=True,
    )
    return receipt
