"""Bind selected donor outputs to at most two development recipients; never refit."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from exact.core.entities.configs.config import ConfigModel
from exact.experiments.fitting_recipes import _supervision
from exact.utils.artifact_transfer import (
    freeze_transfer_manifest,
    transfer_feature_contract,
    validate_transfer_config,
)
from exact.utils.fitted_artifacts import fingerprint, freeze_json
from exact.utils.provenance import (
    dataset_signature_for_paths,
    file_provenance,
    sha256_file,
)


def _find_artifact(output, declared, filename):
    paths = [Path(declared)] if declared else sorted((output / "fitting").glob("**/" + filename))
    if len(paths) != 1 or not paths[0].is_file():
        raise ValueError(f"Selected donor requires one immutable {filename} artifact")
    return paths[0].resolve()


def materialize_transfer(source, suite, manifests, selections):
    """Resolve three E16 arms from one selected E18 run and frozen recipient inputs."""
    from exact.experiments.harness import (
        ExperimentSource,
        _inventory_config,
        deep_merge,
    )
    from exact.experiments.schema import ExperimentConfig

    settings = source.config.frozen_constants.get("donor_transfer")
    if not settings:
        return source
    producer = settings.get("producer", "E18")
    if producer != "E18":
        raise ValueError("The bounded transfer producer is the selected E18 donor head")
    stages = source.config.screen
    if (
        not 1 <= len(stages.tasks) <= 2
        or stages.seeds != [17]
        or any(task.split_role != "development" for task in stages.tasks)
    ):
        raise ValueError("E16 binds one or two frozen development recipients at seed 17")
    if {arm.id for arm in source.config.arms} != {
        "label_free",
        "in_pair_supervised",
        "donor_transfer",
    }:
        raise ValueError("E16 requires the three declared supervision comparators")
    selection = selections.get("experiments", selections).get(producer, {})
    decisions = selection.get("decisions", [])
    if selection.get("status") not in {"selected", "screened_out"} or len(decisions) != 1:
        raise ValueError("E16 requires one completed selected-head decision")
    donor_arm = decisions[0].get("selected_arm") or decisions[0].get("baseline")
    eligible = [
        item
        for item in manifests
        if item.get("experiment_id") == producer
        and item.get("arm_id") == donor_arm
        and item.get("stage") == "screen"
        and item.get("seed") == 17
        and item.get("status") == "complete"
    ]
    if len(eligible) != 1:
        raise ValueError("E16 requires one completed selected donor cell")
    donor_manifest = eligible[0]
    output = Path(donor_manifest["fingerprint_payload"]["output_dir"]).resolve()
    donor = ConfigModel.load_config(output / "_inputs" / "resolved.config.yaml")
    if donor.fingerprint() != donor_manifest.get("resolved_config_hash"):
        raise ValueError("Selected donor resolved configuration hash mismatch")
    if (
        donor.matching.calibration.threshold_mode != "fixed"
        or donor.matching.threshold is None
        or donor.matching.channels.graph.mode != "off"
        or donor.matching.nil.mode != "off"
        or donor.matching.relation_prediction != "none"
    ):
        raise ValueError(
            "Selected donor has unsupported adaptive threshold, graph, NIL or relation state"
        )
    llm = donor.llm.experiment
    if llm.enabled and (llm.gate.mode != "off" or llm.exemplars != "off" or llm.distill != "off"):
        raise ValueError("Selected donor has unsupported fitted LLM transfer state")
    artifacts = {
        "selector": _find_artifact(output, donor.selector.rerank.artifact, "selector.json")
    }
    if donor.matching.fusion.mode == "learned_adaptive":
        raise ValueError("Selected donor has unsupported adaptive fusion transfer state")
    fusion_fitted = donor.matching.fusion.mode in {"analytic_fitted", "learned_global"}
    if fusion_fitted:
        artifacts["fusion"] = _find_artifact(output, donor.matching.fusion.artifact, "fusion.json")
    calibration_fitted = donor.matching.calibration.mode != "none"
    if calibration_fitted:
        artifacts["calibration"] = _find_artifact(
            output, donor.matching.calibration.artifact, "score_calibrator.json"
        )
    payloads = {kind: json.loads(path.read_text()) for kind, path in artifacts.items()}
    train_sources = set(payloads["selector"]["fit_provenance"]["training_sources"])
    donor_signature = payloads["selector"]["fit_provenance"]["application"]["dataset_signature"]
    if donor_signature != dataset_signature_for_paths(donor.data.source, donor.data.target):
        raise ValueError("Selected donor artifact belongs to different ontology inputs")
    donor_inputs = {
        side: file_provenance(getattr(donor.data, side)) for side in ("source", "target")
    }
    # Retain the exact donor score-producing recipe. The LF control disables its
    # fitted heads; the supervised control refits those heads on recipient train.
    mapping = donor.model_dump(mode="json")
    recipe = {
        key: mapping[key]
        for key in (
            "pipeline",
            "matching",
            "dataset",
            "selector",
            "llm",
        )
    }
    recipe["selector"]["rerank"]["artifact"] = str(artifacts["selector"])
    if fusion_fitted:
        recipe["matching"]["fusion"]["artifact"] = str(artifacts["fusion"])
    if calibration_fitted:
        recipe["matching"]["calibration"]["artifact"] = str(artifacts["calibration"])
    recipe["supervision"] = {**_supervision(), "transfer_artifact": None}
    recipe["candidates"] = {
        "encoder_finetune": {"training": None},
        "cross_encoder": {"training": None},
    }
    destination = output.parents[4] / "policies" / source.config.experiment_id
    applications = {}
    for task in stages.tasks:
        recipient = _inventory_config(source, task, "screen")
        if any(
            path is None or not Path(path).is_file()
            for path in (
                recipient.data.source,
                recipient.data.target,
                recipient.data.source_universe,
            )
        ):
            raise ValueError(
                "Transfer recipient requires explicit ontologies and frozen source population"
            )
        report_sources = set(recipient.data.source_universe.read_text().splitlines())
        if train_sources & report_sources:
            raise ValueError("Transfer recipient reporting sources overlap donor training labels")
        signature = dataset_signature_for_paths(recipient.data.source, recipient.data.target)
        if signature == donor_signature or signature in applications:
            raise ValueError("Transfer recipients must be distinct from the donor and each other")
        if recipient.matching.entity_kinds != donor.matching.entity_kinds:
            raise ValueError("Transfer cannot change the selected entity-kind contract")
        config = ConfigModel.from_mapping(deep_merge(recipient.model_dump(mode="python"), recipe))
        recipient_inputs = {
            side: file_provenance(getattr(recipient.data, side)) for side in ("source", "target")
        }
        provenance = {
            "producer": producer,
            "selected_arm": donor_arm,
            "donor_config_sha256": sha256_file(output / "_inputs" / "resolved.config.yaml"),
            "selection_sha256": fingerprint(selection),
            "donor_inputs": donor_inputs,
            "recipient_inputs": recipient_inputs,
            "recipient_population": {
                "path": str(recipient.data.source_universe.resolve()),
                "sha256": sha256_file(recipient.data.source_universe),
            },
            "recipient_source_ids_sha256": fingerprint(sorted(report_sources)),
            "donor_label_source_overlap": 0,
            "shared_ontology_bytes": sorted(
                {value["sha256"] for value in donor_inputs.values()}
                & {value["sha256"] for value in recipient_inputs.values()}
            ),
        }
        path = destination / (task.id + "." + signature + ".json")
        freeze_transfer_manifest(
            path,
            artifacts,
            recipient_signature=signature,
            feature_contract=transfer_feature_contract(config),
            score_threshold=donor.matching.threshold,
            entity_kinds=donor.matching.entity_kinds,
            application_provenance=provenance,
        )
        applications[signature] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "task": task.id,
        }
    bundle_path = destination / ("applications." + fingerprint(applications)[:20] + ".json")
    freeze_json(
        bundle_path,
        {"schema_version": 1, "kind": "cross_pair_transfer_bundle", "applications": applications},
    )
    declaration = source.config.model_dump(mode="json")
    declaration["base_config"] = str(source.base_config_path)
    for arm in declaration["arms"]:
        overlay = deepcopy(recipe)
        if arm["id"] == "donor_transfer":
            overlay["supervision"]["transfer_artifact"] = str(bundle_path.resolve())
            overlay["data"] = {"train_candidates": None}
        else:
            overlay["selector"]["rerank"]["artifact"] = None
            overlay["matching"]["fusion"]["artifact"] = None
            overlay["matching"]["calibration"]["artifact"] = None
            if arm["id"] == "in_pair_supervised":
                components = (
                    ["rerank", "accept"]
                    + (["fusion"] if fusion_fitted else [])
                    + (["calibration"] if calibration_fitted else [])
                )
                overlay["supervision"] = {**_supervision(*components), "transfer_artifact": None}
            else:
                overlay["selector"]["label_free_mode"] = (
                    arm.get("overlay", {})
                    .get("selector", {})
                    .get("label_free_mode", overlay["selector"]["label_free_mode"])
                )
                if fusion_fitted:
                    overlay["matching"]["fusion"]["mode"] = "analytic_shipped"
                overlay["matching"]["calibration"]["mode"] = "none"
        arm["overlay"] = deep_merge(arm["overlay"], overlay)
        for task in stages.tasks:
            config = ConfigModel.from_mapping(
                deep_merge(
                    _inventory_config(source, task, "screen").model_dump(mode="python"),
                    arm["overlay"],
                )
            )
            if arm["id"] == "donor_transfer":
                validate_transfer_config(config)
    declaration["frozen_constants"]["resolved_donor_transfer"] = {
        "producer": producer,
        "selected_arm": donor_arm,
        "bundle_sha256": sha256_file(bundle_path),
    }
    resolved = ExperimentConfig.model_validate(declaration)
    path = destination / (
        "resolved." + fingerprint(resolved.model_dump(mode="json"))[:20] + ".json"
    )
    freeze_json(path, resolved.model_dump(mode="json"))
    return ExperimentSource(resolved, path)
