"""Authenticate a completed teacher panel and carry its quotas into TRAIN/DEV.

This module performs offline review only. A selected teacher is not primary-fit
admission, grounded corpus coverage, or permission to repeat calibration.
"""

from collections import Counter
import json
from pathlib import Path
import sqlite3

from exact.repair.records import canonical_hash, read_record
from exact.repair.semantic_fidelity import _receipt_response
from tools.repair.batch import read, sha
from tools.repair.historical_regression import binding
from tools.repair.shared_release import bound, immutable, validate_completion

DATA_PERMISSIONS = {
    "train": "User-authorized corrective shared generated TRAIN; exact grounded complete plans only; no TEST outcomes.",
    "development": "User-authorized frozen DEV selection slots; exact freshly decoded complete plans; no TEST outcomes.",
}


def review(prepared_path, registry_path, output):
    from tools.repair.corrective_calibration import summarize
    from tools.repair.teacher_panel import panel_contract

    prepared, registry = read(prepared_path), read(registry_path)
    output = Path(output)
    summary_path = Path(prepared["summary_manifest"]["path"])
    summary = bound(prepared["summary_manifest"])
    if not summary.get("postprocessing_only") or len(summary["slots"]) != 32:
        raise ValueError("Expected the complete frozen 32-slot postprocessing contract")
    receipts, rows = [], []
    models = {}
    for profile, manifest_ref in prepared["manifests"].items():
        manifest = bound(manifest_ref)
        proposal, prior, _ = panel_contract(manifest)
        candidates = [
            r
            for r in registry["runs"]
            if r["id"] == f"lower-cost-teacher-panel-approved-001-{profile}"
        ]
        if len(candidates) != 1:
            raise ValueError("Missing or ambiguous registered panel worker")
        run = candidates[0]
        attempt = Path(run["completion_path"]).parent
        completion = read(run["completion_path"])
        receipt = {k: binding(attempt / (k + ".json")) for k in ("completion", "step", "outputs")}
        receipt.update(
            batch=binding(Path(completion["batch"])),
            dispatch_nonce=run["dispatch_nonce"],
            step_id=run["step_id"],
            expected_status="complete",
        )
        terminal, outputs, batch, _ = validate_completion(receipt)
        if batch["frozen_files"].get(manifest_ref["path"]) != manifest_ref["sha256"]:
            raise ValueError("Panel manifest is not a frozen worker input")
        work = Path(terminal["work"])
        for relative, digest in outputs.items():
            if sha(work / relative) != digest:
                raise ValueError("Panel terminal output changed")
        report = read(work / "report.json")
        expected = [r for r in summary["slots"] if r["profile"] == profile]
        if (
            report["scheduled"] != 8
            or report["recorded"] != 8
            or {r["id"] for r in report["rows"]} != {r["id"] for r in expected}
            or any(Path(r["output_directory"]) != work for r in expected)
        ):
            raise ValueError("Panel denominator or worker output binding changed")
        with sqlite3.connect(
            Path(manifest["ledger_directory"]).joinpath("requests.sqlite3").as_uri() + "?mode=ro",
            uri=True,
        ) as db:
            wires = db.execute(
                "SELECT r.request_id,r.identity,a.number,a.status,a.raw,a.sha256,"
                "a.elapsed_seconds,a.usage FROM requests r JOIN attempts a USING(request_id)"
            ).fetchall()
        for slot in expected:
            directory = work / slot["id"]
            saved = read(directory / "receipt.json")
            schedule = read_record(read(directory / "schedule.json"))
            packet = read_record(bound(slot["packet"]))
            if schedule.packet != packet or len(schedule.slots) != 1:
                raise ValueError("Panel packet/schedule identity changed")
            parameters = schedule.slots[0]["parameters"]
            from tools.repair.annotation_profile import request_profile
            from exact.repair.annotation_controls import response_format

            controls = request_profile(manifest, profile)
            context = json.loads(parameters["messages"][1]["content"])
            expected_controls = dict(
                max_tokens=controls["max_output_tokens"],
                provider=manifest["profiles"][profile]["provider"],
                response_format=response_format(packet, slot["swapped"]),
            )
            if controls["reasoning"] is not None:
                expected_controls["reasoning"] = controls["reasoning"]
            if (
                context["swapped"] != slot["swapped"]
                or parameters["model"] != manifest["profiles"][profile]["model"]
                or any(
                    canonical_hash(parameters.get(k)) != canonical_hash(v)
                    for k, v in expected_controls.items()
                )
                or (controls["reasoning"] is None and "reasoning" in parameters)
            ):
                raise ValueError("Panel slot changed its frozen order or request controls")
            payload = {
                k: parameters[k]
                for k in ("model", "messages", "max_tokens", "temperature", "provider")
            }
            payload.update(
                {k: parameters[k] for k in ("reasoning", "response_format") if k in parameters}
            )
            matches = [
                w
                for w in wires
                if canonical_hash(json.loads(w[1])["payload"]) == canonical_hash(payload)
            ]
            if len(matches) != 1 or matches[0][3] != 200:
                raise ValueError("Panel slot lacks its single definitive HTTP200 receipt")
            request_id, identity, number, status, raw, digest, seconds, usage = matches[0]
            response = _receipt_response(
                dict(
                    request_id=request_id,
                    attempt=number,
                    identity=json.loads(identity),
                    raw_response=bytes(raw).decode(),
                    sha256=digest,
                ),
                parameters,
            )
            if response["model"] != manifest["profiles"][profile]["model"]:
                raise ValueError("Panel model differs from frozen profile")
            row = dict(
                id=slot["id"],
                profile=profile,
                swapped=slot["swapped"],
                receipt=binding(directory / "receipt.json"),
                status=saved["status"],
                error=saved.get("error"),
                request_id=request_id,
                attempt=number,
                raw_sha256=digest,
                service_seconds=seconds,
                provider=response.get("provider"),
                model=response["model"],
            )
            if saved["status"] == "complete":
                labels = directory / "labels.json"
                if sha(labels) != saved["labels_sha256"]:
                    raise ValueError("Panel labels changed")
                aggregate = read_record(read(labels)["comparisons"][0]["comparison"])
                row.update(
                    labels=binding(labels),
                    eligible=aggregate.global_target_eligible,
                    decision=aggregate.decision,
                )
            rows.append(row)
        receipts.append(receipt)
        models[profile] = manifest_ref
    if len(rows) != 32 or len({r["request_id"] for r in rows}) != 32:
        raise ValueError("Panel requests must be distinct and complete")
    # This invokes the existing strict gate; no raw-response repair or relabeling.
    gate = summarize(summary_path, output)
    phase = read(Path(summary["ledger_directory"]) / "phase-reservations.json")
    if phase["request_limits"] != proposal["request_limits"] or any(
        phase["reservations"].get(k) != v for k, v in prior["reservations"].items()
    ):
        raise ValueError("Panel review would lose prior request history")
    immutable(output / "phase-state.json", phase)
    result = dict(
        schema="exact-repair/teacher-panel-review/v1",
        status=gate["status"],
        preparation=binding(Path(prepared_path)),
        gate=binding(output / "calibration-gate.json"),
        receipts=receipts,
        rows=rows,
        scheduled=32,
        status_counts=dict(Counter(r["status"] for r in rows)),
        phase_state=binding(output / "phase-state.json"),
        cumulative_requests=len(phase["reservations"]),
        reserved_usd=sum(r["reserved_cost_usd"] for r in phase["reservations"].values()),
        selected_profile=gate["selected_profile"],
        paid_calls=0,
        test_outcomes_opened=False,
        primary_fit_admitted=False,
    )
    immutable(output / "review.json", result)
    if gate["selected_profile"]:
        selected = dict(
            schema="exact-repair/qualified-teacher/v1",
            review=binding(output / "review.json"),
            gate=result["gate"],
            manifest=models[gate["selected_profile"]],
            phase_state=result["phase_state"],
            selected_profile=gate["selected_profile"],
            request_limits=phase["request_limits"],
            limitations=gate["limitations"],
        )
        immutable(output / "selected-teacher.json", selected)
    return result


def qualified_contract(manifest):
    """Use the approved panel's exact selected profile without reopening its clock."""
    from tools.repair.teacher_panel import panel_contract

    selected = bound(manifest["qualified_teacher"])
    gate, review = bound(selected["gate"]), bound(selected["review"])
    source, phase = bound(selected["manifest"]), bound(selected["phase_state"])
    proposal, _, previous = panel_contract(source)
    profile = selected["selected_profile"]
    if not (
        selected["schema"] == "exact-repair/qualified-teacher/v1"
        and gate["status"] == review["status"] == "qualified"
        and gate["selected_profile"] == review["selected_profile"] == profile
        and gate["metrics"][profile] == dict(valid=8, correct=8, order_consistent=4, eligible=True)
        and review["scheduled"] == len(review["rows"]) == 32
        and review["gate"] == selected["gate"]
        and review["phase_state"] == selected["phase_state"]
        and manifest["phase"] in {"train", "development"}
        and manifest["profile"] == manifest["teacher_profile"] == profile == source["profile"]
        and manifest["request_limits"] == selected["request_limits"] == proposal["request_limits"]
        and phase["request_limits"] == manifest["request_limits"]
        and manifest["request_budget_amendment"] == source["request_budget_amendment"]
        and manifest["lineage_id"] == source["lineage_id"]
        and Path(manifest["ledger_directory"]).resolve()
        == Path(source["ledger_directory"]).resolve()
        and manifest["cost_ceiling_usd"] == source["cost_ceiling_usd"]
        and not manifest.get("teacher_panel")
        and compatible_input_profile(manifest, source, profile)
        and all(
            manifest[k] == source[k]
            for k in (
                "prompt_version",
                "prompt_hash",
                "rubric_version",
                "criterion_weights",
                "test_profile",
                "selection_model_ids",
            )
        )
        and manifest["data_permissions"] == DATA_PERMISSIONS[manifest["phase"]]
        and manifest["profiles"][profile] == source["profiles"][profile]
        and manifest["profiles"][manifest["test_profile"]]
        == source["profiles"][source["test_profile"]]
        and manifest["prices_per_million"][profile] == source["prices_per_million"][profile]
        and manifest["calibration_gate"] == selected["gate"]
        and len(manifest.get("slots", [])) <= 256
    ):
        raise ValueError("Qualified teacher successor changed its gate, profile, quota or lineage")
    return proposal, phase, source, previous


def compatible_input_profile(manifest, source, profile):
    """Changing only explicit input limits does not change teacher qualification."""
    from tools.repair.annotation_profile import request_profile

    if set(manifest["request_profiles"]) != {profile}:
        return False
    baseline = source["request_profiles"][profile]
    if manifest["request_profiles"] == {profile: baseline}:
        return True
    controls = manifest["request_profiles"][profile]
    optional = {"max_input_tokens", "max_input_bytes", "input_amendment"}
    if {k: v for k, v in controls.items() if k not in optional} != baseline:
        return False
    controls = request_profile(manifest, profile)
    if "input_amendment" in controls:
        amendment = bound(controls["input_amendment"])
        return amendment.get("previous_profile_hash") == canonical_hash(baseline)
    return controls == baseline
