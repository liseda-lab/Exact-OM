"""Admit exact lossless TRAIN packets while retaining every unavailable row.

This preparation performs no native replay or hosted transmission. Runtime
validation authenticates the fixed native denominator before any account check.
"""

import argparse
from collections import Counter
import copy
from pathlib import Path
import time

from exact.repair.records import canonical_hash, read_record
from tools.repair.batch import _stage_remaining, freeze, prepare_dispatch, read, sha
from tools.repair.grounded_supervision import packet_admission
from tools.repair.historical_regression import binding
from tools.repair.semantic_packet_encoding import REVISION, decode, encode, encoding_identity
from tools.repair.shared_release import authenticate, bound, immutable, validate_completion

SCHEMA = "exact-repair/train-packet-admission/v1"


def native_outputs(receipt):
    terminal, outputs, batch, job = validate_completion(receipt)
    if job["id"] != "prepare-grounded-train-packets-001":
        raise ValueError("Packet admission requires the declared native preparation")
    work = Path(terminal["work"])
    for relative, digest in outputs.items():
        authenticate(dict(path=str(work / relative), sha256=digest))
    return work, outputs


def output_bound(reference, work, outputs):
    path = Path(reference["path"])
    if outputs.get(str(path.relative_to(work))) != reference["sha256"]:
        raise ValueError("Packet input is not a terminal native output")
    return bound(reference)


def transform(original, original_ref, manifest, output):
    row = dict(
        original,
        original_row=original_ref,
        original_status=original["status"],
        original_packet=original["packet"],
        transport_proof=None,
    )
    if original["packet"] is not None:
        packet = read_record(bound(original["packet"]))
        compact, proof = encode(packet)
        packet_path = output / "packets" / (compact.content_hash + ".json")
        proof_path = output / "proofs" / (compact.content_hash + ".json")
        immutable(packet_path, compact.to_dict())
        immutable(proof_path, proof)
        row.update(
            packet=binding(packet_path),
            transport_proof=binding(proof_path),
            **packet_admission(compact, manifest, row["swapped"]),
        )
    row["primary_weak_label_eligible"] = row["status"] == "eligible"
    return row


def validate_row(row, manifest, original):
    expected = dict(
        original,
        original_row=row["original_row"],
        original_status=original["status"],
        original_packet=original["packet"],
        transport_proof=None,
    )
    if original["packet"] is not None:
        raw = read_record(bound(original["packet"]))
        packet = read_record(bound(row["packet"]))
        proof = bound(row["transport_proof"])
        if decode(packet, proof).to_dict() != raw.to_dict() or encode(raw) != (packet, proof):
            raise ValueError("Packet does not use the frozen exact full-context encoding")
        expected.update(
            packet=row["packet"],
            transport_proof=row["transport_proof"],
            **packet_admission(packet, manifest, original["swapped"]),
        )
    expected["primary_weak_label_eligible"] = expected["status"] == "eligible"
    if canonical_hash(expected) != canonical_hash(row):
        raise ValueError("Admitted row changed a native identity, status or packet")


def validate_rows(manifest):
    contract = manifest["packet_admission"]
    if contract["schema"] != SCHEMA or manifest["phase"] != "train":
        raise ValueError("Unknown TRAIN packet admission")
    preparation = bound(contract["encoding"])
    if (
        preparation["revision"] != REVISION
        or preparation["encoding_identity"] != encoding_identity()
    ):
        raise ValueError("Lossless transport profile changed")
    authenticate(preparation["encoder"])
    if (
        sha(Path(__file__).with_name("semantic_packet_encoding.py"))
        != preparation["encoder"]["sha256"]
    ):
        raise ValueError("Running transport encoder differs from frozen source")
    budget = bound(contract["budget"])
    if budget["request_limits"] != manifest["request_limits"] or budget["costs_reset"]:
        raise ValueError("Packet admission budget changed")
    work, outputs = native_outputs(bound(contract["native_receipt"]))
    rows = bound(contract["rows"])["rows"]
    if canonical_hash(rows) != canonical_hash(manifest["slots"]) or len(rows) > 256:
        raise ValueError("Frozen annotation denominator changed")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate annotation row")
    for row in rows:
        original = output_bound(row["original_row"], work, outputs)
        if original["packet"]:
            output_bound(original["packet"], work, outputs)
        output_bound(original["evidence_contract"], work, outputs)
        for receipt in original["native_receipts"]:
            output_bound(receipt, work, outputs)
        validate_row(row, manifest, original)
    return rows


def validate_pretransmission_recovery(manifest):
    """One explicit technical successor; absence of wire alone never grants retry."""
    import json
    import sqlite3

    proof = bound(manifest["pretransmission_recovery"])
    original = bound(proof["predecessor_manifest"])
    if (
        proof.get("schema") != "exact-repair/controlled-byte-guard-recovery/v1"
        or proof.get("cause") != "controlled_profile_legacy_8000_byte_guard"
        or proof.get("same_cause_unsuccessful_attempts") != 1
        or proof.get("max_unsuccessful_attempts") != 2
        or proof.get("costs_reset") is not False
        or original.get("pretransmission_recovery")
        or {k: v for k, v in manifest.items() if k != "pretransmission_recovery"} != original
    ):
        raise ValueError("Technical recovery changed its original scientific contract")
    terminal, outputs, _, _ = validate_completion(bound(proof["predecessor_run"]))
    work = Path(terminal["work"])
    prior = bound(proof["phase_before"])
    current = read(Path(manifest["ledger_directory"]) / "phase-reservations.json")
    if any(current["reservations"].get(k) != v for k, v in prior["reservations"].items()):
        raise ValueError("Technical recovery erased earlier reservations")
    path = Path(manifest["ledger_directory"]) / "requests.sqlite3"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        wires = [json.loads(r[0]) for r in db.execute("SELECT identity FROM requests")]
        for row in original["slots"]:
            if row["status"] != "eligible":
                continue
            refs = proof["slots"][row["id"]]
            receipt = output_bound(refs["receipt"], work, outputs)
            schedule = read_record(output_bound(refs["schedule"], work, outputs))
            run = read_record(output_bound(refs["run"], work, outputs))
            previous = prior["reservations"].get(canonical_hash(("train", row["id"])))
            if (
                receipt.get("status") != "annotation_unavailable"
                or receipt.get("error")
                != "Evidence cannot fit the frozen input budget; no silent truncation"
                or receipt.get("confirmed_failure") is not None
                or run.max_input_bytes != 8000
                or previous is None
                or previous["state"] != "unresolved"
                or previous["request_basis"]
                != canonical_hash((original, row["id"], schedule.packet.content_hash))
                or schedule.packet != read_record(bound(row["packet"]))
                or len(schedule.slots) != 1
                or db.execute(
                    "SELECT 1 FROM repair_annotation_reserves WHERE run_id=?", (run.run_id,)
                ).fetchone()
            ):
                raise ValueError("Recovery is not a proven pre-transmission byte-guard failure")
            parameters = schedule.slots[0]["parameters"]
            payload = {
                k: parameters[k]
                for k in (
                    "model",
                    "messages",
                    "max_tokens",
                    "temperature",
                    "provider",
                    "reasoning",
                    "response_format",
                )
                if k in parameters
            }
            if any(
                w.get("role") == parameters["role"]
                and canonical_hash(w.get("payload")) == canonical_hash(payload)
                for w in wires
            ):
                raise ValueError("Original request reached the wire ledger; no automatic retry")
    return proof


def prepare(campaign, output, repository):
    from tools.repair.corrective_semantics import _request_budget, validate_manifest

    campaign, output, repository = map(Path, (campaign, output, repository))
    registry = read(campaign / "supervisor/registry.json")
    run = next(r for r in registry["runs"] if r["id"] == "prepare-grounded-train-packets-001")
    attempt = Path(run["completion_path"]).parent
    descriptor = read(attempt / "descriptor.json")
    if run["dispatch_nonce"] != descriptor["launch"]["nonce"]:
        raise ValueError("Registered native dispatch identity changed")
    for ref in descriptor["launch"]["bindings"]:
        authenticate(ref)
    terminal = read(attempt / "completion.json")
    receipt = dict(
        completion=binding(attempt / "completion.json"),
        step=binding(attempt / "step.json"),
        outputs=binding(attempt / "outputs.json"),
        batch=binding(Path(terminal["batch"])),
        dispatch_nonce=run["dispatch_nonce"],
        step_id=run["step_id"],
        expected_status="complete",
    )
    work, outputs = native_outputs(receipt)
    report_ref = binding(work / "report.json")
    report = output_bound(report_ref, work, outputs)
    encoding_ref = registry["lossless_packet_preparation"]
    encoding = bound(encoding_ref)
    authenticate(encoding["encoder"])
    authenticate(encoding["source_manifest"])
    immutable(output / "native-receipt.json", receipt)
    originals, rows, manifests, row_refs = [], [], [], []
    for index, reference in enumerate(report["shards"]):
        manifest = output_bound(reference, work, outputs)
        shard_rows = []
        for original in manifest["slots"]:
            original_ref = binding(work / "rows" / (canonical_hash(original["id"]) + ".json"))
            if output_bound(original_ref, work, outputs) != original:
                raise ValueError("Native manifest differs from committed row")
            row = transform(original, original_ref, manifest, output)
            validate_row(row, manifest, original)
            rows.append(row)
            originals.append(original)
            shard_rows.append(row)
        row_path = output / f"rows-{index:02}.json"
        immutable(row_path, dict(rows=shard_rows))
        row_refs.append(binding(row_path))
        manifests.append(dict(manifest, slots=shard_rows))
    if (
        len(rows) != 308
        or len({r["id"] for r in rows}) != 308
        or sum(not r["swapped"] for r in rows) != 256
        or len({r["case_id"] for r in rows}) != 128
    ):
        raise ValueError("Full TRAIN denominator changed")
    immutable(
        output / "phase-before.json", read(campaign / "annotations/ledger/phase-reservations.json")
    )
    phase = read(output / "phase-before.json")
    ledger = read(campaign / "ledger.json")
    immutable(output / "stage-before.json", ledger)
    eligible = [r for r in rows if r["status"] == "eligible"]
    manifest = manifests[0]
    limits, amendment = _request_budget(manifest)
    prior = amendment["qualified_phase"]
    if phase["request_limits"] != limits or any(
        phase["reservations"].get(k) != v for k, v in prior["reservations"].items()
    ):
        raise ValueError("Cumulative annotation history changed")
    records = list(phase["reservations"].values())
    train = [r for r in records if r["phase"] == "train"]
    if train:
        raise ValueError(
            "TRAIN already admitted; reconcile existing slots instead of preparing again"
        )
    prices = manifest["prices_per_million"][manifest["profile"]]
    cap = manifest["request_profiles"][manifest["profile"]]["max_output_tokens"]
    per_call = (8000 * prices["input"] + cap * prices["output"]) / 1e6
    exposure = sum(r["reserved_cost_usd"] for r in records)
    deadline = min(
        manifest["deadline_epoch"],
        ledger["stages"]["acquisition"]["started_epoch"]
        + ledger["stage_limits"]["acquisition"]["elapsed_seconds"],
    )
    if (
        len(eligible) > limits["train"]
        or len(records) + len(eligible) > 704
        or exposure + len(eligible) * per_call > 35
        or len(eligible) * 90 > 28800
    ):
        raise ValueError("Cumulative annotation allowance insufficient")
    jobs = []
    for index, value in enumerate(manifests):
        seconds = 94 * sum(r["status"] == "eligible" for r in value["slots"]) + 120
        job_id = f"annotate-controlled-train-{index:02}-001"
        jobs.append(
            dict(
                id=job_id,
                seconds=seconds + 60,
                slice_seconds=seconds + 60,
                cleanup_seconds=30,
                stage="acquisition",
                budget_stages=["acquisition", "learning"],
                priority=220,
                gpu_devices=[],
                resources=dict(cpus=1, gpus=0, memory_mb=2048, gres="none"),
                deadline_epoch=deadline,
                deadline_policy="defer",
                commands=[
                    [
                        "{python}",
                        "-m",
                        "tools.repair.corrective_semantics",
                        str(output / f"train-{index:02}.json"),
                        "{work}",
                    ]
                ],
            )
        )
        value.update(
            authorized=True,
            seconds=seconds,
            deadline_epoch=deadline,
            execution_gate="bound complete-packet and cumulative budget admission",
        )
    remaining = min(_stage_remaining(copy.deepcopy(ledger), j, time.time()) for j in jobs)
    reservation = sum(j["seconds"] for j in jobs)
    if reservation > 0.7 * remaining:
        raise ValueError("Hosted TRAIN exceeds 70 percent of remaining capacity")
    budget = dict(
        request_limits=limits,
        prior_requests=len(records),
        prior_reserved_usd=exposure,
        new_max_requests=len(eligible),
        new_max_reserved_usd=len(eligible) * per_call,
        hosted_service_reserved_seconds=90 * len(eligible),
        hosted_service_limit_seconds=28800,
        worker_reserved_seconds=reservation,
        remaining_stage_seconds=remaining,
        scheduling_fraction=0.7,
        deadline_epoch=deadline,
        costs_reset=False,
        phase_before=binding(output / "phase-before.json"),
        stage_before=binding(output / "stage-before.json"),
    )
    immutable(output / "admission.json", budget)
    manifest_refs = []
    for index, value in enumerate(manifests):
        value["packet_admission"] = dict(
            schema=SCHEMA,
            encoding=encoding_ref,
            rows=row_refs[index],
            native_receipt=binding(output / "native-receipt.json"),
            budget=binding(output / "admission.json"),
        )
        validate_manifest(value)
        validate_rows(value)
        path = output / f"train-{index:02}.json"
        immutable(path, value)
        manifest_refs.append(binding(path))
    prepared = dict(
        schema=SCHEMA,
        status="prepared_not_queued",
        native_report=report_ref,
        encoding=encoding_ref,
        manifests=manifest_refs,
        admission=binding(output / "admission.json"),
        scheduled=308,
        originals=256,
        swaps=52,
        statuses=dict(Counter(r["status"] for r in rows)),
        original_statuses=dict(Counter(r["status"] for r in rows if not r["swapped"])),
        family_control_statuses=dict(
            Counter(
                "/".join((r["family"], r["control"], r["status"])) for r in rows if not r["swapped"]
            )
        ),
        independent_semantic_evidence=False,
        cohort="generated_only",
        missing_real_coverage=True,
        native_replays=0,
        new_paid_calls=0,
        primary_fitting_admitted=False,
        test_outcomes_opened=False,
    )
    immutable(output / "prepared.json", prepared)
    native_batch = bound(receipt["batch"])
    spec = dict(
        repository=str(repository),
        campaign=str(campaign),
        allocation="14451",
        python=native_batch["python"],
        ledger=str(campaign / "ledger.json"),
        capacity=registry["capacity"],
        source_store=str(campaign / "sources"),
        protocol_source=native_batch["protocol_source"],
        jobs=jobs,
        input_files=[str(p) for p in output.rglob("*.json")],
        purpose="Authorized fixed TRAIN semantic annotation; exact lossless packets and complete unavailable denominator",
    )
    immutable(output / "batch-spec.json", spec)
    batch = freeze(output / "batch-spec.json", campaign / "batches" / output.name)
    descriptors = []
    dependencies = ["prepare-grounded-train-packets-001"]
    for job in jobs:
        descriptors.append(
            prepare_dispatch(
                batch,
                job["id"],
                campaign / "attempts" / job["id"] / "001",
                tmux_socket=campaign / "supervisor/tmux.sock",
                depends_on=dependencies,
            )
        )
        dependencies = [job["id"]]
    result = dict(
        preparation=binding(output / "prepared.json"),
        batch=binding(batch),
        descriptors=descriptors,
        status="prepared_not_queued",
    )
    immutable(output / "dispatch-prepared.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    prepare(args.campaign, args.output, args.repository)


if __name__ == "__main__":
    main()
