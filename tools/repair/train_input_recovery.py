"""Re-admit only never-transmitted oversized TRAIN packets, without native replay."""

import argparse
import copy
from dataclasses import replace
from pathlib import Path
import time

from exact.repair.records import read_record
from tools.repair.annotation_profile import input_limits, request_profile
from tools.repair.batch import _stage_remaining, freeze, prepare_dispatch, read
from tools.repair.grounded_supervision import packet_admission
from tools.repair.historical_regression import binding
from tools.repair.semantic_packet_encoding import encode, decode
from tools.repair.shared_release import authenticate, bound, immutable, validate_completion

SCHEMA = "exact-repair/train-input-recovery/v1"
OVERSIZED = {"unavailable_packet_bytes", "unavailable_packet_tokens"}


def reprofile(original, controls):
    # Coverage is transport metadata; every scientific field remains identical.
    return replace(
        original, coverage={**original.coverage, "byte_budget": input_limits(controls)[1]}
    )


def predecessor(contract):
    from tools.repair.train_packet_admission import validate_rows

    old = bound(contract["predecessor_manifest"])
    validate_rows(old)
    terminal, outputs, batch, _ = validate_completion(contract["predecessor_receipt"])
    if (
        batch["frozen_files"].get(contract["predecessor_manifest"]["path"])
        != contract["predecessor_manifest"]["sha256"]
    ):
        raise ValueError("Predecessor manifest was not frozen for its worker")
    path = Path(terminal["work"]) / "report.json"
    authenticate(dict(path=str(path), sha256=outputs["report.json"]))
    report = read(path)
    results = {r["id"]: r for r in report["rows"]}
    if report["status"] != "complete" or set(results) != {r["id"] for r in old["slots"]}:
        raise ValueError("Predecessor annotation denominator incomplete")
    eligible = []
    for row in old["slots"]:
        if row["status"] in OVERSIZED:
            result = results[row["id"]]
            if (
                result.get("attempted") is not False
                or result.get("artifact") is not None
                or result["status"] != row["status"]
            ):
                raise ValueError("Only never-transmitted oversized packets may be re-admitted")
            eligible.append(row)
    return old, eligible


def validate_rows(manifest):
    contract = manifest["packet_admission"]
    if contract.get("schema") != SCHEMA or manifest["phase"] != "train":
        raise ValueError("Invalid input recovery contract")
    old, eligible = predecessor(contract)
    controls = request_profile(manifest, manifest["profile"])
    if contract["input_amendment"] != controls.get("input_amendment"):
        raise ValueError("Recovery amendment changed")
    current = manifest["slots"]
    if [r["id"] for r in current] != [r["id"] for r in eligible]:
        raise ValueError("Recovery changed the fixed oversized denominator")
    phase_before = bound(contract["phase_before"])
    phase = read(Path(manifest["ledger_directory"]) / "phase-reservations.json")
    if any(phase["reservations"].get(k) != v for k, v in phase_before["reservations"].items()):
        raise ValueError("Input recovery erased earlier costs")
    for row, previous in zip(current, eligible, strict=True):
        if any(
            r.get("phase") == "train" and r.get("slot") == row["id"]
            for r in phase_before["reservations"].values()
        ):
            raise ValueError("Recovery row already had a request reservation")
        native = read_record(bound(previous["original_packet"]))
        amended = reprofile(native, controls)
        if read_record(bound(row["original_packet"])) != amended:
            raise ValueError("Reprofiling changed original scientific evidence")
        packet, proof = read_record(bound(row["packet"])), bound(row["transport_proof"])
        if encode(amended) != (packet, proof) or decode(packet, proof) != amended:
            raise ValueError("Recovery lost complete native evidence")
        expected = dict(
            previous,
            original_packet=row["original_packet"],
            packet=row["packet"],
            transport_proof=row["transport_proof"],
            **packet_admission(packet, manifest, previous["swapped"]),
        )
        expected["primary_weak_label_eligible"] = expected["status"] == "eligible"
        if row != expected:
            raise ValueError("Recovery changed a case, split, assignment or admission result")
    # The teacher, prompt, rubric, phase and original slot identities stay frozen.
    for key in (
        "parent_splits",
        "lineage_id",
        "profiles",
        "prompt_version",
        "prompt_hash",
        "rubric_version",
        "criterion_weights",
        "request_limits",
        "ledger_directory",
    ):
        if manifest[key] != old[key]:
            raise ValueError("Input recovery changed scientific or ledger dependencies")
    return current


def prepare(campaign, output, repository, amendment_path):
    from tools.repair.corrective_semantics import validate_manifest
    from tools.repair.corrective_campaign import source_identity

    campaign, output, repository = map(Path, (campaign, output, repository))
    review_ref = binding(
        campaign / "annotations/controlled-train-byte-recovery-001/completion-review.json"
    )
    review = bound(review_ref)
    prior = bound(review["preparation"])
    phase = read(campaign / "annotations/ledger/phase-reservations.json")
    immutable(output / "phase-before.json", phase)
    amendment_ref = binding(Path(amendment_path))
    amendment = bound(amendment_ref)
    ledger = read(campaign / "ledger.json")
    deadline = min(
        1791997200.0,
        ledger["stages"]["acquisition"]["started_epoch"]
        + ledger["stage_limits"]["acquisition"]["elapsed_seconds"],
    )
    manifests, jobs = [], []
    for index, (ref, receipt) in enumerate(
        zip(prior["manifests"], review["receipts"], strict=True)
    ):
        contract = dict(
            schema=SCHEMA,
            predecessor_manifest=ref,
            predecessor_receipt=receipt,
            input_amendment=amendment_ref,
            phase_before=binding(output / "phase-before.json"),
        )
        old, eligible = predecessor(contract)
        m = copy.deepcopy(old)
        m.pop("pretransmission_recovery", None)
        controls = m["request_profiles"][m["profile"]]
        controls.update(
            max_input_tokens=amendment["max_input_tokens"],
            max_input_bytes=amendment["max_input_bytes"],
            input_amendment=amendment_ref,
        )
        request_profile(m, m["profile"])
        rows = []
        for row in eligible:
            raw = reprofile(read_record(bound(row["original_packet"])), controls)
            packet, proof = encode(raw)
            paths = [
                output / kind / (packet.content_hash + ".json")
                for kind in ("originals", "packets", "proofs")
            ]
            for path, value in zip(paths, (raw.to_dict(), packet.to_dict(), proof)):
                immutable(path, value)
            updated = dict(
                row,
                original_packet=binding(paths[0]),
                packet=binding(paths[1]),
                transport_proof=binding(paths[2]),
                **packet_admission(packet, m, row["swapped"]),
            )
            updated["primary_weak_label_eligible"] = updated["status"] == "eligible"
            rows.append(updated)
        seconds = 94 * sum(r["status"] == "eligible" for r in rows) + 120
        m.update(
            slots=rows,
            packet_admission=contract,
            seconds=seconds,
            deadline_epoch=deadline,
            execution_gate="User-authorized complete-evidence input-profile correction; primary training remains gated",
        )
        validate_manifest(m)
        validate_rows(m)
        path = output / f"train-{index:02}.json"
        immutable(path, m)
        manifests.append(binding(path))
        job_id = f"annotate-train-input-recovery-{index:02}-001"
        jobs.append(
            dict(
                id=job_id,
                stage="acquisition",
                budget_stages=["acquisition", "learning"],
                priority=230,
                seconds=seconds + 60,
                slice_seconds=seconds + 60,
                cleanup_seconds=30,
                resources=dict(cpus=1, gpus=0, memory_mb=2048, gres="none"),
                gpu_devices=[],
                deadline_epoch=deadline,
                deadline_policy="defer",
                commands=[
                    ["{python}", "-m", "tools.repair.corrective_semantics", str(path), "{work}"]
                ],
            )
        )
    slots = [r for ref in manifests for r in bound(ref)["slots"]]
    count = sum(r["status"] == "eligible" for r in slots)
    prices = m["prices_per_million"][m["profile"]]
    per_call = (
        amendment["max_input_tokens"] * prices["input"]
        + amendment["max_output_tokens"] * prices["output"]
    ) / 1e6
    reservations = list(phase["reservations"].values())
    train_used = sum(r["phase"] == "train" for r in reservations)
    prior_cost = sum(r["reserved_cost_usd"] for r in reservations)
    if (
        train_used + count > m["request_limits"]["train"]
        or len(reservations) + count > 704
        or prior_cost + count * per_call > 35
        or 90 * (train_used + count) > 28800
    ):
        raise ValueError("Input recovery exceeds cumulative campaign budget")
    remaining = min(_stage_remaining(copy.deepcopy(ledger), j, time.time()) for j in jobs)
    if sum(j["seconds"] for j in jobs) > 0.7 * remaining:
        raise ValueError("Recovery cannot fit existing acquisition stage")
    immutable(
        output / "admission.json",
        dict(
            prior_reserved_usd=prior_cost,
            prior_train_requests=train_used,
            new_max_requests=count,
            new_max_reserved_usd=count * per_call,
            remaining_stage_seconds=remaining,
            costs_reset=False,
            primary_training_admitted=False,
        ),
    )
    source = source_identity()
    spec = dict(
        repository=str(repository),
        campaign=str(campaign),
        allocation="14451",
        python="/home/pgcotovio/Exact-OM/.venv/bin/python",
        ledger=str(campaign / "ledger.json"),
        capacity=read(campaign / "supervisor/registry.json")["capacity"],
        source_store=str(campaign / "sources"),
        protocol_source=read(campaign / "batches/controlled-train-byte-recovery-001/batch.json")[
            "protocol"
        ],
        jobs=jobs,
        input_files=[str(p) for p in output.rglob("*.json")] + [str(amendment_path)],
        purpose="Complete-evidence TRAIN input correction; no optimizer work or TEST access",
    )
    immutable(output / "batch-spec.json", spec)
    batch = freeze(output / "batch-spec.json", campaign / "batches" / output.name)
    descriptors = [
        prepare_dispatch(
            batch,
            j["id"],
            campaign / "attempts" / j["id"] / "001",
            tmux_socket=campaign / "supervisor/tmux.sock",
            depends_on=[],
        )
        for j in jobs
    ]
    result = dict(
        schema=SCHEMA,
        status="prepared_not_queued",
        source_commit=source["revision"],
        manifests=manifests,
        batch=binding(batch),
        descriptors=descriptors,
        predecessor=review_ref,
        scheduled_responses=len(slots),
        original_comparisons=sum(not r["swapped"] for r in slots),
        scheduled_swaps=sum(r["swapped"] for r in slots),
        primary_training_admitted=False,
        remaining_action="Aggregate these new responses with retained old labels, then strengthen deficient TRAIN families and pass measured readiness before either primary arm",
    )
    immutable(output / "prepared.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "output", "repository", "amendment"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    prepare(a.campaign, a.output, a.repository, a.amendment)


if __name__ == "__main__":
    main()
