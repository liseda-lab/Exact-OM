"""Authenticate newly available native TRAIN packets before any hosted request."""

from pathlib import Path

from exact.repair.records import canonical_hash, read_record
from tools.repair.annotation_profile import request_profile
from tools.repair.batch import read, sha
from tools.repair.grounded_supervision import packet_admission
from tools.repair.recovered_train_packets import TASK, TASK_REVISION
from tools.repair.semantic_packet_encoding import REVISION, decode, encode, encoding_identity
from tools.repair.shared_release import authenticate, bound, immutable, validate_completion
from tools.repair.train_input_recovery import reprofile
from tools.repair.historical_regression import binding

SCHEMA = "exact-repair/recovered-train-packet-admission/v1"


def transform(original, original_ref, manifest, output):
    row = dict(original, original_row=original_ref, original_status=original["status"],
               original_packet=None, transport_proof=None)
    if original["packet"]:
        raw = reprofile(read_record(bound(original["packet"])),
                        request_profile(manifest, manifest["profile"]))
        packet, proof = encode(raw)
        refs = []
        for kind, value in (("originals", raw.to_dict()), ("packets", packet.to_dict()), ("proofs", proof)):
            path = output / kind / (packet.content_hash + ".json")
            immutable(path, value)
            refs.append(binding(path))
        row.update(original_packet=refs[0], packet=refs[1], transport_proof=refs[2],
                   **packet_admission(packet, manifest, row["swapped"]))
    row["primary_weak_label_eligible"] = row["status"] == "eligible"
    return row


def validate_rows(manifest):
    contract = manifest["packet_admission"]
    if contract["schema"] != SCHEMA or manifest["phase"] != "train":
        raise ValueError("Expected recovered TRAIN packet admission")
    template = bound(contract["template"])
    for key in ("parent_splits", "lineage_id", "profiles", "profile", "teacher_profile",
                "prompt_version", "prompt_hash", "rubric_version", "criterion_weights",
                "request_limits", "ledger_directory", "request_profiles", "qualified_teacher"):
        if manifest[key] != template[key]:
            raise ValueError("Recovered packet teacher, protocol or ledger changed")
    amendment = bound(contract["task_amendment"])
    if (amendment["task_revision"] != TASK_REVISION or amendment["task"] != TASK
            or amendment["retry_prior_responses"] is not False):
        raise ValueError("Recovered packet task changed")
    encoding = bound(contract["encoding"])
    if (encoding["revision"] != REVISION or encoding["encoding_identity"] != encoding_identity()
            or sha(Path(__file__).with_name("semantic_packet_encoding.py")) != encoding["encoder"]["sha256"]):
        raise ValueError("Lossless full-evidence encoding changed")
    authenticate(encoding["encoder"])
    before = bound(contract["phase_before"])
    current = read(Path(manifest["ledger_directory"]) / "phase-reservations.json")
    if any(current["reservations"].get(k) != v for k, v in before["reservations"].items()):
        raise ValueError("Earlier request history changed")
    budget = bound(contract["budget"])
    if budget["costs_reset"] or budget["request_limits"] != manifest["request_limits"]:
        raise ValueError("Cumulative budget changed")
    outputs_by_path, scheduled = {}, set()
    for receipt in contract["native_receipts"]:
        terminal, outputs, batch, job = validate_completion(receipt)
        if job["commands"][0][2] != "tools.repair.recovered_train_packets":
            raise ValueError("Unexpected native packet producer")
        plan_path = job["commands"][0][-2]
        plan = bound(dict(path=plan_path, sha256=batch["frozen_files"][plan_path]))
        if plan["task_amendment"] != contract["task_amendment"]:
            raise ValueError("Native producer task changed")
        work = Path(terminal["work"])
        for relative, digest in outputs.items():
            outputs_by_path[str(work / relative)] = digest
        report = bound(dict(path=str(work / "report.json"), sha256=outputs["report.json"]))
        if report["plan"] != dict(path=plan_path, sha256=batch["frozen_files"][plan_path]):
            raise ValueError("Native report plan changed")
        scheduled.update(ref["path"] for ref in report["rows"])

    def output_bound(ref):
        if outputs_by_path.get(ref["path"]) != ref["sha256"]:
            raise ValueError("Packet input is not a terminal native output")
        return bound(ref)

    rows = bound(contract["rows"])["rows"]
    if rows != manifest["slots"] or len(rows) > 256 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Frozen recovered row schedule changed")
    for row in rows:
        if row["original_row"]["path"] not in scheduled:
            raise ValueError("Recovered row was not scheduled")
        original = output_bound(row["original_row"])
        old = bound(original["prior_native_row"])
        if (old["pair"] is not None or old["packet"] is not None
                or row["task_amendment"] != contract["task_amendment"]):
            raise ValueError("Prior bound pair cannot be annotated again")
        if any(r["phase"] == "train" and r.get("slot") == row["id"]
               for r in before["reservations"].values()):
            raise ValueError("Recovered slot already had a paid reservation")
        for ref in original["native_receipts"]:
            output_bound(ref)
        expected = dict(original, original_row=row["original_row"],
                        original_status=original["status"], original_packet=None, transport_proof=None)
        if original["packet"]:
            raw = read_record(output_bound(original["packet"]))
            if raw.task != TASK or raw.split != "train":
                raise ValueError("Native packet task or split changed")
            amended = reprofile(raw, request_profile(manifest, manifest["profile"]))
            if read_record(bound(row["original_packet"])) != amended:
                raise ValueError("Input reprofiling altered scientific evidence")
            packet, proof = read_record(bound(row["packet"])), bound(row["transport_proof"])
            if encode(amended) != (packet, proof) or decode(packet, proof) != amended:
                raise ValueError("Recovered packet lost full evidence")
            expected.update(original_packet=row["original_packet"], packet=row["packet"],
                            transport_proof=row["transport_proof"],
                            **packet_admission(packet, manifest, row["swapped"]))
        expected["primary_weak_label_eligible"] = expected["status"] == "eligible"
        if canonical_hash(expected) != canonical_hash(row):
            raise ValueError("Recovered row changed its frozen native identity")
    return rows
