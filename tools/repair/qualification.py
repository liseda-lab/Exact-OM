"""Run frozen native qualification suites with per-suite evidence and safe resume."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from exact.repair.api import write_artifact
from tools.repair.batch import read, sha


def checked_result(path, identity):
    """Only reuse successful, dependency-bound suites with intact test evidence."""
    value = read(path)
    if value["identity"] != identity:
        raise ValueError("Qualification checkpoint dependencies changed")
    for item in (value["junit"], value["log"]):
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Qualification evidence changed")
    return value if value["status"] == "complete" else None


def run(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = read(manifest_path)
    source = Path(__file__).resolve().parents[2]
    dependencies = [
        (str(p.relative_to(source)), sha(p))
        for p in sorted([*source.glob("exact/repair/*.py"), *source.glob("tools/repair/*.py")])
    ]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in manifest["groups"]:
        identity = hashlib.sha256(
            json.dumps(
                [
                    sha(manifest_path),
                    dependencies,
                    [(name, sha(source / name)) for name in group["tests"]],
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        directory = output / group["id"]
        directory.mkdir(exist_ok=True)
        receipt = directory / "completion.json"
        if receipt.exists():
            previous = checked_result(receipt, identity)
            if previous:
                rows.append(previous)
                continue
        # Retain failed invocation logs across replacements.
        attempt = len(list(directory.glob("attempt-*.log"))) + 1
        log = directory / f"attempt-{attempt}.log"
        junit = directory / f"attempt-{attempt}.xml"
        started = time.time()
        write_artifact(
            output / "progress.json",
            dict(
                group=group["id"],
                status="running",
                completed_groups=len(rows),
                started_epoch=started,
            ),
        )
        with log.open("w") as stream:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *group["tests"],
                    "--junitxml=" + str(junit),
                ],
                cwd=source,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if not junit.exists():
            raise RuntimeError("Qualification suite produced no JUnit evidence: " + group["id"])
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        counts = {
            key: sum(int(s.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        passed = (
            process.returncode == 0
            and counts["tests"] > 0
            and not (counts["failures"] or counts["errors"] or counts["skipped"])
        )
        result = dict(
            identity=identity,
            group=group["id"],
            status="complete" if passed else "failed",
            exit_code=process.returncode,
            counts=counts,
            elapsed_seconds=time.time() - started,
            junit=dict(path=str(junit), sha256=sha(junit)),
            log=dict(path=str(log), sha256=sha(log)),
        )
        write_artifact(receipt, result)
        rows.append(result)
        if not passed:
            raise RuntimeError("Qualification failed or skipped required evidence: " + group["id"])
    report = dict(
        schema="exact-repair/native-qualification/v1",
        status="complete",
        groups=rows,
        gates={
            "G0": "requires_requirement_audit",
            "G1": "requires_historical_regression",
            "G2": "requires_backend_scope_audit",
        },
        interpretation="Native test evidence; not automatic completion of research gates",
    )
    write_artifact(output / "report.json", report)
    write_artifact(output / "progress.json", dict(status="complete", completed_groups=len(rows)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__":
    main()
