#!/usr/bin/env python3
"""Export the closed T2 preliminary results without running experiments.

Keep original report bytes and absolute provenance paths. The export manifest
maps those paths to portable members. Raw datasets, held-out payloads, caches,
model weights, optimizer checkpoints and duplicate code snapshots stay on server.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path, help="New .tar.gz file; never overwritten")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    repository = Path(__file__).resolve().parents[4]
    closure = campaign / "artifacts/expanded-preliminary-closure-001"
    final = campaign / "work/xr21-expanded-preliminary-final/final"
    report = json.loads((final / "report.json").read_text())
    completion = json.loads((closure / "completion.json").read_text())
    if not completion["local_scope_complete"] or not (campaign / "STOP").exists():
        raise ValueError("Export requires authenticated local completion and user stop")
    files = {}
    provenance = {}

    def add(path, expected=None, member=None):
        path = Path(path).resolve()
        relative = path.relative_to(campaign)
        data = path.read_bytes()
        digest = sha256(data)
        if expected is not None and digest != expected:
            raise ValueError(f"Source digest changed: {path}")
        member = member or "campaign/" + relative.as_posix()
        files[member] = data
        provenance[member] = {"source_path": str(path), "sha256": digest,
                              "size_bytes": len(data)}

    for name in ["report.json", "scope-readiness.json"]:
        add(final / name)
    add(final / "REPORT.md", member="SUMMARY.md")
    for ref in report["source_reports"].values():
        add(ref["path"], ref["sha256"])
        sibling = Path(ref["path"]).with_name("report.md")
        if sibling.exists():
            add(sibling)
    for name in ["completion.json", "handoff.json", "costs.json",
                 "attempt-history.json", "authentication.json", "verified-files.json",
                 "scheduler-final.json", "post-publication-verification.json"]:
        add(closure / name)
    for name in ["registry.json", "state.json", "status.json", "policy.json",
                 "shutdown.json", "publication.json", "STOP", "PAUSE"]:
        add(campaign / "supervisor" / name)
    add(campaign / "STOP")
    add(campaign / "plan.json")
    for path in sorted((campaign / "studies/20261003").glob("*.json")):
        add(path)
    for path in sorted((campaign / "manifests").glob("*.json")):
        add(path)
    registry = json.loads((campaign / "supervisor/registry.json").read_text())
    for key in ["preliminary_scope_amendment", "future_cluster_storage_policy"]:
        ref = registry[key]
        add(ref["path"], ref["sha256"])

    pilot_path = campaign / "work/xr21-t2-report/revision-002/report.json"
    add(pilot_path)
    pilot = json.loads(pilot_path.read_text())
    models = []
    for arm in pilot["arms"]:
        source = arm["provenance"]
        for key in ["protocol", "training_report", "completion"]:
            ref = source[key]
            add(ref["path"], ref["sha256"])
        models.append({"arm": arm["id"], "selected_epoch": source["selected_epoch"],
                       "model": source["model"], "weights_included": False})
    files["configuration/selected-models.json"] = json_bytes(models)

    # Preserve attempt outcomes, including original failures and replacements.
    for run in registry["runs"]:
        for key in ["completion_path", "status_path"]:
            path = Path(run.get(key, "/nonexistent"))
            if path.is_file():
                add(path)

    # Batch manifests repeat whole-checkout hashes. Retain commands, resources,
    # package versions and every input hash, replacing only those duplicate lists.
    batches = []
    for path in sorted((campaign / "batches").glob("*/batch.json")):
        raw = path.read_bytes()
        batch = json.loads(raw)
        frozen = batch.pop("frozen_files", {})
        code = str(batch.get("code", ""))
        batch["export_source"] = {"path": str(path), "sha256": sha256(raw)}
        batch["export_omitted_duplicate_code_hashes"] = True
        batch["frozen_input_hashes"] = {
            name: digest for name, digest in frozen.items()
            if not (code and (name.startswith(code + "/") or name == code + ".tar"))
        }
        batches.append(batch)
        for key in ["protocol", "protocol_source"]:
            if batch.get(key):
                protocol = Path(batch[key]).resolve()
                if protocol.is_relative_to(campaign):
                    add(protocol, frozen.get(str(protocol)))
    files["configuration/batches.json"] = json_bytes(batches)

    commits = set()

    def collect_commits(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if (isinstance(item, str) and re.fullmatch(r"[0-9a-f]{40}", item)
                        and any(token in key for token in ["commit", "revision", "head"])):
                    commits.add(item)
                collect_commits(item)
        elif isinstance(value, list):
            for item in value:
                collect_commits(item)

    collect_commits(registry)
    collect_commits(batches)
    files["configuration/source-commits.json"] = json_bytes(sorted(commits))
    evaluation = json.loads(Path(report["source_reports"]["evaluation"]["path"]).read_text())
    rows = []
    for branch, results in evaluation["branches"].items():
        for arm, values in results["by_arm"].items():
            rows.append({"branch": branch, "arm": arm, **{
                key: values.get(key) for key in ["scheduled_rows", "quality_usable_rows",
                    "quality_unavailable_rows", "conditional_quality_mean"]}})
    for name, values in [("fresh-evaluation-by-arm", rows),
                         ("corrections", report["corrections"])]:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in values)
        files[f"tables/{name}.csv"] = stream.getvalue().encode()

    files["README.md"] = Path(__file__).with_name("RESULTS-HANDOFF.md").read_bytes()
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                   text=True).strip()
    manifest = {"schema": "exact-repair/important-results-export/v1",
                "campaign": str(campaign), "exporter_commit": head,
                "scientific_source_commit": completion["scientific_report_source_commit"],
                "closure_source_commit": completion["source_commit"],
                "results_only": True, "scientific_rows_replayed": 0,
                "omitted": ["raw datasets", "unopened held-out payloads", "caches",
                            "model weights and optimizer state", "duplicate code snapshots",
                            "raw worker logs; diagnostic outcomes retained in reports"],
                "source_files": provenance,
                "files": {name: {"sha256": sha256(data), "size_bytes": len(data)}
                          for name, data in sorted(files.items())}}
    files["export-manifest.json"] = json_bytes(manifest)
    files["SHA256SUMS"] = "".join(
        f"{sha256(data)}  {name}\n" for name, data in sorted(files.items())
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data in sorted(files.items()):
                    info = tarfile.TarInfo("exact-repair-preliminary-results-20261007/" + name)
                    info.size = len(data)
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(data))
    digest = sha256(args.output.read_bytes())
    args.output.with_name(args.output.name + ".sha256").write_text(
        f"{digest}  {args.output.name}\n")
    print(json.dumps({"archive": str(args.output), "sha256": digest,
                      "size_bytes": args.output.stat().st_size, "members": len(files),
                      "uncompressed_bytes": sum(map(len, files.values()))}, indent=2))


if __name__ == "__main__":
    main()
