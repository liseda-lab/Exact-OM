"""Validate explicit local E06 captures and write a versioned, leakage-checked schedule."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from exact.repair.study import COHORTS, StudyCaseV2
from tools.repair.prepare import load_protocol


def _release(entry: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    cohort, version = entry["cohort"], str(entry["source_version"]).lower().replace("-", "_")
    data = protocol["data"]
    if cohort == "conference_2025":
        revision = str(data["conference"]["revision"])
        if version not in {revision, "conference_" + revision}:
            raise ValueError(
                "Conference capture version differs from the declared protocol release"
            )
        if entry.get("reference") != data["conference"]["reference"]:
            raise ValueError(
                "Conference reference identity must match the declared partial reference"
            )
        if entry.get("matcher_stratum") not in data["conference"]["matcher_strata"]:
            raise ValueError("Conference matcher stratum is not declared by the protocol")
    elif cohort.startswith("bioml_"):
        release = cohort.removeprefix("bioml_")
        if release not in data["bio_ml"]["releases"] or version not in {
            release,
            "bioml_" + release,
            "bio_ml_" + release,
        }:
            raise ValueError("Bio-ML capture version differs from the declared protocol release")
        if (
            release == "2026_whole"
            and entry.get("release_revision") != data["bio_ml"]["data_revision_2026"]
        ):
            raise ValueError("Bio-ML whole-pair data revision must match the pinned local release")


def _ontology_identity(case: StudyCaseV2, entry: Mapping[str, Any], side: str) -> tuple[str, str]:
    descriptor = entry.get(side)
    if (
        not isinstance(descriptor, Mapping)
        or not descriptor.get("ontology_id")
        or not descriptor.get("snapshot_identity")
    ):
        raise ValueError(f"{side} requires a declared ontology ID and captured snapshot identity")
    identity = str(descriptor["snapshot_identity"])
    if case.problem is not None:
        if identity != getattr(case.problem, f"{side}_identity"):
            raise ValueError(f"{side} snapshot identity differs from the captured repair input")
        actual = {digest for _, _, digest in getattr(case.problem, f"{side}_documents") if digest}
        if set(descriptor.get("document_sha256", ())) != actual:
            raise ValueError(
                f"{side} resolved document hashes differ from the captured input/imports"
            )
    return str(descriptor["ontology_id"]).lower(), identity


def prepare_study(
    protocol_path: str | Path,
    manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Check local capture provenance and persist every requested case outcome.

    Release checks establish consistency with the pinned protocol and local
    captures. They do not certify upstream authenticity or imply that unprovided
    benchmark pairs were downloaded, executed, or successfully verified.
    """
    manifest_path, output = Path(manifest_path), Path(output)
    protocol = load_protocol(Path(protocol_path))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "exact-repair/local-study-inputs/v2":
        raise ValueError("unsupported local study-input manifest")
    entries = manifest["cases"]
    if len({entry["case_id"] for entry in entries}) != len(entries):
        raise ValueError("study case IDs must be unique")
    captured: list[StudyCaseV2] = []
    identities: list[tuple[tuple[str, str], ...]] = []
    errors: list[list[str]] = []
    for entry in entries:
        metadata = {key: entry[key] for key in ("case_id", "cohort", "source_version", "group_id")}
        split = entry.get("split", "")
        if metadata["cohort"] not in COHORTS or split not in {"train", "development", "test"}:
            raise ValueError("local captures require an explicit supported cohort and saved split")
        row_errors: list[str] = []
        try:
            path = manifest_path.parent / entry["artifact"]
            case = read_record(json.loads(path.read_text()))
            if not isinstance(case, StudyCaseV2) or any(
                getattr(case, key) != value for key, value in metadata.items()
            ):
                raise ValueError("local case artifact differs from its declared schedule identity")
            if case.declared_split and case.declared_split != split:
                raise ValueError("local case artifact differs from its declared split")
            case = replace(case, declared_split=split)
        except (OSError, ValueError, KeyError, TypeError) as error:
            case = StudyCaseV2(
                **metadata,
                declared_split=split,
                availability="unavailable" if isinstance(error, OSError) else "invalid",
                detail=str(error),
            )
        try:
            _release(entry, protocol)
        except (ValueError, KeyError, TypeError) as error:
            row_errors.append(str(error))
        collected_pair: list[tuple[str, str]] = []
        if entry["cohort"] != "generated":
            for side in ("source", "target"):
                try:
                    collected_pair.append(_ontology_identity(case, entry, side))
                except (ValueError, KeyError, TypeError) as error:
                    row_errors.append(str(error))
        captured.append(case)
        identities.append(tuple(collected_pair))
        errors.append(row_errors)

    # Collect holdout identities before validating splits so changing a display
    # alias cannot move an already captured held-out ontology into adaptation.
    heldout = set(manifest.get("heldout_ontology_identities", ()))
    conference_holdout = str(protocol["data"]["conference"]["whole_ontology_holdout"]).lower()
    for case, pair in zip(captured, identities):
        if case.cohort == "conference_2025":
            heldout.update(identity for name, identity in pair if name == conference_holdout)
    by_pair: dict[tuple[str, ...], list[int]] = defaultdict(list)
    by_group: dict[str, list[int]] = defaultdict(list)
    by_name: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (case, pair) in enumerate(zip(captured, identities)):
        by_group[case.group_id].append(index)
        if len(pair) == 2:
            by_pair[tuple(sorted(identity for _, identity in pair))].append(index)
        if case.declared_split != "test" and any(identity in heldout for _, identity in pair):
            errors[index].append("whole-ontology holdout cannot enter training or development")
        for name, identity in pair:
            by_name[(case.cohort, name)][identity].append(index)
    for indexes in by_pair.values():
        if len({(captured[i].group_id, captured[i].declared_split) for i in indexes}) > 1:
            for index in indexes:
                errors[index].append(
                    "all matcher/corruption siblings of one ontology pair must share group and split"
                )
    for indexes in by_group.values():
        if len({captured[i].declared_split for i in indexes}) > 1:
            for index in indexes:
                errors[index].append("a structural parent/pair group cannot cross saved splits")
    for values in by_name.values():
        if len(values) > 1:
            for indexes in values.values():
                for index in indexes:
                    errors[index].append(
                        "one declared ontology ID has inconsistent capture identities within a release"
                    )
    for index, reasons in enumerate(errors):
        if reasons:
            case = captured[index]
            captured[index] = replace(
                case,
                availability="invalid",
                detail="; ".join(filter(None, (case.detail, *reasons))),
            )

    artifact_dir = output.parent / (output.stem + "_cases")
    scheduled = []
    rows = []
    for case, pair in zip(captured, identities):
        artifact = artifact_dir / f"{canonical_hash(case.case_id)}.json"
        write_artifact(artifact, case.to_dict())
        scheduled.append(
            {
                "case_id": case.case_id,
                "cohort": case.cohort,
                "source_version": case.source_version,
                "group_id": case.group_id,
                "declared_split": case.declared_split,
                "artifact": str(artifact.relative_to(output.parent)),
            }
        )
        rows.append(
            {
                "case_id": case.case_id,
                "status": case.availability,
                "detail": case.detail,
                "case_hash": case.content_hash,
                "split": case.declared_split,
                "ontology_identities": pair,
            }
        )
    cohort_counts = {}
    for cohort in sorted(
        {case.cohort for case in captured} | {"conference_2025", "bioml_2026_whole"}
    ):
        indexes = [i for i, case in enumerate(captured) if case.cohort == cohort]
        available = [i for i in indexes if captured[i].availability == "available"]
        expected = (
            protocol["data"]["conference"]["expected_pairs"]
            if cohort == "conference_2025"
            else (
                protocol["data"]["bio_ml"]["expected_whole_pairs"]
                if cohort == "bioml_2026_whole"
                else None
            )
        )
        cohort_counts[cohort] = {
            "requested_cases": len(indexes),
            "available_cases": len(available),
            "requested_pairs": len(
                {
                    tuple(sorted(identity for _, identity in identities[i]))
                    for i in indexes
                    if len(identities[i]) == 2
                }
            ),
            "available_pairs": len(
                {
                    tuple(sorted(identity for _, identity in identities[i]))
                    for i in available
                    if len(identities[i]) == 2
                }
            ),
            "protocol_expected_pairs": expected,
        }
    ontology_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for case, pair in zip(captured, identities):
        for name, identity in pair:
            ontology_groups[identity].append(
                {
                    "ontology_id": name,
                    "cohort": case.cohort,
                    "case_id": case.case_id,
                    "split": case.declared_split,
                }
            )
    report = {
        "schema": "exact-repair/study-preparation/v2",
        "shared_ontology_overlap": {
            key: rows
            for key, rows in sorted(ontology_groups.items())
            if len({row["case_id"] for row in rows}) > 1
        },
        "protocol_hash": canonical_hash(protocol),
        "manifest_hash": canonical_hash(manifest),
        "requested": len(captured),
        "statuses": dict(Counter(case.availability for case in captured)),
        "cohorts": cohort_counts,
        "rows": rows,
        "heldout_snapshot_identities": sorted(heldout),
        "scope": "local captured identity/release consistency; missing benchmark pairs remain unprovided",
    }
    schedule = {
        "schema": "exact-repair/study-schedule/v2",
        "cases": scheduled,
        "seed": manifest.get("seed", protocol["data"]["split_seed"]),
        "preparation": report,
    }
    if "arms" in manifest:
        schedule["arms"] = manifest["arms"]
    write_artifact(output, schedule)
    return report


def main(argv: list[str] | None = None) -> int:
    """Prepare only explicitly supplied local captures; never execute a study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare_study(args.protocol, args.manifest, args.output)
    print(
        json.dumps(
            {
                "requested": report["requested"],
                "statuses": report["statuses"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
