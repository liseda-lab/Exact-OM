"""Frozen structural splits and paired native-v3 corpus export; no outcome queries."""

from __future__ import annotations

import argparse
import dataclasses
from collections import Counter
from pathlib import Path

import networkx as nx

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, promote_input_v3, read_record
from tools.repair.batch import read, sha
from tools.repair import expanded_profile as profile
from tools.repair.corpus import MECHANISMS, _case, coherent_control
from tools.repair.prepare import case_from_dict, case_to_dict, save_preparation

SPLITS = ("train", "development", "test", "fresh_evaluation")
TARGETS = dict(train=128, development=32, test=64, fresh_evaluation=64)


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def bound(value):
    path = Path(value["path"])
    if sha(path) != value["sha256"]:
        raise ValueError("Source binding changed: " + str(path))
    return read(path)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if canonical_hash(read(path)) != canonical_hash(value):
            raise ValueError("Frozen corpus artifact changed: " + str(path))
    else:
        write_artifact(path, value)
    return binding(path)


def validate_profile(plan):
    """Check executed source, ownership, all development receipts and selection."""
    report = bound(plan["profile_report"])
    manifest = bound(plan["profile_plan"])
    completion = bound(plan["profile_completion"])
    batch = bound(plan["profile_batch"])
    runtime = bound(plan["profile_runtime"])
    if completion["status"] != "complete" or completion["exit_code"] != 0:
        raise ValueError("Profile has no successful completion receipt")
    if completion["batch"] != plan["profile_batch"]["path"]:
        raise ValueError("Completion belongs to a different profile batch")
    if (completion["step_id"], completion["dispatch_nonce"]) != (
        plan["profile_step"],
        plan["profile_nonce"],
    ):
        raise ValueError("Profile ownership mismatch")
    for path, digest in batch["frozen_files"].items():
        if sha(path) != digest:
            raise ValueError("Frozen profile source changed: " + path)
    if batch["frozen_files"].get(plan["profile_plan"]["path"]) != plan["profile_plan"]["sha256"]:
        raise ValueError("Profile plan not bound to execution")
    if (
        batch["frozen_files"].get(plan["profile_runtime"]["path"])
        != plan["profile_runtime"]["sha256"]
    ):
        raise ValueError("Profile runtime not bound to execution")
    dependencies = []
    for name in ("expanded_profile.py", "corpus.py", "prepare.py"):
        old = Path(batch["code"]) / "tools/repair" / name
        if sha(old) != sha(Path(__file__).with_name(name)):
            raise ValueError("Structural generator changed since profile: " + name)
        dependencies.append((name, sha(old)))
    identity = canonical_hash(
        (
            plan["profile_plan"]["sha256"],
            dependencies,
            profile.IDENTITY_VERSION,
            nx.__version__,
            runtime["dependencies"],
            runtime["code_hashes"],
            runtime["ontology_implementations"],
        )
    )
    profile.checked_checkpoint(Path(plan["profile_report"]["path"]), identity)
    inventory = bound(report["inventory"])
    profile.checked_checkpoint(Path(report["inventory"]["path"]), identity)
    if (
        report["status"] != "complete"
        or report["test_outcomes_opened"]
        or inventory["test_outcomes_opened"]
    ):
        raise ValueError("Profile scope/completion mismatch")
    for source in inventory["sources"]:
        bound(source)
    selected, missing = profile.select_parents(
        inventory["candidates"],
        inventory["exposed"],
        counts=manifest["parent_counts"],
        seed=manifest["seed"],
    )
    for actual, expected in (
        (selected, inventory["selected"]),
        (missing, inventory["missing"]),
        (missing, report["missing_parents"]),
    ):
        if canonical_hash(actual) != canonical_hash(expected):
            raise ValueError("Provisional selection or missing denominator changed")
    expected = {}
    base = Path(plan["profile_report"]["path"]).parent
    for parent in selected:
        if parent["split"] != "profile":
            continue
        for control in ("corrupted", "coherent"):
            key = canonical_hash((parent["key"], control))
            record = read(base / "development_cases" / (key + ".json"))
            case = case_from_dict(record)
            if (
                case.split != "development"
                or case.control != control
                or case.structural_parent != parent["key"]
            ):
                raise ValueError("Profile case scope mismatch")
            for mode in ("cold", "warm"):
                expected[str(base / "rows" / (key + "-" + mode + ".json"))] = (
                    canonical_hash((identity, record["hash"], mode)),
                    parent,
                    control,
                    mode,
                )
    outcomes, summaries, seen = Counter(), [], set()
    for item in report["rows"]:
        if item["path"] not in expected or item["path"] in seen:
            raise ValueError("Unexpected or duplicate development profile row")
        seen.add(item["path"])
        row = bound(item)
        row_identity, parent, control, mode = expected[item["path"]]
        profile.checked_checkpoint(Path(item["path"]), row_identity)
        if (row["parent"], row["family"], row["control"], row["cache_mode_requested"]) != (
            parent["key"],
            parent["family"],
            control,
            mode,
        ) or not row["cleanup_complete"]:
            raise ValueError("Profile row provenance or cleanup mismatch")
        status = (row.get("result") or {}).get("status", "unknown")
        if (item["call_status"], item["result_status"]) != (row["call_status"], status):
            raise ValueError("Profile row summary mismatch")
        outcomes[row["call_status"] + "/" + status] += 1
        summaries.append(
            dict(
                parent=row["parent"],
                family=row["family"],
                control=control,
                mode=mode,
                call_status=row["call_status"],
                result_status=status,
                elapsed_seconds=row["elapsed_seconds"],
                receipt=item,
            )
        )
    if seen != set(expected) or report["recorded"] != len(seen) or report["scheduled"] != 64:
        raise ValueError("Profile denominator mismatch")
    return (
        manifest,
        inventory,
        dict(
            scheduled=report["scheduled"],
            recorded=len(seen),
            outcomes=dict(outcomes),
            rows=summaries,
        ),
    )


def split_schedule(plan, manifest, inventory, summary):
    if manifest["parent_counts"] != dict(
        profile=2, train=8, development=2, test=4, fresh_evaluation=4
    ):
        raise ValueError("Registered parent targets require a separately named amendment")
    if inventory["identity_version"] != profile.IDENTITY_VERSION:
        raise ValueError("Structural fingerprint version mismatch")
    return dict(
        schema="exact-repair/expanded-corpus-split/v1",
        study=plan["study"],
        profile_inventory=bound(plan["profile_report"])["inventory"],
        profile_report=plan["profile_report"],
        seed=manifest["seed"],
        selected=inventory["selected"],
        missing=inventory["missing"],
        targets=TARGETS,
        training_target_cases=224,
        fresh_evaluation_target_cases=64,
        profile_summary=summary,
        generator_aliases=inventory["generator_aliases"],
        identity_version=profile.IDENTITY_VERSION,
        test_outcomes_opened=False,
        decision="Retain every provisional parent and missing target without outcome-based replacement",
        viability="Development compilation only; partial/timeout rows retained. No solve, label or learning viability claim",
        diversity="Eight constructions with path-length variants; fingerprints do not establish broad semantic or structural diversity",
        scientific_budget_amendment=False,
    )


def freeze_split(plan_path, destination):
    plan = read(plan_path)
    manifest, inventory, summary = validate_profile(plan)
    return immutable(destination, split_schedule(plan, manifest, inventory, summary))


def audit_structures(manifest, inventory, output, identity):
    """Independently recompute every historical and candidate fingerprint."""
    recomputed = profile.prepare_inventory(manifest, output / "structural-audit", identity)
    for key in (
        "sources",
        "exposed",
        "candidates",
        "selected",
        "missing",
        "generator_aliases",
        "alias_fingerprint_groups",
        "identity_version",
        "requested_counts",
        "selected_counts",
    ):
        if canonical_hash(recomputed[key]) != canonical_hash(inventory[key]):
            raise ValueError("Structural/exposure audit mismatch: " + key)
    return binding(output / "structural-audit/inventory.json")


def materialize_pair(parent, study, seed):
    """Regenerate evidence after split freeze, with new cohort/parent identities."""
    cohort = "fresh-evaluation" if parent["split"] == "fresh_evaluation" else "training"
    group = study + ":" + cohort + ":" + parent["key"]
    split = "test" if parent["split"] == "fresh_evaluation" else parent["split"]
    case = _case(
        group,
        parent["family"],
        split,
        parent["depth"],
        0,
        seed,
        score_noise=0.1,
        feature_dropout=0.0,
        misleading_label_fraction=0.1,
    )
    case = dataclasses.replace(case, problem=promote_input_v3(case.problem), schema_revision="v3")
    if set(profile.parent_fingerprints(case)) != set(parent["fingerprints"]):
        raise ValueError("Regenerated parent structure changed")
    return [case_to_dict(variant) for variant in (case, coherent_control(case))]


def export_pair(records, parent, output):
    if len(records) != 2:
        raise ValueError("Expected one corrupted case and one coherent control")
    cases = [case_from_dict(record) for record in records]
    if [case.control for case in cases] != ["corrupted", "coherent"]:
        raise ValueError("Missing paired control")
    if (
        len({case.structural_parent for case in cases}) != 1
        or len({case.split for case in cases}) != 1
    ):
        raise ValueError("Paired cases cross structural parents or splits")
    if any(case.schema_revision != "v3" for case in cases):
        raise ValueError("Native v3 cases required")
    result = []
    for record, case in zip(records, cases):
        key = canonical_hash(case.case_id)
        evaluator = immutable(output / "evaluator" / (key + ".json"), record)
        observable = immutable(output / "observable" / (key + ".json"), case.problem.to_dict())
        if read_record(bound(observable)).content_hash != case.problem.content_hash:
            raise ValueError("Observable native-v3 round trip failed")
        result.append(
            dict(
                case_id=case.case_id,
                structural_parent=case.structural_parent,
                group_id=parent["group_id"],
                source_parent=parent["key"],
                family=parent["family"],
                split=parent["split"],
                control=case.control,
                status="materialized",
                fingerprints=profile.parent_fingerprints(case),
                evaluator=evaluator,
                observable=observable,
                case_hash=record["hash"],
                input_hash=case.problem.content_hash,
            )
        )
    return result


def audit_release_rows(rows, inventory):
    """Actual clean/corrupted cores must remain disjoint from all other groups."""
    owners = {}
    for old in inventory["exposed"]:
        for fingerprint in old["fingerprints"]:
            owners[fingerprint] = "historically_exposed"
    for parent in inventory["selected"]:
        for fingerprint in parent["fingerprints"]:
            if owners.setdefault(fingerprint, parent["group_id"]) != parent["group_id"]:
                raise ValueError("Selected parent overlaps exposure or another group")
    seen = set()
    for row in rows:
        if row["status"] != "materialized":
            continue
        if row["case_id"] in seen:
            raise ValueError("Duplicate released case")
        seen.add(row["case_id"])
        for fingerprint in row["fingerprints"]:
            if owners.setdefault(fingerprint, row["group_id"]) != row["group_id"]:
                raise ValueError("Materialized variant overlaps exposure or another group")
        for field in ("evaluator", "observable"):
            bound(row[field])


def run(plan_path, output):
    from exact.repair.study import runtime_manifest
    from exact.repair.workers import bounded_call

    plan_path, output = Path(plan_path), Path(output)
    plan = read(plan_path)
    output.mkdir(parents=True, exist_ok=True)
    manifest, inventory, summary = validate_profile(plan)
    schedule = bound(plan["split_schedule"])
    if canonical_hash(schedule) != canonical_hash(
        split_schedule(plan, manifest, inventory, summary)
    ):
        raise ValueError("Frozen split does not match validated profile selection")
    runtime = runtime_manifest()
    identity = canonical_hash((sha(plan_path), sha(Path(__file__)), runtime))
    immutable(output / "runtime.json", runtime)
    immutable(output / "split.json", schedule)
    audit = audit_structures(manifest, inventory, output, identity)
    rows = []
    exposed_families = {
        MECHANISMS.get(row["family"], row["family"]) for row in inventory["exposed"]
    }
    for parent in schedule["selected"]:
        if parent["split"] == "profile":
            continue
        receipt = output / "parents" / (canonical_hash(parent["key"]) + ".json")
        row_identity = canonical_hash((identity, parent))
        saved = profile.checked_checkpoint(receipt, row_identity)
        if saved is None:
            write_artifact(
                output / "progress.json",
                dict(stage="materializing", parent=parent["key"], recorded=len(rows)),
            )
            result = bounded_call(
                materialize_pair,
                parent,
                plan["study"],
                schedule["seed"],
                timeout=plan["per_parent_seconds"],
                memory_mb=plan["per_parent_memory_mb"],
            )
            if not result.cleanup_complete:
                raise RuntimeError(
                    "Corpus worker cleanup incomplete; inspect descendants before continuation"
                )
            if result.status == "complete":
                outcomes = export_pair(result.value, parent, output)
            else:
                outcomes = [
                    dict(
                        source_parent=parent["key"],
                        group_id=parent["group_id"],
                        family=parent["family"],
                        split=parent["split"],
                        control=control,
                        status="unavailable_materialization",
                        call_status=result.status,
                        detail=result.detail,
                    )
                    for control in ("corrupted", "coherent")
                ]
            saved = profile.checkpoint(
                receipt,
                row_identity,
                rows=outcomes,
                resources=dict(result.resource_usage),
                call_status=result.status,
            )
        rows.extend(saved["rows"])
    for index, missing in enumerate(schedule["missing"]):
        if missing["split"] in SPLITS:
            rows.extend(
                dict(missing, missing_parent_index=index, control=control)
                for control in ("corrupted", "coherent")
            )
    audit_release_rows(rows, inventory)
    releases = {}
    for split in SPLITS:
        subset = [
            dict(
                row,
                family_exposure=(
                    "seen_family" if row["family"] in exposed_families else "unseen_family"
                ),
            )
            for row in rows
            if row["split"] == split
        ]
        if len(subset) != TARGETS[split]:
            raise ValueError("Released denominator differs from frozen target: " + split)
        payload = dict(
            schema="exact-repair/expanded-corpus-release/v1",
            study=plan["study"],
            split=split,
            scheduled=TARGETS[split],
            rows=subset,
            split_schedule=plan["split_schedule"],
            status_counts=dict(Counter(row["status"] for row in subset)),
            teacher_queries=0,
            test_outcomes_opened=False,
            scientific_claim=False,
        )
        release = immutable(output / "releases" / (split + ".json"), payload)
        cases = [
            case_from_dict(bound(row["evaluator"]))
            for row in subset
            if row["status"] == "materialized"
        ]
        path = output / "releases" / (split + "-preparation.json")
        if path.exists():
            from tools.repair.prepare import load_preparation

            prior, caches, info = load_preparation(path)
            if (
                caches
                or canonical_hash([case_to_dict(c) for c in prior])
                != canonical_hash([case_to_dict(c) for c in cases])
                or info != payload
            ):
                raise ValueError("Existing native preparation release changed")
        elif cases:
            save_preparation(path, cases, payload)
        else:
            empty = dict(
                schema="exact-repair/training-preparation/v3", report=payload, cases=[], caches={}
            )
            immutable(path, dict(empty, hash=canonical_hash(empty)))
        releases[split] = dict(manifest=release, preparation=binding(path))
    return profile.checkpoint(
        output / "completion.json",
        identity,
        schema="exact-repair/expanded-corpus-completion/v1",
        status="complete",
        scope="Corpus materialization only; no completed evaluation/training or G0-G2 claim",
        scheduled_cases=288,
        training_target_cases=224,
        fresh_evaluation_target_cases=64,
        status_counts=dict(Counter(row["status"] for row in rows)),
        releases=releases,
        split_schedule=plan["split_schedule"],
        structural_audit=audit,
        profile_summary=summary,
        scientific_budget_amendment=False,
        teacher_queries=0,
        test_outcomes_opened=False,
        api_spend_usd=0,
        followup_stages=plan["followup_stages"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-split", "run"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze-split":
        freeze_split(args.plan, args.output)
    else:
        run(args.plan, args.output)


if __name__ == "__main__":
    main()
