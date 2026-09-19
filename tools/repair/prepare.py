"""Local, evaluator-only corpus preparation and versioned training manifests.

No matcher, reference download or experiment is started by importing this module.
Real cases require explicit clean/observed alignments and intended probes; their
supervision never enters the observable graph.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pyowl_core as owl

from exact.repair.candidates import make_candidate
from exact.repair.learning import ProbeOutcome, RepairLabel, TeacherCache, TeacherProbe
from exact.repair.records import (
    RepairInputV2,
    RevisionObjectV2,
    canonical_hash,
    read_record,
)
from exact.repair.workers import bounded_call
from tools.repair.corpus import GeneratedCase, coherent_control, generate_corpus


def load_protocol(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Resolve declared JSON inheritance with cycle detection and finite numbers."""
    path = path.resolve()
    if path in seen:
        raise ValueError("Cyclic training protocol inheritance")

    def reject_number(value):
        raise ValueError(f"Nonfinite protocol number: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate protocol key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), parse_constant=reject_number, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Training protocol must be an object")
    parent = value.pop("extends", None)
    if parent is None:
        return cast(dict[str, Any], value)
    if not isinstance(parent, str) or not parent:
        raise ValueError("Protocol inheritance requires a nonempty path")

    def merge(base, overrides):
        result = dict(base)
        for key, item in overrides.items():
            result[key] = (
                merge(result[key], item)
                if isinstance(result.get(key), dict) and isinstance(item, dict)
                else item
            )
        return result

    return cast(dict[str, Any], merge(load_protocol(path.parent / parent, seen | {path}), value))


def case_to_dict(case: GeneratedCase) -> dict[str, Any]:
    """Serialize evaluator artifacts separately from the public repair input."""

    def encode(value):
        return owl.canonical_bytes(value).hex()

    value = {
        "case_id": case.case_id,
        "structural_parent": case.structural_parent,
        "family": case.family,
        "split": case.split,
        "problem": case.problem.to_dict(),
        "probes": [
            {
                "probe_id": probe.probe_id,
                "axiom_hex": encode(probe.axiom),
                "family": probe.family,
                "desired": probe.desired,
                "nonvacuity": (
                    None if probe.nonvacuity is None else [encode(v) for v in probe.nonvacuity]
                ),
            }
            for probe in case.probes
        ],
        "intended_assignment": case.intended_assignment,
        "generation_seed": case.generation_seed,
        "mirrored": case.mirrored,
        "origin": case.origin,
        "control": case.control,
        "intended_theory": [encode(value) for value in case.intended_theory],
        "intended_active": [encode(value) for value in case.intended_active],
        "variation": case.variation,
        "ambiguity_group": case.ambiguity_group,
    }
    return {"schema": "exact-repair/teacher-case/v2", "hash": canonical_hash(value), "case": value}


def _probe(value: Mapping[str, Any]) -> TeacherProbe:
    def decode(item):
        return owl.decode_canonical(bytes.fromhex(item))

    if type(value.get("desired", True)) is not bool:
        raise ValueError("Probe intention must be an explicit boolean")
    return TeacherProbe(
        str(value["probe_id"]),
        decode(value["axiom_hex"]),
        str(value["family"]),
        value.get("desired", True),
        (
            None
            if value.get("nonvacuity") is None
            else tuple(decode(item) for item in value["nonvacuity"])
        ),
    )


def case_from_dict(payload: Mapping[str, Any]) -> GeneratedCase:
    """Reject stale/corrupt evaluator artifacts and use authoritative core decoding."""
    value = payload.get("case")
    if payload.get("schema") != "exact-repair/teacher-case/v2" or canonical_hash(
        value
    ) != payload.get("hash"):
        raise ValueError("Teacher case schema or content hash mismatch")
    if not isinstance(value, Mapping):
        raise ValueError("Teacher case must be an object")
    problem = read_record(value["problem"])
    if not isinstance(problem, RepairInputV2):
        raise ValueError("Teacher case requires a repair input")
    return GeneratedCase(
        value["case_id"],
        value["structural_parent"],
        value["family"],
        value["split"],
        problem,
        tuple(_probe(probe) for probe in value["probes"]),
        tuple(value["intended_assignment"]),
        value["generation_seed"],
        value["mirrored"],
        value["origin"],
        value["control"],
        tuple(owl.decode_canonical(bytes.fromhex(item)) for item in value["intended_theory"]),
        tuple(owl.decode_canonical(bytes.fromhex(item)) for item in value["intended_active"]),
        tuple(tuple(item) for item in value["variation"]),
        value["ambiguity_group"],
    )


def cache_from_dict(value: Mapping[str, Any]) -> TeacherCache:
    """Restore a validated cache; unknown labels retain their original masks."""
    return TeacherCache(
        tuple(value["candidate_counts"]),
        tuple(
            RepairLabel(
                tuple(row["assignment"]),
                row["feasible"],
                row["benefit"],
                row["cost"],
                tuple(ProbeOutcome(**probe) for probe in row["semantic_vector"]),
            )
            for row in value["labels"]
        ),
        value["complete"],
        value["stop_reason"],
        tuple(tuple(pair) for pair in value["hashes"]),
        value["elapsed_seconds"],
    )


def save_preparation(
    path: Path,
    cases: Sequence[GeneratedCase],
    report: Mapping[str, Any],
    caches: Mapping[str, TeacherCache] | None = None,
) -> None:
    """Write a reproducible local training manifest atomically."""
    payload = {
        "schema": "exact-repair/training-preparation/v2",
        "report": dict(report),
        "cases": [case_to_dict(case) for case in cases],
        "caches": {key: asdict(value) for key, value in (caches or {}).items()},
    }
    from exact.repair.api import write_artifact

    write_artifact(path, {**payload, "hash": canonical_hash(payload)})


def load_preparation(
    path: Path,
) -> tuple[tuple[GeneratedCase, ...], dict[str, TeacherCache], dict[str, Any]]:
    value = json.loads(path.read_text())
    identity = value.pop("hash", None)
    if canonical_hash(value) != identity:
        raise ValueError("Training preparation content hash mismatch")
    if value.get("schema") != "exact-repair/training-preparation/v2":
        raise ValueError("Unsupported preparation manifest")
    cases = tuple(case_from_dict(case) for case in value["cases"])
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate preparation case IDs")
    parents: dict[str, str] = {}
    for case in cases:
        if case.split not in {"train", "development", "test"}:
            raise ValueError("Unknown case split")
        if parents.setdefault(case.structural_parent, case.split) != case.split:
            raise ValueError("A clean structural parent cannot cross splits")
    return (
        cases,
        {key: cache_from_dict(cache) for key, cache in value["caches"].items()},
        value["report"],
    )


def generated_from_protocol(
    protocol: Mapping[str, Any], *, seed: int | None = None
) -> tuple[GeneratedCase, ...]:
    """Honor numeric grouped counts, corruption siblings and template holdouts."""
    data = protocol["data"]
    evidence = data["evidence"]
    if (
        evidence["simulator"] != "label_jaccard_plus_noise_v2"
        or evidence["latent_corruption_features"]
        or not evidence["counterbalance_provenance"]
    ):
        raise ValueError("Unsupported or leaking evidence simulator")
    if data["split_unit"] != "clean_structural_parent":
        raise ValueError("Generated splits must group clean structural parents")
    holdout = protocol["experiments"]["composition_holdout"]
    cases = generate_corpus(
        split_counts=data["groups_per_family"],
        siblings_per_parent=data["corruptions_per_group"],
        seed=data["split_seed"] if seed is None else seed,
        families=data["families"],
        heldout_families=(holdout,),
        score_noise=evidence["noise_std"],
        misleading_label_fraction=evidence["misleading_label_fraction"],
    )
    cases = tuple(_with_object_count(case, data["generated_objects"]) for case in cases)
    # Clean controls are counted per parent, independently of corruption count.
    controls: list[GeneratedCase] = []
    if data["clean_controls_per_group"]:
        parent_cases: dict[str, GeneratedCase] = {}
        for case in cases:
            parent_cases.setdefault(case.structural_parent, case)
        for case in parent_cases.values():
            controls.extend(
                coherent_control(case, case_id=f"{case.structural_parent}:clean-{index}")
                for index in range(data["clean_controls_per_group"])
            )
    return cases + tuple(controls)


def _with_object_count(case: GeneratedCase, counts: Sequence[int]) -> GeneratedCase:
    """Add observed coherent distractors to meet the declared object-count stratum."""
    allowed = sorted(
        count for count in counts if type(count) is int and count >= len(case.problem.objects)
    )
    if not allowed or any(type(count) is not int or count < 1 for count in counts):
        raise ValueError(
            "Generated object counts cannot omit a mechanism's required editable objects"
        )
    count = allowed[
        int(hashlib.sha256(case.structural_parent.encode()).hexdigest(), 16) % len(allowed)
    ]
    objects, fixed, monitored = (
        list(case.problem.objects),
        list(case.problem.fixed_axioms),
        list(case.problem.policy.monitored_classes),
    )
    intended, theory, evidence = (
        list(case.intended_assignment),
        list(case.intended_theory),
        list(case.problem.evidence),
    )
    for index in range(len(objects), count):
        prefix = "urn:exact:observed-distractor:" + canonical_hash((case.case_id, index))
        source, target = (owl.Class(owl.IRI(prefix + suffix)) for suffix in (":source", ":target"))
        axiom = owl.SubClassOf(source, target)
        object_id = f"distractor-{index}"
        keep = make_candidate(object_id, (axiom,), ("keep",))
        delete = make_candidate(object_id, (), ("delete",), cost_features=(("delete", 1.0),))
        objects.append(
            RevisionObjectV2(
                object_id,
                "mapping",
                (axiom,),
                (keep, delete),
                source_entity=source,
                target_entity=target,
            )
        )
        declarations = (owl.Declaration(source), owl.Declaration(target))
        fixed.extend(declarations)
        monitored.extend((source, target))
        intended.append(0)
        theory.extend((*declarations, axiom))
        evidence.append(
            (
                object_id,
                {"source_text": "topic entity", "target_text": "topic role", "score": 1 / 3},
            )
        )
    return replace(
        case,
        problem=replace(
            case.problem,
            objects=tuple(objects),
            fixed_axioms=tuple(fixed),
            policy=replace(case.problem.policy, monitored_classes=tuple(monitored)),
            evidence=tuple(evidence),
        ),
        intended_assignment=tuple(intended),
        intended_theory=tuple(theory),
        variation=(*case.variation, ("editable_objects", count)),
    )


def _read_alignment(path: Path):
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() in {".rdf", ".xml"}:
        from exact.io.writers.oaei_rdf import read_alignment

        return read_alignment(path)
    from exact.utils.data import read_table

    return read_table(path)


def _real_case(entry: Mapping[str, Any], root: Path) -> GeneratedCase:
    """Load only explicitly supplied local assets; imports follow the core loader."""
    from exact.repair.api import prepare_repair

    source_path, target_path = (root / entry[key] for key in ("source", "target"))
    source, target = owl.load_snapshot(str(source_path)), owl.load_snapshot(str(target_path))
    clean = prepare_repair(source, target, _read_alignment(root / entry["clean_alignment"]))
    probes = tuple(_probe(value) for value in entry["probes"])
    if not probes:
        raise ValueError("Real adaptation requires explicitly declared training-side probes")
    identity = canonical_hash(tuple(sorted((clean.source_identity, clean.target_identity))))
    intended_theory = (
        *clean.fixed_axioms,
        *(ax for obj in clean.objects for ax in obj.original_axioms),
    )
    from tools.repair.train import _verify_intended

    clean_case = GeneratedCase(
        entry["case_id"],
        "pair:" + identity,
        "real_structure",
        entry["split"],
        clean,
        probes,
        tuple(0 for _ in clean.objects),
        entry.get("seed", 0),
        False,
        origin="training_side_real",
        intended_theory=intended_theory,
    )
    if not _verify_intended(clean_case):
        raise ValueError("Clean intended pair must be verified before loading its corruption")
    evidence = json.loads((root / entry["evidence"]).read_text()) if entry.get("evidence") else None
    observed = prepare_repair(
        source, target, _read_alignment(root / entry["observed_alignment"]), evidence=evidence
    )
    intended = []
    clean_by_id = {obj.object_id: obj.original_axioms for obj in clean.objects}
    for obj in observed.objects:
        intended.append(
            next(
                (
                    i
                    for i, c in enumerate(obj.candidates)
                    if c.axioms == clean_by_id.get(obj.object_id)
                ),
                -1,
            )
        )
    return GeneratedCase(
        entry["case_id"],
        "pair:" + identity,
        entry.get("family", "real_structure"),
        entry["split"],
        observed,
        probes,
        tuple(intended),
        entry.get("seed", 0),
        False,
        origin="training_side_real",
        intended_theory=intended_theory,
        variation=(
            ("source_sha256", hashlib.sha256(source_path.read_bytes()).hexdigest()),
            ("target_sha256", hashlib.sha256(target_path.read_bytes()).hexdigest()),
            ("supervision", entry["supervision"]),
        ),
    )


def prepare_real_manifest(
    path: Path, *, deadline_seconds: float = 120.0, call_seconds: float = 30.0
) -> tuple[tuple[GeneratedCase, ...], dict[str, Any]]:
    """Prepare local ontology-pair groups with no held-out adaptation supervision."""
    from tools.repair.train import _verify_intended

    value = json.loads(path.read_text())
    if value.get("schema") != "exact-repair/local-real-pairs/v2":
        raise ValueError("Unsupported local real-pair manifest")
    started, cases, rows = time.monotonic(), [], []
    split_by_pair: dict[str, str] = {}
    heldout = {str((path.parent / item).resolve()) for item in value.get("heldout_ontologies", ())}
    for entry in value["cases"]:
        if entry["split"] not in {"train", "development", "test"}:
            raise ValueError("Real cases require a saved ontology-pair split")
        if entry["supervision"] not in {"public_training", "user_declared", "synthetic_corruption"}:
            raise ValueError("Real adaptation supervision must be explicitly permitted")
        if entry["split"] != "test" and any(
            str((path.parent / entry[key]).resolve()) in heldout for key in ("source", "target")
        ):
            raise ValueError("Whole-ontology holdout cannot enter training or development")
        remaining = deadline_seconds - (time.monotonic() - started)
        if remaining <= 0:
            rows.append({"case_id": entry["case_id"], "status": "preparation_deadline"})
            continue
        result = bounded_call(_real_case, entry, path.parent, timeout=min(call_seconds, remaining))
        if result.status != "complete":
            rows.append(
                {"case_id": entry["case_id"], "status": result.status, "detail": result.detail}
            )
            continue
        case = result.value
        if split_by_pair.setdefault(case.structural_parent, case.split) != case.split:
            raise ValueError("An ontology pair cannot cross train/development/test")
        remaining = deadline_seconds - (time.monotonic() - started)
        verified = bounded_call(
            _verify_intended, case, timeout=min(call_seconds, max(0.0, remaining))
        )
        if verified.status != "complete" or verified.value is not True:
            rows.append({"case_id": case.case_id, "status": "unverified_intended_parent"})
            continue
        cases.append(case)
        rows.append(
            {"case_id": case.case_id, "status": "prepared", "input_hash": case.problem.content_hash}
        )
    return tuple(cases), {
        "requested": len(value["cases"]),
        "produced": len(cases),
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
        "manifest_hash": canonical_hash(value),
        "scope": "local declared training-side supervision; no held-out reference injection",
    }
