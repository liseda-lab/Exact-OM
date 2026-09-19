"""Standalone and sequential matcher adapters for Exact-Repair."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .records import (
    BudgetsV2,
    ObjectiveV2,
    PolicyV2,
    RepairInputV2,
    RepairResultV2,
    RevisionObjectV2,
    canonical_hash,
    make_objective,
)

# Evaluation/teacher channels must never become deployment features. All other
# observed matching channels are retained rather than reduced to one score.
_PRIVATE_KEYS = frozenset(
    {
        "reference",
        "reference_alignment",
        "gold",
        "gold_label",
        "target_label",
        "ground_truth",
        "corruption",
        "corruption_trace",
        "latent_parent",
        "teacher",
        "teacher_label",
        "future_explanation",
        "test_statistics",
        "oracle",
        "hidden_clean_theory",
        "clean_theory",
        "corruption_position",
        "reference_membership",
        "teacher_optimum",
        "teacher_labels",
        "split_id",
        "split_ids",
        "post_repair_explanations",
        "correctness",
    }
)


def observable_evidence(value: Any, *, path: str = "") -> tuple[Any, tuple[str, ...]]:
    """Preserve observed features while recording omitted evaluation-only channels."""
    omitted: list[str] = []
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            key = str(key)
            location = f"{path}.{key}" if path else key
            if key.lower().replace("-", "_") in _PRIVATE_KEYS:
                omitted.append(location)
            else:
                result[key], exclusions = observable_evidence(child, path=location)
                omitted.extend(exclusions)
        return result, tuple(omitted)
    if isinstance(value, (list, tuple)):
        items = []
        for i, child in enumerate(value):
            item, exclusions = observable_evidence(child, path=f"{path}[{i}]")
            items.append(item)
            omitted.extend(exclusions)
        return items, tuple(omitted)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return observable_evidence(value.tolist(), path=path)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"nonfinite matching evidence at {path}")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"matching evidence at {path} must be JSON-compatible")
    return value, ()


def ontology_occurrences(source: Any, target: Any) -> tuple[tuple[str, str, Any], ...]:
    """Enumerate exact side/document/axiom occurrences, preserving shared imports."""
    from pyowl_core import coerce_snapshot

    result = []
    for side, provider in (("source", source), ("target", target)):
        snapshot = coerce_snapshot(provider)
        if not snapshot.is_complete:
            raise ValueError(f"{side} imports are incomplete; repair requires resolved snapshots")
        if tuple(snapshot.iter_extensions()):
            raise ValueError(f"{side} contains extension constructs unsupported by repair records")
        for axiom in snapshot.iter_axioms():
            origins: tuple[Any, ...] = snapshot.origin_index.origins_for(axiom)
            if not origins:
                origins = (None,)
            for origin in origins:
                document_key = (
                    origin.document_key
                    if origin is not None
                    else str(snapshot.structural_fingerprint)
                )
                ordinal = origin.occurrence if origin is not None else 0
                occurrence_id = canonical_hash((side, str(document_key), ordinal, axiom))
                result.append((occurrence_id, f"{side}:{document_key}:{ordinal}", axiom))
    return tuple(sorted(result, key=lambda row: row[0]))


def _document_identities(snapshot: Any) -> tuple[tuple[str, str, str], ...]:
    """Retain resolved import identities without materializing layered core views."""
    import pyowl_core as owl

    pending, seen, documents = [snapshot], set(), set()
    while pending:
        view = pending.pop()
        if id(view) in seen:
            continue
        seen.add(id(view))
        if isinstance(view, owl.OntologyOverlay):
            pending.append(view.base)
        elif isinstance(view, owl.OntologyComposite):
            pending.extend(member.view for member in view.members)
        elif hasattr(view, "iter_documents"):
            for key, document in view.iter_documents():
                documents.add(
                    (
                        str(key),
                        str(document.document_iri.value) if document.document_iri else "",
                        (
                            document.provenance.source_sha256.hex()
                            if document.provenance.source_sha256
                            else ""
                        ),
                    )
                )
        else:
            # A custom OntologyView may expose only semantic identity. Do not
            # invent acquisition locations or file hashes for such providers.
            documents.add((f"view:{view.structural_fingerprint}", "", ""))
    return tuple(sorted(documents))


def prepare_repair(
    source: Any,
    target: Any,
    mappings: Any,
    *,
    matcher_identity: str = "external",
    evidence: Mapping[str, Any] | None = None,
    ontology_objects: Sequence[RevisionObjectV2] = (),
    expressions: Sequence[Any] = (),
    endpoint_alternatives: Mapping[str, Sequence[tuple[str, Any]]] | None = None,
    policy: PolicyV2 | None = None,
    budgets: BudgetsV2 | None = None,
) -> RepairInputV2:
    """Adapt any alignment without invoking a matcher or discarding rich evidence.

    Existing core snapshots are reused. Ontology objects name occurrences from
    :func:`ontology_occurrences`; their originals are removed only at that exact
    occurrence. Property/individual mappings remain typed logical axioms.
    """
    import pyowl_core as owl

    from exact.io.writers._frames import canonical_frame

    from .candidates import mapping_candidates

    source, target = owl.coerce_snapshot(source), owl.coerce_snapshot(target)
    occurrences = ontology_occurrences(source, target)
    selected = {obj.occurrence_id: obj for obj in ontology_objects}
    if len(selected) != len(ontology_objects):
        raise ValueError("an ontology occurrence may be selected only once")
    found = set()
    fixed = set()
    for identity, origin, axiom in occurrences:
        if identity in selected:
            obj = selected[identity]
            if obj.kind != "ontology_axiom" or set(obj.original_axioms) != {axiom}:
                raise ValueError("ontology patch original differs from its asserted occurrence")
            found.add(identity)
        else:
            fixed.add(axiom)
    if found != set(selected):
        raise ValueError("unknown ontology occurrence in repair request")
    if isinstance(mappings, Mapping):
        mappings = [mappings]
    if (
        isinstance(mappings, (list, tuple))
        and mappings
        and all(isinstance(row, Mapping) for row in mappings)
    ):
        import pandas as pd

        # Preserve explicit missing values across heterogeneous JSON rows. Pandas
        # otherwise coerces None to NaN when another mapping has a numeric score.
        mappings = pd.DataFrame(mappings, dtype=object)
    if isinstance(mappings, (list, tuple)) and not mappings:
        import pandas as pd

        mappings = pd.DataFrame(columns=["SrcEntity", "TgtEntity"])
    frame = canonical_frame(mappings, require_score=False)
    for column, default in (("eligible", True), ("locked", False), ("Kind", "class")):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: (
                    default
                    if value is None or (isinstance(value, float) and math.isnan(value))
                    else value
                )
            )
    objects = []
    features: dict[str, Any] = {}
    entities = {
        "class": owl.Class,
        "object_property": owl.ObjectProperty,
        "data_property": owl.DataProperty,
        "individual": owl.NamedIndividual,
    }
    relations = {"equivalent": "=", "source_subsumed_by_target": "<", "source_subsumes_target": ">"}
    for index, row in enumerate(frame.to_dict(orient="records")):
        kind = str(getattr(row["Kind"], "value", row["Kind"]))
        row["Kind"] = kind
        target_kind = row.get("TgtKind", kind)
        if str(getattr(target_kind, "value", target_kind)) != kind:
            raise ValueError("cross-kind correspondences cannot be coerced into OWL mappings")
        eligible, locked = row.get("eligible", True), row.get("locked", False)
        if type(eligible) is not bool or type(locked) is not bool:
            raise ValueError("mapping eligibility and locks must be booleans")
        if kind not in entities:
            raise ValueError(f"unsupported mapping entity kind: {kind}")
        left, right = str(row["SrcEntity"]), str(row["TgtEntity"])
        if not left.strip() or not right.strip() or left == "nan" or right == "nan":
            raise ValueError("mapping endpoints must be nonempty IRIs")
        relation = relations.get(str(row["Relation"]), str(row["Relation"]))
        supplied_id = row.get("object_id")
        object_id = (
            str(supplied_id)
            if supplied_id is not None
            and not (isinstance(supplied_id, float) and math.isnan(supplied_id))
            else f"mapping:{index}:{canonical_hash((left, right, kind, relation))[:16]}"
        )
        constructor = entities[kind]
        candidates = mapping_candidates(
            object_id,
            constructor(owl.IRI(left)),
            constructor(owl.IRI(right)),
            relation,
            entity_kind=kind,
            eligible=eligible,
            locked=locked,
            expressions=expressions,
            endpoint_alternatives=(endpoint_alternatives or {}).get(object_id, ()),
        )
        row["object_id"] = object_id
        keep = next(c for c in candidates if "keep" in c.action_tags)
        objects.append(
            RevisionObjectV2(
                object_id,
                "mapping",
                keep.axioms,
                candidates,
                eligible=eligible,
                locked=locked,
                source_entity=constructor(owl.IRI(left)),
                target_entity=constructor(owl.IRI(right)),
            )
        )
        score = row.get("Score")
        if score is not None and not math.isfinite(float(score)):
            raise ValueError("mapping scores must be finite or explicitly absent")
        observed, omitted = observable_evidence(row)
        features[object_id] = {
            "mapping": observed,
            "score_missing": score is None,
            "matcher_identity": matcher_identity,
            "omitted": omitted,
        }
    external, omitted = observable_evidence(dict(evidence or {}))
    for key, value in external.items():
        if key in features:
            features[key]["matching_features"] = value
        else:
            features[key] = value
    features["evidence_omissions"] = list(omitted)
    all_objects = tuple(objects) + tuple(ontology_objects)
    original_axioms = (
        tuple(source.iter_axioms())
        + tuple(target.iter_axioms())
        + tuple(ax for obj in objects for ax in obj.original_axioms)
    )
    signature = {
        entity
        for ax in original_axioms
        for entity in owl.signature(ax)
        if entity.kind == owl.EntityKind.CLASS and entity != owl.OWL_NOTHING
    }
    # A supplied policy can add obligations, but cannot accidentally shrink the
    # original monitored class signature to mapped endpoints or initial witnesses.
    policy = dataclasses.replace(
        policy or PolicyV2(),
        exceptions=tuple(
            owl.Class(owl.IRI(c)) if isinstance(c, str) else c
            for c in (policy or PolicyV2()).exceptions
        ),
        monitored_classes=tuple(
            sorted(
                signature
                | {
                    owl.Class(owl.IRI(c)) if isinstance(c, str) else c
                    for c in (policy or PolicyV2()).monitored_classes
                },
                key=canonical_hash,
            )
        ),
    )
    return RepairInputV2(
        tuple(sorted(fixed, key=canonical_hash)),
        all_objects,
        policy,
        str(source.structural_fingerprint),
        str(target.structural_fingerprint),
        matcher_identity,
        tuple(sorted(features.items())),
        tuple(source.iter_axioms()),
        tuple(target.iter_axioms()),
        budgets or BudgetsV2(),
        source_documents=_document_identities(source),
        target_documents=_document_identities(target),
    )


def matching_run_evidence(run_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Read an existing run's mappings, complete explanations and alternatives."""
    from exact.runs.reader import RunReader

    reader = RunReader.open(Path(run_dir))
    mappings = reader.mappings("global")
    evidence: dict[str, Any] = {"matching_run": str(Path(run_dir).resolve())}
    try:
        evidence["explanations"] = list(reader.iter_explanations())
    except FileNotFoundError:
        evidence["explanations_missing"] = True
    try:
        evidence["alternative_candidates"] = reader.mappings("local").to_dict(orient="records")
    except FileNotFoundError:
        evidence["alternatives_missing"] = True
    if reader.layout.source_decisions_path.is_file():
        evidence["source_decisions"] = json.loads(reader.layout.source_decisions_path.read_text())
    # Published explanations can omit intermediate model channels. Preserve the
    # available inference checkpoint/audit/candidate data as optional evidence;
    # never read training, validation, test-label or fitted-statistic artifacts.
    from exact.runs.manifest import sha256_file

    captured = []
    checkpoint_dir = reader.layout.checkpoints_dir
    paths = sorted(checkpoint_dir.glob("inference_*.json"))
    for directory in sorted(checkpoint_dir.glob("inference_*")):
        if directory.is_dir() and directory.name.endswith(("_audit", "_candidates", "_overlay")):
            paths.extend(sorted(directory.glob("shard-*.jsonl*")))
    for path in paths:
        if path.name.endswith(".jsonl.zst"):
            import zstandard

            with zstandard.open(path, "rt", encoding="utf-8") as stream:
                content = [json.loads(line) for line in stream if line.strip()]
        elif path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                content = [json.loads(line) for line in stream if line.strip()]
        else:
            content = json.loads(path.read_text())
        content, omitted = observable_evidence(content)
        captured.append(
            {
                "path": reader.layout.relative(path),
                "sha256": sha256_file(path),
                "content": content,
                "omitted": omitted,
            }
        )
    evidence["inference_artifacts"] = captured
    return mappings, evidence


def repair_alignment(
    source: Any, target: Any, mappings: Any, *, objective: ObjectiveV2 | None = None, **options: Any
) -> RepairResultV2:
    """Run standalone, or immediately after any matcher using its in-memory output."""
    from .kernel import repair

    problem = prepare_repair(source, target, mappings, **options)
    # An explicit untrained edit-cost control, not confidence-as-semantic-benefit.
    objective = objective or make_objective(
        problem.objects,
        profile=(
            ("edit", 1.0),
            ("delete", 1.0),
            ("ontology_edit", 2.0),
            ("human_authored_ontology_edit", 2.0),
        ),
    )
    return repair(problem, objective)


def write_artifact(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a replay envelope without altering ontology inputs."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
