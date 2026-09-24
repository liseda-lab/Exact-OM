"""Freeze reference-free inputs without changing the selected matching recipe.

Local runs contain at most one original query per source, so pool-dependent
normalization, NIL decisions and listwise judges never see a source-unioned pool.
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from exact.utils.fitted_artifacts import freeze_json
from exact.utils.provenance import dataset_signature_for_paths, sha256_file


def _write(path: Path, content: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Frozen public input identity conflict: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(content)
        temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _verify(binding: dict[str, Any]) -> Path:
    path = Path(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise ValueError(f"Public inference binding changed: {path}")
    return path


def prepare_population(
    ontology: Path, destination: Path, *, entity_kinds: list[str], filter_ignored: bool = False
) -> dict[str, Any]:
    """Enumerate the native ontology closure, never a benchmark reference/query file."""
    from exact.core.entities.kinds import normalize_entity_kinds
    from exact.ontology import load_ontology
    from exact.ontology.provenance import _core_provenance

    kinds = normalize_entity_kinds(entity_kinds)
    ontology = Path(ontology).resolve()
    manifest = destination.with_suffix(destination.suffix + ".manifest.json")
    if manifest.is_file():
        record, _ = validate_population(manifest, destination)
        if (
            record["ontology"]["sha256"] != sha256_file(ontology)
            or record["entity_kinds"] != [kind.value for kind in kinds]
            or record["filter_ignored_alignment_classes"] != filter_ignored
        ):
            raise ValueError("Existing native population was prepared with different inputs/policy")
        return record
    source = load_ontology(ontology)
    core = cast(dict[str, Any], _core_provenance(source.owl_snapshot()))
    if core["backend"] != "native" or core["closure"]["complete"] is not True:
        raise ValueError("Public ontology populations require a complete native closure")
    excluded = set(source.excluded_from_alignment()) if filter_ignored else set()
    by_kind = {kind.value: sorted(set(source.entities(kind)) - excluded) for kind in kinds}
    population = sorted({iri for values in by_kind.values() for iri in values})
    if not population:
        raise ValueError("The full eligible ontology population is empty")
    record = {
        "schema_version": 1,
        "kind": "full_native_ontology_population",
        "ontology": _binding(ontology),
        "entity_kinds": [kind.value for kind in kinds],
        "filter_ignored_alignment_classes": filter_ignored,
        "source_cap": None,
        "reference_labels_used": False,
        "ontology_core": core,
        "counts_by_kind": {kind: len(values) for kind, values in by_kind.items()},
        "count": len(population),
        "population": _write(destination, "\n".join(population) + "\n"),
    }
    freeze_json(destination.with_suffix(destination.suffix + ".manifest.json"), record)
    return record


def validate_population(path: Path, population: Path | None = None) -> tuple[dict[str, Any], Path]:
    record = json.loads(path.read_text())
    if (
        record.get("kind") != "full_native_ontology_population"
        or record.get("source_cap") is not None
        or record.get("reference_labels_used") is not False
        or record.get("ontology_core", {}).get("backend") != "native"
        or record.get("ontology_core", {}).get("closure", {}).get("complete") is not True
    ):
        raise ValueError("Global inference requires a full native ontology population manifest")
    bound = _verify(record["population"])
    _verify(record["ontology"])
    if population is not None and population.resolve() != bound.resolve():
        raise ValueError("Source universe differs from the native population binding")
    values = bound.read_text().splitlines()
    if len(values) != record["count"] or values != sorted(set(values)):
        raise ValueError("Native population manifest count/uniqueness mismatch")
    return record, bound


def validate_run_population(reader, record, side):
    from exact.core.entities.configs.config import ConfigModel

    config = ConfigModel.load_config(reader.layout.config_path)
    ontology = getattr(config.data, side)
    if (
        ontology is None
        or sha256_file(ontology) != record["ontology"]["sha256"]
        or sorted(config.matching.entity_kinds) != sorted(record["entity_kinds"])
        or config.dataset.filter_ignored_alignment_classes
        != record["filter_ignored_alignment_classes"]
        or config.run.source_cap is not None
    ):
        raise ValueError("Saved run ontology or population policy differs from native manifest")
    stats_path = reader.layout.root / "stats/run_stats.json"
    stats = json.loads(stats_path.read_text())
    core = stats.get("ontology_stack", {}).get(side, {}).get("core", {})
    expected = record["ontology_core"]
    if (
        core.get("backend") != "native"
        or core.get("closure", {}).get("complete") is not True
        or core.get("fingerprints") != expected["fingerprints"]
        or sorted(
            row["source_sha256"] for row in core.get("closure", {}).get("source_documents", [])
        )
        != sorted(row["source_sha256"] for row in expected["closure"]["source_documents"])
    ):
        raise ValueError("Saved run native ontology closure differs from population manifest")


def _inference_config(config: Path, source: Path, target: Path) -> dict[str, Any]:
    from exact.core.entities.configs.config import ConfigModel
    from exact.experiments.rationale_policy import apply_rationale_policy

    resolved = apply_rationale_policy(ConfigModel.load_config(config).model_dump(mode="json"))
    selected = ConfigModel.load_config(config)
    # Resolve the original recipe, then apply only its immutable fitted artifacts.
    for name in (
        "retrieval",
        "fusion",
        "rerank",
        "llm",
        "accept",
        "calibration",
        "structure",
        "relation",
    ):
        resolved["supervision"]["components"][name] = selected.supervision.resolve_component(
            name,
            training_available=bool(selected.data.refs.get("train")),
            profile_binding={"dataset_signature": dataset_signature_for_paths(source, target)},
        )[0]
    resolved["data"].update(
        track=None,
        task=None,
        descriptor=None,
        revision=None,
        source=str(source.resolve()),
        target=str(target.resolve()),
        refs={},
        train_candidates=None,
        reference_role=None,
        candidates=None,
        source_universe=None,
        candidate_source="generated",
    )
    resolved["run"].update(source_cap=None, experiment_audit=True)
    resolved["matching"]["nil"]["pool_miss_development_reference"] = None
    resolved["matching"]["nil"]["training_source_labels"] = None
    resolved["matching"]["relation_training_file"] = None
    for name in ("encoder_finetune", "cross_encoder"):
        resolved["candidates"][name]["training"] = None
    for row in resolved["pipeline"]:
        for key in tuple(row["params"]):
            if "reference" in key or key == "training_source_labels":
                row["params"].pop(key)
    # Unlabelled population gates are recomputed for the public candidate population.
    gate = resolved["llm"]["experiment"]["gate"]
    if gate["mode"] in {"source_top_fraction", "pair_top_fraction"}:
        gate["artifact"] = None
    exact_policy = resolved["matching"]["anchor_rescoring"]["exact_policy"]
    if exact_policy is not None:
        resolved["dataset"]["filter_exact_matches"] = exact_policy == "hard"
    return resolved


def prepare_public_inference(
    config: Path,
    destination: Path,
    *,
    source: Path,
    target: Path,
    track: str,
    public_candidates: Path | None = None,
) -> Path:
    """Write immutable configs and original-query assignment; do not run a model."""
    from exact.core.entities.configs.config import ConfigModel
    from exact.core.entities.configs.yaml_io import dump_yaml_document
    from exact.experiments.submission import NIL_IRI, TRACKS, _pools

    if track not in TRACKS:
        raise ValueError(f"Unsupported submission track: {track}")
    destination = destination.resolve()
    base = _inference_config(config, source, target)
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": "reference_free_inference",
        "track": track,
        "selected_config": _binding(config),
        "source": _binding(source),
        "target": _binding(target),
        "reference_labels_used": False,
        "run_eval": False,
        "generate_llm_rationales": False,
        "runs": [],
    }
    configs: list[tuple[dict[str, Any], list[int]]]
    if track.endswith("global"):
        if public_candidates is not None:
            raise ValueError("Global inference cannot use a local candidate/query population")
        base["data"].update(execution_mode="global_alignment", candidate_provenance="generated")
        for side, ontology in (("source", source), ("target", target)):
            path = destination / f"{side}.population.txt"
            prepare_population(
                ontology,
                path,
                entity_kinds=base["matching"]["entity_kinds"],
                filter_ignored=base["dataset"]["filter_ignored_alignment_classes"],
            )
            record[side + "_population_manifest"] = _binding(path.with_suffix(".txt.manifest.json"))
        base["data"]["source_universe"] = str(destination / "source.population.txt")
        configs = [(base, [])]
    else:
        if public_candidates is None:
            raise ValueError("Local inference requires the original public queries")
        if track == "biokg-typed":
            if base["matching"]["relation_prediction"] not in {"none", "learned_three_way"}:
                raise ValueError(
                    "BioKG block output requires a head with saved full relation distributions"
                )
            base["io"]["output_formats"] = list(
                dict.fromkeys(base["io"]["output_formats"] + ["typed-tsv"])
            )
        if base["dataset"]["drop_exact_match_sources"]:
            raise ValueError(
                "Local submission must score every candidate; drop_exact_match_sources is incompatible"
            )
        queries = _pools(public_candidates, track)
        record["public_candidates"] = _binding(public_candidates)
        record["query_count"] = len(queries)
        record["queries"] = queries
        record["score_scope"] = "original_query"
        shards: list[list[int]] = []
        occurrences: dict[str, int] = defaultdict(int)
        for index, query in enumerate(queries):
            shard = occurrences[query["source"]]
            occurrences[query["source"]] += 1
            if len(shards) <= shard:
                shards.append([])
            shards[shard].append(index)
        configs = []
        for shard, indices in enumerate(shards):
            local = json.loads(json.dumps(base))
            table = io.StringIO(newline="")
            writer = csv.writer(table, delimiter="\t", lineterminator="\n")
            writer.writerow(["SrcEntity", "TgtCandidates"])
            for index in indices:
                query = queries[index]
                candidates = [target for target in query["candidates"] if target != NIL_IRI]
                if not candidates:
                    raise ValueError("A local query must have at least one real candidate")
                writer.writerow([query["source"], repr(candidates)])
            pool = _write(destination / f"shard-{shard}.candidates.tsv", table.getvalue())
            population = _write(
                destination / f"shard-{shard}.sources.txt",
                "\n".join(sorted(queries[i]["source"] for i in indices)) + "\n",
            )
            local["data"].update(
                candidates=pool["path"],
                source_universe=population["path"],
                execution_mode="local_ranking",
                candidate_provenance="benchmark_supplied",
                candidate_source="track",
            )
            configs.append((local, indices))
    for index, (resolved, indices) in enumerate(configs):
        from exact.utils.frozen_inference import freeze_inference_manifest

        validated = ConfigModel.from_mapping(resolved, warn_v1=False)
        artifact_path = destination / f"inference-{index}.frozen.json"
        freeze_inference_manifest(validated, artifact_path)
        validated.supervision.inference_artifact = artifact_path
        config_path = destination / f"inference-{index}.yaml"
        binding = _write(
            config_path, dump_yaml_document(validated.model_dump(mode="json", by_alias=True))
        )
        record["runs"].append(
            {
                "config": binding,
                "run_dir": str(destination / f"run-{index}"),
                "query_indices": indices,
                "run_eval": False,
                "inputs": {
                    name: _binding(Path(resolved["data"][name]))
                    for name in ("candidates", "source_universe")
                    if resolved["data"].get(name)
                },
            }
        )
    path = destination / "inference.json"
    freeze_json(path, record)
    return path
