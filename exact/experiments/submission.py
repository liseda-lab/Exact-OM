"""Export saved predictions against public populations without loading reference labels."""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from exact.io.writers.oaei_rdf import OaeiRdfWriter
from exact.runs.reader import RunReader
from exact.utils.data import read_table
from exact.utils.fitted_artifacts import freeze_json
from exact.utils.provenance import sha256_path

NIL_IRI = "https://oaei.ontologymatching.org/2026/diso/NIL"
TRACKS = ("bioml-local", "bioml-global", "oaei-kg-global", "diso-ranking", "biokg-typed")


def _iri(value: Any) -> str:
    if not isinstance(value, str) or not urlsplit(value).scheme or any(c.isspace() for c in value):
        raise ValueError(f"Expected an absolute entity IRI: {value!r}")
    return value


def _unique(values: list[Any], description: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {description}")


def _number(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError("A saved finite score is required for every candidate")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("A saved finite score is required for every candidate")
    return score


def _trace(reader: RunReader) -> dict[str, Any] | None:
    path = reader.layout.source_decisions_path
    if not path.exists():
        return None
    trace = json.loads(path.read_text())
    if (
        trace.get("schema_version") != 2
        or trace.get("stage") != "after_cardinality_and_relation_typing"
    ):
        raise ValueError("Submission requires a final source-decision trace")
    sources = [_iri(row["Src"]) for row in trace["records"]]
    _unique(sources, "source-decision records")
    _unique(trace["source_universe"], "source-universe entries")
    if set(sources) != set(trace["source_universe"]):
        raise ValueError("Incomplete source-decision population")
    return dict(trace)


def _pools(path: Path, track: str) -> list[dict[str, Any]]:
    size: int | None
    if track == "diso-ranking":
        queries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if any(type(row.get("qid")) is not int or row["qid"] < 0 for row in queries):
            raise ValueError("DISO requires original nonnegative integer qid values")
        _unique([row["qid"] for row in queries], "DISO qid values")
        size = 50
    else:
        with path.open(newline="") as stream:
            table = csv.DictReader(stream, delimiter="\t")
            if set(table.fieldnames or []) != {"SrcEntity", "TgtCandidates"}:
                raise ValueError("Use the public gold-stripped SrcEntity/TgtCandidates pool")
            queries = [
                {"source": row["SrcEntity"], "candidates": ast.literal_eval(row["TgtCandidates"])}
                for row in table
            ]
        for index, row in enumerate(queries):
            row["qid"] = index
        size = 50 if track == "biokg-typed" else None
    if not queries:
        raise ValueError("Public candidate population is empty")
    for row in queries:
        if track == "diso-ranking" and set(row) - {"qid", "source", "type", "candidates"}:
            raise ValueError("DISO export accepts only public pool fields, never answer fields")
        _iri(row["source"])
        candidates = row.get("candidates")
        if (
            not isinstance(candidates, list)
            or not candidates
            or (size is not None and len(candidates) != size)
        ):
            raise ValueError(
                f"Invalid {track} query candidate count (expected {size or 'original nonempty pool'})"
            )
        candidates = [_iri(value) for value in candidates]
        _unique(candidates, "public pool candidates")
        if track == "diso-ranking" and NIL_IRI not in candidates:
            raise ValueError("Every DISO pool must include the official NIL IRI")
    return queries


def _scores(reader: RunReader, trace: dict[str, Any] | None, field: str, include_nil: bool):
    scores: dict[tuple[str, str], float] = {}

    def add(source, target, value):
        pair = (_iri(source), _iri(target))
        if pair in scores:
            raise ValueError(f"Duplicate scored candidate pair: {pair!r}")
        scores[pair] = _number(value)

    if trace is not None:
        for row in trace["records"]:
            source = row["Src"]
            for candidate in row["candidates"]:
                add(source, candidate["target"], candidate.get(field))
            if include_nil and (source, NIL_IRI) not in scores and field == "Q_match":
                # Candidate and NIL probabilities must share the saved joint scale.
                values = [row.get("ontology_nil_probability")]
                values += [candidate.get("Q_nil") for candidate in row["candidates"]]
                available = [_number(value) for value in values if value is not None]
                if available:
                    if max(available) - min(available) > 1e-12:
                        raise ValueError("Inconsistent saved NIL probabilities")
                    add(source, NIL_IRI, available[0])
    else:
        for row in reader.iter_explanations():
            add(row["src_iri"], row["tgt_iri"], row.get("confidences", {}).get(field))
    return scores


def _relation_scores(reader):
    path = reader.layout.alignment_dir / "relation_scores.tsv"
    if not path.is_file():
        raise ValueError(
            "BioKG requires all saved candidate/relation scores; sparse mappings cannot be submitted"
        )
    frame = read_table(path)
    scores = {}
    for row in frame.itertuples():
        pair = (_iri(row.SrcEntity), _iri(row.TgtEntity))
        if row.Relation not in {"=", "<", ">"} or row.Relation in scores.setdefault(pair, {}):
            raise ValueError("Invalid or duplicate saved candidate/relation score")
        scores[pair][row.Relation] = _number(row.Score)
    if any(set(row) != {"=", "<", ">"} for row in scores.values()):
        raise ValueError("BioKG requires scores for all three relations for every candidate")
    return scores


def _coverage(scores, expected):
    if set(scores) != expected:
        raise ValueError(
            f"Candidate population mismatch: missing={len(expected - set(scores))}, extra={len(set(scores) - expected)}"
        )


def _query_run_scores(manifest, public_candidates, queries, track, field):
    from exact.experiments.public_inference import _verify

    plan = json.loads(manifest.read_text())
    if (
        plan.get("kind") != "reference_free_inference"
        or plan.get("track") != track
        or plan.get("score_scope") != "original_query"
        or plan.get("queries") != queries
        or plan.get("run_eval") is not False
    ):
        raise ValueError("Invalid original-query inference manifest")
    if _verify(plan["public_candidates"]).resolve() != public_candidates.resolve():
        raise ValueError("Original query input differs from the inference binding")
    results = [None] * len(queries)
    for run in plan["runs"]:
        _verify(run["config"])
        for binding in run.get("inputs", {}).values():
            _verify(binding)
        reader = RunReader.open(Path(run["run_dir"]))
        if sha256_path(reader.layout.config_path) != run["config"]["sha256"]:
            # Runtime may normalize YAML; compare the resolved mappings without loading refs.
            from exact.core.entities.configs.config import ConfigModel

            if ConfigModel.load_config(reader.layout.config_path) != ConfigModel.load_config(
                Path(run["config"]["path"])
            ):
                raise ValueError("Saved run configuration differs from frozen query config")
        indices = run["query_indices"]
        selected = [queries[index] for index in indices]
        _unique([query["source"] for query in selected], "sources in original-query shard")
        trace = _trace(reader)
        scores = (
            _relation_scores(reader)
            if track == "biokg-typed"
            else _scores(reader, trace, field, track == "diso-ranking")
        )
        _coverage(
            scores, {(row["source"], target) for row in selected for target in row["candidates"]}
        )
        for index in indices:
            if results[index] is not None:
                raise ValueError("Duplicate original query assignment")
            row = queries[index]
            results[index] = {target: scores[row["source"], target] for target in row["candidates"]}
    if any(value is None for value in results):
        raise ValueError("Missing original query run")
    return results


def export_submission(
    run_dir: Path,
    output: Path,
    track: str,
    *,
    public_candidates: Path | None = None,
    source_universe: Path | None = None,
    population_manifest: Path | None = None,
    target_population_manifest: Path | None = None,
    query_runs: Path | None = None,
    score_field: str = "S_final",
    source_uri: str | None = None,
    target_uri: str | None = None,
) -> Path:
    """Validate and export a frozen run; require full declared populations for global output.

    Ranking exports require every public candidate's saved score. DISO accepts a
    saved joint NIL probability only with ``score_field='Q_match'``. No default
    NIL score, calibration, model call, reference input, or missing-score fill is used.
    Global source universes must come from the full eligible ontology signature,
    not reference-bearing sources or a capped development population.
    """
    if track not in TRACKS:
        raise ValueError(f"Unsupported submission track: {track}")
    reader = RunReader.open(run_dir)
    output = Path(output).resolve()
    if output.is_relative_to(reader.layout.root):
        raise ValueError("Export outside the immutable run directory")
    trace = _trace(reader)
    inputs = {}
    if trace is not None:
        inputs["source_decisions"] = sha256_path(reader.layout.source_decisions_path)
    if track.endswith("global"):
        if public_candidates is not None:
            raise ValueError(
                "Global submissions require full ontology populations, not candidate pools"
            )
        from exact.experiments.public_inference import (
            validate_population,
            validate_run_population,
        )

        if population_manifest is None:
            raise ValueError("Global export requires a full native population manifest")
        source_population_record, source_universe = validate_population(
            population_manifest, source_universe
        )
        inputs["population_manifest"] = sha256_path(population_manifest)
        if (
            source_universe is None
            or trace is None
            or trace["source_universe_status"] != "declared"
        ):
            raise ValueError(
                "Global export requires a declared full source-universe file and trace"
            )
        population = [
            _iri(line.strip()) for line in source_universe.read_text().splitlines() if line.strip()
        ]
        _unique(population, "public source-universe entries")
        if not population or set(population) != set(trace["source_universe"]):
            raise ValueError("Global run does not cover the full declared source population")
        if target_population_manifest is None:
            raise ValueError("Global export requires a target native population manifest")
        target_population_record, target_population = validate_population(
            target_population_manifest
        )
        validate_run_population(reader, source_population_record, "source")
        validate_run_population(reader, target_population_record, "target")
        targets = set(target_population.read_text().splitlines())
        inputs["target_population_manifest"] = sha256_path(target_population_manifest)
        source_uri, target_uri = _iri(source_uri), _iri(target_uri)
        mapping_path = reader.layout.mapping_path("global")
        if mapping_path.is_file():
            frame = reader.mappings("global")
        else:
            mapping_path = reader.layout.alignment_dir / "paper.maps_global.tsv"
            frame = read_table(mapping_path)
        pairs = list(zip(frame["SrcEntity"].map(_iri), frame["TgtEntity"].map(_iri)))
        _unique(pairs, "global mapping pairs")
        population_set = set(population)
        if any(source not in population_set for source, _ in pairs):
            raise ValueError("Global mapping source is outside the declared population")
        if any(target not in targets for _, target in pairs):
            raise ValueError("Global mapping target is outside the declared population")
        if "Relation" in frame and not frame["Relation"].eq("=").all():
            raise ValueError("These global submissions require equivalence relations")
        if any(not 0 <= _number(value) <= 1 for value in frame["Score"]):
            raise ValueError("Global confidence must be in [0, 1]")
        frame["Relation"] = "="
        inputs["mappings"] = sha256_path(mapping_path)
        inputs["source_universe"] = sha256_path(source_universe)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        OaeiRdfWriter().write(
            frame, temporary, options={"source_uri": source_uri, "target_uri": target_uri}
        )
        content = temporary.read_bytes()
        temporary.unlink()
        query_count = len(population)
    else:
        if public_candidates is None or source_universe is not None:
            raise ValueError("Ranking export requires the original public candidate pool")
        queries = _pools(public_candidates, track)
        if query_runs is not None:
            scores_by_query = _query_run_scores(
                query_runs, public_candidates, queries, track, score_field
            )
            inputs["query_runs"] = sha256_path(query_runs)
        else:
            by_source: dict[str, set[str]] = {}
            for query in queries:
                pool = set(query["candidates"])
                if query["source"] in by_source and by_source[query["source"]] != pool:
                    raise ValueError(
                        "Different original pools share a source; use query_runs to preserve query-dependent scores"
                    )
                by_source[query["source"]] = pool
            scores = (
                _relation_scores(reader)
                if track == "biokg-typed"
                else _scores(reader, trace, score_field, track == "diso-ranking")
            )
            expected = {(row["source"], target) for row in queries for target in row["candidates"]}
            _coverage(scores, expected)
            if trace is not None and set(trace["source_universe"]) != set(by_source):
                raise ValueError("Run and public query source populations differ")
            scores_by_query = [
                {target: scores[row["source"], target] for target in row["candidates"]}
                for row in queries
            ]
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        if track == "bioml-local":
            writer.writerow(["SrcEntity", "TgtCandidates"])
        if track == "biokg-typed":
            writer.writerow(["SrcEntity", "TgtEntity", "Relation", "Score"])
        for row, scores in zip(queries, scores_by_query):
            if track == "biokg-typed":
                from exact.io.writers.typed_tsv import RELATION_TO_TYPED

                writer.writerows(
                    (
                        row["source"],
                        target,
                        RELATION_TO_TYPED[relation],
                        format(scores[target][relation], ".17g"),
                    )
                    for target in row["candidates"]
                    for relation in ("=", "<", ">")
                )
                continue
            ranked = sorted(row["candidates"], key=lambda target: (-scores[target], target))
            if track == "diso-ranking":
                stream.write(json.dumps({"qid": row["qid"], "ranking": ranked}) + "\n")
            else:
                writer.writerow([row["source"], repr(ranked)])
        content = stream.getvalue().encode()
        query_count = len(queries)
        inputs["public_candidates"] = sha256_path(public_candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != content:
        raise FileExistsError(f"Refusing to replace a different submission: {output}")
    if not output.exists():
        temporary = output.with_suffix(output.suffix + ".partial")
        with temporary.open("wb") as binary_stream:
            binary_stream.write(content)
            binary_stream.flush()
            os.fsync(binary_stream.fileno())
        temporary.replace(output)
    freeze_json(
        output.with_suffix(output.suffix + ".manifest.json"),
        {
            "schema_version": 1,
            "track": track,
            "run_dir": str(reader.layout.root),
            "score_field": None if track.endswith("global") else score_field,
            "query_count": query_count,
            "inputs": inputs,
            "output_sha256": sha256_path(output),
            "reference_labels_used": False,
            "models_invoked": False,
        },
    )
    return output
