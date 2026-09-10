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
TRACKS = ("bioml-local", "bioml-global", "oaei-kg-global", "diso-ranking")


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
        _unique([row["source"] for row in queries], "Bio-ML query sources")
        size = 100
    if not queries:
        raise ValueError("Public candidate population is empty")
    for row in queries:
        if track == "diso-ranking" and set(row) - {"qid", "source", "type", "candidates"}:
            raise ValueError("DISO export accepts only public pool fields, never answer fields")
        _iri(row["source"])
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != size:
            raise ValueError(f"Every {track} query must contain exactly {size} candidates")
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


def export_submission(
    run_dir: Path,
    output: Path,
    track: str,
    *,
    public_candidates: Path | None = None,
    source_universe: Path | None = None,
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
        scores = _scores(reader, trace, score_field, track == "diso-ranking")
        expected = {(row["source"], target) for row in queries for target in row["candidates"]}
        if set(scores) != expected:
            raise ValueError(
                f"Candidate population mismatch: missing={len(expected - set(scores))}, extra={len(set(scores) - expected)}"
            )
        if trace is not None and set(trace["source_universe"]) != {
            row["source"] for row in queries
        }:
            raise ValueError("Run and public query source populations differ")
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        if track == "bioml-local":
            writer.writerow(["SrcEntity", "TgtCandidate", "Score"])
        for row in queries:
            ranked = sorted(
                row["candidates"], key=lambda target: (-scores[row["source"], target], target)
            )
            if track == "diso-ranking":
                stream.write(json.dumps({"qid": row["qid"], "ranking": ranked}) + "\n")
            else:
                writer.writerows(
                    (row["source"], target, format(scores[row["source"], target], ".17g"))
                    for target in ranked
                )
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
