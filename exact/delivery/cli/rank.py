"""Fit and apply local ranking without changing original query membership."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path


def read_queries(path, *, labels):
    """Read positional BioML TSV or explicit JSON queries without merging sources."""
    from exact.runtime.local_query_ranker import _queries

    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError("Queries JSON must contain a list")
    else:
        rows = []
        with path.open(newline="") as stream:
            for qid, row in enumerate(csv.DictReader(stream, delimiter="\t")):
                item = dict(
                    qid=qid,
                    source=row["SrcEntity"],
                    candidates=ast.literal_eval(row["TgtCandidates"]),
                )
                if labels:
                    item["gold_target"] = row["TgtEntity"]
                rows.append(item)
    result = _queries(rows, labels=labels)
    if labels and any(q["gold_target"] not in q["candidates"] for q in result):
        raise ValueError("TRAIN query gold target must belong to its original candidate pool")
    return result


def write_rankings(path, result, queries):
    """Publish the BioML 2026 positional LIST format and verify its permutation."""
    from exact.acceptance.io import atomic
    import io

    if len(result["queries"]) != len(queries):
        raise ValueError("Incomplete original query output")
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t")
    writer.writerow(["SrcEntity", "TgtCandidates"])
    for predicted, query in zip(result["queries"], queries, strict=True):
        targets = [c["target"] for c in predicted["candidates"]]
        if (
            predicted["qid"] != query["qid"]
            or predicted["source"] != query["source"]
            or Counter(targets) != Counter(query["candidates"])
        ):
            raise ValueError("Output changed original query membership or positional order")
        writer.writerow([query["source"], repr(targets)])
    payload = buffer.getvalue().encode()
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Output already contains a different ranking; use a new path")
    atomic(path, payload)
    with path.open(newline="") as stream:
        restored = list(csv.DictReader(stream, delimiter="\t"))
    for saved, original in zip(restored, queries, strict=True):
        targets = ast.literal_eval(saved["TgtCandidates"])
        if saved["SrcEntity"] != original["source"] or Counter(targets) != Counter(
            original["candidates"]
        ):
            raise ValueError("Published ranking failed independent permutation check")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="exact rank", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "apply"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--candidates",
            required=True,
            type=Path,
            help="Native/decision JSONL pair rows (or parquet)",
        )
        sub.add_argument(
            "--queries",
            required=True,
            type=Path,
            help="BioML TSV or JSON list: qid,source,candidates; TRAIN also gold_target",
        )
        sub.add_argument("--model", required=True, type=Path, help="Fitted JSON model file")
        if name == "fit":
            sub.add_argument("--config", required=True, type=Path)
            sub.add_argument(
                "--confirmed-negatives",
                action="store_true",
                required=True,
                help="Confirm other TRAIN candidates are labelled negatives",
            )
            sub.add_argument(
                "--fold-membership", type=Path, help="Optional JSON source-to-fold map"
            )
        else:
            sub.add_argument("--anchors", type=Path, help="Original score-one anchor JSON sidecar")
            sub.add_argument(
                "--output", required=True, type=Path, help="BioML candidate-ranking TSV"
            )
    args = parser.parse_args(argv)
    from exact.acceptance.io import read_frame

    try:
        rows = read_frame(args.candidates)
        if "S_final" not in rows and "S_pair_final" in rows:
            rows["S_final"] = rows.S_pair_final
        queries = read_queries(args.queries, labels=args.command == "fit")
        return _run(args, rows, queries)
    except (ValueError, OSError, KeyError, SyntaxError) as exc:
        parser.error(str(exc))


def _run(args, rows, queries):
    from exact.runtime import local_query_ranker
    from exact.utils.provenance import sha256_file

    support = {(q["source"], t) for q in queries for t in q["candidates"]}
    if args.command == "fit":
        from exact.core.entities.configs.config import ConfigModel

        gold = {(q["source"], q["gold_target"]) for q in queries}
        if set(zip(rows.Src, rows.Tgt)) != support:
            raise ValueError(
                "Every TRAIN query candidate requires a native scored row; score TRAIN with filter_exact_matches=false"
            )
        rows["confirmed_label"] = [int((s, t) in gold) for s, t in zip(rows.Src, rows.Tgt)]
        local_query_ranker.fit(
            rows,
            queries,
            args.model,
            selector_config=ConfigModel.load_config(args.config),
            fold_membership=(
                json.loads(args.fold_membership.read_text()) if args.fold_membership else None
            ),
            application=dict(
                split_role="train",
                negative_label_policy="confirmed_negatives",
                source_ids=[],
                candidate_sha256=sha256_file(args.candidates),
                queries_sha256=sha256_file(args.queries),
                config_sha256=sha256_file(args.config),
            ),
        )
    else:
        anchors = json.loads(args.anchors.read_text()) if args.anchors else []
        anchor_pairs = {(a["Src"], a["Tgt"]) for a in anchors}
        actual = set(zip(rows.Src, rows.Tgt)) if {"Src", "Tgt"} <= set(rows) else set()
        if actual | anchor_pairs != support:
            raise ValueError(
                "Scored rows plus native anchors must equal complete original query support"
            )
        result = local_query_ranker.apply(args.model, rows, queries, anchors=anchors)
        write_rankings(args.output, result, queries)
        from exact.native.score import write_json

        write_json(args.output.with_suffix(".scores.json"), result)
        write_json(
            args.output.with_suffix(".receipt.json"),
            dict(
                fit_identity=result["fit_identity"],
                candidate_occurrences=result["candidate_occurrences"],
                query_count=result["original_query_count"],
                output_sha256=sha256_file(args.output),
                input_sha256=sha256_file(args.candidates),
                queries_sha256=sha256_file(args.queries),
                model_sha256=sha256_file(args.model),
                anchors_sha256=sha256_file(args.anchors) if args.anchors else None,
                scores_sha256=sha256_file(args.output.with_suffix(".scores.json")),
                acceptance=False,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
