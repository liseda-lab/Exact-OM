"""Validated scored-pair inputs and atomic portable artifacts."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .features import RAW, FEATURES, features


def digest(path):
    """Return a streaming SHA256 without retaining the file in memory."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic(path, payload):
    """Publish bytes durably, without exposing a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    """Publish a JSON artifact with deterministic encoding."""
    atomic(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def read_frame(path, columns=None):
    """Read CSV/TSV, JSONL or parquet, preserving candidate row order."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True)
        return frame[columns] if columns is not None else frame
    return pd.read_csv(
        path,
        sep="\t" if path.suffix.lower() == ".tsv" else ",",
        usecols=columns,
        keep_default_na=False,
    )


@dataclass
class Candidates:
    """Feature matrix and lexically ordered entity codes for one full population."""

    x: np.ndarray
    src: np.ndarray
    tgt: np.ndarray
    sources: list
    targets: list

    def pairs(self, indices):
        """Resolve selected row indices without changing multiplicity or order."""
        return [(self.sources[self.src[i]], self.targets[self.tgt[i]]) for i in indices]


def prepare(frame):
    """Recompute relative features over the complete supplied candidate graph.

    Input must already contain the corrected ontology endpoint population and
    effective pair scores. This stage makes no model or provider requests.
    """
    missing = set(["Src", "Tgt", *RAW]) - set(frame.columns)
    if missing:
        raise ValueError("Missing candidate columns: " + ", ".join(sorted(missing)))
    if frame[["Src", "Tgt"]].isna().any().any():
        raise ValueError("Missing entity identifier")
    frame = frame.copy(deep=False)
    src_values = frame.Src.astype(str)
    tgt_values = frame.Tgt.astype(str)
    if (src_values.str.len() == 0).any() or (tgt_values.str.len() == 0).any():
        raise ValueError("Empty entity identifier")
    if pd.DataFrame({"s": src_values, "t": tgt_values}).duplicated().any():
        raise ValueError("Duplicate global candidate pair; local query pools are a different task")
    raw = frame[RAW].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite candidate evidence")
    sources, targets = sorted(set(src_values)), sorted(set(tgt_values))
    sm, tm = {s: i for i, s in enumerate(sources)}, {t: i for i, t in enumerate(targets)}
    src = src_values.map(sm).to_numpy(dtype=np.int32)
    tgt = tgt_values.map(tm).to_numpy(dtype=np.int32)
    x = np.empty((len(frame), len(FEATURES)), dtype=np.float32)
    features(raw, src, tgt, len(sources), len(targets), x)
    return Candidates(x, src, tgt, sources, targets)


def references(path):
    """Read positive one-to-one TRAIN/VALID pairs; no private evaluator is imported."""
    frame = read_frame(path)
    aliases = [("Src", "Tgt"), ("SrcEntity", "TgtEntity"), ("source", "target")]
    columns = next((pair for pair in aliases if set(pair) <= set(frame.columns)), None)
    if columns is None:
        raise ValueError("Reference needs Src/Tgt, SrcEntity/TgtEntity, or source/target columns")
    if "Relation" in frame and not frame.Relation.isin(["=", "<", ">", "<=", ">="]).all():
        raise ValueError("TRAIN/VALID input must contain positive references only")
    if frame[list(columns)].isna().any().any():
        raise ValueError("Missing reference endpoint")
    rows = [(str(s), str(t)) for s, t in frame[list(columns)].itertuples(index=False, name=None)]
    if any(not s or not t for s, t in rows):
        raise ValueError("Empty reference endpoint")
    pairs = set(rows)
    if len(pairs) != len(rows):
        raise ValueError("Duplicate reference pair")
    if len({s for s, _ in pairs}) != len(pairs) or len({t for _, t in pairs}) != len(pairs):
        raise ValueError("This acceptance policy requires a one-to-one positive reference")
    if not pairs:
        raise ValueError("Reference contains no positive pairs")
    return pairs


def partial_metrics(pairs, gold, given=()):
    """Reference-covered metric: source OR target coverage, excluding supplied pairs."""
    pred = set(pairs) - set(given)
    ss, tt = {s for s, _ in gold}, {t for _, t in gold}
    judged = {(s, t) for s, t in pred if s in ss or t in tt}
    tp = len(pred & gold)
    return {
        "TP": tp,
        "FP": len(judged) - tp,
        "FN": len(gold) - tp,
        "P": tp / len(judged) if judged else 0.0,
        "R": tp / len(gold) if gold else 0.0,
        "F1": 2 * tp / (len(judged) + len(gold)) if judged or gold else 0.0,
        "selected": len(pairs),
        "unjudged": len(pred - judged),
    }


def export(directory, candidates, indices, probabilities, policy):
    """Write deterministic CSV and OAEI RDF, then verify independent RDF parsing."""
    directory = Path(directory)
    completion = directory / "completion.json"
    if completion.exists():
        existing = json.loads(completion.read_text())
        if existing["policy"] != policy:
            raise ValueError("Output contains a different frozen policy; use a new directory")
        if any(
            digest(directory / name) != checksum for name, checksum in existing["outputs"].items()
        ):
            raise ValueError("Published mapping artifact changed")
        return completion
    pairs = candidates.pairs(indices)
    if len({s for s, _ in pairs}) != len(pairs) or len({t for _, t in pairs}) != len(pairs):
        raise ValueError("Selection is not one-to-one")
    rows = sorted((s, t, float(probabilities[i])) for (s, t), i in zip(pairs, indices))
    frame = pd.DataFrame(rows, columns=["SrcEntity", "TgtEntity", "Score"])
    frame["Relation"] = "="
    atomic(directory / "mappings.csv", frame.to_csv(index=False, float_format="%.9g").encode())
    ns = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"
    rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    tree = ET.Element("{" + rdf + "}RDF")
    alignment = ET.SubElement(tree, "{" + ns + "}Alignment")
    for source, target, score in rows:
        cell = ET.SubElement(ET.SubElement(alignment, "{" + ns + "}map"), "{" + ns + "}Cell")
        ET.SubElement(cell, "{" + ns + "}entity1", {"{" + rdf + "}resource": source})
        ET.SubElement(cell, "{" + ns + "}entity2", {"{" + rdf + "}resource": target})
        ET.SubElement(cell, "{" + ns + "}relation").text = "="
        ET.SubElement(
            cell,
            "{" + ns + "}measure",
            {"{" + rdf + "}datatype": "http://www.w3.org/2001/XMLSchema#float"},
        ).text = format(score, ".9g")
    atomic(directory / "alignment.rdf", ET.tostring(tree, encoding="utf-8", xml_declaration=True))
    restored = [
        (
            cell.find("{" + ns + "}entity1").get("{" + rdf + "}resource"),
            cell.find("{" + ns + "}entity2").get("{" + rdf + "}resource"),
        )
        for cell in ET.parse(directory / "alignment.rdf").iter("{" + ns + "}Cell")
    ]
    if set(restored) != set(pairs) or len(restored) != len(pairs):
        raise ValueError("RDF publication changed mappings")
    write_json(
        directory / "completion.json",
        {
            "status": "complete",
            "policy": policy,
            "mappings": len(rows),
            "candidate_pairs": len(candidates.src),
            "outputs": {
                name: digest(directory / name) for name in ["mappings.csv", "alignment.rdf"]
            },
        },
    )
    return directory / "completion.json"
