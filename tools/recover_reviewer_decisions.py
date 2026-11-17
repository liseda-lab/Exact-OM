"""Extract a small, hash-verified decision receipt on the original data server.

Read-only: no inference, reference labels, ontology copies or dependency installation.
Only numeric matcher fields and explicitly requested query identities are exported.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

NATIVE_HASH = "d5b5b54e19ca04de29f5843e376bceb921fd4d9f5bd1e9e277b307c0e6581459"
SUBMISSION_HASH = "9af4541dc44b35b8e32531214ecc03913f9e936352037812eff98a2ac50a8577"
APPLY_HASH = "4bedf786fef14316db6036392e0804e1d8ac7c43f69ffab496385c9aa1c186cb"
NUMERIC_FIELDS = frozenset(
    "S_base S_struct S_lctx s_label s_label_star s_hier s_attr s_sim s_diff s_strsim "
    "U U_dis U_ind Q_struct q_attr q_diff q_hier q_label q_lex q_sim q_strsim "
    "I_attr I_ctx I_diff I_hier I_label I_lex I_sim I_strsim I_struct w_c w_i w_struct".split()
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Reader:
    def __init__(self):
        self.inventory = {}

    def read(self, path, expected=None):
        path = Path(path)
        raw = path.read_bytes()
        sha = digest(raw)
        if expected and sha != expected:
            raise ValueError(f"Changed original artifact: {path}")
        self.inventory[str(path)] = sha
        return raw

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def bound(self, ref):
        return self.json(ref["path"], ref["sha256"])

    def shard(self, ref):
        receipt = self.bound(ref)
        if receipt["status"] != "complete":
            raise ValueError("Incomplete shard")
        compressed = self.read(receipt["data"]["path"], receipt["data"]["sha256"])
        # Use the data server's existing decoder; do not modify its Python environment.
        raw = subprocess.run(
            ["zstd", "-dc"], input=compressed, capture_output=True, check=True
        ).stdout
        if digest(raw) != receipt["raw_sha256"] or len(raw) != receipt["raw_bytes"]:
            raise ValueError("Changed uncompressed shard")
        return json.loads(raw), receipt


def recover(root, query_ids):
    reader = Reader()
    native = reader.json(root / "local-native/test/result.json", NATIVE_HASH)
    submission = reader.json(root / "local-apply/submission-complete.json")
    if submission["status"] != "complete" or submission["mode"] != "local_ranking":
        raise ValueError("Expected the completed original local-ranking run")
    if submission["submission"]["sha256"] != SUBMISSION_HASH:
        raise ValueError("Wrong final submission")
    identity = reader.json(root / "local-apply/outputs/identity.json")
    if identity["native"]["sha256"] != NATIVE_HASH:
        raise ValueError("Application and native results are unrelated")
    source_manifest = reader.bound(identity["source"])
    worker = next(b for b in source_manifest["bindings"] if b["sha256"] == APPLY_HASH)
    reader.read(worker["path"], APPLY_HASH)
    population = reader.bound(native["population"])
    inputs = reader.bound(population["query_identity"])
    input_by_id = {q["qid"]: q for q in inputs}
    if set(query_ids) - input_by_id.keys():
        raise ValueError("Unknown original query")
    anchors = {(r["Src"], r["Tgt"]) for r in population["anchors"]}
    outputs, applied, selected_shards, refs = {}, {}, set(), {}
    for ref in submission["outputs"]:
        block, receipt = reader.shard(ref)
        hits = [q for q in block["queries"] if q["qid"] in query_ids]
        if not hits:
            continue
        selected_shards.add(receipt["key"])
        for q in hits:
            original = input_by_id[q["qid"]]
            targets = [c["target"] for c in q["candidates"]]
            if (
                q["source"] != original["source"]
                or len(targets) != 100
                or set(targets) != set(original["candidates"])
            ):
                raise ValueError("Output is not the original query permutation")
            if q["qid"] in outputs:
                raise ValueError("Duplicate output query")
            outputs[q["qid"]] = q
            refs[q["qid"]] = receipt["data"]["sha256"]
        for row in block["pair_scores"]:
            applied[(row["Src"], row["Tgt"])] = row
    if set(outputs) != set(query_ids):
        raise ValueError("Missing output query")
    pairs = {(q["source"], c["target"]) for q in outputs.values() for c in q["candidates"][:5]}
    raw_rows, native_refs = {}, {}
    for ref in native["shards"]:
        receipt = reader.bound(ref)
        if receipt["key"] not in selected_shards:
            continue
        block, receipt = reader.shard(ref)
        for row in block["rows"]:
            pair = row["Src"], row["Tgt"]
            if pair in pairs:
                if pair in raw_rows and raw_rows[pair] != row:
                    raise ValueError("Ambiguous native pair")
                raw_rows[pair] = row
                native_refs[pair] = receipt["data"]["sha256"]
    bank = reader.bound(identity["bank"])
    answers, answer_refs = {}, {}
    for ref in bank["shards"]:
        if not Path(ref["path"]).name.startswith("test-"):
            continue
        block, receipt = reader.shard(ref)
        for answer in block["answers"]:
            pair = tuple(answer["pair"])
            if pair in pairs:
                if pair in answers and answers[pair] != answer:
                    raise ValueError("Ambiguous model response")
                answers[pair] = answer
                answer_refs[pair] = receipt["data"]["sha256"]
    # The TSV is a ranked candidate list, never an accepted alignment.
    tsv = reader.read(submission["submission"]["path"], SUBMISSION_HASH).decode()
    saved = list(csv.DictReader(tsv.splitlines(), delimiter="\t"))
    positions = {q["qid"]: i for i, q in enumerate(inputs)}
    result = {}
    for qid, q in outputs.items():
        line = saved[positions[qid]]
        if line["SrcEntity"] != q["source"] or ast.literal_eval(line["TgtCandidates"]) != [
            c["target"] for c in q["candidates"]
        ]:
            raise ValueError("Published ranking differs from saved query output")
        case_id = f"ncit-doid-q{qid:04d}"
        candidates = []
        for rank, c in enumerate(q["candidates"][:5], 1):
            pair = q["source"], c["target"]
            protected = bool(c.get("protected_exact"))
            if protected != (pair in anchors):
                raise ValueError("Exact anchor disagrees with final ranking")
            row, scores, answer = raw_rows.get(pair), applied.get(pair), answers.get(pair)
            if not protected and (row is None or scores is None):
                raise ValueError("Missing scored pair")
            if scores and scores["cached_response_reused"] != (answer is not None):
                raise ValueError("Model response reuse disagrees with bank")
            if scores and answer and scores["p_llm"] != answer["probability"]:
                raise ValueError("Model signal differs from applied response")
            numeric = {k: v for k, v in (row or {}).items() if k in NUMERIC_FIELDS}
            if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in numeric.values()):
                raise ValueError("Invalid numeric matcher field")
            candidates.append(
                dict(
                    candidate_id=f"{case_id}-candidate-{rank}",
                    target=c["target"],
                    rank=rank,
                    final_score=c["score"],
                    utility=c.get("utility"),
                    protected_exact=protected,
                    native_scores=numeric,
                    applied=scores,
                    model_answer=(
                        {
                            k: answer[k]
                            for k in [
                                "probability",
                                "cached_response_reused",
                                "live_provider_call_attempted",
                                "selector_stage_consultation",
                            ]
                        }
                        if answer
                        else None
                    ),
                    artifact_refs=[
                        "sha256:" + h
                        for h in [refs[qid], native_refs.get(pair), answer_refs.get(pair)]
                        if h
                    ],
                )
            )
        result[case_id] = dict(
            source=q["source"],
            query_id=qid,
            candidate_count=len(q["candidates"]),
            candidates=candidates,
        )
    return dict(
        schema="exact-reviewer-decision-recovery/1",
        mode="local_ranking",
        native_sha256=NATIVE_HASH,
        submission_sha256=SUBMISSION_HASH,
        worker_sha256=APPLY_HASH,
        cases=result,
        verified_artifacts=reader.inventory,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = recover(args.run, {int(x) for x in args.queries.split(",")})
    with args.output.open("x") as stream:
        json.dump(data, stream, allow_nan=False, indent=2)
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "cases": len(data["cases"]),
                "pairs": sum(len(c["candidates"]) for c in data["cases"].values()),
                "bytes": args.output.stat().st_size,
                "sha256": digest(args.output.read_bytes()),
            }
        )
    )
