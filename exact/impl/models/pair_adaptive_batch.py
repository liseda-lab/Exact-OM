"""Bounded staged execution of the existing evidence channel computations.

Opt-in until real-device discrete parity is admitted. Selection, evidence budgets,
precision and downstream candidate grouping remain owned by the existing scorer.
"""
from collections import defaultdict
import os

import torch


def score_evidence_job(scorer, name, job):
    prepared = getattr(scorer, "_prepared_evidence_result", None)
    if prepared is not None and prepared[0] == name:
        job.close()
        return prepared[1]
    return run_evidence_jobs(scorer, [job], batched=False)[0]


def call_prepared(scorer, name, method, payload, *args, **kwargs):
    """Keep existing numerical-cache decorators around complete ordinary records."""
    if payload is None:
        return method(*args, **kwargs)
    previous = getattr(scorer, "_prepared_evidence_result", None)
    scorer._prepared_evidence_result = (name, payload)
    try:
        return method(*args, **kwargs)
    finally:
        scorer._prepared_evidence_result = previous


def _encode_unique(scorer, texts, kind):
    values = list(dict.fromkeys(texts))
    encode = scorer.encode_labels_batch if kind == "label_matrix" else scorer.encode_contexts_batch
    width = max(1, int(os.getenv("EXACT_PAIR_CONTEXT_TEXT_BATCH", "64")))
    pieces = []
    start = 0
    while start < len(values):
        try:
            vectors = encode(values[start:start + width])
        except torch.cuda.OutOfMemoryError:
            if width == 1:
                raise
            width = max(1, width // 2)
        else:
            pieces.append(vectors)
            start += width
    encoded = torch.nn.functional.normalize(torch.cat(pieces, dim=0), dim=-1)
    return {text: encoded[index] for index, text in enumerate(values)}


def _batch_matrices(scorer, requests, kind):
    """Bucket by exact shape, without padding or unrelated pair cross-products."""
    output = [None] * len(requests)
    live = [(index, request) for index, request in enumerate(requests)
            if request[1] and request[2] and (kind != "context_matrix" or scorer.use_context)]
    for index, request in enumerate(requests):
        if not request[1] or not request[2] or (kind == "context_matrix" and not scorer.use_context):
            output[index] = torch.zeros((len(request[1]), len(request[2])))
    if not live:
        return output
    vectors = _encode_unique(scorer, (text for _, req in live for side in req[1:3] for text in side), kind)
    buckets = defaultdict(list)
    for index, req in live:
        buckets[(len(req[1]), len(req[2]))].append((index, req))
    max_elements = max(1, int(os.getenv("EXACT_PAIR_CONTEXT_MATRIX_ELEMENTS", "1048576")))
    for (rows, cols), bucket in buckets.items():
        width = max(1, min(64, max_elements // (rows * cols)))
        for start in range(0, len(bucket), width):
            block = bucket[start:start + width]
            left = torch.stack([torch.stack([vectors[t] for t in req[1]]) for _, req in block])
            right = torch.stack([torch.stack([vectors[t] for t in req[2]]) for _, req in block])
            matrices = scorer._sim01(torch.bmm(left, right.transpose(1, 2))).detach().cpu()
            for (index, _), matrix in zip(block, matrices):
                output[index] = matrix
    return output


def _batch_context_similarity(scorer, requests):
    output = [scorer.tau] * len(requests)
    pairs = []
    for index, (_, source, target, budget) in enumerate(requests):
        if not scorer.use_context or not source or not target:
            continue
        left = scorer._join_channel_sentences(source, budget)
        right = scorer._join_channel_sentences(target, budget)
        if left and right:
            pairs.append((index, left, right))
    if not pairs:
        return output
    vectors = _encode_unique(scorer, (text for _, left, right in pairs for text in (left, right)), "similarity")
    left = torch.stack([vectors[source] for _, source, _ in pairs])
    right = torch.stack([vectors[target] for _, _, target in pairs])
    scores = scorer._sim01(torch.sum(left * right, dim=-1)).detach().cpu().tolist()
    for (index, _, _), score in zip(pairs, scores):
        output[index] = float(score)
    return output


def run_evidence_jobs(scorer, jobs, *, batched):
    """Advance existing channel bodies only after their dependencies resolve."""
    results = [None] * len(jobs)
    pending = [(index, job, None) for index, job in enumerate(jobs)]
    try:
        while pending:
            waiting = defaultdict(list)
            for index, job, value in pending:
                try:
                    request = job.send(value)
                except StopIteration as complete:
                    results[index] = complete.value
                else:
                    waiting[request[0]].append((index, job, request))
            pending = []
            for kind, group in waiting.items():
                requests = [request for _, _, request in group]
                if not batched:
                    methods = {"label_matrix": scorer._encode_label_matrix,
                               "context_matrix": scorer._encode_context_matrix,
                               "similarity": scorer._context_similarity_from_sentences}
                    values = [methods[kind](*request[1:]) for request in requests]
                elif kind == "similarity":
                    values = _batch_context_similarity(scorer, requests)
                else:
                    values = _batch_matrices(scorer, requests, kind)
                pending.extend((index, job, value) for (index, job, _), value in zip(group, values))
        return results
    finally:
        for job in jobs:
            job.close()


def prepare_structural_block(scorer, pairs, families):
    try:
        return _prepare_structural_block_once(scorer, pairs, families)
    except torch.cuda.OutOfMemoryError:
        if len(pairs) < 2:
            raise
    midpoint = len(pairs) // 2
    return (prepare_structural_block(scorer, pairs[:midpoint], families)
            + prepare_structural_block(scorer, pairs[midpoint:], families))


def _prepare_structural_block_once(scorer, pairs, families):
    """Prepare one bounded internal tile; group-dependent fusion stays outside it."""
    jobs, keys = [], []
    rows = [{"hierarchy": {}} for _ in pairs]
    for index, (source, target, src, tgt) in enumerate(pairs):
        for family in families:
            jobs.append(scorer._hierarchy_family_job(
                family, src.get("hierarchy", {}).get(family, []),
                tgt.get("hierarchy", {}).get(family, []), source, target))
            keys.append((index, family))
        jobs.append(scorer._similarity_channel_job(src.get("object_triples", []), tgt.get("object_triples", []), staged=True))
        keys.append((index, None))
    for (index, family), payload in zip(keys, run_evidence_jobs(scorer, jobs, batched=True)):
        if family is None:
            rows[index]["similarity"] = payload
        else:
            rows[index]["hierarchy"][family] = payload
    jobs = [scorer._attribute_channel_job(
                src.get("attributes", []), tgt.get("attributes", []),
                src.get("labels", []), tgt.get("labels", []),
                rows[index]["hierarchy"], rows[index]["similarity"])
            for index, (_, _, src, tgt) in enumerate(pairs)]
    for row, payload in zip(rows, run_evidence_jobs(scorer, jobs, batched=True)):
        row["attributes"] = payload
    return rows
