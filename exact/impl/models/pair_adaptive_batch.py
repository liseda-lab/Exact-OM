"""Bounded staged execution of the existing evidence channel computations.

Scientific use is opt-in for the reviewed staged recipe. Selection, evidence
budgets, precision and downstream grouping remain owned by the existing scorer.
"""

from collections import defaultdict
import os

import torch


BATCHING_RECIPE = "staged-32x64-mm-v1"
SCIENTIFIC_BATCHING_ENV = {
    "EXACT_PAIR_CONTEXT_BATCHING": "1",
    "EXACT_PAIR_CONTEXT_SCHEDULE": "staged",
    "EXACT_PAIR_CONTEXT_ENCODER": "batch",
    "EXACT_PAIR_CONTEXT_MATRIX_KERNEL": "mm",
    "EXACT_PAIR_CONTEXT_ARITHMETIC": "device",
    "EXACT_PAIR_CONTEXT_BLOCK_PAIRS": "32",
    "EXACT_PAIR_CONTEXT_TEXT_BATCH": "64",
    "EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY": "0",
    "EXACT_ENCODER_CUDA_GRAPHS": "0",
}


def execution_options():
    """Independent qualification switches; defaults retain the legacy numerics."""
    options = {}
    for name, default, allowed in (
        ("SCHEDULE", "legacy", {"legacy", "staged"}),
        ("ENCODER", "legacy", {"legacy", "batch"}),
        ("MATRIX_KERNEL", "mm", {"mm", "bmm"}),
        ("ARITHMETIC", "device", {"device", "cpu"}),
    ):
        value = os.getenv("EXACT_PAIR_CONTEXT_" + name, default)
        if value not in allowed:
            raise ValueError(f"Invalid EXACT_PAIR_CONTEXT_{name}: {value!r}")
        options[name.lower()] = value
    return options


def validate_batching_admission():
    """Admit the fixed reviewed recipe; keep other variants in qualification."""
    from exact.impl.models.encoder_graph import graph_enabled

    graph_enabled()
    stable_identity = os.getenv("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", "0")
    if stable_identity not in {"0", "1"}:
        raise ValueError("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY must be 0 or 1")
    recipe = os.getenv("EXACT_PAIR_CONTEXT_RECIPE")
    namespace = os.getenv("EXACT_PRIMITIVE_NAMESPACE", "")
    if recipe is not None:
        if recipe != BATCHING_RECIPE:
            raise ValueError(f"Unknown EXACT_PAIR_CONTEXT_RECIPE: {recipe!r}")
        mismatches = [
            name for name, value in SCIENTIFIC_BATCHING_ENV.items() if os.getenv(name) != value
        ]
        if mismatches:
            raise ValueError(f"Batching recipe {recipe} requires its fixed settings: {mismatches}")
        prefix = f"scientific-{recipe}-"
        if (
            os.getenv("EXACT_EXPERIMENT_ROLE") != "development"
            or not namespace.startswith(prefix)
            or namespace == prefix
        ):
            raise RuntimeError(
                f"Batching recipe {recipe} is only admitted for development runs "
                f"in an isolated {prefix}* namespace."
            )
        return
    if os.getenv("EXACT_PAIR_CONTEXT_BATCHING", "0") != "1" and stable_identity == "0":
        return
    execution_options()
    if (
        os.getenv("EXACT_EXPERIMENT_ROLE") != "throughput_fixture"
        or not namespace.startswith("qualification-")
        or namespace == "qualification-"
    ):
        raise RuntimeError(
            "Experimental encoder/batching execution is only admitted for throughput_fixture runs "
            "in an isolated qualification-* namespace or the explicitly reviewed development recipe."
        )


def scientific_execution_payload():
    """Bind checkpoints to the admitted execution without changing legacy identities."""
    if os.getenv("EXACT_PAIR_CONTEXT_RECIPE") is None:
        return None
    validate_batching_admission()
    return {
        "recipe": BATCHING_RECIPE,
        "namespace": os.environ["EXACT_PRIMITIVE_NAMESPACE"],
        "settings": dict(SCIENTIFIC_BATCHING_ENV),
    }


def score_evidence_job(scorer, name, job):
    validate_batching_admission()
    prepared = getattr(scorer, "_prepared_evidence_result", None)
    if prepared is not None and prepared[0] == name:
        job.close()
        return prepared[1]
    return run_evidence_jobs(
        scorer, [job], batched=os.getenv("EXACT_PAIR_CONTEXT_BATCHING", "0") == "1"
    )[0]


def call_prepared(scorer, name, method, payload, *args, cache_hit=False, **kwargs):
    """Keep existing numerical-cache decorators around complete ordinary records."""
    if cache_hit:
        # peek_channel has already validated and restored this exact payload.
        return payload
    if payload is None:
        return method(*args, **kwargs)
    previous = getattr(scorer, "_prepared_evidence_result", None)
    scorer._prepared_evidence_result = (name, payload)
    try:
        return method(*args, **kwargs)
    finally:
        scorer._prepared_evidence_result = previous


def _encode_unique(scorer, texts, kind, *, normalize=True):
    values = list(dict.fromkeys(texts))
    encode = scorer.encode_labels_batch if kind == "label_matrix" else scorer.encode_contexts_batch
    width = max(1, int(os.getenv("EXACT_PAIR_CONTEXT_TEXT_BATCH", "64")))
    pieces = []
    start = 0
    while start < len(values):
        try:
            vectors = encode(values[start : start + width])
        except torch.cuda.OutOfMemoryError:
            if width == 1:
                raise
            width = max(1, width // 2)
        else:
            pieces.append(vectors)
            start += width
    encoded = torch.cat(pieces, dim=0)
    if normalize:
        encoded = torch.nn.functional.normalize(encoded, dim=-1)
    return {text: encoded[index] for index, text in enumerate(values)}


def _batch_matrices(scorer, requests, kind):
    """Keep encoding, matrix kernels and reduction placement independently testable."""
    options = execution_options()
    output = [None] * len(requests)
    live = [
        (index, request)
        for index, request in enumerate(requests)
        if request[1] and request[2] and (kind != "context_matrix" or scorer.use_context)
    ]
    for index, request in enumerate(requests):
        if (
            not request[1]
            or not request[2]
            or (kind == "context_matrix" and not scorer.use_context)
        ):
            output[index] = torch.zeros((len(request[1]), len(request[2])), device=scorer.device)
    if not live:
        return output
    encode = scorer.encode_labels_batch if kind == "label_matrix" else scorer.encode_contexts_batch
    vectors = None
    if options["encoder"] == "batch":
        vectors = _encode_unique(
            scorer,
            (text for _, req in live for side in req[1:3] for text in side),
            kind,
            normalize=False,
        )
    encoded = {}
    for index, request in live:
        # Normalize the original side shape even when encoder calls are regrouped.
        # Legacy mode must also preserve left/right cache-miss call boundaries.
        encoded[index] = tuple(
            torch.nn.functional.normalize(
                (
                    encode(list(side))
                    if vectors is None
                    else torch.stack([vectors[text] for text in side])
                ),
                dim=-1,
            )
            for side in request[1:3]
        )
    if options["matrix_kernel"] == "mm":
        for index, _ in live:
            left, right = encoded[index]
            output[index] = scorer._sim01(left @ right.T)
        if options["arithmetic"] == "cpu":
            output = [matrix.detach().cpu() for matrix in output]
        return output
    buckets = defaultdict(list)
    for index, req in live:
        buckets[(len(req[1]), len(req[2]))].append((index, req))
    max_elements = max(1, int(os.getenv("EXACT_PAIR_CONTEXT_MATRIX_ELEMENTS", "1048576")))
    for (rows, cols), bucket in buckets.items():
        width = max(1, min(64, max_elements // (rows * cols)))
        for start in range(0, len(bucket), width):
            block = bucket[start : start + width]
            left = torch.stack([encoded[index][0] for index, _ in block])
            right = torch.stack([encoded[index][1] for index, _ in block])
            matrices = scorer._sim01(torch.bmm(left, right.transpose(1, 2)))
            if options["arithmetic"] == "cpu":
                matrices = matrices.detach().cpu()
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
    vectors = None
    if execution_options()["encoder"] == "batch":
        vectors = _encode_unique(
            scorer,
            (text for _, left, right in pairs for text in (left, right)),
            "similarity",
            normalize=False,
        )
    scores = []
    for _, source, target in pairs:
        enc = torch.nn.functional.normalize(
            (
                scorer.encode_contexts_batch([source, target])
                if vectors is None
                else torch.stack([vectors[source], vectors[target]])
            ),
            dim=-1,
        )
        # Retain the exact legacy reduction shape; pack only the final transfer.
        scores.append(scorer._sim01(torch.sum(enc[0] * enc[1], dim=-1)))
    scores = torch.stack(scores).detach().cpu().tolist()
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
                    methods = {
                        "label_matrix": scorer._encode_label_matrix,
                        "context_matrix": scorer._encode_context_matrix,
                        "similarity": scorer._context_similarity_from_sentences,
                    }
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
    validate_batching_admission()
    try:
        return _prepare_structural_block_once(scorer, pairs, families)
    except torch.cuda.OutOfMemoryError:
        if len(pairs) < 2:
            raise
    midpoint = len(pairs) // 2
    return prepare_structural_block(scorer, pairs[:midpoint], families) + prepare_structural_block(
        scorer, pairs[midpoint:], families
    )


def _prepare_structural_block_once(scorer, pairs, families):
    """Prepare one bounded internal tile; group-dependent fusion stays outside it."""
    from exact.experiments.numerical_cache import peek_channel

    jobs, keys = [], []
    rows = [{"hierarchy": {}, "cache_hits": set()} for _ in pairs]
    for index, (source, target, src, tgt) in enumerate(pairs):
        for family in families:
            args = (
                family,
                src.get("hierarchy", {}).get(family, []),
                tgt.get("hierarchy", {}).get(family, []),
            )
            cached = peek_channel(
                scorer._score_hierarchy_family, *args, src_iri=source, tgt_iri=target
            )
            if cached is not None:
                rows[index]["hierarchy"][family] = cached
                rows[index]["cache_hits"].add(("hierarchy", family))
            else:
                jobs.append(scorer._hierarchy_family_job(*args, source, target))
                keys.append((index, family))
        args = (src.get("object_triples", []), tgt.get("object_triples", []))
        cached = peek_channel(scorer._score_similarity_channel, *args)
        if cached is not None:
            rows[index]["similarity"] = cached
            rows[index]["cache_hits"].add(("similarity", None))
        else:
            jobs.append(scorer._similarity_channel_job(*args, staged=True))
            keys.append((index, None))
    for (index, family), payload in zip(keys, run_evidence_jobs(scorer, jobs, batched=True)):
        if family is None:
            rows[index]["similarity"] = payload
        else:
            rows[index]["hierarchy"][family] = payload
    jobs, indices = [], []
    for index, (_, _, src, tgt) in enumerate(pairs):
        args = (
            src.get("attributes", []),
            tgt.get("attributes", []),
            src.get("labels", []),
            tgt.get("labels", []),
            rows[index]["hierarchy"],
            rows[index]["similarity"],
        )
        cached = peek_channel(scorer._score_attribute_channel, *args)
        if cached is not None:
            rows[index]["attributes"] = cached
            rows[index]["cache_hits"].add(("attributes", None))
        else:
            jobs.append(scorer._attribute_channel_job(*args))
            indices.append(index)
    for index, payload in zip(indices, run_evidence_jobs(scorer, jobs, batched=True)):
        rows[index]["attributes"] = payload
    return rows
