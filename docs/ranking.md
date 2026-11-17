# Local query ranking

`exact rank` fits and applies the final pairwise local-query head. Each supplied
query has its own competition. Queries sharing a source are preserved separately;
their candidates are never merged into a global inference task. The head does
not perform acceptance, NIL classification, or global target conflict removal.

```console
exact rank fit --candidates train-scored.jsonl --queries local.train.cands.tsv \
  --config local-ranking.yaml --model local-head.json --confirmed-negatives
exact rank apply --candidates test-scored.jsonl --queries local.test.cands.tsv \
  --model local-head.json --output submission.tsv
```

The configuration must declare `CandidateSetSelector` after the pair scorer,
`selector.rerank.mode: pairwise`, `selector.rerank.model: current_linear`,
`matching.calibration.mode: none`, and `matching.nil.mode: off`. Secondary
selector LLM decisions must be disabled. The decision probabilities, if used,
must already be mixed into the scored table.

The final benchmark's local model used five source-grouped folds and all
available public TRAIN queries. It preserves query-specific competition and
within-query reciprocity, with a linear pairwise rank model. The model artifact
records its feature contract, folds, original query occurrences, and input/source
hashes. `--fold-membership folds.json` accepts an explicit source-to-fold map.

## Queries and labels

Use original BioML TSV files in their released row order. TRAIN has
`SrcEntity`, `TgtEntity` (the known positive), and `TgtCandidates` (a Python list
of target IRIs). Application needs `SrcEntity` and `TgtCandidates`. The CLI assigns
the original row index as the query identifier. Application does not read a
`TgtEntity` label even if that column is present.

Alternatively, provide a JSON list:

```json
[
  {"qid": 0, "source": "urn:source:1", "candidates": ["urn:target:a", "urn:target:b"]},
  {"qid": 1, "source": "urn:source:1", "candidates": ["urn:target:b", "urn:target:c"]}
]
```

TRAIN JSON additionally requires `gold_target` in each original pool. Query IDs
must be unique nonnegative integers; repeated sources are allowed. Target members
within a query must be unique. All TRAIN sources must be disjoint from reporting
sources. `--confirmed-negatives` explicitly asserts that the benchmark labels
the other TRAIN candidates as negatives. Do not use that option to turn an
incomplete positive reference into negative labels.

## Scored evidence and exact anchors

The scorer table is JSONL or parquet with unique `Src`/`Tgt` pairs, effective
`S_pair_final` (or `S_final`), native channel scores and
`selector_evidence_items`. Do not omit compact evidence or substitute fabricated
values. Metadata present during fitting must also be present at application.
One scored row may be reused in several original queries; outputs retain every
query occurrence. Extra or missing pairs outside the original support are errors.

All TRAIN candidate pairs require real scored features, including exact matches;
score TRAIN with `dataset.filter_exact_matches: false`. Application may preserve
the original native hard-anchor sidecar:

```console
exact rank apply --candidates test-scored.jsonl --queries local.test.cands.tsv \
  --anchors native-anchors.json --model local-head.json --output submission.tsv
```

The sidecar is a JSON list of `Src`, `Tgt`, `S_final: 1.0`. These are existing
native exact matches, not inferred labels. They bypass the learned local head
with score one; other candidates still receive a complete ranking. Anchors must
belong to the original query pool, and scored rows plus anchors must cover it
exactly. Global acceptance uses a different anchor policy.

## Outputs and restart

The official BioML 2026 LIST submission format contains `SrcEntity` and
`TgtCandidates`, where each value is a plain ordered list of IRIs, best first.
It is not a list of `(IRI, score)` tuples. The released evaluator matches queries
by position; the CLI verifies query order and exact candidate permutations after
writing. A `.scores.json` sidecar retains probabilities, utilities, anchors and
query IDs; `.receipt.json` records hashes and counts.

Compatible fitting resumes completed source-fold checkpoints. Changed inputs,
configuration, or implementation require a fresh model path. Finished submissions
cannot be overwritten with different rankings. Previously frozen research heads
retain their original source hashes; they are not silently relabelled as models
fitted by the cleaned release. New users should fit the portable release model
on their TRAIN data.
