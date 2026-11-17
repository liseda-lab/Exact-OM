# Fitted global acceptance

The acceptance stage converts a complete graph of scored candidate pairs into a
one-to-one global alignment. It uses the same per-edge features and selection
algorithms as the final Exact-2.0 target-competition experiments. Candidate
generation and pair scoring happen before this stage; it makes no LLM requests.

```console
exact acceptance fit --candidates scored.parquet --train train.tsv \
  --valid valid.tsv --output fitted --threads 4
exact acceptance apply --candidates scored.parquet --model fitted \
  --policy partial --output alignment-partial --threads 4
exact acceptance apply --candidates scored.parquet --model fitted \
  --policy full --output alignment-full --threads 4
```

Use positive public TRAIN pairs to fit and optionally public VALID pairs to
choose the partial policy's assignment method. Never pass TEST references as
either input. The apply command accepts no references. TRAIN and VALID must be
one-to-one and have disjoint source and target endpoints. Candidate membership
must be generated independently of TEST reference coverage.

## Input table

CSV, TSV, JSONL, and parquet are accepted. Preserve original candidate row order: it
breaks equal-score feature ranks. Use the complete population, including sources
without a reference, because target-relative features depend on all contenders.
Do not apply the model independently to shards: features and assignment require
the entire graph. The current reader holds the compact scored table and feature
matrix in memory; it does not load full evidence checkpoints or model weights
for pair scoring. Plan several GiB RAM for millions of edges.

Required identifiers are `Src` and `Tgt`, with these numeric columns:

```text
S_base S_pair_final U s_label S_struct s_hier s_sim s_diff s_attr
cand_sim cand_sim_prob cand_share_log_ratio repair_distinctive
original_exact_anchor src_obj_ic_mean tgt_obj_ic_mean
src_hier_total_count tgt_hier_total_count src_obj_count tgt_obj_count
src_attr_count tgt_attr_count
```

`repair_distinctive` is the historical name of the per-source distinctive-
evidence feature; this module performs no ontology repair. `S_pair_final` must
already contain the effective gated decision contribution. Primary ontology
endpoints and alignment exclusions must already be applied, with relative
retrieval features recomputed. Imported context must not become a mapping
endpoint merely because it appears in the context graph. Exact anchors are
features, not unconditional accepted mappings. Duplicate global pairs and
nonfinite features are rejected. This is not the local-ranking query interface.

Reference tables accept `Src`/`Tgt`, `SrcEntity`/`TgtEntity`, or `source`/`target`.
Missing retrieved positive pairs remain in threshold recall denominators.

## Training and policies

The default classifier is histogram gradient boosting with 80 iterations,
7 leaves, depth 3, learning rate 0.08 and L2 regularization 1. A logistic model
is available with `--family logistic`. The seed, five-fold membership, declared
threshold grid and source/configuration hashes are recorded. `--fold-membership`
can supply a JSON source-to-fold mapping for compatibility with an existing run.

Each held-out fold excludes all fitting edges touching either its TRAIN source
or target endpoints. Known-source candidates and strong reverse contenders are
reference-relative negatives. Unknown background is kept separate and receives
the explicit weak weight 0.02, adjusted for sampling probability. This is an
assumption about incomplete references, not confirmed negative biological truth.

Both thresholds are selected from TRAIN out-of-fold predictions. The partial
objective judges predictions whose source **or** target has reference coverage.
The full objective adds an inverse-probability background error surrogate. VALID,
when supplied, chooses among greedy, winner-only, and sparse assignment for the
partial policy; it does not refit model weights or choose a threshold.

- **Greedy:** accept descending scores, trying alternatives when a target is taken.
- **Winner:** consider only a source's best candidate; a target conflict leaves it unmatched.
- **Assignment:** maximize total log-odds benefit over threshold, with a private
  unmatched option and up to five alternatives per source.

The default new-data fit does not reproduce the historical benchmark's fallback
selection against an older model. In particular, the final NCIT–DOID full result
retained a different calibrated rank/winner-acceptance head. It must not be
represented as a new histogram model's result. All thresholds are fitted on the
user's data; the benchmark's task-specific thresholds are not universal defaults.

That retained numeric head has a separate compatibility route. Its pair
calibration, linear rank model, genuine-winner acceptance model, calibrated
pair floor, and target conflicts are applied without refitting:

```console
exact acceptance import-retained --head frozen-selector.json \
  --temperature 0.75 --output retained
exact acceptance apply-retained --candidates scored.parquet \
  --model retained --output alignment-retained
```

The temperature must match the fitted original selector. Import keeps numeric
coefficients and provenance hashes and removes original campaign paths. The
compatibility route does not offer a new nested-OOF fit for this older head;
use `fit` for the final edge classifier on new data.

## Frozen models and output

To apply a trusted previously fitted final edge model without refitting:

```console
exact acceptance import-model --model historical-model.joblib --trusted-model \
  --threshold 0.1 --mode assignment --policy partial --output imported
exact acceptance apply --candidates scored.parquet --model imported \
  --policy partial --output alignment
```

The input pickle must contain the qualified `(scaler, estimator)` tuple with
33 features. Only import models from trusted authors: joblib uses pickle, which
can execute code. Bundles record the exact scikit-learn version and verify the
model hash and feature schema before loading. Keep the bundle directory together
when moving it; it contains no server-specific dependencies or absolute paths.

Fit checkpoints are atomic per fold. Repeating the same fit resumes completed
models; changed inputs, source, or settings require a new output directory.
Application writes `mappings.csv`, `alignment.rdf`, and a hashed
`completion.json`. RDF is independently parsed back to verify mapping identity
and one-to-one cardinality. No original results or input tables are overwritten.
