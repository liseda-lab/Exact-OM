# Staged matching workflow

The stages share immutable pair evidence. Neither native scoring nor decision
routing opens a reference alignment. Fitting explicitly accepts TRAIN data;
validation may qualify an operating policy. Keep TEST labels outside fitting.

## Global matching

```console
exact config profile native -o native.yaml
exact score -s source.owl -t target.owl -y native.yaml -o native \
  --mode global --source-prefix https://example.org/source/ \
  --target-prefix https://example.org/target/ --device cuda:0
exact decisions prepare --candidates native/pairs.jsonl \
  --sources native/sources.txt --output decisions --policy global
exact decisions run --plan decisions/plan.json --cache elephant.sqlite \
  --model-dir /models/elephant-1
exact decisions apply --candidates native/pairs.jsonl \
  --plan decisions/plan.json --cache elephant.sqlite --output scored.jsonl
exact acceptance fit --candidates scored.jsonl --train train.tsv \
  --valid valid.tsv --output acceptance
exact acceptance apply --candidates scored.jsonl --model acceptance \
  --policy full --output full-alignment
exact acceptance apply --candidates scored.jsonl --model acceptance \
  --policy partial --output partial-alignment
```

Check `exact acceptance fit --help` for background sampling and fold controls.
Fitting evaluates a fixed, recorded threshold grid. The example uses the generic final edge-classifier recipe;
it does not rerun the historical model-family tournament.

The native profile retrieves at most 20 candidates with the frozen hybrid
lexical/MiniLM recipe and scores them with SapBERT/BGE evidence. For global
matching, provide owning IRI prefixes independently of reference coverage.
The scoped adapter supports primary RDF/XML named classes, honors deprecated
and alignment-exclusion annotations, and retains imported ontology context.
It generates the original pool, scores that pool, then filters endpoints and
recomputes relative features. Removed candidates are not backfilled. A local
candidate pool cannot be supplied to global scoring.

The default corrected global gate admits the most ambiguous 20% of the full
eligible source universe and at most five pairs per admitted source. Elephant's
probability is mixed with weight `clip(0.8 * U, 0, 1)`; this is not a constant
50% mixture. No summary or rationale model is called by these stages. See the
[decision guide](decisions.md) for the historical NCIT–DOID gate exception.

To run without pair decisions, fit/apply acceptance directly to native
`pairs.jsonl`. This is an explicit no-LLM configuration, not a silent fallback
for missing decision responses.

## Local ranking

```console
exact config profile local -o local.yaml
exact score -s source.owl -t target.owl -y local.yaml -o train-native \
  --mode local --candidates train.cands.tsv --device cuda:0
exact score -s source.owl -t target.owl -y local.yaml -o test-native \
  --mode local --candidates test.cands.tsv --device cuda:0
```

Use `local-train` decision preparation for TRAIN and `local-application` for
the reporting pool, then fit/apply `exact rank` as documented in
[local ranking](ranking.md). The published local TRAIN route consults all
eligible TRAIN sources; reporting uses 20%. This training cost is distinct from
deployment cost. The portable local profile scores TRAIN anchors explicitly so
the fitted feature population remains complete. Supplied application anchor
sidecars remain explicit hard bypasses.

Local query identities and occurrences remain separate even when the same
source appears repeatedly. The shared native scoring table stores each pair
once; the ranker reconstructs every original query and writes a BioML
`SrcEntity / TgtCandidates` TSV permutation. It never performs global
acceptance or target-conflict resolution.

## Scope and scaling

The staged global path is the finalized class-matching recipe. The existing
`exact align` path also supports configured property, individual, RDF and
CSV-KG inputs; these capabilities are not claims of completed benchmark
qualification on other tracks.

Native scoring checkpoints at source-shard boundaries. Decision inference
commits each local response; fitting checkpoints each fold. Repeating identical
commands resumes compatible work. Changed inputs, source code, models or
execution identities require new output directories. Final acceptance currently
loads the scored edge table into memory; size RAM for that table and its
33-feature matrix, and use one acceptance process per large task. Native and
decision stages stream their outputs.

Frozen historical models can be imported with the explicit compatibility
commands in [acceptance](acceptance.md). Model files, licensed ontologies and
private benchmark references are not shipped in the package.
