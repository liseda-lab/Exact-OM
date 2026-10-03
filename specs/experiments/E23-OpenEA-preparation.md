# E23 natural-KG replacement: OpenEA v2.0

This approved development-only case replaces the otherwise inapplicable **E23 natural-KG
comparison only**. K0/StarWars and other experiment families remain unchanged. Preserve the
existing `natural_graph_off`, `natural_inductive`, `graph_only`, and `graph_shuffled` arms.

Use the official OpenEA **v2.0** `D_W_15K_V1/721_5fold/1` split. Its official public archive is
[Figshare article version 3](https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760/3),
file 34234391, SHA256 `37adcaf4a7ed33530a39b637da7329e891456bfc89557d8b1ac307c67eb8f5bc`.
The upstream [OpenEA documentation](https://github.com/nju-websoft/OpenEA) recommends this
version because encoded entity URIs remove a name-bias shortcut. Do not decode those IDs.

```bash
python tools/prepare_openea.py --cache /tmp/exact-openea \
  --output data/experiments-v2/openea-v2-D_W_15K_V1-fold1
```

The full archive stays node-local. Only four graph/attribute files and fold 1's `train_links`
and `valid_links` are extracted. `ent_links`, `test_links` and other folds are never opened.
Public triples become separate CSV-KG sources with individual entity kinds; attributes remain
literals. Empty literal values (five in the pinned DBpedia file) are omitted from the derived
view and counted; original bytes are retained. Only explicit name/label attributes supply labels. Alignment links are never graph
edges. The preparation checks graph endpoint coverage, 15,000 entities per side, 3,000 training
links, 1,500 validation links, bijection within each split and disjoint train/validation endpoints.

`bindings-fragment.json` introduces **K0_OpenEA**, with train/validation references and its own
source identity. `train.sources.txt` selects 2,000 official training sources using the existing
nested seed-17 selection. All official validation sources remain available; the screen uses
the existing 300-source cap. `training-label-scope.json` pins the official training links.

Generate both candidate pools with the existing frozen retrieval recipe on this case. Do not
inherit StarWars candidate files, insert gold targets into validation pools, or add a different
retriever. After generating the training pool, call
`label_training_candidates(frame, train_links, expected_sha256=...)` before freezing it.
The helper retains every retrieved row and score. It assigns 1 to official training matches,
0 only to cross-pairs between distinct official training matches, and unknown elsewhere.
The existing `confirmed_negatives` fitter excludes unknown pairs. These negatives follow the
benchmark's bijective alignment contract; they are not independently verified real-world
non-equivalence, ontology-wide NIL, or a completeness claim for arbitrary KG entities.

The four arms must share the resulting candidate-pool identities, sources and training scope.
This new individual case cannot reuse class/StarWars fitted-head identities. No preparation
result is empirical readiness or evidence of model quality. `preparation.json` records the
exact accessed archive members and hashes every output; repeat preparation verifies them.

The own-pool preparation is a separate queued CPU/GPU preparation step, without scoring or
head fitting:

```bash
python tools/prepare_openea_pools.py \
  --config data/experiments-v2/locks/R_v2.config.yaml \
  --prepared data/experiments-v2/openea-v2-D_W_15K_V1-fold1-02 \
  --output data/experiments-v2/openea-v2-D_W_15K_V1-fold1-pools-01 \
  --device cuda:0
```

Use metadata revision 2, whose source sampling explicitly uses the `individual` kind; the
existing class sampler retains its default. This tool calls the normal dataset ontology
loader, source-universe freezer and candidate generator with the supplied configuration.
It passes no alignment reference to retrieval. Pool receipts pin that recipe, its implementation,
public KG hashes, selected sources, and resulting rows; completed roles resume without another
encoder pass. The approved R_v2 baseline uses its pinned MiniLM candidate retriever with k=20.

Merge the resulting `bindings-fragment.json` into the campaign's explicit bindings before
calling `prepare_campaign`. The `e23_natural_case` override changes E23 alone. Its own pools
replace StarWars dependencies and policy inheritance; the existing rich class ablations and
E12 keep their original bindings. Metadata-only preparation remains blocked until pool receipts
exist. All four natural-KG arms share one train pool and one validation pool, with caps and seed
checked against their receipt. Retrieval controls already applied during pool generation are
disabled during provided-pool loading, so they cannot rerank or expand the frozen rows a second
time. This preparation does not establish empirical readiness or authorize using test labels.

For an automatic immutable campaign handoff, add both `--campaign-bindings <original.yaml>`
and `--campaign-output <new-directory>` to the pool command. After both roles finish, the CLI
writes `campaign-bindings.json = deep_merge(original_bindings, own_pool_fragment)` and calls
`prepare_campaign` with the same base config. The supplied original bindings must not override
E23 back to K0 or reinstate its former pool inheritance; these conflicts fail explicitly.

Before any validation retrieval or graph fitting, the train-pool receipt checks supervision
support using the graph learner's unchanged seed-shuffled source-group folds (up to three).
Each training and held-out partition must contain positives and permitted negatives. If a
partition is unusable, the original retrieved rows and failure counts remain saved, preparation
stops before producing executable bindings, and retries reuse that receipt. This requires an
explicit scientific decision; the tool never changes seeds, injects missing gold targets, or
widens the negative-label scope to force a fit. The support report does not measure accuracy.
