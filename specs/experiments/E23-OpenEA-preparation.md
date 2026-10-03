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
