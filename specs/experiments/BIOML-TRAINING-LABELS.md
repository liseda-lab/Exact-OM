# BioML local training labels

The official local-ranking contract supports benchmark negative labels **only for the supplied training distractors**. It does not establish ontology-wide reference completeness or natural NIL.

The [official task definition](https://bio-ml.oaei-ml.org/tasks/local/) describes the pool as a gold target plus sampled hard negatives and explicitly permits the gold-bearing train/validation pools for ranker supervision. The inspected [contract source](https://github.com/liseda-lab/OAEI-Bio-ML/blob/ec436a97f49875227dedf634faa5280b1376bda9/tasks/local/ranking_task_index.md) is pinned at `ec436a97f49875227dedf634faa5280b1376bda9`; its SHA256 is recorded by the converter.

The [dataset README at the installed revision](https://huggingface.co/datasets/OAEI-ML/bio-ml/blob/c454644a334ab43754bc1070c0fb7fdd56a90a1d/README.md) distinguishes local ranking’s standard, unrepaired reference from the repaired global reference. Therefore the converter writes a matching local training reference from the public train-pool gold. A local training positive absent from the repaired global train file remains positive through its explicit `confirmed_label=1`.

The [historical generation script](https://github.com/liseda-lab/OAEI-Bio-ML/blob/ec436a97f49875227dedf634faa5280b1376bda9/data_scripts/generate_cand_maps.py) passes the full reference to DeepOnto’s negative sampler. [DeepOnto’s documentation](https://krr-oxford.github.io/DeepOnto/bio-ml/#candidate-mapping-generation) states that known reference mappings are excluded from negatives. This supports the intended sampling semantics; it is not a claim that the historical script reproduces the installed 2026 release byte for byte.

`prepare_confirmed_bioml_training` unions all public train gold alternatives for a source before assigning `0` to the remaining provided candidates. Unlisted pairs remain unknown. It rejects train/reporting source overlap and gold absent from its pool. Each output records the original input hash, official dataset and contract revisions, label counts and `negative_scope=provided_training_candidate_pairs_only`. Case bindings retain `reference_completeness=known_incomplete` and use `negative_policy=confirmed_only`.

Recreate the machine’s base bindings first if needed, then run this committed converter:

```sh
.venv/bin/python data/experiments-v2/make-local-bindings.py
.venv/bin/python tools/prepare_bioml_training.py \
  --bindings data/experiments-v2/local-bindings.yaml \
  --dataset-root data/experiments-v2/bioml-primary \
  --dataset-revision c454644a334ab43754bc1070c0fb7fdd56a90a1d
```

The second command verifies the installed input lock and changes only training inputs and their declared negative policy/provenance for D0/D1 and the designated H0/H1/H2 training pools. It reads frozen reporting **source IDs** solely to reject overlap. It neither reads reporting gold nor trains a model. H2 training remains an execution-time G4 dependency; preparing its public designated train labels does not run that fit.
