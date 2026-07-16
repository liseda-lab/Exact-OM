# Evaluation

Run evaluation with an alignment (`--run_eval`) or invoke `exact-eval` on an existing mapping.
The historical `bioml-eval` command remains an alias.

```console
exact-eval \
  --alignment_file runs/demo/alignment/maps_global.tsv \
  --output_dir runs/demo/evaluation \
  --full_reference_file reference.tsv \
  --source_ontology_file source.owl \
  --target_ontology_file target.owl
```

The builtin backend always runs locally. The optional `bioml` backend requires
`exact-om[bioml-eval]` and invokes the pinned upstream evaluator through its Python API rather
than shelling out or mutating `sys.path`.

## Metrics

| Metric | Interpretation |
| --- | --- |
| Precision | Fraction of predicted mappings present in the reference. |
| Recall | Fraction of reference mappings recovered. |
| F1 | Harmonic mean of precision and recall. |
| MRR | Mean reciprocal rank of the reference target in each local list. |
| Hits@K | Fraction of queries whose reference target appears in the first K candidates. |
| Candidate-oracle recall | Reference fraction present anywhere in the evaluated candidate pool. |

Training mappings are removed from both predictions and the scored reference where they act as
track null mappings. For multi-kind runs, references outside `matching.entity_kinds` are
reported and excluded.

## Reports and timing

Backend reports live in `evaluation/`; summary metrics and evaluator provenance are merged into
`stats/run_stats.json`. Evaluation has its own `Postprocess.Evaluation` timing stage. A skipped
optional backend is not counted as compute, and a resumed run does not attribute restored work
to the current session.

Use `tools/analyze_alignment_run.py` to split false negatives into candidate-absent,
wrong-selected, abstained, and rejected buckets. It reads either run layout through
`RunReader`.
