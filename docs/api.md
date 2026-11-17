# Python API

The existing high-level API remains available:

```python
from exact import AlignmentRunner

runner = AlignmentRunner(
    source_ontology_file="source.owl",
    target_ontology_file="target.owl",
    output_dir="runs/alignment",
    config_file="config.yaml",
)
runner.run()
```

Use `help(AlignmentRunner)` for the supported constructor fields. Command-line
entry points return integer process statuses; library/helper return contracts are unchanged.

The staged workflow also exposes callable Python modules:
`exact.native.score.score`, `exact.decisions`,
`exact.acceptance.model`, and `exact.runtime.local_query_ranker`.
Their arguments mirror the CLI's explicit files, model bundles and policies.
Prefer the CLI for reproducible job scripts: it records the same artifacts and
avoids depending on internal scorer methods.

The existing registry supports custom datasets, models, trainers and
evaluators. Extensions must preserve candidate membership, reference isolation,
and explanation/score consistency. A custom backend is responsible for its own
dependency and model provenance.
