# Design protocol, not executable runtime

`development.json` fixes sequencing, default bounds, case sizes and initial data identity. Its null model bindings and unresolved import/path/run/policy/case bindings are intentional design-time unknowns. B0 defines and validates stage-scoped execution locks: ontology/import/runtime bindings precede context indexing; actual run hashes precede run import; each model binding precedes its own dispatch. A future Exact run or challenger model does not block independent context preparation or provisional baseline generation. Pending stages cannot run with missing dependencies. No fake paths, profiles or results may fill them. Existing Exact/inspect commands do not accept this blueprint today.

`contract.schema.json` is Draft-07 JSON Schema for target shared primitives and three fixture envelopes. It is deliberately not a full route/OpenAPI specification; 01/04 require the backend to define and test all resource models and cross-resource invariants. The schema's `$id` is a namespace, not a downloadable dependency.

Schema-check seed fixtures with a standards-compliant JSON Schema validator. `validate_specs.py` checks schema validity, fixture shape, page counts/missingness, synthetic-output labeling, protocol consistency and local document links. It needs the development-only `jsonschema` Python package; it does not load Exact, contact a provider or run benchmarks.

From the repository root, with `jsonschema` installed in a development environment:

```sh
python specs/explanation-framework/protocol/validate_specs.py
```

Passing this check validates the design artifacts only. The B0–B5 runtime checks remain pending.


## Product and study amendment (2026-09-20)

`study-design.json` is the configurable two-condition mixed-case blueprint. The recommended
24-case and shorter 20-case presets are design examples; final participant workload, real
adjudication keys, consent text and analysis are locked before study publication. They do not
block implementing the service. `study.schema.json` adds participant-facing shared envelopes,
not a complete implemented survey API. `fixtures/study/` contains nine explicitly synthetic
examples and no invitation secret or answer key.

The researcher-only `study-schedules.synthetic.json` and `study-scoring-oracles.synthetic.json`
exercise assignment and scoring semantics; never mount them as participant resources. The
validator checks both JSON Schemas, entity identity consistency, participant fixture shapes,
eight balanced schedules, nine scoring examples and rejection of ambiguous ranking payloads.
It still does not test real database persistence, browser behavior or Render deployment.
