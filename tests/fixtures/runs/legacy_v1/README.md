# Pre-WP-L run golden

This fixture is a layout-v1 run produced with the pre-WP-L artifact shape:

- a monolithic `full_explanations.json` consumer artifact;
- independent audit, candidate, and final-overlay JSONL streams;
- a schema-v2 compact inference checkpoint pointing at those streams; and
- deliverable/stat files under the legacy nested alignment directory.

The payload is deliberately plain JSONL (the legacy manifest's supported
`compression: none` mode) so changes remain reviewable and deterministic.
`tests/run_artifacts_acceptance_test.py` opens it through `RunReader`, restores
the checkpoint through `SemanticAlignmentRunner`, and migrates the three
streams into the layout-v2 explanation store.
