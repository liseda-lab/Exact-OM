# Migration from Exact v1

The `exact align` flags and high-level Python API remain available. The bare
legacy form `exact -s ... -t ... -o ...` still dispatches to alignment.
Successful CLI invocations now return process status zero rather than passing
Python result objects to the console wrapper. Library result values are unchanged.

Use `exact config migrate old.yaml --output v2.yaml` for unversioned
configuration. Native OWL processing no longer needs a JVM. Retained hidden
JVM flags warn and have no effect.

The final Exact-2.0 recipe is the explicit score → decisions → acceptance/rank
workflow. Existing `align` does not silently replace a user's pipeline or fit
a different acceptance model. Start from the bundled profiles and read
[the workflow](docs/pipeline.md).

Old fitted artifacts bind their implementation and input schemas. Do not
rename paths or edit checksums to make incompatible models load. Import
qualified historical global heads using the acceptance compatibility commands,
or refit with the new package. Keep source-specific provenance with old results.

Repair, interactive inspection and study-service commands are excluded.
Use `exact run info/export` for practical explanation access. Historical code
and experiments remain in the repository's research branches.
