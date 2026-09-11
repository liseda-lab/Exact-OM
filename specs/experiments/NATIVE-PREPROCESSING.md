# Native preprocessing and bounded measurement

The preprocessing repair changes execution, not the ontology inputs, projection profile,
candidate populations or matching hypothesis. Earlier measurements already used the native
OWL loader, while projector `auto` selection could use Python. The installed projector also
allows scalar compilation before a native edge pass. A native loader or final backend label
alone therefore did not establish native projection.

Exact now requires native parsing and encoded-native projection. Legacy projector `auto`
configurations resolve to native without fallback; explicit Python selection is rejected.
The adapter requires an encoded-native report with zero scalar compilation/materialization
counters before caching edges. A missing compiler, unsupported input or incompatible report
fails explicitly. Import policies, strict parsing and hashed local import bindings remain in
force; inputs are not reparsed with a permissive backend.

The narrow bridge in `exact/ontology/native_projection.py` uses the installed upstream
compiler and edge policy, with a checked **0.2.0 / API 1** contract. It does not implement a
second projection algorithm. Upgrading that dependency requires compatibility and parity
checks. The installed compiler cannot retain the tested mmap owner's buffers; that owner is
explicitly unsupported, with no scalar fallback. Normal document loading uses a native
snapshot. These constraints concern OWL compilation; declared CSV-KG edges do not pass through
an OWL compiler.

Graph construction now fetches labels lazily through the existing cache. Annotation,
exclusion and class-hierarchy preparation use public typed axiom partitions, and unused origin
lookups are disabled. These remove repeated whole-ontology scans; label rendering, graph
wrappers and per-entity feature assembly still include Python work. Existing per-entity caches
remain in use. Dataset timing also records ontology setup before a dataset-cache hit.

## Benchmark workflow

`tools/benchmark_native_preprocessing.py` reads the selected ontology and locked imports from
a resolved Exact configuration. It uses the asserted hierarchy and actual pair-adaptive raw
feature functions, with deterministic verbalization and CPU wrapper settings. It never opens
reference/candidate/training files, fits a model, loads an encoder or sends an LLM request.

Run each ontology in a fresh process with a new output path and an external timeout. For
example, this limits a diagnostic invocation to 15 minutes:

```console
timeout --signal=TERM --kill-after=15s 900s \
  .venv/bin/python tools/benchmark_native_preprocessing.py \
  --config data/experiments-v2/g0-validation-04/cold64/run/_inputs/resolved.config.yaml \
  --side source --entity-limit 64 \
  --output data/experiments-v2/native-preprocessing/new-source.json
```

Use `--side target` and another output path for DOID. For an exact comparison with saved cold64
features, also pass `--entities PATH` containing the saved 64 source IRIs, one per line.
Without that option the tool records a deterministic diagnostic sample; it does not claim to
reproduce the saved candidate population. A small entity limit bounds feature queries, not
full-ontology loading or projection.

The report records input/import and implementation hashes, native execution evidence, entity
selection, counts and signature/exclusion/label/edge/feature digests. Ordered phases measure
load, signature, projection, edge hashing, exclusions, labels, graph construction, feature
wrapper initialization, cold feature lookup, cached feature lookup and provenance. Times are
incremental in that order; RSS is the cumulative process high-water mark. Cached lookup time
is not cold throughput. Compare digests with identical input bytes, settings and selected
entities when assessing a repair.

Reports are written at phase boundaries. If a timeout kills native code, the last phase may
remain `running`; that is incomplete evidence. Existing reports cannot be overwritten. The
six focused tests in `tests/native_preprocessing_benchmark_test.py` cover native fixture
execution, stable digests, pinned imports, forbidden model/network calls and failure records.
The native bridge has separate parity and fallback-rejection tests in
`tests/ontology_native_projection_test.py`.

This benchmark is operational evidence, not a G0 pass or campaign admission. It omits retrieval,
encoding, fitted heads, hosted judgment, extraction and end-to-end recovery checks. Preserve
all earlier timings and attempts, including cold64's original worker duration and cumulative
budget charge. Attempt 05's cold-reload forecast term is a conservative policy calculation,
not a measured warm duration. New measurements must support any revised forecast; no new
full-ontology speed or matching-quality result is claimed here. See
[G0-VALIDATION.md](G0-VALIDATION.md) for admission and retained attempt history.
