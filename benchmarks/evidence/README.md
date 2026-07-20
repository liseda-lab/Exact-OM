# Release evidence

This directory contains small, reviewable records for acceptance checks whose real inputs or
historical environments cannot be committed. Large and licensed captures stay under the
gitignored `exp/` tree; the evidence records pin their inputs and outputs by SHA-256.

## WP-B Conference parity

`wp_b_conference.json` compares the Java-free ontology backend at `a089386` with the last
working pre-removal implementation at `330044c`. Both runs used the CPU-only v1-compatible
`wp_b_conference_legacy.yaml`; Exact 2.0 migrated that config in memory. CUDA and hosted LLM
access were not used.

To recapture the current snapshots after materializing `cmt-conference`:

```bash
python tools/capture_backend_baseline.py --backend exact \
  --output-dir exp/baselines/exact DATA/cmt-conference/source.owl \
  DATA/cmt-conference/target.owl
```

To recapture the historical backend directly, run the current capture tool in an isolated
Python 3.10 environment with JDK 15, `mowl-borg==1.0.1`, `JPype1==1.5.0`,
`numpy==1.26.4`, and `class-resolver==0.5.4`:

```bash
python tools/capture_backend_baseline.py --backend mowl --jvm-heap 4g \
  --output-dir exp/baselines/mowl DATA/cmt-conference/source.owl \
  DATA/cmt-conference/target.owl
```

The historical end-to-end commands themselves run from an isolated worktree at `330044c`;
the capture helper intentionally invokes mOWL directly and does not depend on that old tree's
Exact wrappers.

The local-ranking candidate table used for the second end-to-end check was the 20 generated
targets per class source from the fixed global run, paired with the 12 class reference rows.
Its hash and row count are in the JSON record, so regeneration drift is visible.

## WP-M NCIT–DOID scale baseline

`wp_m_ncit_doid_baseline.json` freezes the last private-parser/private-projector measurements
before WP-M moves ontology ownership to `pyowl-core` and projection to the shared projector.
The legally redistributable Bio-ML files remain below the gitignored `data/` tree; the evidence
record contains their byte sizes and SHA-256 digests, plus cold parse and projection counts and
timings. It deliberately contains no machine-local path.

The comparison after M2/M3 must use inputs with those exact digests and report, separately,
source loading, snapshot construction, time to first projected edge, full projection, peak and
incremental RSS, native/Python backend, cache state, and output fingerprints. A result with a
different input digest is a new benchmark, not a replacement for this baseline.

`wp_m_ncit_doid_candidate.json` is the first hash-matched post-migration capture. It proves one
load, shared object identity, no second ontology representation, and a fast projection-cache hit,
but it is deliberately marked `blocked`: Python loading and projection exceed the 25% wall-time
gate. The NCIT projection's net 754-edge difference is fully classified and digest-bound: 762
RB-019 subrole-expansion additions, eight RB-009 removals of edges incorrectly emitted by the old
private Exact approximation, and zero residuals. Pinned shared-projector semantics pass; the
performance regressions and comparable pinned-runner RSS evidence must still be resolved before
M5 or the 2.1.0 release can be accepted. The raw benchmark remains reproducible with:

```bash
PYTHONPATH=.:../pyOWLCore/src:../pyOwl2Vec-Star-projector/src \
  python benchmarks/owl_stack_scale.py \
  data/bioml_zenodo/ncit-doid/source.owl \
  data/bioml_zenodo/ncit-doid/target.owl \
  --output /tmp/exact-wp-m-ncit-doid.json
```

## WP-N native-consumer handoff

Schema 3 of `benchmarks/owl_stack_scale.py` keeps the WP-M fields and additionally records CPU
time, canonical projection and hierarchy-result digests, public core operation deltas, exact
consumer ingestion provenance, encoded compiler counters, copy/materialization coverage, and
optional reasoner compilation plus first/complete-result timing. It records `null` rather than
inventing a phase measurement when a consumer does not yet expose publication/compiler timing.

Capture Python, native in-process, and verified-wire reasoner cases as separate cold processes;
do not compare a warm encoded publication against a cold parser run. For example:

```bash
PYTHONPATH=.:../pyOWLCore/src:../pyOwl2Vec-Star-projector/src:../pyELK/src:../pyHermiT/src \
  python benchmarks/owl_stack_scale.py \
  data/bioml_zenodo/ncit-doid/source.owl \
  data/bioml_zenodo/ncit-doid/target.owl \
  --load-backend native --projector-backend native \
  --reasoner hermit --reasoner-backend native \
  --require-encoded-consumers \
  --output /tmp/exact-wp-n-ncit-doid-native.json
```

`--require-encoded-consumers` is deliberately fail-closed: it rejects scalar fallback even when
the semantic output is correct. Leave it off only for the scalar comparison capture. GO and the
largest licensed workflow use the same command shape and remain external release evidence.
