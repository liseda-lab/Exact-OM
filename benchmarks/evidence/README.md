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
