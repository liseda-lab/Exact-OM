# User study workflow

The user-study tools turn a completed local-ranking run into a balanced, reviewable selection.
They read both run layouts through `RunReader`; a legacy monolithic explanation file is not
required for a layout-v2 run.

```console
exact-user-study \
  --run-dir runs/omim-ordo \
  --top-k 5 \
  --per-rank 4 \
  --shortlist-per-rank 8 \
  --generate-rationales
```

The analysis computes pair metrics, source-panel completeness, gold rank, score gaps, evidence
coverage, ambiguity, and failure buckets. It produces a deterministic shortlist before a human
review sheet is merged; manual selections are never silently overwritten.

Typical derived artifacts include:

- pair and source-panel tables;
- balanced shortlist and selected mappings;
- selected explanation records and rationale coverage;
- failure taxonomy and diagnostics;
- a compact inspection bundle and analysis notebook.

Keep the original run immutable. Store reviewer decisions and bundle outputs under
`analysis/user_study/`, and record the source run manifest/config fingerprint in study
metadata. Use `exact-inspect serve` for a fixed iframe deployment or `exact-inspect open` for
local audit.
