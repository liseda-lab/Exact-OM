# Third post-implementation frontend review — 2026-10-04

**Reviewed checkout: `e36cfa5` (R10–R11 fixes on backend `96b76fe`).** The review found one
product regression introduced by the R11 correction and one test whose evidence was racy. The
required behavior is in [19](../19-frontend-integration-corrections.md) F21. Neither depends on
a backend change.

| ID | Reopens | Priority | Reproduction | Observed defect |
|---|---|---|---|---|
| R12 | R03 (J07; frontend subset of J09) | P2 | Real study service, tutorial lesson 1: the practice source card lists two parents, “crate” and “lidded object”. | Both parent buttons are disabled; before R11 they opened the parent. The frozen-resource adapter (`resourceIndex.ts`, `entityContextFrom`) turns each recorded subclass axiom into a bare IRI parent fact, but after R11 the card requires the typed `hierarchy_projection` before it navigates. The same adapter serves v1 study publications, so their case cards lose parent navigation too. |
| R13 | R02 (J04/J05) evidence | P3 | `study-v2-integration.spec.ts`, “a comparison failure blocks with its own label; a profile failure likewise”, run repeatedly. | Intermittent failure. Right after the answer control becomes enabled, the test asserts synchronously that exactly one `case_ready` request was recorded. The traffic recorder adds a request only after it finishes, so the UI can recover before the acknowledgement is recorded. A diagnostic with a bounded asynchronous assertion passed three consecutive runs and saw HTTP 200. |

**Why earlier tests missed R12:** the full tutorial journey navigates through the hierarchy tree,
which already carried typed edges, never through the card's parent buttons. R11's regressions
exercised only the scoped API adapter.

The reviewer's diagnostics are starting points; the maintained regressions are listed in the
third follow-up handoff.
