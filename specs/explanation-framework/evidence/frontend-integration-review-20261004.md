# Second post-implementation frontend review — 2026-10-04

**Reviewed checkout: `ad9461c` (R06–R09 fixes on backend `96b76fe`).** Independent verification
passed:
- the four original reproductions;
- 68 unit tests;
- the production build;
- specification validation;
- 37 browser integration tests.

It also reproduced two further frontend defects, which reopen R01 and R02 in
[17](../17-integration-corrective-programme.md). The required behavior is in
[19](../19-frontend-integration-corrections.md) F20–F21. Neither depends on a backend change.

| ID | Reopens | Priority | Reproduction | Observed defect |
|---|---|---|---|---|
| R10 | R02 (J04/J05) | P2 | Fault injection on a usable explanation case:<br>1. Refuse optional hierarchy reads with 403, so two capability checks start.<br>2. Hold both checks.<br>3. Fail the first; the case blocks.<br>4. Restore valid responses and Retry; ranking becomes enabled.<br>5. Release the second, older failure. | The older failure blocks the recovered case again. The check's callback attached its failure to whatever attempt was current when it resolved (`attemptRef.current` at `caseContent.tsx:168`), not to the attempt that issued it. |
| R11 | R01 (J03) | P2 | Real synthetic data: an object property with 55 named superproperties.<br>1. Open its full context; 50 parents show.<br>2. Load the other 5 and choose one. | Navigation opens the parent as `kind=class`. Continuation records kept only the parent IRI. The shells (`BrowseView`, the pair workspace's full-context dialog and card links) then checked only the first page to decide whether to keep the type. Data properties fail the same way. |

**Why earlier tests missed them:**
- R10 needs two overlapping checks with a retry between their resolutions; the earlier 403 test used one.
- R11 needs a non-class entity with more than one page of parents; earlier paging fixtures paged class parents only.

The review's temporary reproductions are diagnostic starting points. The maintained regressions are listed in
the follow-up handoff.
