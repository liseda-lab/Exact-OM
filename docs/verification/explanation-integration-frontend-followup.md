# Study integration frontend follow-up: R06–R09 — 2026-10-03

**Implementation complete and verified on the synthetic fixture; not a release-readiness claim.**
Four frontend defects reproduced after the S3 record (`8755431`) reopened R01 and R02:
- R06: truncated fact categories;
- R07: no full context in the study;
- R08: a malformed cached response was reused by Retry;
- R09: readiness ignored the admitted components.

All four are fixed, and their regressions pass against the real study service and the main app. R01 and R02 are
joint-verified again on that evidence. R03 stays blocked on the manual J10 audit, which must
now also cover the controls added here. The [review record](../../specs/explanation-framework/evidence/frontend-integration-review-20261003.md)
has the reproductions; [19](../../specs/explanation-framework/19-frontend-integration-corrections.md)
F16–F19 the requirements; the [receipt](explanation-integration-frontend-followup-receipt.json) the
exact results.

| Commit | Content |
|---|---|
| `8b13600` | Specifications first: findings R06–R09 recorded, R01/R02 reopened, F16–F19 specified |
| `ad03312` | R06 continuation in the shared card and hierarchy browser (+ adjacent fixes) |
| `c0af27a` | R07 full context in the shared pair workspace |
| `3a1fcd0` | R08/R09 validated cache, attempt-tagged readiness, component-aware render check |
| `237df9c`, `23aa74f` | Fixture option and regressions; one test-timing stabilization |
| this handoff | the following documentation commit |

The backend was not changed. The scoped routes already expose every continuation, context and
configuration these fixes use, and no contract gap was found. The only non-frontend change is an
additive, default-off `--paged-facts` option in the synthetic fixture builder (test data only).

## Outcomes

| Finding | Outcome | How |
|---|---|---|
| **R06** fact pagination | **Fixed and verified** | `useContinuation` continues each displayed page from its own cursor: definitions, alternate definitions, synonyms, every defining-fact category and the other recorded facts through `/entity-facts`, and parents through the parent hierarchy route. Items append in the service's order, deduplicated by fact or edge identity, keeping `data-fact-id` so citations and "show where it appears" still work. Counts read "N of M shown", or "more exist" when only a cursor is known; "all are shown" appears only when complete. Loading, failure and retry are local. The hierarchy browser's parent list and expanded ancestor branches had the same first-page limit and now continue too. |
| **R07** study full context | **Fixed and verified** | `PairWorkspace` passes "Open full context" to its hierarchy browsers when `original_context` is admitted. It opens the shared full-context card over the case's own scoped source. A generated description is requested only when `entity_description` is admitted and the entity is the source or one of its five candidates. The dialog keeps the ranking, draft, inspected candidate, tab and hierarchy focus. Choosing a parent in it navigates the hierarchy. Opening reports the declared `definition_open` type. The baseline has no workspace, so nothing changes there. |
| **R08** retry and cache | **Fixed and verified** | The shared cache validates the required structure before accepting a focal response:<br>• contexts: identity, label record, label list, the definition/synonym/parent pages, every category page and completeness;<br>• available explanations: claims, entities, task and subjects.<br>An unusable 200 is a local failure for every consumer and is evicted. A render failure resets the cache before Retry. Readiness transitions carry their attempt, so late results cannot overwrite a block or ready another attempt. While blocked, the workspace is hidden, so nothing refetches behind the participant's back, and Retry refetches only what failed. The answer, candidate, focus and tab are kept. |
| **R09** component-aware readiness | **Fixed and verified** | The render check now expects exactly what the condition shows: the question; two cards only with `original_context` or `entity_description`; descriptions only with `entity_description`; the comparison only with `pair_comparison`. Lazy tabs are never awaited. Absent components must be absent; admitted ones must render and settle. |

**Adjacent issues found in the re-review and fixed:**
- `EntityCard` read the generated description even when it was not shown. In the study that read can be refused (403) for non-focal entities or description-free scopes, and the old suspicion rule would then block the case. Both were observed on the baseline: the original-context-only publication issued description requests.
- A 403 from any optional read blocked the case. Now it blocks only if the scope's own `/capabilities` is refused as well; otherwise it stays a local panel error.
- The hierarchy browser's parent list and ancestor branches were first-page only with no total (see R06), and appended children were not deduplicated.
- The graph's "expand parents" added a first page silently; it now states when more parents exist and points to the hierarchy view.

## Verification

**Fresh fixture:** `--navigation-size 65 --paged-facts 26`, 59 files. The paged entity has 26 facts per category and 55 parents; the API returns 20 of 26 and 50 of 55 on first read, with continuations of 6 and 5. Variant revisions were published through the admin API with their own form versions:
- `-cmp`: `pair_comparison`;
- `-ctx`: `original_context`;
- `-desc`: `entity_description`;
- `-nav`: `hierarchy`, `evidence_table`, `evidence_graph`.

**Real service:** `serve_explanation_v2_e2e`, HTTPS on loopback with a self-signed certificate, PostgreSQL 16.2, the compiled export at `23aa74f` and the normal CSP. There were two main-app servers on the same frozen contexts:
- the study's policy, for J03 parity;
- policy widened with `alternate_definitions`, for the main-app paging test.

**Baseline reproduction:** the new regressions were run against the reviewed baseline `8755431`, built in a temporary worktree and served alongside. **8 failed and 2 passed:**
- **R06 (main app):** the "20 of 26 definitions shown" count was absent.
- **R06/R07 (study):** "Open full context" was missing.
- **R08:** the malformed context reached rendering instead of being rejected, and the case never recovered as required.
- **R09, comparison-only, hierarchy/evidence-only and the recorded-absence variant:** never became answerable.
- **R09, original-context-only:** descriptions were read although not shown.
- **Passed on the baseline:** the late-read guard and the description-only layout. Both are defensive regressions.

**Final results at `23aa74f`:**
- type check and production build pass; 68 unit tests (7 new: continuation, structure, render expectations, shared cache);
- browser: **80 passed, 2 skipped**. The skips are `legacy.spec`, because the historical run directory is unavailable. This run includes the fresh, unseeded participant journey, every R06–R09 regression, all earlier v2 integration tests, the exploration, library, demo and accessibility suites, and v1 study and recovery;
- backend suites that use the fixture builder: 40 passed, 1 PostgreSQL-only skip; black and isort clean; spec validator: no errors.

## Remaining issues and limits

- **J10 manual audit (open):** no screen-reader or manual keyboard pass was performed. It must now also cover the "Load more …" controls, the full-context dialog and the blocked-case state.
- **Intermittent reload stall (observed, unresolved, backend to investigate):** once in about ten runs of the R03 assessment test, a `page.reload()` never reached its load event. At the same time the study service logged `h11 LocalProtocolError: Too little data for declared Content-Length` from Starlette's `BaseHTTPMiddleware`, raised when a client disconnects mid-response. It did not recur in six targeted reruns. The frontend does not cause the disconnect handling.
- **Citation labels:** a citation to a fact not on a loaded page shows a generic "cited fact" label, though it still opens the exact record. "Show where it appears" works once the page with that fact is loaded.
- **Graph truncation note:** covered by the type check only; no scored graph node in the fixture has more than 50 parents.
- **Alternate definitions in the study:** the study policy withholds them, so their paging was verified in the main app only.
- **Tutorial pages:** the synthetic tutorial resources expose no continuation reader. A prepared page with a cursor would honestly say "not available in this view"; current fixtures have none.
- **Release gates (unchanged):**
  - the real `backend-release` package;
  - Docker, Render deployment and restore;
  - trusted HTTPS;
  - usability and duration pilot;
  - consent/information approval, case adjudication and the historical DOID/import-closure gates.

No push, deployment, recruitment or participant contact was performed.
