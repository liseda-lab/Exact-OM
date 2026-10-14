# Joint human-review acceptance — 14 October 2026

The focused spec 24 corrections are implemented and the final **69-test browser matrix plus one opposite-order complete journey passed**, with no retries, failures, skips or flaky tests. The build is available for private human review with the limitations below. It does **not** have unqualified clean technical acceptance: original HR1 causation is unresolved, a synthetic backend latency gate failed, and wider repository quality checks remain failing. Research/deployment readiness is not established.

This handoff follows [spec 24](../../specs/explanation-framework/24-human-review-readiness-corrections.md), the [frontend handoff](explanation-readiness-frontend-corrections-20261013.md) and the [14 October independent review](explanation-readiness-frontend-independent-review-20261014.md). The companion [backend handoff](explanation-readiness-backend-corrections-20261013.md) provides the implementation, quality diagnosis and exact startup/stop commands. [Machine-readable receipt](explanation-readiness-acceptance-20261014/receipt.json), [test results](explanation-readiness-acceptance-20261014/tests.json), [bindings](explanation-readiness-acceptance-20261014/binding.json) and [quality details](explanation-readiness-acceptance-20261014/quality.json) are adjacent, sanitized evidence.

## Final binding and method

Starting checkout: `de2cddac9190a94dcc0362ce96bfa69c8692c80a`. Logical commits: `3211cd7` mechanical formatting; `c230bc2` HR2 frontend corrections; `392b104` static delivery/diagnostics and regressions; `fd47dcacaa09d58ca8dd6f86c52a1a6204ce67b0` preservation-test correction. Documentation is committed afterwards without changing the frozen runtime. Unrelated dirty files remain disclosed and hash-checked in the receipt; they are not part of this delivery.

| Identity | Exact value |
|---|---|
| Frozen backend source (58 files) | `3cb520b13a4471342a9eea34e67c5e80a420b71277693c6b568f3496d884954c` |
| Final production export | `3289bf6daf9312da0187ec95047827c22587c283492047026f10dff9fcdf8350` |
| Reused package definition | `b89ae882e62ceb9ad810aa063b739429c08027fef26fe0ef0dc75ca0007bc18a` |
| Package ID | `sha256:55acbf149864181695c4aa19817131682ef615f307c13993172a87fd5eef322c` |
| Unchanged source publication | `3893f7607b229fbbd3957d382ad3d27063293a642e18f15ddce8605eddfe209f` |
| New managed publication | `352ed1d79d71cdb13a5eb5e77506c82541ffefd9db958a6da8e188bd52fbd47b` |
| Environment / revision | `6205ddfa9a264beab2a54612ddcc8379` / `review-20261014T123129-31902966fc` |
| Validity | Closes `2026-11-13T12:31:29.114541+00:00` |
| TLS certificate SHA-256 | `e5e4c3c0a6a0e73a61e3f7df7ff48d93a8a305e9317981683294d64a002074ab` |

All final functional browser checks used the same final code/export/data binding, compiled frontend, actual PostgreSQL 16.2 service, HTTPS and normal CSP. One worker and zero retries were explicit. Chrome's certificate exception was restricted to the recorded local certificate SPKI; the API client trusted the same local certificate through `NODE_EXTRA_CA_CERTS`. Owner-browser trust was not changed. The separate retained API read-probe helper uses its pre-existing `ignoreHTTPSErrors` allowance; its results do not certify CSS caching or transfer hashes.

The source is `/Users/pedrocotovio/Library/Caches/exact-om-dev/readiness-closure-20261014/source-fd47dcac`; export and environment are under `/Users/pedrocotovio/Library/Caches/exact-om-review/readiness-closure-20261014/`, named `frontend-fd47dcac` and `environment-fd47dcac`. This directory also retains original raw logs/JUnit/Playwright results. The canonical checkout is `/Users/pedrocotovio/Developer/Repos/OM/Exact-OM`.

## A24 acceptance matrix

| Requirement | Fresh observation | Disposition |
|---|---|---|
| A24-01 styling/reload | Two full unseeded 24-case journeys, first baseline then first explanation. Nine computed-style/layout boundaries each; actual lesson-two, wrong question-five and consultation reloads. Each journey had four completed CSS loads: one network, one memory and two disk; no failed/pending CSS or export mismatch. | Follow-up checks pass; **original HR1 remains unresolved**. Fault cancellation separately verifies the detector. |
| A24-02 static delivery | 81 certificate-validated HTTPS checks: CSS GET/HEAD MIME, body/hash/length, public caching, missing assets, protected HTML/API/errors, CSP/profile boundaries. Six client-close attempts followed by successful independent reads. Fourteen final static unit cases include disabled logging, redaction/rotation, cancellation, incomplete send and offload unknowns. | Confirmed cache defect fixed and verified. Server completion means ASGI send completion, not proof of TCP/client receipt. |
| A24-03 prepared entry | Fresh real prepared landing, examples and All sources, deliberate unprepared links, navigation/reload, bounded target paging. Twelve promoted/extended HR2 boundary cases pass, including absence/empty/failure, wrong typed identity, budget, exhaustion, retry and intervening choices. | HR2-A/B fixed and verified; synthetic boundaries remain labelled. |
| A24-04 hierarchy entry | Curated default/label, alternate roots, continuation, typed/basis changes, independent/pinned focus and legacy metadata regressions pass. | Accepted HR4 preserved; no new ontology edges or obsolescence inference. |
| A24-05 study isolation | Existing Luna/recovery/journey suites pass: participant surfaces have no unrestricted catalogue, baseline/current-scope denials, shared workspaces, explicit per-case optional/combinable methods and analysis exports. Fresh API parity below. | Verified within the recorded current development publication. |
| A24-06 quality | 142 frontend units, typecheck/build pass; contract snapshots/import boundaries/docstrings pass. Broad backend:393 passed / 2 failed / 3 skipped; one atime test defect corrected and 25 affected checks pass. Pinned owned formatting/static lint pass. | **Not full CI green**: unresolved synthetic G4 latency plus wider quality failures below. |
| A24-07 rendering | Complete matrix includes 10 visual/Axe/overflow checks. Additional 10-pass capture run saved 25 views: desktop light/dark,375px/100%,375px/200%,320px/200%. All 25 and four HR2 negative-state captures inspected. Keyboard/focus checks remain in entry/journey suites. | No new visual blocker observed; narrow 200% wrapping/long vertical reading remains a usability limitation. Human accessibility review open. |
| A24-08 operations/preservation | Exact hashes independently rechecked; same package, provenance and partial-ELK semantics. 120 pairs / 260 descriptions / 24 scopes and the complete 4,189-root inventories pass API parity. Owned restart preserved all 47 test-session states; actual HTTPS reissue/exchange resumed a completed 24-case test session. Shared guard measurements below. | Implemented boundaries verified with the stated limits; no new full bundle or source-inode mutation. |

## Exact browser execution and evidence boundaries

| Final matrix file | Tests |
|---|---:|
| `assessment-recovery.spec.ts` | 8 |
| `luna-corrections.spec.ts` | 14 |
| `readiness-entry.spec.ts` | 19 |
| `readiness-visual.spec.ts` | 10 |
| `review-hierarchy.spec.ts` | 2 |
| `review-independent-edges.spec.ts` | 12 |
| `static-delivery.spec.ts` | 2 |
| `study-review-journey.spec.ts` | 1 |
| `style-boundary.spec.ts` | 1 |
| **Total** | **69** |

The final matrix took 828.544 s including runner overhead; its baseline-first journey took 107.270 s. Exactly one further complete journey covered explanation-first (runner 91.919 s). Assignment was obtained through ordinary test-session allocation; neither schedule nor assigned order was rewritten. There were 32 assigned test sessions before the second journey; the next ordinary schedule was verified as explanation-first. Both full journeys used rendered consent/setup/background, all six tutorial lessons, incorrect assessment answers and recovery, saved/reloaded drafts, all 24 cases across both conditions, changed consultation methods, completion and analysis/4 exports. Invitations were API-created; training completion was not seeded. A deliberate invalid tutorial action in the journey tests rejection and is labelled diagnostic, not a successful action.

The initial focused run added 14 passes (12 HR2 + 2 real cache). The full matrix already passed the visual assertions but its optional screenshot directory was not set. The additional 10-test visual run enabled `EXACT_SCREENSHOT_DIR` to retain captures; it does not replace any failed acceptance result. Its study views use seeded synthetic state solely to inspect empty case controls, separate from unseeded journeys. The CSS fault test intentionally aborts a stylesheet and verifies detection; it is not evidence that HR1's original cause reproduced.

Reproduce the browser batch from the synced frontend cache after setting the private environment locally:

```sh
TASK=/Users/pedrocotovio/Library/Caches/exact-om-review/readiness-closure-20261014
PY=/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python
NODE=/Users/pedrocotovio/Library/Caches/exact-om-dev/node-v24.21.0-darwin-x64/bin/node
FE=/Users/pedrocotovio/Library/Caches/exact-om-dev/fe
export EXACT_E2E_URL=https://127.0.0.1:19675
export EXACT_E2E_REVIEW_MAIN_URL="$EXACT_E2E_URL"
export EXACT_E2E_STUDY_V2_CONFIG="$TASK/harness.json"
export EXACT_E2E_EXPORT_DIR="$TASK/frontend-fd47dcac"
export EXACT_E2E_CERT_SPKI='WbLmculjnG86/SodK/bBJ2ZzMS+pH6TdB7Of11Vfk9Q='
export NODE_EXTRA_CA_CERTS="$TASK/environment-fd47dcac/tls/localhost.pem"
export EXACT_CHROMIUM_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
# Choose NEW output paths; do not overwrite original failures or receipts.
export EXACT_E2E_STYLE_DIAGNOSTICS="$TASK/new-review-styles"
export EXACT_SCREENSHOT_DIR="$TASK/new-review-views"
export PLAYWRIGHT_JSON_OUTPUT_FILE="$TASK/new-review-results.json"
# Run the guard from the canonical checkout; invoke Playwright with the FE directory as cwd.
"$PY" -B -m tools.review_storage_budget check --growth-gb 1
"$PY" -B -m tools.review_storage_budget run --growth-gb 1 -- "$PY" -B -c 'import subprocess,sys; raise SystemExit(subprocess.call(sys.argv[2:],cwd=sys.argv[1]))' "$FE" "$NODE" "$FE/node_modules/@playwright/test/cli.js" test e2e/luna-corrections.spec.ts e2e/study-review-journey.spec.ts e2e/review-hierarchy.spec.ts e2e/assessment-recovery.spec.ts e2e/readiness-entry.spec.ts e2e/readiness-visual.spec.ts e2e/style-boundary.spec.ts e2e/review-independent-edges.spec.ts e2e/static-delivery.spec.ts --workers=1 --retries=0 --output="$TASK/new-review-output"
```

Use this for a new purposeful verification, not an automatic repeat-until-green loop. Keep the FE source pinned to the recorded frontend tree; do not resync later application edits into a check advertised as this binding. The full exact command records and per-test results are in the adjacent receipt.

Representative inspected captures: [desktop prepared entry](explanation-readiness-acceptance-20261014/hr2-landing-desktop-light.png), [dark hierarchy](explanation-readiness-acceptance-20261014/hr4-suggested-start-desktop-dark.png), [375px/200% hierarchy continuation](explanation-readiness-acceptance-20261014/hr4-entry-list-375-200pct-light.png), [320px/200% study reload](explanation-readiness-acceptance-20261014/study-case-after-reload-320-200pct-dark.png), [missing-catalogue recovery](explanation-readiness-acceptance-20261014/hr2-absent-catalogue-source-browsing.png), [bounded target search](explanation-readiness-acceptance-20261014/hr2-prepared-target-search-paused.png). Viewport captures are not full-page coverage: some enlarged narrow views show controls above the content and require scrolling to reach the loaded context.

## Findings, failures and remaining technical work

**HR1 remains unresolved.** The original 13 October batch was 24 passed / 1 failed at the first baseline case after the question-five reload. Its [failed viewport](explanation-human-review-readiness-20261013/unstyled-study-baseline.png), [previous styled tutorial](explanation-human-review-readiness-20261013/styled-tutorial-before-failure.png) and [receipt](explanation-human-review-readiness-20261013/receipt.json) remain unchanged. All19 original evidence hashes and all 8 hashes in the 14 October independent review match. The new finite investigation did not reproduce the failure. The confirmed cache override prevented ordinary asset reuse and is repaired, but no evidence establishes it as the incident's cause. Historical R16 and profile-read incidents remain distinct. On a future failure, compare bounded CSS completion/hash and correlated ASGI observations with stylesheet-link/CSP/navigation/computed-style events. Matching complete bytes would direct the next investigation toward browser lifecycle; incomplete correlated transfers toward the serving/transport boundary. Current tests do not decide between those hypotheses.

**Backend regression failure retained.** The broad 398-case run reported 393 passed / 2 failed / 3 skipped. One preservation test incorrectly treated read-induced atime changes as immutable-file mutation; `fd47dca` fixes only that test, retaining content, inode, link, permissions, mtime and ctime assertions, and all 25 affected module tests pass. The other failure remains a synthetic G4 performance measurement:100 mixed reads / four readers, all responses successful/bounded, context p95 = 0.5156448660418391 s against ≤0.5 s. Five of 25 entity-context calls crossed 500 ms. The publisher correctly rejected a ledger declaring the failed gate passed. This is not the delivered real-data package, and unchanged code alone does not prove an inherited cause. Backend performance owns one planned isolated baseline/final comparison with route/application/serialization/CPU timings; preserve the gate and both outcomes.

**HR3 scoped correction is complete; full CI is not.** Black 24.10.0 / isort 5.13.2 mechanically cleaned 46 tracked explanation files in a separate commit; non-import AST and import inventories match. Full checks leave 19 Black / 10 isort files in unrelated modules/preserved untracked experiments. Mypy retains exactly 48 baseline error identities across 13 files. Flake8's 52 current entries comprise 45 tracked baseline equivalents and 7 preserved untracked cluster/HermiT findings; the clean baseline had 722 entries. Owned delivery files pass lint. Module owners must diagnose broader semantic findings; no suppressions, changed style configuration or dependency-lock churn was used. Import boundaries pass 5/5 and priority docstrings 82.6% exceed 80%. Python 3.10/3.11, distribution builds and the full CI job matrix were not rerun locally.

The three broad-suite skips are opt-in real PostgreSQL process/restart and v1/v2 backup/restore tests without their dedicated cluster variables. Fresh managed restart/state preservation below is a separate check; prior backup/restore evidence remains limited to its original local fixture. Actual Docker image execution and production-volume deployment/restore remain unverified technical release work. Legacy packetless citation ambiguity retains the previously documented compatibility boundary.

An early TypeScript check caught a new pagination-helper inference error (TS7022); an explicit query/page type corrected it before the final build. Original logs remain retained. The first static red test reproduced the study cache override; its public-demo companion initially had a fixture-admission setup error. Neither harness correction is presented as a second production defect.

## Data, operations, storage and access

The [fresh API preservation receipt](explanation-readiness-acceptance-20261014/api-preservation.json) passes all 120 prepared pairs, 260 descriptions and 24 scopes, plus all 4,189 roots through 43 pages per asserted/structural main and scoped inventory. The original matcher/context artifacts and protected predecessor statistics are unchanged. It preserves 3,443 selected evidence records, 291 approved clauses and all 17 previously published definition supplements; no supplement was generated in this pass. Partial ELK remains incomplete positive evidence, so absence is unknown. These two full API-driven journeys are separate from the unseeded browser journeys.

The current citation counter is **337 scoped resolutions**, versus 336 in the predecessor run. Both runs resolved the same **324 distinct fact-response bodies**. One existing fact was checked in one additional scope: the verifier de-duplicates by scope/typed subject/fact while globally processing an artifact only at its first encounter, so traversal order affects repeated checks. This is not a changed corpus count. The [citation-count audit](explanation-readiness-acceptance-20261014/citation-count-audit.json) records the exact repeated response and unchanged response-body inventory.

Owned status → stop → stopped status → start → ready status all succeeded. All **47 pre-existing test sessions in this new revision retained byte-identical saved state** across the restart. The lifecycle tool then created a fresh private development invitation. A separate actual-HTTPS recovery check used supported administrator reissue on one completed test session created by this pass, exchanged its invitation and read cookie-authenticated state: completed stage, all 24 cases and unchanged saved state. Missing-Origin exchange was explicitly rejected with 403. The small helper initially lacked its module path, then omitted the required Origin header; both failures are retained and were corrected in the harness, with no product/assertion change. That one owned test had two invitation rotations in total; historical/other sessions were not reissued. See [lifecycle](explanation-readiness-acceptance-20261014/lifecycle.json) and [resume](explanation-readiness-acceptance-20261014/post-restart-resume.json).

Bounded API read probes passed during normal operation and after restart: **200 successful reads each**, plus ten intentional aborts, ten reloads with outstanding reads and ten stream-cancellation attempts per probe, with subsequent independent reads. Only expected intentional abort/cancellation failures were observed. Normal-operation maximum read latency was 647 ms; after-restart maximum 940 ms (diagnostic mixed-read p95 approximately 469/533 ms). These probes assert response/body/length and recovery invariants, not a latency acceptance gate. Client cancellation does not establish incomplete server transmission. [Normal probe](explanation-readiness-acceptance-20261014/normal-read-probe.json), [after-restart probe](explanation-readiness-acceptance-20261014/after-restart-read-probe.json). Retained diagnostic rotations remained 0600 and below 1 MiB per file with at most two backups; the [bounded-window audit](explanation-readiness-acceptance-20261014/diagnostics-audit.json) found no observed error types or incomplete finishes, and service logs had no recorded Content-Length protocol exception. Rotated windows do not prove that every historical event was observed.

Main app: **https://127.0.0.1:19675/**. Study entry is **https://127.0.0.1:19674/participate/** with a private fresh test invitation. The backend handoff contains the exact frozen-source start/status/fresh-session/stop commands and preserved predecessor rollback binding. `fresh-session` writes the invitation to a private local file; do not put the invitation, token, cookie or storage contents into a commit or chat. Owner-browser trust of this local certificate remains a setup step, not evidence of public TLS deployment. Do not reopen expired historical revisions by editing their database.

Measured combined usage after verification/packaging was **52,310,274,470 bytes (52.310 GB)** at 2026-10-14T13:26:58.391672+00:00, with **114,955,427,840 bytes** filesystem free. The largest observed point sample was 52,310,274,470 bytes. A conservative simultaneous reservation of 3 GB (up to 1.5 GB foreground plus the 1.5 GB aggregate detached-service allowance) puts estimated peak usage at **55,310,274,470 bytes (55.310 GB)**. Initial combined usage was 51,800,378,750 bytes. All were below the 100 GB limit and retained the 5 GB safety / 30 GB filesystem-free reserves. These are measured samples plus conservative reservations, not a continuous peak or OS quota. The [storage receipt](explanation-readiness-acceptance-20261014/storage.json) includes all retained samples; evidence is about 4.2 MB. No full data bundle or second dependency environment was created, and no protected inode was modified. Retained runtimes were reused; a small pinned quality-tool overlay was installed in the existing verification cache.

**Implementation status:** the assigned HR2-A/B, verified-public-cache correction, bounded diagnostics and owned formatting work are implemented and pass their affected checks. Accepted hierarchy/design/citation behavior is preserved. **Integrated local human review:** available with the explicit HR1, performance and quality limitations; no clean technical verdict. **Full CI:** not passed. **Research/deployment readiness:** not established. Manual screen-reader/accessibility review, participant usability/pilot evaluation, study-owner approval and final matcher/case/ontology admission remain open alongside the separate technical work above.

No further frontend implementation round is required by the passing affected checks. The concrete future frontend/integration task is to use the retained bounded diagnostics if HR1 recurs and investigate the evidenced boundary; narrow 200% usability belongs in the human review. No generation batch, matcher campaign, full physical bundle, HermiT attempt, public deployment or participant contact occurred.
