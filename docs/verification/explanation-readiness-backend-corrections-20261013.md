# Backend human-review readiness corrections — 14 October 2026

This is the backend handoff required by spec 24. See the [joint acceptance](explanation-readiness-acceptance-20261013.md), [sanitized receipt](explanation-readiness-acceptance-20261014/receipt.json), [exact commands and test results](explanation-readiness-acceptance-20261014/tests.json) and [quality details](explanation-readiness-acceptance-20261014/quality.json). The user explicitly included the two small frontend corrections in this closure assignment. The functional browser matrix passes on the final bound implementation; original HR1 causation and a measured synthetic resource-performance gate remain unresolved. Full CI and research launch readiness are not established.

## Findings and implementation

| Finding | Disposition and evidence |
|---|---|
| HR1, original loss of application styling | **Unresolved / not reproduced in two final complete browser journeys.** The confirmed public-asset cache override is fixed and verified separately. Nine style boundaries passed in each allocation order, including lesson/assessment reloads. No observation establishes that the cache override caused the original incident. |
| HR2-A, explicit examples URL without usable catalogue | **Fixed and verified.** Ordinary sources remain usable with absent, empty, incompatible-run or failed metadata; useful retries and deliberate choices survive recovery. |
| HR2-B, prepared targets beyond initial pages | **Fixed and verified.** Authoritative typed targets resolve through bounded, cancellable candidate pagination; exhaustion, failure and the search budget expose explicit recovery. No unprepared candidate is silently substituted. |
| HR3, pinned formatting and quality | **Owned mechanical/static-delivery checks pass; full quality remains failing.** Commit `3211cd7` formats 46 tracked explanation-framework files using Black 24.10.0 and isort 5.13.2. Non-import AST and imported-symbol inventories match. Wider failures and the backend regression failure are itemized below. |
| HR4, curated hierarchy entry | **Accepted behavior preserved and regression-verified.** The suggested default, independent focus, other roots and pagination remain available. No inferred obsolescence or invented hierarchy edge was added. |

`exact_inspect/public_assets.py` admits only existing, non-symlink, hash-named public assets in configured `_next/static` namespaces, binds actual SHA-256 and file identity at mounting, and rejects subsequent mutation. The main and study protection layers honor a trusted internal route marker for immutable caching on eligible successful GET/HEAD responses. File responses retain their body/range handling; the SHA-256 ETag identifies actual bytes independently of the build filename's hash convention.

HTML, APIs, private study resources, non-admitted files and errors retain their no-store policies. Strict CSP, traversal protections, profile allowlists, authentication and baseline/current-scope boundaries remain intact. No participant wire model or prepared-data schema changed; frozen runtime contracts pass their check.

Opt-in `ReadDiagnostics` adds `exact-static-transfer/1` records and safe correlation headers only for admitted assets. Records contain asset hash/size, method/status, random request ID, declared/completed byte counts, duration and completion/disconnection/error type. HEAD is distinct from a zero-byte GET; offloaded transfer counts remain unknown. The observer does not consume request bodies, buffer streams, suppress cancellation/errors or retry transfers. Rotation is bounded to three 1 MiB files with restrictive permissions. Browser diagnostics validate correlation values and sanitize exception categories. A cached response can replay an earlier ID without a new server request.

Frontend changes concern prepared-example availability/target resolution and regressions. After initial candidate loading, at most 20 further pages are searched automatically before an explicit continuation. Abort controllers and stale-response checks preserve intervening source/pair choices. The two retained reproductions are promoted into 12 cases covering catalogue failures, typed identity, exhaustion, budget, retry and races. Optional screenshots capture four asserted public fallback/error states. A 375px/200% visual variant supplements 375px/100% and 320px/200% coverage.

## Exact implementation, runtime and data binding

| Binding | Recorded identity |
|---|---|
| Starting checkout | `de2cddac9190a94dcc0362ce96bfa69c8692c80a` |
| Mechanical commit | `3211cd7` — pinned explanation-framework formatting |
| Frontend behavior commit | `c230bc2` — prepared targets and ordinary-source fallback |
| Delivery/diagnostics commit | `392b104` — verified static caching and bounded observations |
| Final implementation/test commit | `fd47dcacaa09d58ca8dd6f86c52a1a6204ce67b0` — preservation test distinguishes access time from immutable state |
| Frozen source | `/Users/pedrocotovio/Library/Caches/exact-om-dev/readiness-closure-20261014/source-fd47dcac` |
| Backend fingerprint | `3cb520b13a4471342a9eea34e67c5e80a420b71277693c6b568f3496d884954c`, 58 files |
| Frontend export | `/Users/pedrocotovio/Library/Caches/exact-om-review/readiness-closure-20261014/frontend-fd47dcac` |
| Export tree SHA-256 | `3289bf6daf9312da0187ec95047827c22587c283492047026f10dff9fcdf8350` |
| Unchanged package | `postmigration-20261011/combined/package/package-definition-review-20261013.json`; SHA-256 `b89ae882e62ceb9ad810aa063b739429c08027fef26fe0ef0dc75ca0007bc18a` |
| Unchanged source publication | `postmigration-20261011/combined/publication-definition-review-20261013.json`; SHA-256 `3893f7607b229fbbd3957d382ad3d27063293a642e18f15ddce8605eddfe209f` |
| Package ID | `sha256:55acbf149864181695c4aa19817131682ef615f307c13993172a87fd5eef322c` |
| Managed environment | `/Users/pedrocotovio/Library/Caches/exact-om-review/readiness-closure-20261014/environment-fd47dcac`; environment ID `6205ddfa9a264beab2a54612ddcc8379` |
| Study revision | `review-20261014T123129-31902966fc`; closes `2026-11-13T12:31:29.114541+00:00` |
| Bound publication SHA-256 | `352ed1d79d71cdb13a5eb5e77506c82541ffefd9db958a6da8e188bd52fbd47b` |
| TLS certificate SHA-256 | `e5e4c3c0a6a0e73a61e3f7df7ff48d93a8a305e9317981683294d64a002074ab` |
| Certificate SPKI | `WbLmculjnG86/SodK/bBJ2ZzMS+pH6TdB7Of11Vfk9Q=` |
| Ports | PostgreSQL 19673; study HTTPS 19674; exploration HTTPS 19675 |
| Runtimes | Python 3.12.3, FastAPI 0.116.2, Uvicorn 0.35.0, Pydantic 2.13.4, psycopg 3.3.6, pyowl-core 0.2.1, PostgreSQL 16.2, Node 24.21.0, Playwright 1.63.0, Google Chrome 155.0.8059.79 |

A read-only independent binding check rehashed frozen backend/export/package/source-publication identities and compared all 165 tracked frontend files between the canonical checkout, frozen checkout and synced test/build cache: no mismatch. Unrelated dirty specifications, review receipts, untracked experiment/cluster tools, storage tooling and archives were preserved. The baseline inventory records their original state; none is silently included in the runtime claim.

No new generation, matching, ontology reconstruction, full physical data bundle or HermiT run was performed. Protected hardlinks were not unlocked or edited. The retained data includes the full 30,374-pair matcher pool, 120 prepared pairs, 260 explanations, all previously verified citation responses, original evidence/scores, approved prose and positive-only partial-ELK semantics. Fresh API verification passes all 120 pairs, 260 descriptions and 24 case scopes, including 4,189 roots across 43 pages in both main/scoped asserted and structural inventories. It preserves 3,443 selected evidence records, 291 previously approved clauses and the 17 definition supplements already published before this pass. The current traversal performed 337 scoped citation resolutions versus the predecessor’s 336; both have the same 324 distinct fact-response bodies. The difference is a repeated fact resolved in one additional scope due to traversal/first-encounter de-duplication, not a changed citation or artifact. See the adjacent citation-count audit and API receipt.

## Verification, original failures and limitations

Evidence root: `/Users/pedrocotovio/Library/Caches/exact-om-review/readiness-closure-20261014`. Command arrays, exit statuses, JUnit/browser results, quality logs and sanitized binding records are retained there; the joint handoff accounts for A24-01–A24-08.

| Check | Recorded result |
|---|---|
| Frontend typecheck / production build | Passed; `frontend-checks-final.json`. |
| Frontend unit suite | **142 passed**, zero failures/skips; 2.912 s. |
| Focused browser regressions/cache checks | **14 passed**, zero failures/skips/retries; 47.774 s. |
| Final integrated browser matrix | **69 passed**, zero failures/skips/flaky tests; one worker, retries 0; 828.544 s. Includes all required four-file suites plus entry, visual, style, HR2 and static delivery regressions. |
| Baseline-first complete journey in matrix | Passed in 107.270 s; all 24 cases, both conditions, saved assessment recovery, per-case method changes, completion and analysis exports. Nine style boundaries passed. |
| Additional explanation-first complete journey | **1 passed**, zero failures/skips/retries; runner 91.919 s. Same complete browser requirements and nine style boundaries; no API-seeded training. |
| Actual HTTPS static delivery | **81 checks passed**; CSS GET/HEAD MIME, lengths/hashes, immutable cache, HTML/API/error protections, CSP/profile boundaries and independent subsequent reads. |
| Browser static caching | Main and study each observed one network transfer followed by one memory-cache load. Each complete journey observed CSS sources: one network, one memory, two disk; no failed/pending CSS transfers or export mismatches. |
| Focused backend delivery/serving/follow-up/image-content checks | **22 passed**. |
| Broad backend regression run | **393 passed, 2 failed, 3 skipped** (398 collected), 872.811 s. Failures preserved and explained below; this run is not clean. |
| Preservation test correction | **25 passed**, 10.456 s, after correcting the atime comparison in `fd47dca`. |
| Frozen runtime contract checks | Passed. |
| Import boundaries / docstring threshold | Five contracts kept, zero broken; docstrings 82.6% against 80% minimum. |
| Owned delivery files: pinned Black/isort and flake8 | Passed. |
| Full pinned Black/isort checks across CI paths | Failed on **19 Black files / 10 isort files** outside the owned correction; exact lists retained. |
| Full mypy | **48 errors in 13 files**; all error identities equal the baseline after normalizing line numbers. |
| Full flake8 | **52 entries**: 45 tracked baseline equivalents (normalize referenced F811 line numbers) plus seven findings in preserved untracked cluster/HermiT files absent from the clean baseline. The clean baseline has 722 entries, dominated by mechanical formatting. |
| Additional rendered-capture pass | **10 passed**, producing **25 PNGs** under `visual-final/`. The complete matrix already passed these visual/Axe/focus/overflow assertions; its optional `EXACT_SCREENSHOT_DIR` was absent, so this repeat supplied retained captures rather than replacing a failed matrix. The integration owner inspected all 25 final PNGs and the four focused HR2 negative-state captures, covering desktop light/dark, narrow 375px/200% and 320px/200% views and 375px/100%; no new visual blocker was observed. |

Rendered limits: at 200% text the narrow views require substantial vertical scrolling and wrapped IRIs; several viewport captures show controls above the content. This is bounded rendered inspection, not a manual screen-reader or participant-usability acceptance. The retained PNGs support later human review without replacing it.

The static regression reproduced the study middleware's immutable-to-private-no-store override before correction. Its initial public-demo setup also failed because the fixture lacked explicit development-demo admission; that harness error was corrected and is not another product defect. Original failure outputs are retained separately. The first frontend typecheck also caught TS7022 in the new pagination helper; explicit query/page types corrected it before the final passing build.

The broad regression's immutable-copy test compared complete `stat` tuples, incorrectly treating ordinary access-time changes caused by reading as content mutation. The corrected test preserves inode/link/size/mtime/ctime and content checks while ignoring atime, and its 25-test affected run passes. No immutable artifact or shared inode was changed to satisfy the test.

The other failure is a **measured unresolved synthetic resource-performance gate** in `test_handoff_binds_separate_public_http_measurements`. Its synthetic package is `sha256:4f7524cc0223c26eced9cfdfd196900c40bedf01d3d49455b0f8dbd958c3373c`, not the final real review package. Across 100 mixed TestClient reads and four threads, all responses were successful and bounded, but `context_p95_seconds=0.5156448660418391` exceeded the unchanged 0.5 s gate. The separate interpolated overall p95 was 0.5142434274312109 s; pair p95 was unavailable. Five of 25 entity-context reads exceeded 500 ms, with a 0.5827062600 s maximum. The handoff publisher correctly rejected a ledger declaring G4 passed. Unchanged verifier code does not establish inherited causation; service/context work, shared-client/thread overhead and host contention remain possible. The next discriminating experiment is one planned isolated-process baseline/final comparison on the saved fixture with the same four-reader workload and separate route/application/serialization/CPU observations, retaining both outcomes and the threshold. No blind retry or gate relaxation was used.

Three opt-in PostgreSQL process/recovery tests were skipped because their dedicated cluster variables were absent. They cover owned restart preservation and v1/v2 backup/restore. Prior isolated recovery evidence remains reusable within its recorded scope; these skips are not new recovery passes. Docker image execution and production-volume deployment/restore remain separate unverified technical release work.

The retained `tools/verify_service_reads.mjs` uses `ignoreHTTPSErrors: true` and does not explicitly disable cache; its API observations do not certify CSS caching or export hashes. The new static checks provide that evidence separately.

Actual client-close probes read only 1,024 bytes from a 429,952-byte JavaScript response and then closed the connection; server observations reported a terminal ASGI send. This is application-send completion, not TCP acknowledgment or proof that the client received the remaining bytes. Thus these are interruption attempts with successful subsequent reads, not proof of server-side interruption or reproduction of HR1. Pure-ASGI tests separately exercise cancellation/send failure and incomplete production without concealing exceptions.

Both final complete journeys retain wrong-answer recovery and ordinary cached reloads at the actual training/assessment boundaries. The original HR1 failed receipt and screenshots remain unchanged: all 19 evidence hashes in `explanation-human-review-readiness-20261013/receipt.json` and all eight in `explanation-readiness-frontend-independent-review-20261014/receipt.json` match, and their local evidence links resolve. A future reproduction should compare CSS completion/hash and safe server correlation with stylesheet lifecycle/computed-style events at the failing boundary; complete matching bytes point toward client lifecycle, while an incomplete correlated transfer calls for transport-boundary investigation. Historical R16 and profile-read incidents are not conflated with HR1.

Bounded API read probes passed during normal operation and after restart: **200 successful reads each**, plus ten intentional aborts, ten reloads with outstanding reads and ten stream-cancellation attempts per probe, with subsequent independent reads. Only expected intentional abort/cancellation failures were observed. Normal-operation maximum read latency was 647 ms; after-restart maximum 940 ms (diagnostic mixed-read p95 approximately 469/533 ms). These probes assert response/body/length and recovery invariants, not a latency acceptance gate. Client cancellation does not establish incomplete server transmission. [Normal probe](explanation-readiness-acceptance-20261014/normal-read-probe.json), [after-restart probe](explanation-readiness-acceptance-20261014/after-restart-read-probe.json). Retained diagnostic rotations remained 0600 and below 1 MiB per file with at most two backups; the [bounded-window audit](explanation-readiness-acceptance-20261014/diagnostics-audit.json) found no observed error types or incomplete finishes, and service logs had no recorded Content-Length protocol exception. Rotated windows do not prove that every historical event was observed.

## Reproducible private local operation

From the canonical checkout, use these exact values:

```sh
PY=/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python
WT=/Users/pedrocotovio/Library/Caches/exact-om-dev/readiness-closure-20261014/source-fd47dcac
E=/Users/pedrocotovio/Library/Caches/exact-om-review/readiness-closure-20261014/environment-fd47dcac
EXPORT=/Users/pedrocotovio/Library/Caches/exact-om-review/readiness-closure-20261014/frontend-fd47dcac
ASSETS=/Users/pedrocotovio/Library/Caches/exact-om-review/postmigration-20261011/combined
PGBIN=/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/lib/python3.12/site-packages/pgserver/pginstall/bin
export PYTHONDONTWRITEBYTECODE=1
"$PY" -B -m tools.review_storage_budget check --growth-gb 1.5
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -B -m tools.review_environment status --root "$E"
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" EXACT_REVIEW_READ_DIAGNOSTICS=1 "$PY" -B -c 'import resource; resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728)); resource.setrlimit(resource.RLIMIT_CORE,(0,0)); from tools.review_environment import main; main()' start --root "$E"
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -B -m tools.review_environment fresh-session --root "$E"
/usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -B -m tools.review_environment stop --root "$E"
```

Detached PostgreSQL and the two loopback HTTPS workers have an explicit **1,500,000,000-byte aggregate reservation**, 128 MiB per-file limits and core dumps disabled (`detached-reservation.json`). The file limit is not an aggregate quota. Start detached services outside the foreground storage wrapper only after the shared guard check; never daemonize a writer from inside that wrapper.

For a distinct new environment, choose a new durable root and three free ports; guard the foreground preparation with a conservative peak estimate. The retained inputs need no copying:

```sh
"$PY" -B -m tools.review_storage_budget run --growth-gb 1 -- /usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$WT" "$PY" -B -m tools.review_environment prepare --root "$E" --publication "$ASSETS/publication-definition-review-20261013.json" --assets "$ASSETS" --package "$ASSETS/package/package-definition-review-20261013.json" --frontend "$EXPORT" --python "$PY" --pgbin "$PGBIN" --postgres-port 19673 --study-port 19674 --exploration-port 19675
```

For this existing root, prepare verifies/reuses the saved binding; it does not reset the database. Source/export mutations require a separate environment instead of editing saved hashes. Final owned status/stop/start/resume/fresh-session outcomes: Owned status → stop → stopped status → start → ready status all succeeded. All **47 pre-existing test sessions in this new revision retained byte-identical saved state** across the restart. The lifecycle tool then created a fresh private development invitation. A separate actual-HTTPS recovery check used supported administrator reissue on one completed test session created by this pass, exchanged its invitation and read cookie-authenticated state: completed stage, all 24 cases and unchanged saved state. Missing-Origin exchange was explicitly rejected with 403. The small helper initially lacked its module path, then omitted the required Origin header; both failures are retained and were corrected in the harness, with no product/assertion change. That one owned test had two invitation rotations in total; historical/other sessions were not reissued. See [lifecycle](explanation-readiness-acceptance-20261014/lifecycle.json) and [resume](explanation-readiness-acceptance-20261014/post-restart-resume.json).

Main app: `https://127.0.0.1:19675/`; study origin: `https://127.0.0.1:19674/`. `fresh-session` writes a private test invitation under `$E/invitations/`; read it locally and keep it out of committed material. The private harness is `/Users/pedrocotovio/Library/Caches/exact-om-review/readiness-closure-20261014/harness.json`. Never print or commit credentials, invitation fragments, cookies or browser storage. These are development sessions, not participant research data.


The fresh unused test invitation is saved privately at `/Users/pedrocotovio/Library/Caches/exact-om-review/readiness-closure-20261014/environment-fd47dcac/invitations/e14e722b13c043d49d6209669fd12613.json`. Open that file locally to obtain the invitation; its secret is intentionally absent here. The certificate is `$E/tls/localhost.pem`; verify the recorded SHA-256 before trusting this local certificate in the owner browser. Browser trust was not modified by this task.

Automated browser checks used the recorded certificate's validated SPKI through `EXACT_E2E_CERT_SPKI`, plus `NODE_EXTRA_CA_CERTS` for the request client; normal application CSP remained active. This avoids treating certificate-error cache suppression as application behavior. Owner-browser trust in this local certificate remains an explicit local setup matter; no system-wide trust or public TLS deployment was performed.

Rollback uses the preserved predecessor `/Users/pedrocotovio/Library/Caches/exact-om-review/backend-integration-20261013/environment-frontend-fffa6d8`, its frozen source `/Users/pedrocotovio/Library/Caches/exact-om-dev/s2dev/source-fffa6d8`, and ports 19663–19665. Run that environment's ownership-checked lifecycle with its own source. Do not overwrite exports, adopt other services or modify historical publications/databases.

## Storage and delivery conclusion

Initial combined usage was **51,800,378,750 bytes**, with **117,650,501,632 bytes** filesystem free. Measured combined usage after verification/packaging was **52,310,274,470 bytes (52.310 GB)** at 2026-10-14T13:26:58.391672+00:00, with **114,955,427,840 bytes** filesystem free. The largest observed point sample was 52,310,274,470 bytes. A conservative simultaneous reservation of 3 GB (up to 1.5 GB foreground plus the 1.5 GB aggregate detached-service allowance) puts estimated peak usage at **55,310,274,470 bytes (55.310 GB)**. Initial combined usage was 51,800,378,750 bytes. All were below the 100 GB limit and retained the 5 GB safety / 30 GB filesystem-free reserves. These are measured samples plus conservative reservations, not a continuous peak or OS quota. The [storage receipt](explanation-readiness-acceptance-20261014/storage.json) includes all retained samples; evidence is about 4.2 MB. No full data bundle or second dependency environment was created, and no protected inode was modified. Retained runtimes were reused; a small pinned quality-tool overlay was installed in the existing verification cache. The shared 100,000,000,000-byte cap, 5 GB safety reserve and 30 GB filesystem-free reserve remain binding; all output roots stay in the combined inventory.

The focused behavior corrections and both complete browser orders pass on the recorded implementation. The joint handoff accounts for API preservation and lifecycle and retains unresolved HR1 and the measured synthetic G4 failure. There is no unqualified clean technical acceptance or green full-CI claim. No further frontend implementation blocker was demonstrated by the final 69-test matrix; remaining follow-up depends on new evidence rather than a presumed additional frontend round.

Technical release work includes unresolved incident/performance investigation, inherited quality failures, actual Docker execution, production-volume deployment/restore and the recorded packetless legacy-citation compatibility boundary. Manual screen-reader/accessibility work, participant usability/pilot evaluation, owner review and final matcher/case/ontology/study admission remain separate human/research gates. No public deployment, launch approval or participant contact is claimed.

The companion joint handoff maps every A24-01–A24-08 requirement to fresh or explicitly limited evidence. Full command arrays, exact changed paths, runtime-schema hashes and original failure digests are in the adjacent receipt. Documentation-only commits do not alter the recorded frozen service identity.
