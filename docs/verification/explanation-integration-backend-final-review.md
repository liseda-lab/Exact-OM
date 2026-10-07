# Backend final integration review — 2026-10-04

**The two reproduced backend defects are corrected. Full acceptance remains open:** the
historical reload stall is still unexplained, J10 requires a manual accessibility audit,
and the release/study-owner gates remain. No frontend implementation change is required
by this review. This is synthetic integration evidence, not launch approval.

The [receipt](explanation-integration-backend-final-receipt.json) records commits, contract
and fixture hashes, commands, test results, diagnostics and limitations. This review
supplements the dated [S1 handoff](explanation-integration-backend-handoff.md),
[frontend follow-up 3](explanation-integration-frontend-followup-3.md) and
[acceptance record](explanation-integration-acceptance.md); their historical evidence is
preserved. Specifications 14–19 and the existing acceptance protocol govern this work.

## Reviewed checkout and changes

Started from clean tracked files on `dev` at
`2646d1f5d620f9a88f40308cbb6974b464c7a91e`, the independently reviewed frontend baseline.
The unrelated untracked `exact-repair-first-tests-20260929.tar.gz` was preserved.

| Commit | Content |
|---|---|
| `fa03f97b8384e2c2e347b6031e3019fd8c2447cb` | R14: explicit v2 action entity kinds, historical-resource compatibility, HTTP regressions, specification and generated OpenAPI |
| `8d5054ba8bca49ba3951d558bba05a6fe73985be` | R15: direct ASGI protection middleware and framing/cancellation regressions; integrated from isolated commit `416ce697` |
| `9bdd9ab44ad1bfbb866dade25c6c32a60daba50a` | Bounded HTTPS reload/disconnect tooling; integrated from `97e229f` and the counter-label clarification `bd2c35d` |
| Following documentation commit | This final review, receipt, contract guide and status updates; deliberately not self-referential |

The backend runtime tested in the final combined run is `8d5054b`; `9bdd9ab` adds only
diagnostic tools. The frontend source is unchanged from `2646d1f`. A separate reviewer
examined both fixes and the diagnostic tooling. That review found a historical-resource
compatibility risk and ambiguous diagnostic counter names; both were corrected before
delivery. The integrating backend agent executed the final combined suites. Independent
source review and executed integration tests are separate evidence, not interchangeable.

Parallel ownership was isolated: root owned action validation, schemas, handoff and the
root database/service (18973/18974); the disconnect investigator used a separate managed
checkout, index, branch, fixture and PostgreSQL/HTTPS service (19013/19014). The reviewer
was read-only. No shared mutable database or index was used by those agents.

## Findings and dispositions

| ID | Severity / owner | Root cause and disposition |
|---|---|---|
| R14 | P2 / backend | Shared `EntityRef` defaulted an omitted action kind to `class` before tutorial validation. **Fixed and verified:** `TutorialActionEntity` requires an explicit kind at the HTTP boundary; helper identity matching also fails closed. |
| R15 | P2 / backend | The `BaseHTTPMiddleware` response relay could append an empty final body after producer cancellation, despite an incomplete declared Content-Length. **Fixed and verified at the reproduced boundary:** direct ASGI protection preserves the producer's body/completion and original exception. |
| R16 | P2, provisional / backend integration, with frontend investigation if needed | Historical intermittent `page.reload()` stall. **Unresolved:** no stall was reproduced in this review, and no causal connection to R15 has been established. The h11 exception may have accompanied the original incident; these tests cannot decide that. |

R01/R02/R04 retain their scoped joint verification; R03 remains blocked on J10. R14/R15
closure does not close R16 or confer full joint acceptance.

### R14: request validation and compatibility

The new real-router regression fails on the archived `2646d1f` source: the parent action
without `entity.kind` returns **200 rather than 422**. The initial reproduction also
observed `context.parent` added to completed requirements. On the correction, omitted,
null and unknown kinds fail request validation; a valid kind attached to the wrong frozen
identity or ontology fails domain validation. Rejections leave state, history and mutation
receipts unchanged. Explicit recorded class and object-property evidence succeeds; all
four declared kinds are accepted by the action model, subject to exact scope membership.
An additional live HTTPS/PostgreSQL check verifies omitted/null/unknown/wrong typed
identity responses (422, unchanged state), followed by an explicit parent action (200).

The action-only model leaves the shared v1/publication model unchanged. Frozen publication,
resource and tutorial schemas are byte-identical to the baseline; only runtime OpenAPI
changes. Historically admitted resource files can omit class kinds. They are parsed through
their original `ExplanationResource` schema in memory before indexing, without changing
their bytes, hashes, publication or provenance. A regression admits such a resource and
successfully records a new explicit typed parent action while preserving its exact bytes
and republish receipt.

A historical omitted-kind action receipt is reconstructed using the old request models,
stored and recovered with a new store instance. An omitted-kind retry now returns 422;
resending its original normalized explicit kind under the same key reproduces the old
canonical request hash and immutable receipt. No stored receipts are rewritten. V1
publication/resume/export and v2 restart/backup/restore checks continue to pass. See the
[active contract guide](explanation-study-v2-backend-contract.md) for client behavior.

### R15/R16: exact failure boundary and bounded reproduction

On the unmodified baseline middleware, an ASGI producer declares **8 bytes**, emits
**4 bytes** with `more_body=true`, then raises `CancelledError`. The relay sends an
additional empty body with `more_body=false`; real h11 framing raises exactly
`LocalProtocolError: Too little data for declared Content-Length`. A separate baseline
regression shows cancellation before body bytes being swallowed. This is an isolated
producer/middleware/framing reproduction, not a reproduction of the historical browser
stall or proof of its original trigger.

The correction changes only the origin/security-header wrapper. It forwards the original
ASGI messages and exceptions; it does not remove Content-Length, pad/truncate bodies,
suppress failures, relax CSP, alter authentication or update dependencies. Tests cover
failure/cancellation before and after partial bytes, disconnected-send propagation and
producer cleanup, subsequent requests, exact document CSP, HEAD length/body behavior,
duplicate Set-Cookie, origin/auth checks and request-size limits.

The live diagnostic used the compiled frontend through the actual study app, self-signed
loopback HTTPS, PostgreSQL and normal CSP. It seeded consent/setup/background through the
API to reach the tutorial; this diagnostic is separate from the unseeded acceptance journey.

| Scenario | Repetitions | Result |
|---|---:|---|
| Normal tutorial reload | 20 | All loaded and restored the lesson |
| Reload after receiving a slow diagnostic stream's first chunk | 20 | All reloaded; original stream cancellation remained observable |
| Explicit AbortController after a received stream chunk | 20 | All rejected with AbortError |
| Deliberately cancelled 8-byte producer after 4 bytes | 20 | All browser reads failed; no false successful completion |
| Reload after receiving 1,024 bytes of an authenticated original tutorial resource | 20 | All reloaded; each subsequent full resource matched its reference hash and Content-Length, and state length/stage matched |

There were 81 subsequent health checks, all successful. The 40 standard-scenario reloads
took 195–325 ms (median 245.5 ms); the resource scenario's 21 reloads, including one setup
reload, took 200–387 ms (median 261 ms). Assertions used a 15-second reload and lesson
visibility deadline. No CSP violation or h11 exception occurred. The service recorded
40 slow-producer finalizations, 20 original **expected injected** CancelledErrors and 20
delayed-resource observations. Those injected exceptions remain visible; they are not
concealed or counted as clean production requests.

Instrumentation records producer bytes/completion, disconnect receive messages and
exception class. Producer counters are **attempted application output, not proof of
network delivery**. The browser independently verifies the first received chunk and
subsequent full resource hash/length. Resource delays change only delivery timing/chunking
of the original authenticated body. Diagnostic endpoints are confined to the synthetic
loopback launcher, not mounted in the production app.

These finite scenarios did not reproduce the historical stall. Retain R16 until a future
occurrence supplies correlated browser navigation/request timing, ASGI producer/transport
events and a service traceback that identifies the failing boundary. No backend/frontend
causal attribution or timeout/assertion weakening is justified by the current evidence.

## Verification and acceptance matrix

| Check | Result and scope |
|---|---|
| Combined backend suite | **168 passed, 2 skipped** on `8d5054b`; skips are exclusively the two PostgreSQL recovery cases |
| PostgreSQL core suite | 68 passed plus one harness failure: the configured receipt directory was outside the test's permitted synthetic workspace. Restart/restore assertions preceding that directory guard had passed. |
| PostgreSQL corrected rerun | **9 passed**, no skips, on the combined fixes: all five action-contract tests, corrective HTTP tests and both v1/v2 restart/backup/restore tests. Every one of the 69 distinct core cases now has passing evidence. |
| Frontend unit/build | **75 passed**; TypeScript check and Next production export passed. Initial scratch unit invocation lacked the checked-in backend example file; copying that unchanged fixture resolved the setup error. |
| Combined browser suite | **42 passed**, no skips/retries: full fresh UI journey and every current v2 integration regression, using the compiled frontend and unmodified final service |
| Contract/specification checks | Runtime snapshots match generated models; spec validator passes; focused lint/syntax and scoped diff checks pass |
| Release package/container | Not verified: designated real release package absent; Docker daemon unavailable, despite the installed CLI |

The final fresh journey begins with a new invitation and performs consent, tool-neutral
setup, background, all six training lessons, all five assessed items with an incorrect
retry, both study conditions, per-case consultation (including multiple methods and none),
final questionnaire and completion. It does not use API-completed training or substituted
success responses. Browser faults/delays and the seeded disconnect diagnostics are labelled
separately in the receipt.

| Acceptance IDs | Evidence and limit |
|---|---|
| J01/J02 | Rechecked descriptor/capability contracts, authorization, invitation/session generation, stage/condition, stale scope and cursor isolation. Browser baseline/help/stale-scope subset passes. Other security permutations are backend evidence. |
| J03 | Current browser navigation, paging, typed parents, exact facts/citations and full context pass. Main-app comparison evidence is reused from frontend follow-up 3; the main app was not rerun here and its implementation is unchanged. |
| J04/J05 | Required content, malformed/failed responses, readiness acknowledgements, retries, drafts, late access checks, session replacement, second tabs and pause/resume pass. Seeded/intercepted failures are not happy-path acceptance evidence. |
| J06/J07 | Durable lesson/assessment positions, conflicts/offline recovery and the complete fresh journey pass. PostgreSQL verifies persistence across actual restart and backup/restore. |
| J08/J09 | Timing oracles, JSON/CSV analysis-3 unknown-coverage semantics, consultation exports, v1 compatibility, immutable old exports and restored state pass. Page-seconds are not presented as unique elapsed coverage or attention. |
| J10 | Automated focus, axe, light/dark reflow and actual 320–2560 px widths at 100%/200% text pass. **Manual screen-reader and keyboard audit not performed.** |
| J11 | Study CSP, forms, ranking, consultation and researcher exports rechecked. Main-app/library/demo and v1 browser results are reused from the independent frontend evidence; v1 backend regressions were rerun. |
| J12 | Build, contracts, relevant suites and focused review pass within this scope. External release and human gates remain open. |

## Environment and reproduction

Python 3.12.14; FastAPI 0.116.2; Starlette 0.48.0; uvicorn 0.35.0; h11 0.16.0;
anyio 4.15.1; Pydantic 2.13.5; psycopg 3.3.6; PostgreSQL 16.2; pyowl-core 0.2.1.
Frontend: Node 24.19.0, Next 15.5.26, React 19.1.0, Playwright 1.63.0 and
Chrome 154.0.8037.97. Locked frontend dependencies and the build copy were outside iCloud;
the repository remained the sole source. Local certificates are self-signed, with browser
certificate checking bypassed only for the local harness; application CSP was unchanged.

The fresh 59-file synthetic fixture was built with `--navigation-size 65 --paged-facts 26`.
Publication SHA-256: `0f6205be34f125a9f3f8385d77ac9aab06f57df4c78560a5f7a413fc6f156a15`.
Admitted publication hash: `07af91d5fb946491ac072f04de5ad7f604156f478d2e4ff1eca5a253c48872f3`.
The receipt includes the revision, inventory, variant hashes, tutorial hash and all four
runtime contract hashes. Native preparation succeeds on this synthetic fixture; it does
not establish admission of the real biomedical release inputs.

```sh
python -m tools.build_explanation_v2_fixture <empty-fixture-dir> --navigation-size 65 --paged-facts 26
python -m tools.serve_explanation_v2_e2e --fixture <fixture-dir> \
  --database-url <dedicated-postgresql-url> --tls-cert <cert> --tls-key <key> \
  --config <private-config> --port 18974 --frontend-dir <compiled-out>
EXACT_E2E_STUDY_V2_CONFIG=<private-config> EXACT_CHROMIUM_PATH=<chrome> \
  npx playwright test e2e/study-v2-backend.spec.ts e2e/study-v2-integration.spec.ts
python -m tools.freeze_explanation_contracts --check
python specs/explanation-framework/protocol/validate_specs.py
```

Run Playwright from the frontend directory. Build/typecheck/unit commands are its checked-in
package scripts; include `docs/verification/explanation-integration-backend-examples.json`
when copying it to a scratch parent. The receipt lists exact backend suite files and
PostgreSQL environment variables. Recovery evidence must use a directory below
`data/explanation-framework/postgres/`; only explicitly marked synthetic clusters may be
restarted. Never run these recovery tests against participant databases.

For bounded fault investigation, substitute `python -m tools.review_study_disconnects`
for the synthetic launcher above on a separate port/database, then run:

```sh
node tools/review_study_reload.mjs <private-config> <frontend-with-node_modules> <chrome> <receipt.json> 20 standard
node tools/review_study_reload.mjs <private-config> <frontend-with-node_modules> <chrome> <receipt.json> 20 resource
```

Keep harness credentials and full raw local logs private; no tokens, invitation URLs,
participant records or dumps are committed. The checked-in receipt contains only
synthetic identifiers, summaries and artifact hashes. Rebuilding produces new revision
IDs/timestamps and consequently new hashes.

## Handoff and remaining gates

**Implementation correctness:** R14 and the reproduced R15 boundary are corrected, reviewed
and regression-verified. Current frontend typed actions remain compatible; no frontend
code follow-up is required. Older clients emitting implicit kinds must send their recorded
type, including retries. R16 remains an unresolved integration finding.

**Joint acceptance:** the fresh combined journey and affected automated matrix pass. Full
acceptance still requires R16 disposition and the J10 manual screen-reader/keyboard audit,
including the changed parent links and recovery controls. The current work does not
replace the human audit or claim fresh execution of the unchanged main-app/v1 browser suite.

**Launch readiness:** not established. Still required:

- The admitted real release package at `data/explanation-framework/backend-release/package/package.json`,
  with its matching frozen contexts, explanations, resources and admission evidence. It is absent;
  the older deployment bundle is not an equivalent input.
- An available Docker daemon and image build/smoke tests, trusted-HTTPS deployment and
  deployment backup/restore verification. The local Docker socket is unavailable; no image
  or public deployment was attempted.
- Manual accessibility, formative usability and duration pilot.
- Study-owner consent/information approval and real case adjudication.
- Original pinned DOID admission and complete import-closure resolution under the existing
  gates; synthetic native preparation does not waive them.

External inspection remains optional and combinable, reported separately per case.
Protégé remains a recommendation. No push, public deployment, recruitment or participant
contact was performed.
