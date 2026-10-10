# Backend S1 follow-up

**2026-10-10.** Starting checkout: `7a6028c032962d0afae0e7de5a479090f330f3e1`.
Implementation commit: `4f9dad94ecf288164e3d5f5ee922387e8822c4b2`.
This follows the [independent S1 review](explanation-product-review-backend-s1-review.md),
specifications 20–22 and the accepted S0 frontend. The accompanying
[receipt](explanation-product-review-backend-s1-followup/receipt.json) records the final
implementation identity, artifact hashes and verification status.

## Decision and acceptance boundary

The literal-origin defect is corrected, and a separately frozen real review bundle has
substantially more usable evidence. The existing publication and all recorded candidate
decisions are preserved. Frontend S2 can use the existing versioned contracts; no wire schema
changed in this follow-up. This is still a **qualified backend handoff**, not completed joint
acceptance or launch approval.

The user explicitly made real inference optional after the bounded ELK attempts. Both
attempts failed and produced no taxonomy. That exception does not turn failed classification
into verified inference. The remaining provider-authentication, native DOID import-admission,
coverage, performance and frontend gates remain explicit below. Long runs may continue after
handoff; a running check is never a passing receipt.

## Findings and dispositions

| Finding | Change and evidence | Disposition / owner |
|---|---|---|
| S1-01, P1 | Saved literal features often omit language or datatype. The old resolver treated omission as an explicit empty qualifier. The corrected resolver accepts only an unambiguous admitted literal for the exact typed focal subject, predicate and lexical value. Explicit conflicting qualifiers and foreign subjects remain rejected. | Backend defect corrected and native regression verified. Evidence coverage improved; unresolved originals and unsupported frontend projections remain visible. S2 owns rendering and comparison semantics. |
| S1-02, P1 → user-authorized optional exception | Tried native `pyelk-reasoner 0.2.1` with `pyowl-core 0.2.1`, strict unsupported-feature rejection, two workers, a 600-second wall budget and a 2-GiB parser budget. DOID failed native admission on `OBJECT_EXACT_CARDINALITY`; NCIT failed native parsing at its memory limit. | No inferred edges or taxonomy published. Optional under the user's latest instruction. Asserted/structural browsing remains available. |
| S1-03, P1 | Added a fail-fast real preflight: one comparison, its independent profiles and a missing-definition profile before the remaining approved batch. The first real request returned HTTP 401. The preflight stopped after that one attempt. | Blocked on a working OpenRouter credential. No remaining batch, prose approvals or independent prose claim audit occurred. Existing admitted typed descriptions remain labelled as such. |
| S1-04, P2 | Retrieved and verified all 15 pinned imports. Strict native admission still rejects the original DOID root; the declared root derivative parses, but admission with the original closure fails on RDF axiom reification. A two-file reproduction isolates declaration-dependent RDF mapping across imports. | Upstream native-parser boundary; no import is now missing from the retained input set. Original/full-closure admission remains open. The delivered derivative/root-only scope and download/context identities are unchanged. |
| S1-05, P2 | Inspected retained `current-12-root` and `restriction-profile-2` exports. They contain scored positive pairs, not the missing decision contrasts. A separate nine-pair, three-source DOID diagnostic smoke completed with three recorded exact-prefilter routes and six recorded below-threshold rejections. Its portable package passed all nine pair API requests. | New trace examples delivered separately; they do not rewrite historical NCIT/DOID events or become study cases. Sparse selected evidence, threshold-positive-but-not-emitted cases and selected universal/cardinality/inverse coverage are still not demonstrated. |
| S1-06, P2 | Reuse a readiness-verified decision-index digest only for the exact unchanged inode/stat identity. Verify study workspaces before readiness, including publication reuse. Deduplicate repeated preflight hashes within one invocation, retaining change detection. Add worker phase receipts and distinguish Git lookup timeout from failure. | Mechanisms and integrity regressions verified. Fresh service measurements are recorded separately; no claim that historical startup or reload findings are closed merely because unit tests pass. |

### Evidence recovery and graph coverage

The frozen 24-source/120-pair bundle contains **3,443 selected features**. Linked features
increase from **301 to 1,079**; **2,364 remain unresolved**. The diagnostic breakdown is:

- 355 retained origins fail admission for the exact focal subject;
- 133 retained relations name a different subject;
- 574 have no matching admitted focal relation origin;
- 1,302 literal records have no unique matching admitted literal. This category includes
  absent, filtered or ambiguous matches; it is not a claim that the unrestricted source lacks
  the statement.

No neighbour's fact, shared channel or convenient ontology record is substituted. Feature
IDs and recorded projection statuses remain unchanged. All 3,443 producer projection statuses
are still `not_recorded`; an offline frontend projection observation is separate evidence.

The actual S0 pure graph model, using hash-identical source and admitted original ASTs,
produced **1,049 drawn original-record occurrences, 31 `no_graph_form` occurrences and 2,364
unavailable records** with literals expanded. One feature can reference multiple originals,
so record counts and feature counts are deliberately different. This is not a browser render.
S0's automatic channel-derived comparison lines were excluded from acceptance.

There are five pairs with a projected source-only channel and 92 with multiple projected
channels on both sides. Examples include `review-04-candidate-2` (source-only attributes),
`review-03-candidate-1` (two both-sided channels) and `review-17-candidate-2` (17 drawn records).
The independent catalogue records seven preferred-label occurrences longer than 80 characters
and 16 pairs with linked expression originals. Those expression originals remain inspectable;
the current S0 model draws no quantified expression edges for them. S2 must preserve the
unsupported-form/list route instead of inventing simpler edges.

The independent publication audit passed all **260 prepared artifacts and 514 unique claims**
(529 displayed occurrences), with zero failures. It rechecked exact admitted support and
main/study artifact parity using its separate semantic rules. These remain typed template
claims; the audit did not approve provider prose. A full-pool comparison verified all **30,374
candidate pairs**, with zero changes to recorded scores, decisions or source records.

### Ontology and generation provenance

The admitted DOID input remains the declared annotation-declaration derivative
`f552e866a6233add0c02e5fed1ac1b0236c36c7f92247ed8c3a2a522ecc88fbb`, derived from original
`611355c445537fcf4bae2c519f1b3598af5a8fea793274316e35525b7d05e945`.
NCIT root remains `1a7182a7327ebc4181f7d6b0f7e81ed04dd258f1a86bd8f560e4a0d61439d58a`.
The context, download, policy and publication hashes are in the receipt and publication audit.
No original/derivative relabelling, new full-closure admission or description regeneration was
performed. Existing ontology-only facts and inference scope remain distinct from matcher scope.

The native minimal reproduction is in `tests/fixtures/explanation_doid_import_admission`:
an annotation declaration present only in an imported document yields
`RDF_MAPPING_INCOMPLETE`; putting the same declaration in the root succeeds with a complete
three-axiom snapshot. Run `python -m tools.reproduce_review_import_mapping` with that directory.
The full retained-input probe is `tools.review_ontology_admission`; its receipt distinguishes
original root, original closure, derivative root and derivative with original imports.
The required upstream work is closure-aware native RDF mapping/reification in pyOWLCore,
not another guessed ontology parser or silently partial admission.

ELK's DOID parse took 6.223 s and failed native admission after 5.029 s. NCIT's parse failed
after 13.485 s. The parser memory budget is not a process RSS ceiling: observed NCIT peak RSS
was about 4.08 GB. Both receipts have `taxonomy: null`. A small native ELK taxonomy probe worked;
it does not establish successful real-ontology classification.

The bounded OpenRouter scope remains 74 profiles and 12 comparisons. Attempt 3 changed only
the preparation selection-file binding to its current hash; packet identities, model, corrected
prompts and permitted job scope did not expand. One HTTP 401 was recorded, with no token usage
record or claim-review input. The credential had raw-key syntax; no credential is in Git,
receipts, logs or command arguments. Historical malformed HTTP-200 outputs were not normalized
into accepted claims. A working credential and successful preflight are required before the
remaining bounded preparation and independent semantic review can proceed.

## Verification and limitations

- The new literal regression fails against the starting commit's resolver and passes after
  correction. It covers omitted/explicit qualifiers, ambiguity, wrong subjects and entity kinds.
- Seventeen focused tests passed. Thirty-four follow-up verifier tests passed. The broader
  run passed 116 tests; two lifecycle checks were blocked by sandbox socket/process permissions,
  and PostgreSQL was initially skipped. A permitted dedicated PostgreSQL lifecycle rerun passed
  all 35 tests with zero skips. The initial failures are retained as infrastructure failures.
- The runtime schema/OpenAPI freeze check passed. No schema regeneration was needed.
- Fifteen affected frontend graph/tutorial unit checks passed. They do not replace rendered
  S2 acceptance or establish the correctness of S0's automatic comparison lines.
- The auxiliary smoke completed in 68.583 s, using two CPU threads, an existing offline encoder,
  a 12-GiB address-space cap and a 30-minute wall cap. It made no provider calls, performed no
  training and touched no final experiment outputs. Its self-pairs are diagnostic inputs, not
  independent alignment-effectiveness evidence. Its nine real ASGI pair responses passed.
  The exact-prefilter routes have actual completed producer events, not inferred history from
  protected flags. The producer still did not export an exact label/normalization basis. Their
  bypassed scoring is not counted as the missing sparse-selected-evidence stratum.
- Fresh HTTPS/PostgreSQL verification and startup receipts are listed in the machine receipt.
  A pending or failed receipt remains an open check. The prior independently verified 656-request
  journey belongs to the earlier publication and is not relabelled as a fresh result.
- The first new start failed exploration readiness within the unchanged timeout. Its study worker
  completed publication admission in 51.972 s and workspace readiness in 0.409 s (53.612 s total).
  The lifecycle tool stopped its newly owned processes and preserved the database. A background
  rerun is recorded at `<root>/environment-verification/status.json`; it performs startup, a fresh
  complete journey and restart/resume, leaving successful services running. Pending work is
  deliberately handed off rather than keeping the chat waiting for it.
  The rerun reached readiness in **106.066 s** with the preserved cluster/publication. This is
  fresh-worker readiness after the failed initial attempt, not a clean cold-publication result.
- Cloud-backed checkout reads blocked several checks, including Python bytecode, Git object
  reads and unrelated files during Git index refresh. Local bytecode caching and a materialized,
  hash-identical graph module avoided some of those delays. These observations do not prove the
  cause of the historical Git lookup failure or R16 reload stall.

Historical R16 remains separate. No middleware, dependency, Content-Length, CSP, authentication
or cancellation workaround was introduced. Manual accessibility, usability/pilot, owner approval,
deployment, final ontology admission and research launch gates remain open. No public push or
deployment was performed.

## Local artifacts and startup

The new preparation root is
`/Users/pedrocotovio/Library/Caches/exact-om-review/s1-followup-20261010`.
The immutable study/main input is `refreshed-input-v2`; the separate diagnostic main-app package
is `trace-review-v2/package/package.json`. Earlier staging failures are retained for diagnosis,
not advertised as successful publications. The original delivered environment is untouched.
The auxiliary package also has a local import archive, `trace-review-v2/trace-smoke.zip`.

Use the existing Python executable
`/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python` from the repository checkout.
The owned environment is `<root>/environment`, with PostgreSQL 19203, study HTTPS 19204 and
main HTTPS 19205. The machine receipt says whether those new listeners actually became ready.
Lifecycle commands are `python -m tools.review_environment start|status|stop --root <environment>`.
`fresh-session --root <environment>` creates a private test invitation receipt; do not publish
its token. The configured certificate is `<environment>/tls/localhost.pem`. API checks trust
that exact certificate; browser trust was not changed. A browser certificate gate is not bypassed.

The compiled frontend is accepted S0 from the prior delivered environment. Its missing
`study-review/1` negotiation still correctly produces 409 for forms/3 publications. That guard
must remain until S2 implements the contract. The existing real bundle may be inspected through
the main app, but a complete rendered participant journey requires S2.

Reproduction tools are `refresh_review_evidence`, `audit_review_publication`,
`verify_review_graph`, `run_review_trace_smoke`, `prepare_review_trace_package`,
`review_ontology_admission`, `review_elk_taxonomy` and `verify_review_environment` under `tools/`.
Each preparation requires a new destination. Original inputs, failed attempts and published
revisions are preserved. The selection sidecar remains the hash-bound one in the prior
environment's `verification/publication-audit-inputs/selection.json`.

## Exact frontend S2 work

Use the [existing S2 contract handoff](explanation-product-review-backend-s1-contracts.md)
and spec 21; no additional backend request field is required by this follow-up.

1. Negotiate `study-review/1`; render the server-ordered forms/3 questionnaires and optional,
   combinable external methods, including named Other and per-case consultation reports.
2. Discover current workspace scope IDs from state/case responses. New evidence scopes end
   in `-literal-v3`; never hard-code a prior scope or attach another publication's resources.
3. Preserve explicit typed tutorial navigation and admitted fact/evidence identities. Handle
   structured 422 details without advancing progress; retain assessment retry/recovery behavior.
4. Separate origin resolution, frontend projection and recorded comparison metadata. Use
   admitted originals for the richer recovered literals. Only explicit recorded bridge metadata
   can establish a comparison link; a shared channel is insufficient.
5. Keep list/record access for expression originals with no supported graph form; test the
   named source-only, multi-channel, dense and long-label cases from the catalogue.
6. Use entry-point and hierarchy basis/status APIs. Inference remains failed/unavailable under
   the user's exception. Do not display a loading spinner indefinitely or substitute asserted
   edges while labelling them inferred.
7. Render description preparation/support status honestly. There is no approved provider prose
   in this bundle. Original definitions, typed statements and missingness are distinguishable.
8. Present aggregate About metadata without exposing future study cases. Keep the auxiliary
   DOID trace smoke separate from the 24-case review publication and its scores.
9. Run a fresh complete rendered HTTPS journey after S2, including both conditions, consultation,
   completion, reload/cancellation, keyboard/screen-reader/manual accessibility and usable
   original-record/graph navigation. API success alone does not close this gate.

**Implementation correctness:** corrected boundaries and delivered evidence are supported by
the named checks, with the open findings above. **Joint acceptance:** open until S2 and fresh
rendered verification, plus remaining backend data/performance criteria. **Launch readiness:**
not established; no research invitations, final ontology admission or owner approval are implied.
