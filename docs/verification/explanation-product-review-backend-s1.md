# Backend S1 implementation and interim development-data review

## Scope and acceptance boundary

This delivery supplies the backend B01–B07 runtime corrections and an interim development
review environment from specs 20–22, starting at
`ffd390f35c108abe47f5603b91d28bd1a54e590d`. That is the accepted frontend S0 handoff;
the independent S0 follow-up review remains historical evidence for its exercised browser
paths. This work does not rerun or change the ongoing matching experiments.

The runtime is a **development review**, with real provisional NCIT–DOID cases, test sessions,
unresolved correctness and disabled correctness metrics. Its tutorial remains synthetic and
disjoint. It cannot issue research invitations or silently become a final study publication.
The existing frontend intentionally cannot negotiate forms/3 until S2 implements those forms.

Implementation correctness, joint product acceptance and launch readiness are separate.
Implemented contracts have automated semantic/API evidence; complete real-data acceptance is
not claimed. B02 has selected-coverage and original-DOID admission gaps, B03 lacks a completed real
inferred taxonomy, and B04 lacks successfully reviewed provider prose. Real ontology inference has
recorded failures, and the original DOID/import admission gate remains open. Frontend S2 and
the fresh integrated browser/visual review in S3 are not completed by a backend API journey.
S2 can integrate these frozen contracts now using the explicitly typed preparation and
unavailable-inference states; these remaining backend data gaps must be closed for full
Q05–Q07 acceptance. Human accessibility/usability audits and study-owner approval remain later follow-ups; they
were not used to stop independent engineering work.

## Reviewed and implementation revisions

The starting checkout was `ffd390f35c108abe47f5603b91d28bd1a54e590d`. The runtime used for
final service verification is `a06af6ec1e6e4064e610cb3f2751bdb7e52d9a8b`; later handoff/status
commits do not change its service fingerprint. The machine receipt lists every implementation
commit since the starting checkout. Work was integrated through a task-owned checkout, with
one Git index owner. The final hierarchy verifier is committed at `3a1ecb7ab2967b5955b76864aa68a2cf7456965d`; it does not change the served runtime fingerprint. The task-owned history is integrated into the requested checkout without staging its pre-existing specification, research or frontend-review changes. No push is performed. Specs 20–22 and the independent S0 review were supplied as
pre-existing uncommitted files in the requested checkout; their reviewed hashes are recorded,
and this work leaves those files and the other pre-existing edits unstaged.

## Changes

| Requirement | Delivered behavior and remaining limit |
|---|---|
| B01 | Typed action and admitted-fact validation; optional evidence identity; frozen tutorial reachability checks; persisted lesson/assessment position and restored completed-requirement IDs; additive safe validation identifiers and correlation IDs. Rejection is transactional, replay keeps its original identity, and valid next actions remain usable. Historical publications are read under their frozen questionnaire materialization. |
| B02 | A real completed development run, full recorded candidate pools, a deliberately varied 24-source selection with five recorded targets per source, shared policy/context identities, public-safe About metadata and unresolved review keys. Selection gaps and the DOID derivative/root-only scope are explicit. |
| B03 | Offline native hierarchy preparation and portable indexed entry points; separate asserted, structural and inferred semantics, directness, typed identities and status. Synthetic semantic oracles pass. Bounded real NCIT/DOID runs did not finish; no completed real inferred taxonomy is claimed. |
| B04 | Versioned typed descriptions plus a receipt-bound independent-review path for provider drafts; no real provider prose was approved in this bundle. Citation membership alone never approves prose. Review receipts bind the complete source generation manifest, packet, exact claim inventory and response. Legacy artifacts remain byte-stable. Runtime failures/unreviewed drafts have labelled factual fallback branches; this delivered publication uses only typed template support. |
| B05 | Recorded exact-match route metadata; explicit fact-resolution and graph-projection statuses; policy filtering both at new admission and legacy reads; typed subject identity and retained tombstones/counts for unavailable evidence. No channel union is promoted to feature correspondence. |
| B06 | Frozen forms/3, setup/3, consultation/3, methods/3 and analysis/4; requested experience bands and Other branches; optional combinable external methods; conditional clearing; researcher-only named-method export with spreadsheet escaping. Legacy forms and codebooks remain supported. |
| B07 | Durable local preparation and owned PostgreSQL/HTTPS service lifecycle, compiled frontend identity, private invitation state, explicit fresh-session/revision behavior, retained database state, TLS verification and actionable diagnostics. |

See [S2 contracts](explanation-product-review-backend-s1-contracts.md) for exact wire examples,
routes, negotiation, forms and export semantics. The [actual response excerpts](explanation-product-review-backend-s1/s2-response-examples.json)
contain ten bounded public HTTPS examples and the delivered public form definitions, with
full-response hashes and extraction pointers; they are not complete response validation fixtures. Runtime OpenAPI and JSON schemas are generated,
not hand-maintained substitutes for the implementation.

The recorded implementation conclusion is that the affected backend contracts and safety
boundaries passed their exercised tests and independent reviews. This is a qualified backend
handoff with explicit real-data/preparation gaps. Joint acceptance remains pending S2/S3 and
those gaps; research launch remains unapproved and outside this delivery.

## Requirement ownership and delivery commits

| Requirement / review issue | Contract or implementation commits | Verification boundary / next owner |
|---|---|---|
| B01 / U01–U04 | `758082a`, `016976d`, subsequent runtime fixes through `394af1a` | Actual API rejection/replay/reachability tests; frontend S2 implements guided return/position presentation and preserves S0 assessment recovery. |
| B02 / U09/U12/U18 | `178b8f2`, `00b7a59`, `66f42ce`, `43e8fc2`, `4c9a8c9`, `394af1a` | Real 24-case publication plus independent full-pool and typed-claim audits; backend data owner retains original DOID/import and coverage gaps. |
| B03 / U10–U12 | `365558c`, `00b7a59` | Indexed API and native semantic oracles pass; bounded real preparation failed. Backend data owner must complete real classification; S2 must render the recorded unavailable state. |
| B04 / U06/U08 | `178b8f2`, `66f42ce` | Typed descriptions and provenance rejection tests; no approved real provider prose. Backend preparation owns the authorized corrected batch and semantic review; S2 consumes explicit preparation/support states. |
| B05 / U07/U15 | `758082a`, `7c37436`, `43e8fc2`, `394af1a` | Recorded trace and exact metadata boundaries checked on real and synthetic evidence. S2 owns graph/list presentation using those statuses and correspondence levels. |
| B06 / U13/U14 | `016976d`, `a964a24`, `4c9a8c9` | HTTP, PostgreSQL, persistence and restricted JSON/CSV regressions; S2 implements frozen new forms and explicit negotiation. |
| B07 / U17 | `7c36bf4`, `2682b33`, `66f42ce`, `b70f340`, `a06af6e` | Owned local lifecycle and immutable identity checks; final relocation receipt records cold/warm and real API scope. Owner uses the maintained local runbook; deployment acceptance remains separate. |

These are implementation entry points, not assertions that each requirement has full product
acceptance. The receipt enumerates the complete commit range and frozen contract hashes.

## Real input provenance and coverage

The completed run is `g0-validation-10/local300/run` on the existing development host, ended
2026-09-18T12:13:43.099568+00:00, source commit `c86f540a21fab81d123f9d7b2656a73b17e82632`
with a recorded dirty worktree. Its source-tree and worktree digests are retained in the receipt.
The full candidate universe has 300 sources and 30,374 pairs. The upstream population was
development/validation material sampled with seed 17 and cap 300; it is not a sealed test set.
The new review selection does not consult reference labels, but this does not make the upstream
reference-based population an unbiased evaluation sample.

The review chooses descending recorded `S_final`, breaking ties by recorded candidate-array
position. Array position is not a reconstructed production rank. Unresolved review keys leave
unrecorded production ranks empty; research publication admission retains its stricter rule.
Full candidate pools remain browsable and scores are not renormalized.

| Source stratum | Selected sources | Eligible sources observed |
|---|---:|---:|
| Rich definitions | 23 | 226 |
| Missing/asymmetric definitions | 5 | 162 |
| Multiple inheritance | 20 | 216 |
| Restriction/equivalence-heavy context | 8 | 130 |
| Close alternatives | 16 | 124 |
| Sparse selected matcher evidence | 0 | 0 |

Tags overlap. Eligible counts come from the selector’s recorded top-five candidate windows;
absent feature exports remain unknown and are excluded from the sparse-evidence count.
All 120 selected top-five pairs are recorded emitted/threshold-positive pairs;
8 have saved protected flags and 112 do not. All 120 delivered traces report the
`prefilter` stage and `exact_basis` as `not_recorded`; the saved flags are not exported
exact-route events or rule/basis evidence. A separate [retained producer audit](explanation-product-review-backend-s1/raw-selected-stage-audit.json) matched all 120 typed source-decision candidates and 112 retained explanation records. No per-pair prefilter event or exact basis was retained; `candidate_decisions.json` is absent and undeclared. The eight absent explanations coincide with the eight saved protected flags. This establishes historical export absence, not that the stage never ran. Real exact-stage trace coverage is therefore missing. Rejected/differing-final-decision coverage is
missing from this selected study set. That gap is not concealed by the presence of other
candidates in the full run. Per-pair resolution/projection counts and identities are retained
in the machine-readable catalogue. Matcher-to-original fact resolution is prepared for the
selected typed review entities; most evidence outside that prepared set remains explicitly
unresolved. The full ontology context is still independently indexed and browsable. Counts
of unprepared mappings are not presented as zero selected evidence.

The final independent audit verified 140 unique entity profiles and 120 comparisons: 260
unique artifacts, 514 unique supported claims and 529 claim occurrences across study workspaces.
It checked 1,038 typed packet originals and 394 distinct cited fact IDs against the actual
context indexes, using separately implemented semantic rules, with no failures. Repeated entity
profiles account for 264 result occurrences. Main and study payloads match except the optional-field
`independent_review: null` serialization projection. All artifacts use typed template
support; there is no approved provider prose in this publication.

The selected cases retain 3,443 evidence records: 301 have admitted focal original links and
3,142 are unresolved. Eight pairs have absent feature exports and remain unknown. Raw inventory
has 112 pairs with multiple channels on both sides and 11 with a source-only channel; neither
coverage pattern survives as multiple/source-only **admitted original links**. All 3,443 graph
projection statuses are `not_recorded`, not asserted supported or unsupported. Recorded comparison
identities comprise 688 channel comparisons and 2,641 feature correspondences; 242 bridges are
available as recorded and 3,087 are withheld. Identity resolution alone is not permission to draw
a bridge with unavailable original support.

Available linked expression originals occur in 16 pairs across `review-01`, `review-02`,
`review-12` and `review-18`. Seven preferred display-label occurrences exceed 80 characters in
`review-13`, `review-20`, `review-21` and `review-22`; packet labels have a separate count of 11.
The [owner catalogue](explanation-product-review-backend-s1/case-catalogue.md) gives exact case,
entity, fact and target identities, scores and per-pair counts. These are useful review inputs,
but they do not satisfy full graph-coverage acceptance. Real universal/cardinality/inverse
restriction coverage is absent here; those semantics were tested only with fixtures.

The admitted NCIT context contains 212,055 typed entities and 1,798,045 axioms; DOID contains
16,395 entities and 91,530 axioms. These are complete indexes for the declared policy scope,
not snippets selected around the study cases. NCIT has no imports in its recorded root.
DOID uses the explicitly identified annotation-declaration derivative, with the original and
derivative hashes and transformation receipt retained. Its ignored import `http://purl.obolibrary.org/obo/doid/obo/ext.owl` has no admitted
source hash and remains outside this context. Original DOID strict parsing and a pinned,
verified import closure are still required. This is an interim root-only milestone, not closure of original-input acceptance.

Three ontology-only DOID examples (`DOID_0050155`, `DOID_0050161`, `DOID_0050177`) each show
one asserted versus two structural parent edges. Real inferred comparisons are unavailable:
the DOID bounded 300-second and 900-second runs timed out; the NCIT 300-second run exhausted
its budget during preparation. Failure manifests retain phase timings, package/options/input
identities and zero exported inferred edges. Counts remain unknown where classification failed.

## Findings corrected during implementation review

| Finding | Severity / owner | Cause and disposition |
|---|---|---|
| Future review cases disclosed through About metadata | P1 / backend isolation | Actual HTTP reproduction exposed the whole catalogue before assignment. Participant state now receives aggregate-only metadata (`cases: []`) on fresh responses and old mutation replay; full owner catalogue, immutable publication bytes, stored receipts and request hashes are retained. |
| Implicit or cross-scope tutorial evidence | P1 / backend contract | Strict typed/action-specific admission preserved; no default fact or class is introduced. Actual API regressions check atomic rejection and subsequent success. |
| Unresolved correctness admitted outside review purpose | P1 / backend | Shared key enums exposed a missing purpose guard. Both unresolved key kind and origin now require development review. HTTP rejection leaves no study row. |
| Reviewed prose envelope could be detached from complete generation provenance | P1 / backend | Added full manifest/review/input validation including nested model revalidation and forgery cases. |
| Selected evidence promoted neighbour-origin facts to focal citations | P1 / backend evidence | Actual first-case publication rejected 16 links whose original axioms did not reference the focal typed entity. The general resolver intentionally supports neighbouring origins; selected-evidence admission/read/export lacked the narrower subject boundary. Retain the raw records, selection counts and strict workspace check; require the focal typed binding and preserve unsupported links as explicit unresolved records. The public policy export is rebuilt under a new exporter identity. |
| Legacy workspace evidence could expose unadmitted selected text | P1 / backend | Apply the same policy filter on admission and reads, retaining missingness tombstones and original frozen bytes. |
| Filtered full ontology export encountered absent original syntax | P1 / preparation | Policy-filtered contexts deliberately omit raw syntax. Reconstruct only allowlisted native typed nodes and verify canonical/typed round-trip agreement, rather than parsing another syntax or restoring hidden content. |
| Position-only tutorial save failed revalidation | P2 / backend | Default-filled serialization introduced an omitted legacy alias. Revalidation now retains original field presence while historical request hashes remain unchanged. |
| Structured 422 changed legacy detail shape and misidentified nested array indices | P2 / backend | Retain sanitized legacy detail and derive action indices only immediately after `actions`; regression includes multiple nested indices. |
| New runtime modules absent from study image | P1 / packaging | Explicit image copy closure and cold-import checks updated; no matcher/native runtime dependency is imported by the cold study service. |
| Full saved candidate pool was incomplete in the exploration index | P1 / backend | The importer unioned explanations and mapping/decision records but omitted 155 source-decision candidates without explanations. Added the saved typed candidate universe, retaining exact source-record locations and allowlisted scores; missing stage events remain `not_recorded`, while evidence exports remain `not_exported`. Saved array position is not a rank. |
| Manifest-declared global mapping filename was ignored | P1 / backend provenance | The real manifest retained `alignment/paper.maps_global.tsv`, while the reader looked only for its canonical filename. Bind the unique declared mapping path and hash through the supported reader; do not rename or invent producer history. |
| Cross-side comparison evidence was retained but not imported | P1 / backend | The historical producer saved `cross_side_provenance`, while the adapter read only the newer `evidence_bridges` form. Resolve exact recorded aliases by side, typed entity, channel and family; preserve unresolved/ambiguous records and distinguish entity/channel comparisons from two-feature correspondences. Main and authorized study now share the same policy-filtered metadata. |
| Cached preparation result belonged to another request | P1 / preparation | Independent review reproduced a copied-result cache substitution. Bind cached request, receipt and result to the exact job identity before reuse; the regression uses local stubs and makes no provider requests. |
| Startup kept waiting after an owned worker exited | P2 / local review lifecycle | The readiness loop retried connection errors after the child had already terminated, consuming the full admission timeout. Check the recorded process ownership before and after readiness probes, fail with a safe role-specific diagnostic, and retain rollback of only newly started owned roles. This does not identify or suppress the separate Git-identity failure. |
| Real hierarchy verification assumed the first focal class had asserted edges | P2 / verification | Both services correctly returned empty asserted pages for the first NCIT focal class. Probe up to 200 unique frozen typed classes, preserving status/count/edge/continuation parity even on empty probes; unavailable responses still fail. Regressions cover a valid later class, cross-service disagreement and unavailable status. Preserve the earlier partial journey and bind reused full-inventory evidence by hash. |
| Portable offline audit omitted selection metadata | P2 / review packaging | The runtime input whitelist correctly omitted an owner-only file, but the documented audit expected `selection.json` beside that input. Deliver the exact selection as a separate owner-audit sidecar and pass it explicitly; retain mandatory equality to the frozen publication’s selection hash. Preserve the 696 runtime input files and original inventory, and rerun the audit against the delivered copy. |
| Provider prose omitted required claim structure | P1 / preparation | The first prose prompt did not state the full JSON shape. Preserve malformed saved outputs; add a new prompt identity with exact field/category rules and transport-boundary regressions. No permissive normalization or semantic self-approval. |

Two relocated cold-start attempts failed before serving a request: the study worker
reported unavailable backend Git identity, while an immediate direct identity check against
the same checkout succeeded. Intermittent slow Git reads were also observed in this host's
managed checkout; that is context, not proof of the worker failure's cause. Both failed attempts
are retained separately from successful runs. Six later instrumented worker-entrypoint probes
(three with the source path present and three hidden) completed the same Git command in
0.010–0.013 seconds; the original Git failure cause remains unresolved. No data admission check was bypassed.

The historical R16 intermittent browser reload stall remains unresolved. API early-close and
subsequent full-response checks do not establish the cause of that browser finding. This S1
delivery makes no new middleware/dependency change to claim it fixed.

## Verification and limitations

The accompanying receipt records exact source/build/data hashes, commands, timings, failures,
skips and independent agent reviews. Test totals below must not be summed as distinct coverage.

- Broad backend run: 382 passed, three failed and three skipped. The three failures were
  investigated and corrected; a focused rerun passed 19 checks. The original failure receipt
  is retained. Two Torch-dependent modules could not collect in the available runtime and
  are recorded separately, not counted as passed.
- Participant About disclosure correction: the populated-catalogue HTTP test failed before
  correction, then passed; 33 focused PostgreSQL/API/verifier checks passed after correction.
  The intermediate rerun exposed a test setup mistake (tuple unpacking), which was corrected;
  its failure receipt remains preserved. Twelve other runtime checks passed and the dedicated
  PostgreSQL backup/restore check was skipped in that SQLite invocation, covered separately below.
- Dedicated PostgreSQL regression suite: 138 passed without skips. Additional lifecycle identity
  checks passed 33, and pair metadata checks passed 17. These use separate disposable owned
  clusters, which were stopped after verification.
- Integrated runtime rerun: 44 passed, covering actual API publication, typed tutorial evidence,
  description support/provenance, policy isolation, filtered exports and image closure.
- Latest candidate-pool, saved mapping, preparation-mode and environment-verifier checks:
  57 passed. The first run had one mocked response missing the newly required CSP; the fixture
  now supplies the real policy and the privacy/status assertion is unchanged.
- Typed evidence origins and historical bridge metadata: 34 focused API/export/cache checks
  passed after reproducing the real publication failure. Independent review repeated 19 affected
  checks. The generic neighbour resolver remains supported; publication/read boundaries keep
  strict focal identity and reject foreign evidence, typed pairs, channels, families and facts.
- Local startup failure handling: 34 focused checks passed, one real-PostgreSQL
  check deselected and covered by the dedicated PostgreSQL suites. Independent review
  confirmed that pre-existing database processes are outside startup rollback.
- Corrected hierarchy probe verifier: 52 checks passed. The first-empty-class regression failed against the old verifier before correction; unavailable hierarchy and cross-service mismatch still fail.
- Cache-lineage rejection: 16 passed after independent review reproduced a copied-result defect.
  The preparation suite also passed 23 checks; totals overlap.
- Corrected prompt transport/cache regressions: five passed. These are mocked transport tests;
  real provider attempts are accounted for separately, and none produced approved prose.
  The final typed-claim audit has its own receipt. Its first harness attempt
  expected identical omission of an optional null field; an exhaustive 264-result comparison
  isolated that serialization difference, and the narrow correction was rerun successfully
  without changing publication bytes. The initial maintained-checkout repeat passed in 37.16 seconds. A final repeat against the relocated copy with its explicit selection sidecar is passed in 41.659 seconds with unchanged substantive results and all 28 imported-module hashes matching the earlier success. The previously missing-sidecar attempt and import-order-only executable change are retained in execution history.
- Accepted frontend source copied to persistent local build storage: 112 unit tests, type check
  and compiled production build passed. It is the S0 frontend, not an S2 implementation.
- Specification artifact validator passed: 30 JSON files, 13 schema fixtures, 242 links,
  eight study schedules and nine scoring oracles. This validates the supplied working-tree specifications, including the pre-existing
  uncommitted amendments; it is not service acceptance.
- Native hierarchy/entry-point semantic fixtures exercise direct versus transitive edges,
  equivalence, cycles, multiple inheritance, unsatisfiable classes, imports and typed identities.
  Real native runs are reported separately above.

Successful API verification does not establish browser layout, accessibility or usability.
No final matching experiment, remote deployment, public push, research invitation or real
participant collection is part of this work.

## Delivered service verification

The final environment uses runtime `a06af6ec1e6e4064e610cb3f2751bdb7e52d9a8b`, dedicated
loopback PostgreSQL and verified HTTPS. Its input copy was served with the original preparation
path hidden. Copying took 63.537 seconds, preparation 57.594 seconds and the first complete
start 130.798 seconds. The cold-start HTML bodies matched the exact compiled files, full
content lengths and configured CSP; three post-readiness probes per service took about
1.6–4.8 ms. Peak observed owned-main-process RSS was 135,148 KiB for study and 75,120 KiB
for exploration. Descendant process memory is not summed. “Cold” means first process launch
against fresh PostgreSQL; the OS file cache was not cleared. These measurements are not a
claim that the historical ten-second cold-health target passed.

Warm restart passed in 93.498 seconds against the preserved PostgreSQL data. The supported
Codex in-app browser refused the main URL with `net::ERR_CERT_AUTHORITY_INVALID` before
rendering any page. No warning was bypassed, no trust store changed and no global TLS setting
weakened. There is therefore no new rendered screenshot or browser smoke pass. Trusting this
specific local certificate through the documented owner workflow is necessary before that
browser check can run. Separately, the exact-certificate API client completed strict TLS
handshakes. The old S0 participant UI also lacks forms/3 negotiation until S2.

The first API attempt traversed all 4,570 inventory pages, with full ordered typed-entity parity across both services. It stopped after 681.78 seconds on the verifier’s first-class hierarchy assumption described above. That is a retained partial run, not a successful complete journey. The corrected fresh journey reuses only that hash-bound exhaustive inventory evidence and reruns navigation, search and the full participant workflow. The corrected fresh API journey passed in 178.173 seconds: 632 real HTTPS requests, all 24 cases (12 explanation and 12 ontology baseline), actual 120.009-second tutorial dwell, five rejected and five correct assessment attempts, consultation and completion. All 98 observed participant states withheld the owner catalogue. Default and explicit named-method JSON/CSV exports retained their schema/immutability checks. Three responses were closed after 128 bytes; subsequent requests returned the identical complete 38,106-byte bodies. The observed study log had no h11, content-length, ASGI or traceback signature. These early-close probes do not prove server-side cancellation or browser reload behavior.

Search and nonempty hierarchy parity passed on both full ontologies. The successful asserted probes were DOID’s first and NCIT’s third recorded class; both bounded ancestor graphs had 11 nodes. Neither real probe had a hierarchy continuation page, so that boundary has unit regression evidence only. The resumed run explicitly references the earlier 4,570-page full-inventory receipt and its exact verifier/publication hashes; it does not claim to repeat that walk.

The preserved-environment restart for this final journey took 98.977 seconds. All 696 runtime-copy files retained their original hashes in a final 33.170-second check, and the original preparation directory was restored. The actual publication audit uses one separately delivered owner selection sidecar; that addition does not modify the 696 runtime input files. The final service receipt records the source publication separately from the new serving revision and expiry. Services remain running on their owned loopback ports. The serving revision is `review-20261010T154710-f1025c8952`, closing at `2026-11-09T15:47:10.522210+00:00`. The immutable source publication is `sha256:edf000e8498b000493f0f7d9812188e03b6b0241fd84e1ca7508d7afb41a040a`; the newly prepared serving publication is `sha256:8cbcbcd0246809c9f5cf3593e6e7412098c63d6ed7f36eb82165af7f59369033`. PostgreSQL is 16.2.

Performance is not fully accepted: the fresh journey's first explanation-condition current-case response (case 13) took 25.413 seconds and the slowest main pair-metadata response took 7.861 seconds; their medians were 0.0315 and 0.015 seconds. Full inventory page medians/p95 were 0.134/0.158 seconds for study and 0.092/0.121 for main. These measured outliers and the cold-start durations remain review limitations; they are not disguised by warm probes, and no instrumented cause is claimed here. Code inspection confirms that opening a context or decision index can perform a full integrity hash, cached by file identity/stat for later reads. This is consistent with first-use validation, but there was no internal phase instrumentation to attribute these measured delays solely to hashing. Subsequent explanation-case responses were at most 0.046 seconds and subsequent main pair-metadata responses at most 0.034 seconds.

## Frontend S2 work

1. Consume and validate the frozen S1 schemas and real response examples. Implement forms/3,
   setup/3, consultation/3 and methods/3 before declaring `client_contract: study-review/1`.
   Preserve historical clients and saved outbox identities.
2. Display original definitions separately from prepared descriptions. Use explicit preparation
   kind/support/failure metadata, preserve exact reviewed claim text/citations, and render
   unavailable inference as unavailable rather than an empty successful taxonomy.
3. Add entry-point browsing with typed kind, basis, cursor, directness and completeness; show
   asserted/structural examples and failed native preparation status. Never run preparation
   from a view or infer missing hierarchy edges in the browser.
4. Finish guided step position/return controls against this frozen practice publication,
   action-specific evidence identity and structured rejection details. There is no persisted `step_id`; derive the next compatible guided task from acknowledged requirement completion within the saved lesson. Context/bridge clicks
   do not complete original-evidence requirements.
5. Fetch `/evidence-metadata?candidate_id=…` when study capabilities advertise
   `exact-evidence-metadata/1`, using the current authorized workspace scope. Preserve
   `comparison_records` resolution states and dependency-inventory references.
   Consume recorded exact-match and evidence/projection statuses; preserve typed graph/list
   identity and actual correspondence level. All delivered selected records have graph projection
   `not_recorded`; neither a resolved original nor recorded correspondence establishes
   producer-exported graph support. Preserve list/original access. Complete F08/F09 visual work using the real dense
   and restriction-heavy cases; explicitly retain the sparse-case coverage limitation.
6. Show aggregate About metadata once in study (`cases: []`); use the full case catalogue only
   in the main owner review. Match the current authorized study case to that catalogue during
   owner verification, without fetching future case identities in the participant app. Keep baseline restricted. Complete optional per-case method reporting with conditional text and clearing.
7. Compile S2 into a separate persistent build directory and prepare a separate review root using the same frozen inputs; retain this delivered S0 environment and receipt rather than overwriting its frontend bytes. Run the full fresh browser journey and Q01–Q12 against the compiled S2 frontend, actual HTTPS
   service and PostgreSQL. Inspect real main/study/tutorial views in both themes and record
   focus/reflow/keyboard/contrast evidence. Carry the S0 recovery-archive presentation follow-up.

## Remaining acceptance and launch gates

- Technical: successful full real inferred taxonomy and contrasting inferred examples;
  original DOID/import admission; missing selected-case, exact-stage trace and graph-support coverage; completed S2 consumers and
  S3 real browser/visual acceptance after the local certificate trust prerequisite;
  historical intermittent reload cause/reproduction; unexplained intermittent worker Git lookup failure; first-use/cold-start latency acceptance.
- Later research publication: frozen final matcher and approved/adjudicated cases, final context
  policy and regenerated dependency-matched descriptions, consent/questionnaire approval,
  deployment/restore verification and explicit study-owner release decision.
- Human-only follow-ups: manual accessibility/screen-reader audit, usability/pilot review and
  final user review. These are open, not falsely reported as passed or technical blockers.

## Provider preparation accounting

Attempt 1 was rejected by the expired credential (86 HTTP 401 responses). Attempt 2 used the
replacement credential: 77 recorded attempts, 76 saved HTTP 200 responses (66 profiles and
10 comparisons), and one interrupted request with unknown delivery. The 76 saved outputs did
not meet the required claim schema, so none was approved or published as supported prose.
Their reported usage was 325,098 prompt plus 26,865 completion tokens (351,963 total), with
reported cost USD 0.06488370. This is provider-reported usage, not a verified account charge;
the interrupted request's charge is unknown. All attempt records remain separate and private.

The corrected exact-schema prompt and bounded-dispatch tooling are verified with local
transport regressions. An additional 74-profile/12-comparison batch is prepared under
`sha256:02264c7889aa4525e6bc517717fe39b570f01cccc7326401befa1062ba14bafe` with zero
automatic HTTP retries and zero semantic repairs. It has not been dispatched: additional
external requests beyond the earlier bounded authorization require the pending explicit
approval. The usable interim publication contains 260 `grounded_description` artifacts with
`typed_template_support`; none is provider prose or a `factual_fallback` result. The runtime
also supports labelled factual fallbacks, verified separately. Malformed model text is not
presented as a successful generation batch.

The original input package records license treatment as local research, with redistribution
subject to each ontology's source license. This handoff does not establish a separate license
review or authorize public distribution. Portable local verification preserves that treatment.

## Repeating the publication audit

From the maintained checkout and installed runtime below, rerun the independent typed-claim
rules against the immutable delivered input. Choose a new receipt filename for each run:

```sh
"$HOME/Library/Caches/exact-om-dev/pyenv/bin/python" tools/audit_review_publication.py \
  --runtime-root "$PWD" \
  --selection-file "$HOME/Library/Caches/exact-om-review/s1-delivered-final-20261010/verification/publication-audit-inputs/selection.json" \
  "$HOME/Library/Caches/exact-om-review/s1-delivered-final-20261010/input" \
  "$HOME/Library/Caches/exact-om-review/s1-delivered-final-20261010/verification/owner-repeat.json"
```

This verifies hashes, typed originals, semantic obligations, artifact parity and catalogue
coverage. It makes no provider request and does not turn template validation into a prose or
biomedical correctness approval. The receipt binds both the audit script and imported runtime
module bytes. It writes outside the immutable input and refuses to overwrite its earlier receipt.

## Startup and local operation

The delivered environment is at
`/Users/pedrocotovio/Library/Caches/exact-om-review/s1-delivered-final-20261010`. Its portable inputs
are under `input/`; its compiled accepted S0 frontend is under `frontend/`. Backend runtime
identity is committed in the receipt, and the managed checkout remains available at
`/Users/pedrocotovio/.codex/worktrees/study-timing-integration/Exact-OM`. Keep that checkout
while this environment uses it. The exact delivered service state and timing results are
recorded in the accompanying relocation receipt.

```sh
cd /Users/pedrocotovio/.codex/worktrees/study-timing-integration/Exact-OM
REVIEW_PYTHON="$HOME/Library/Caches/exact-om-dev/pyenv/bin/python"
REVIEW_ROOT="$HOME/Library/Caches/exact-om-review/s1-delivered-final-20261010"
"$REVIEW_PYTHON" -m tools.review_environment status --root "$REVIEW_ROOT"
"$REVIEW_PYTHON" -m tools.review_environment start --root "$REVIEW_ROOT"
"$REVIEW_PYTHON" -m tools.review_environment fresh-session --root "$REVIEW_ROOT"
```

Main review uses `https://127.0.0.1:19185/`; study uses `https://127.0.0.1:19184/participate/`.
PostgreSQL uses loopback port 19183. The fresh-session command returns the path of a private
invitation receipt, not its secret. The compiled S0 participant client does not yet declare
`study-review/1`; receiving its explicit 409 is the compatibility guard until S2 implements
forms/3. The HTTP verifier supplies the negotiated contract and exercises the actual workflow;
this does not make the old participant UI an S2 implementation.

The delivered certificate expires on 9 November 2026 at 15:47:10 UTC. Its SHA-256 DER fingerprint is
`A4:42:89:EC:F4:98:B7:1D:AD:7B:43:AE:E1:B2:24:2D:9D:E4:8E:02:3E:77:74:48:AE:7B:12:85:95:40:6D:86`.
Inspect `tls/localhost.pem` and confirm that fingerprint before trusting it in a browser. Service probes verify that exact certificate and hostname. No global TLS bypass is
required. To stop only the owned processes, preserving the database and frozen revisions:

```sh
"$REVIEW_PYTHON" -m tools.review_environment stop --root "$REVIEW_ROOT"
```

An expired or closed study revision requires `fresh-session --new-revision --days 30`;
an expired certificate requires `renew-certificate`. These are explicit operations that
retain previous files and data. See [the lifecycle runbook](../explanation-review-environment.md)
for the installed-runtime prerequisites, certificate handling, immutable input checks, ownership guards and logs.
Never publish `environment.json`, invitation receipts, raw provider responses or credentials.

To repeat the actual workflow, including a new exhaustive full-index walk, choose a new receipt path:

```sh
"$REVIEW_PYTHON" -m tools.verify_review_environment --root "$REVIEW_ROOT" \
  --receipt "$REVIEW_ROOT/verification/owner-api-repeat.json"
```

This creates a fresh test session and observes the real tutorial dwell, assessment, both study conditions, consultation and completion. It does not render browser views.

The relocation check uses APFS copy-on-write clones with distinct inodes and matching hashes.
The separately delivered owner-audit selection file is 157,066 bytes and is bound to the original selection hash; it is outside the immutable runtime input inventory.
Those runtime clones are independent files, not symbolic links to the preparation directory. This proves
local relocation under the recorded source-hide test; it is not a cross-filesystem/container
installation result. The delivered copy contains 696 files totaling 13,707,483,995 logical bytes; a conventional
copy on another filesystem must provide that space in addition to its PostgreSQL data.
