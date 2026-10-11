# Study data publication after repository recovery

**11 October 2026.** Start: `b661b3f7f6abdd521b7322d21cdb276c4120b5d1`.
The checkout now lives at `/Users/pedrocotovio/Developer/Repos/OM/Exact-OM`.
This continues the [reviewed prose preparation](explanation-product-review-backend-s1-prose-recovery.md)
without changing the frozen development run, ontology scope, provider responses or independent
claim decisions. Historical disk-full and materialization failures remain recorded; their
previously staged handoff is now committed as `77552fd`.

The [receipt](explanation-product-review-backend-s1-publication-recovery/receipt.json) is a dated
snapshot. **Snapshot: running / cache_only_assembly; publication: assembling.** Dependent checks: prose_api: not_completed; https_postgresql: not_completed.
It is not a completed-publication claim unless the live worker reaches `status: passed`,
`phase: complete`. The worker continues independently of this chat. No new paid generation is
needed or enabled. No final matcher experiment, existing publication or database was modified.

## Data and recovery

The frozen local300 development export supplies **24 source cases / 120 presented pairs**, with
all **30,374 recorded candidate pairs** retained. Coverage is 23 rich-definition cases,
5 absent/asymmetric-definition cases, 20 multiple-parent cases, 8 restriction/equivalence cases
and 16 close-alternative cases. These categories overlap. Sparse selected evidence is still
missing. The new receipt records exact input, selection, pool, policy and contract identities.

The working provider batch and independent reviews were retained: 74 profiles and 12
comparisons, with **68 approved claims and 104 rejected claims**. Approved prose exists in 56
profile artifacts and eight comparisons. Eighteen profiles and four comparisons have no
approved prose; two approved comparisons are one-sided. All twelve attempted missing-definition
profiles still require faithful factual fallbacks. This operation assembles the already approved
claims; it does not claim to solve that remaining B04 quality/coverage gap.

NCIT remains the admitted root snapshot. DOID remains the explicitly identified annotation-
declaration derivative/root scope. Fifteen original imports are retained, but original/full-
closure native admission is still open. Download, context, policy and prepared-description
identities must continue to match; no scope is widened or relabelled during recovery.

The previous assembly failed when the disk filled. The later resume could not read cloud-backed
source and timed out. Source and Git are now resident outside iCloud. The new worker uses the
migrated checkout, a nonblocking exclusive lock, saved generation/review caches, a 30 GiB free-space
guard and separate progress/log/verification paths. Existing failure evidence is preserved.

The ordered worker steps are:

1. Resume immutable cache-only assembly; write `publication.json` only after package, resources,
   workspace indexes and catalogue finish.
2. Replay the independent review bindings, validate exact admitted support, test the actual main
   ASGI explanation API and verify study-resource/workspace claim parity.
3. Prepare an isolated PostgreSQL/HTTPS development environment using the retained compiled S0
   frontend, bound backend source and new test-publication identity.
4. Run the actual HTTP journey: tutorial rejection/recovery and server grading, both conditions,
   scoped workspace/catalogue isolation, per-case methods, completion, JSON/CSV exports and
   restart/resume. Preserve TLS, CSP, content-length and timing assertions.

These are API/integration checks. They do not constitute rendered frontend acceptance. The S0
build still requires S2 negotiation/presentation for `study-review/1` and forms/3; its compatibility
refusal is not bypassed to obtain a browser pass.

## Verification performed during recovery

**234 distinct backend tests passed.** The initial selection had one optional PostgreSQL
skip; that exact check was subsequently run with the installed PostgreSQL runtime on isolated
port 19483 and passed. There are no remaining skips in the combined selection. The transport
suite reported the existing routing-alias deprecation warning. The frozen runtime schema/OpenAPI
check passed without changes. Test receipts and their hashes accompany this handoff.

The full-publication and HTTPS results are recorded separately in the live paths below; pending
or running phases are not passes. Read measured cold-start/first-use/restart results before
closing S1-06. Historical R16 reload causation, frontend screenshots, accessibility and usability
were not reverified by this preparation task.

## Optional reasoning

ELK cluster job **14455** completed, but both NCIT and DOID receipts say
`completed_incomplete` and `eligible_for_integration_review: false`. NCIT incompleteness covers
datatypes, datatype definitions and positive unions; DOID covers exact/minimum cardinalities
and positive unions. Partial taxonomy files remain diagnostics and are not integrated as
complete inferred hierarchies. The exact result receipts are retained with this report.

At the one scheduler check, DOID HermiT **14460** was running, corrected-NCIT **14462** was
waiting on its dependency, and admission preflight **14461** had completed. No job was changed,
restarted or awaited. Inference remains the user's optional exception; it does not block this
publication. Any later complete result needs separate provenance/semantic integration review.

## Local paths and startup

Durable root: `/Users/pedrocotovio/Library/Caches/exact-om-review/s1-prose-20261010`.
The original saved provider jobs/reviews remain in `s1-20261010`; their cache-only copies are
inside `prepared-input` in the durable root. Credentials remain outside Git and are removed
from the worker environment.

- Live operation: `resume-worker-20261011.json`; log: `resume-worker-20261011.log`.
- Completion marker: `prepared-input/publication.json`.
- Full prose/API parity: `prose-verification-20261011.json`.
- HTTPS/PostgreSQL status: `environment-verification-20261011/status.json`.
- Detailed journey: `environment-verification-20261011/journey.json`.
- Environment config: `environment/environment.json` (private; do not copy to Git).

After the worker reports success, use the installed interpreter from the new checkout:

```sh
cd /Users/pedrocotovio/Developer/Repos/OM/Exact-OM
/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python -m tools.review_environment status --root /Users/pedrocotovio/Library/Caches/exact-om-review/s1-prose-20261010/environment
```

Replace `status` with `start`, `stop` or `fresh-session` for lifecycle operations. A fresh session
issues private test invitations; do not put them in Git or public logs. Do not start a second
assembly worker or rerun completed verifiers into existing output directories. Inspect the
recorded failing phase before retrying. Successful verification leaves the owned services running.

Planned endpoints are exploration `https://127.0.0.1:19305` and study
`https://127.0.0.1:19304`, with PostgreSQL on 19303. Verify readiness rather than assuming these
links work during assembly. Trust the exact `environment/tls/localhost.pem` certificate locally;
no global TLS/CSP bypass is required. The publication is a private development review with
correctness metrics disabled, not a research recruitment link.

## Frontend S2 and acceptance

Use the [frozen S1 contracts](explanation-product-review-backend-s1-contracts.md) and spec 21:

1. Negotiate `study-review/1`; render server-ordered forms/3, conditional Other fields and optional,
   combinable per-case methods without changing historical forms.
2. Discover scope IDs from actual responses. Keep explicit entity kinds, server-owned grading,
   rejected-action recovery and baseline isolation.
3. Render approved prose separately from original definitions and factual fallbacks. Preserve
   exact claim wording/citations/support status; one-sided prose is not a supported difference.
4. Consume entry points and honest inference status. Preserve unresolved originals, recorded
   bridge meaning and unsupported expression/list alternatives in the real graph.
5. Build a compatible frontend into a new persistent directory/environment. Run the complete
   rendered HTTPS journey, main/study parity, both themes, keyboard/reflow and reload/cancellation
   checks. Keep the existing assessment-recovery fixes. Protégé remains a recommendation.

**Implementation:** retained backend contracts/regressions verified; assembly and integration are
bounded by their own receipts. **Joint acceptance:** open for S2/rendered review and the explicit
prose, original-scope, coverage and performance gaps. **Launch readiness:** not established;
final matcher/case adjudication, owner approval, deployment/restore, accessibility and pilot
checks remain separate. No public push or deployment occurred.
