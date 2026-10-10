# Backend S1: provider prose recovery

**2026-10-10.** This is the B04 follow-up after the user replaced the expired OpenRouter
credential. Starting checkout: `66b588ce979eab1b9b0b446d7eba9d61712b549d`.
The [machine receipt](explanation-product-review-backend-s1-prose-recovery/receipt.json)
records implementation identity, frozen inputs, independent reviews and verification.
The earlier [S1 follow-up](explanation-product-review-backend-s1-followup.md) remains the
record for ontology admission, evidence recovery, inference and the earlier API journey.

## Outcome and limits

The new credential works. The final bounded batch completed **74 entity profiles and 12
comparisons**, all with HTTP 200 and structurally reviewable claims. A separate Codex review
of the saved OpenAI drafts approved **68 of 172 claims**, comprising 57 profile claims in
56 artifacts and 11 comparison claims in eight artifacts. The other 104 claims were rejected.
Provider success, independent semantic approval and publication readiness are separate results.

There are **18 profile artifacts and four comparisons with no approved prose**. All 12 attempted
profiles without an original definition fall in that group: their drafts guessed biomedical
meaning or parent names from labels/opaque identifiers, or merely repeated terminology.
Their existing, honestly labelled factual fallbacks remain available. Six of the approved
comparison artifacts contain useful information about both entities; `review-07` and `review-12`
contain only an approved description of one entity. These do not constitute twelve successful
two-sided prose comparisons. No further paid batch was launched to disguise this coverage gap.

The recovery therefore closes authentication, provider-format and completed-batch failures.
It improves usable prose materially but **does not close all of B04**, joint acceptance or
research launch readiness. Missing-definition synthesis, four wholly rejected comparisons,
two one-sided comparisons and rendered S2 usefulness remain explicit limitations.

## Root causes and corrections

| Finding | Cause and correction | Disposition / owner |
|---|---|---|
| P1: authentication failure | Attempt 3 returned HTTP 401, with the recorded response saying the API key expired. The updated private credential authenticates successfully. It is not stored in Git, command arguments, receipts or public artifacts. | Resolved for the observed requests; future key expiry remains a runtime error. |
| P1: invalid provider output shape | Prompt instructions alone allowed invalid categories and output shapes. The transport now accepts an optional strict JSON response schema; the prose path supplies required fields, allowed categories and no extra properties. The exact schema enters the durable request identity. | Corrected and covered by real calls plus transport/cache regressions. Legacy calls omit the new option. |
| P2: unreadable prose | Template examples encouraged literal copying; long provenance envelopes distracted from the admitted meaning. V5 sends a compact view of the same admitted facts and explicit missingness, without renderer templates. Original complete packets remain frozen for provenance and review. | Improved in reviewed examples; semantic quality still requires claim-by-claim review. |
| P1: batch continuation after transport failure | A full run could continue after a saved job lacked a provider response. It now stops immediately, just like preflight, and cannot silently retry a completed/failed job on resume. | Regression covers success followed by credential failure and repeated invocation without new calls. |
| P1: unsupported model assertions | Drafts inferred broader/narrower disease scope, shifted qualifiers, guessed parent labels, or strengthened material basis into causation. One comparison denied a karyotype explicitly present in its cited definition; another dropped “presumed” from TPO causation. | Rejected by the separate review; no blanket semantic approval, category alias repair or evidence substitution. B04 coverage remains open. |

The provider schema follows the documented [OpenRouter structured-output interface](https://openrouter.ai/docs/guides/features/structured-outputs).
It constrains shape, not truth. Production admitted-fact validation and independent review
are still mandatory. Neither the model nor its output can issue its own approval.

## Frozen preparation and provenance

The model remains `openai/gpt-4o-mini-2024-07-18`, routed only to OpenAI, with provider fallback
disabled, required-parameter support enabled, temperature zero and a 2,000-token output cap.
There were no semantic-repair or HTTP retries. The original 74-profile/12-comparison scope,
recorded development run, candidate identities, scores, ontology facts and policy were retained.
No ongoing matcher experiment or historical publication was altered.

The successful plan is attempt 7, with canonical identity
`sha256:99858ad647b9e8b06cc4e4020823704d23d72ae57d44eb381810af1fc94f780c`.
Small preflights preceded it: attempt 4 demonstrated template copying; attempt 5 stopped on
an invalid category; attempt 6 verified the schema but remained poor prose. Their responses,
prompt versions and immutable caches are retained. Attempt 7 reused its three preflight jobs
and dispatched the remaining 83. It took 232.100 seconds and the provider ledger reported
$0.0453348. All recovery attempts 4–7 together made 93 requests at a reported $0.0522276;
these are provider-reported costs, not an independently reconciled invoice.

The original private preparation is:

`/Users/pedrocotovio/Library/Caches/exact-om-review/s1-20261010`

It holds `provider-plan-attempt-7.json`, `generation-attempt-7/provider-ledger/requests.sqlite3`,
saved request/response pairs, `review-requests-attempt-7` and `independent-reviews-attempt-7`.
The credential remains only in its private credential file. The new publication is assembled
without a credential and with cache-only generation under a separate durable root:

`/Users/pedrocotovio/Library/Caches/exact-om-review/s1-prose-20261010`

The target revision is `development-review-s1-prose-20261010`. Its source is the same recorded
local300 development run, not a new experiment. Original and derivative ontology identities,
root-only/import limitations and failed real ELK inference are unchanged. The new bundle
must be admitted as a new revision; it cannot replace bytes beneath an existing revision.

## Verification

The final combined regression run passed **90 tests, zero failures and zero skips**. Coverage
includes schema transmission, durable schema-bound request identity, legacy cache replay,
immutable original packets despite compact provider payloads, batch failure/resume behavior,
description metadata and enforced separate semantic review. One existing routing-alias
deprecation warning remains. The runtime schema/OpenAPI freeze check passed unchanged.
There is no participant API schema change and no frontend source change.

The actual main ASGI application served both independently approved preflight artifacts with
HTTP 200 and exact claim parity. A rejected draft identity returned 404. This used the actual
admitted NCIT/DOID context and application, not a mocked provider or synthetic ontology. It
does not by itself establish the complete publication, HTTPS/PostgreSQL or browser journey.

`tools.verify_review_prose` is a read-only reproducible check of the assembled publication.
It replays every separately saved review against the exact original response, verifies admitted
facts, checks the main API against the saved artifact, checks study-resource/workspace parity,
and uses the existing independent typed-rendering audit for non-prose fallback claims. It does
not independently redo the semantic judgement or turn a hash match into semantic approval.
Its result and any fresh HTTPS/PostgreSQL checks are separate receipts below; pending work
is not described as passing.

The existing full HTTPS/PostgreSQL result of 5,206 checks belongs to the previous publication.
It remains useful regression evidence for unchanged session isolation, tutorial grading,
questionnaires, completion and export, but is not relabelled as validation of this new prose.
The accepted S0 frontend still needs S2 contract negotiation and presentation; a fresh complete
rendered journey remains open. No new browser cancellation/stall conclusion is claimed.

## Local startup and running work

Assembly runs once in the background. `assembly-status.json` and `assembly.log` in the new root
identify the exact phase or failure. `prepared-input/publication.json` is written last and is
the assembly completion marker. No existing service or database is stopped or modified.

The post-assembly worker records `verification-status.json`, checks all served prose, prepares
a new isolated environment and runs the existing HTTPS/PostgreSQL journey plus restart/resume.
It leaves successful services running. Consult its receipt before treating the new endpoints
as ready. Long work is handed off rather than keeping the conversation waiting.

From the repository, use the existing Python executable:

```text
/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python
```

The new environment root is `<new-root>/environment`, with PostgreSQL 19303, study HTTPS
19304 and exploration HTTPS 19305. After successful preparation, lifecycle operations are:

```text
python -m tools.review_environment status --root <new-root>/environment
python -m tools.review_environment start --root <new-root>/environment
python -m tools.review_environment fresh-session --root <new-root>/environment
python -m tools.review_environment stop --root <new-root>/environment
```

The certificate is `<new-root>/environment/tls/localhost.pem`; trust that exact local certificate
for API verification. Never publish generated invitations, database configuration or credentials.
Do not run a second worker concurrently. If a worker failed, inspect its recorded phase and
preserve the failed receipts before a new verification attempt. No public deployment or push
was performed.

## Exact frontend S2 follow-up and acceptance

Use the [S1 contract handoff](explanation-product-review-backend-s1-contracts.md) and spec 21.
The prepared artifact contract remains `exact-description/2` with review receipts
`exact-description-review/1`. An approved artifact exposes `preparation_kind=grounded_description`,
`support_status=independently_reviewed`, exact cited claim IDs and the bound independent review.
Rejected/unapproved jobs retain their factual fallback metadata. Provider HTTP success or model
name is never the criterion for a “validated generated description” label.

1. Render the approved prose separately from original ontology definitions. Preserve exact text,
   citations, support status and publication scope. Do not remove portions of a claim while
   displaying its original review receipt.
2. Preserve a concise fallback/missing-support presentation for all 12 missing-definition profiles
   and the other rejected outputs. Do not infer biomedical meaning from names or synthesize prose
   in the browser. Show one-sided approved descriptions without implying a disease difference.
3. Obtain scope IDs from the new publication responses. Continue `study-review/1` negotiation,
   forms/3, explicit typed tutorial evidence, recorded bridge semantics, asserted/inferred status,
   per-case consultation and optional combinable external methods. Protégé remains a recommendation.
4. Review the six useful two-sided comparison examples and the two one-sided cases in the actual UI.
   Run the complete rendered HTTPS journey and reload/cancellation checks with this publication
   after S2. Manual accessibility and pilot usefulness remain separate checks.

**Implementation correctness:** authentication and response-shape recovery are verified; independent
review excludes unsupported claims. Complete-publication/runtime verification is bounded by its
recorded worker status. **Joint acceptance:** open for S2, rendered verification and the remaining
B04/data/performance gaps. **Launch readiness:** not established. Original DOID/full-import admission,
manual accessibility, pilot usability, owner approval and deployment remain open; real inference
remains the user's explicitly optional exception after the failed bounded ELK attempts.
