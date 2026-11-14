# Final 40-source study — production preparation, 14 November 2026

## Status

The owner's exported decisions have been applied to a new ordinary, non-synthetic
study publication, `exact2-ncit-doid-production-20261114`. The publication and
versioned frontend are prepared on AppVM. Live activation and public verification
are recorded in the completion addendum below; preparation alone is not deployment.

The active source checkout is `/private/tmp/exact-study-nov10`, branch
`codex/final-study-review`. Unrelated work in the canonical Developer checkout was
not changed. Runtime release commit: `1e8ad4e`; frontend is built on AppVM using
Node 24. The deployment branch is `study-deploy`, pushed through `1e8ad4e`.

## Selection and protocol

The actual exported owner review, not an inferred browser selection, is retained
privately. It keeps H6PD deficiency, Meningothelial Cell Neoplasm and Encephalitis
Lethargica; substitutes Peripheral Neuropathy for Angioleiomyoma and CINCA for
Benign Mixed Tumor of Vulva. The twelve previously approved replacements remain.

There are **40 distinct sources, 200 original source–candidate pairs and 237
distinct concepts**. The pool's reference-rank counts are **17/13/5/2/3**. Each
participant receives ten distinct sources, one from each original rank 1–5 in
each condition. Condition order is counterbalanced; source and within-block order
are sampled. Original candidate tuples, positions, scores and reference keys were
compared with the frozen final catalogue and remain unchanged.

Five cases use the shared explanation workspace. Five show only identities,
original order and scores, with the supplied ontology downloads. Inspection-method
questions occur after each no-tool case only. Unrequested method reports are not
encoded as non-use. Tutorial skip remains explicit and records a skip, never a
pass. Research exports exclude test sessions. Earlier test publications and
responses are not relabelled as research data.

The pool is purposive. Rare rank strata repeat more frequently across participants;
rank 4 has only two available sources. Reference-based correctness is not a claim
that every alternative absent from the benchmark is clinically false. The final
analysis plan preserves case identity, rank, condition order and tutorial status.

Identities:

- Final Exact 2.0 result SHA-256:
  `d5b5b54e19ca04de29f5843e376bceb921fd4d9f5bd1e9e277b307c0e6581459`.
- Selected catalogue SHA-256:
  `7d1a8b260bcc43c860a944306464fae296e40822a9ab4dc1d236bf39335099e2`.
- Private production publication file SHA-256:
  `53dbf31c0567794bccb461e1fbcc7520dcd435c04c8598b331f2566b82714d50`.

The closing time is 14 December 2026, 23:59:59 UTC, consistent with the previously
requested collection month. The month is no longer displayed as participant copy.

## Descriptions and evidence

Unchanged independently reviewed artifacts were reused only when their complete
support packets matched exactly: **158 reused artifacts**. Luna generated the
remaining 279 with four workers, no automatic retries or fallback model, and a
conservative maximum of USD 1.18 under the existing USD 5 ceiling. This is a
planning bound, not a billed-cost statement.

The assistant independently read all **312 claims in 275 reviewable drafts**
against their cited original literals and typed axioms. Decisions: **299 approved,
13 rejected**. Four outputs did not reach reviewable admission. Ten reviewable
drafts had no approved claim. Their responses and rejection reasons remain intact
in the private archive; generated text was not rewritten during admission.

The combined publication has **227 generated entity profiles and 196 generated
comparisons**. Ten entity profiles and four comparisons retain typed original-fact
fallbacks. Rejected examples include an incorrect citation, unsupported scope
claims, unhelpfully generic summaries, silently repaired malformed definitions and
unattributed changes to ambiguous enzyme names. This is grounding review against
the frozen ontologies, not independent biomedical adjudication.

All six displayed concepts have original definitions in 33 cases; five concepts
have them in five cases, and four in two cases. Generated material does not turn
an absent original definition into an asserted ontology fact.
Nine distinct concepts lack an original definition. Eight have admitted generated
profiles; neuroaxonal dystrophy retains a typed description of its original
cerebral-degeneration classification. It is not presented as a newly asserted
clinical definition, and its missing original definition remains explicit.

All **200 pairs have some original-fact-linked evidence**. Across the selected
pool, 3,420 matcher records resolve to original facts and 3,972 remain unresolved.
The unresolved records retain their status; no original provenance, matcher
bridges or full-DL results were invented. Existing partial ELK semantics and
limitations remain. There are 41 admitted workspace scopes including the synthetic
tutorial, and 445 frozen resources. Strict publication admission verifies resources,
fact identities, packet/receipt bindings and workspace confinement.

The existing full ontology contexts are reused. No new full ontology bundle,
matcher execution or classifier execution was created. Context identity checks
preserve inode, size, mtime and ctime; Linux access-time changes are not corruption.

## Participant text and frontend

`tools/final_study_copy.py` supplies the new information version
`final-study-information/20261114`. National Cancer Institute Thesaurus (NCIT) and
Disease Ontology (DOID) are both expanded. The requested study-period, duration
placeholder, clinical/professional-abilities and pending ethics-status passages
are removed. The copy proceeds from purpose and task to recording, one-year
retention, voluntary withdrawal and contact details. It makes no claim of ethics
approval or exemption. The consent text scrolls with the page rather than inside
a small separate box. Desktop and 375 px mobile screenshots were inspected.

The researcher must honour the stated withdrawal/removal policy, including any
retained exports/backups covered by it. Technical deployment does not determine an
institution's review requirements. The owner explicitly authorized deployment;
that authorization is recorded without inventing institutional approval.

Final browser verification exposed a cold case load at the block boundary that
outlasted the old test's 15-second expectation. The original loading skeleton had
no accessible message. It now shows a visible status while the initial case loads,
and a request still pending after 45 seconds offers Retry. Retry fetches the current
presentation again. The regression checks cover stalled-load recovery as well as
normal navigation; this does not prove the cause of every historical blank-page
report.

The frontend build initially reported three dependency advisories. The tracked
lock now uses Next 15.5.29, sharp 0.35.5 and source-map-js 1.2.2, retaining the prior
PostCSS override. The rebuilt frontend's npm audit reports zero vulnerabilities.
See the [Next advisory](https://github.com/advisories/GHSA-2464-fhr4-w9xw),
[sharp advisory](https://github.com/advisories/GHSA-wq5f-xc86-pv6w) and
[source-map advisory](https://github.com/advisories/GHSA-68fv-2mgg-jv7q).
This does not close the repository-wide alerts reported on its default branch.

## Verification and private records

- Targeted preparation/prose checks: 18 passed. Final selection, sampling and
  prose-admission regression run: 16 passed. These are overlapping focused runs,
  not a claim that the entire repository suite passed.
- Frontend type checking, static export and 142 unit tests passed after patching.
- The real publication passed strict admission in an isolated SQLite validation
  database in 128.1 seconds. A repeat checks the already admitted immutable identity.
- The same publication independently passed strict admission on PostgreSQL 16 in
  126.6 seconds. Both complete ten-case service journeys then passed again on
  PostgreSQL, including completion, export exclusion and context preservation.
- Two full ten-case service journeys completed, in opposite condition orders,
  including final questionnaire and explicit completion. Both preserve distinct
  sources and rank balance, selectively request methods and expose no private keys.
- Default exports excluded every test session; test-inclusive exports retained
  enabled reference-based metrics for the ordinary production study.
- Real-data browser results and public verification are in the completion addendum.
- The production Python environment successfully loads the new application factory.
- A private PostgreSQL backup was taken and its archive listing verified.

One initial validation harness omitted the separate completion mutation after the
final questionnaire; the harness was corrected and both journeys completed. The
failed test session remains a test record. A first browser run had three passes
and the cold-load timeout described above; its diagnostic output is preserved.
A second SQLite run passed four of five checks but logged `database is locked`
during first ontology verification: SQLite's test-only `BEGIN IMMEDIATE` read
transaction blocked a simultaneous state request. Final acceptance therefore uses
a separate PostgreSQL instance, matching the production engine, with peer
authentication through a private Unix socket and no TCP listener. It does not
change the live PostgreSQL cluster or weaken publication checks.

Browser tests execute on AppVM, so private researcher/signing test credentials stay
there. Only participant screenshots are transferred for visual inspection. Traces
and video recording are disabled. Both validation databases are separate from live
PostgreSQL. No invitations were sent to participants.

Private durable locations:

- AppVM: `~/exact-study/production-20261114/private/` — exported owner decisions,
  final selection, inputs, publication, reference keys, admission results,
  validation script/report, test configuration and pre-deployment backup.
- AppVM: `~/exact-study/assets/exact2-ncit-doid-production-20261114/` — new small
  explanation assets (about 38 MiB), plus small indexes in the existing workspaces
  directory; the large ontology contexts are shared unchanged.
- AppVM: `~/exact-study/releases/production-20261114/` — source-controlled release
  and built static frontend. Build, admission and browser logs are under
  `~/exact-study/production-20261114/`.
- compute-lis5: `/mnt/datasets/pgcotovio/exact-study-preparation-20261114-production/`
  — frozen generation plan, raw provider responses, per-claim review requests,
  independent decisions/receipts, rejections and combined reviewed results.

The measured combined local footprint after preparation was **52.03 GB** against
the owner's decimal **100 GB** cap, with about 80.96 GB filesystem space free.
Server preparation avoided local full-bundle transfers. The storage guard remained
in use for local growing operations.

## Activation and recovery

`~/exact-study/activate-production-20261114.sh` verifies the release, validation
receipt, backup and clean tracked live checkout, then requests sudo authentication
before changing live files. It advances the live `study-deploy` checkout, retains
old hashed frontend assets for existing tabs, switches the frontend symlink,
restarts `exact-study`, checks readiness, and publishes the new immutable revision.
It creates two private **test** links for final verification/owner review.

The prior commit/frontend target are saved privately in `previous-release.txt`.
Do not automatically restore the database backup after participants have started:
that would discard later responses. Preserve old revisions and sessions. If
activation fails, inspect readiness/logs and the saved prior release before issuing
new invitations. New participant invitations must select the new production
revision; existing links keep their frozen earlier information and cases.

## Completion addendum

All **five final browser checks passed on isolated PostgreSQL** in 2.5 minutes:
two ten-case journeys, final welcome/consent at desktop and mobile widths, real
evidence/hierarchy in both themes, and a deliberately stalled case followed by
successful retry. Both condition orders complete without browser errors. The
frontend source and runtime release are `1e8ad4e`, pushed to `study-deploy`.
The slowest cold `/cases/current` request was 44.008 seconds while the full ontology
files were verified. Warm requests are much faster, but a future process restart
can repeat that initial delay. The new loading message and bounded retry cover
that wait; this change does not claim to optimize the underlying cold hash cost.

AppVM requires interactive sudo authentication for the final systemd restart.
The activation script, private backup, complete publication and acceptance
receipts are prepared. Pending: authenticated live activation, publication receipt
and public verification. Until those results are observed, the release is staged,
not yet confirmed live.

Screen-reader testing, a human usability pilot and institutional review decisions
are not asserted by these automated checks. No new full PostgreSQL concurrency
campaign was run. Original ontology wording issues, incomplete matcher provenance
and partial ELK coverage remain explicitly bounded limitations.
