# Backend and integration follow-up — 13 October 2026

Reviewed checkout: `1b04ac065e69505a07f5f6ed6985ce5ca108be33`, following the
[independent R2-F1 review](explanation-s2-luna-r2-f1-review.md), spec 23 and the relevant
preparation, context, description, persistence and acceptance requirements in specs 20–22.
[Machine-readable receipt](explanation-backend-integration-followup-20261013/receipt.json) and
[reproduction commands](explanation-backend-integration-followup-20261013/commands.json) accompany
this report. The accepted frontend implementation is unchanged. This report distinguishes actual service
verification, synthetic faults, independent semantic review and outstanding release gates.

## Corrections and original failures

| Finding | Severity / owner | Cause and disposition |
|---|---|---|
| Legacy fixture would not start: complete retained category inventory missing | P1 backend preparation | Raw context preparation omitted `completeness.category_counts`; policy export already retained it. New context preparation records the exact SQL inventory. Reuse of an old inventory-less preparation refuses and requests a new revision. Ontology-resource admission was not weakened. |
| Legacy support failed scope admission once typed packets were retained | P1 backend preparation | The old pipeline put `ontology_version_id` in `context_hashes`. New execution locks use context/4 and generation/6; packets bind actual `context.sqlite` bytes. Exact extracts explicitly declare `selection.context_binding="raw-context/1"`. Portable exports may retain that source hash only through their same-policy, immutable export receipt; each typed original and AST is still checked against the served context. Modern reviewed descriptions cannot opt into this legacy exception. |
| Legacy synthetic publication had unresolved keys under a v1 research-shaped contract | P1 fixture | Unresolved keys are legal only in the development-review contract. The legacy **synthetic** fixture now declares its mechanical `c1` key, original five candidates/scores/ranks and construction provenance. This key exercises server-owned grading and is explicitly not an ontology-equivalence adjudication. Current real review publications retain unresolved correctness. |
| Resume after context repair dispatched an unnecessary provider request | P2 backend preparation | Binding to the repaired context bytes changed an otherwise identical legacy exact-extract request. Reuse now requires exact equality of every request field except the previous context hashes, checks the retained response hash, reruns grounding, and writes a new artifact with explicit revalidation provenance. Changed facts or any other input still dispatch normally. Reviewed descriptions and comparisons do not use this path. |
| Asserted DOID first page was mostly unhelpful roots | P2 backend navigation | A context-bound, explicitly curated starting choice returns the existing `DOID_4` disease root on the first page. Every original root remains in continuation. No label-based obsolescence inference, deletion, invented relationship or new deprecation assertion. |
| Some Luna profiles repeated a label/parent despite an available definition | P2 preparation | Audited all 140 profiles. 17 had an available retained definition; preserve every approved clause/citation and append one explicitly quoted full original definition. A separate agent reviewed all 34 clauses in these 17 drafts before publication. 12 weak profiles have no retained original definition; keep their honest templates. 111 profiles did not trigger the low-information heuristic; this is not a new human usability rating of all prose. |
| Old packetless citations could default to the wrong class identity | P1 compatibility / backend and frontend | New exact-extract jobs retain full typed support packets; main API, study admission and frozen resources validate them. Generation/6 replays saved /5–/2 responses without a provider dispatch when the frozen packet and wire settings are identical. Historical packetless artifacts remain immutable; their genuine ambiguity cannot be resolved from annotation IRI alone. See the frontend boundary below. |
| First frontend run reported `network_unreachable` then timed out | Undiagnosed integration incident | Retained first-run evidence locates the failure before the mite-infestation citation was clicked. It does not establish a backend cause. Added opt-in bounded read diagnostics and actual-browser startup/normal-read, abort, reload and stream-cancellation probes. Passing repetitions do not close this incident. No blanket retry, timeout increase, middleware exception suppression or dependency change. |
| Study image omitted transitive reader modules | P1 deployment packaging | The selective COPY list had drifted behind the prepared reader. Copy the complete small `exact_inspect` package. Isolated cold-import tests verify the image module layout without matcher/native/provider dependencies. Actual Docker image execution remains unverified because the local daemon is unavailable. |

Implementation commits: `cc5ecc1` (inventory, typed support, navigation, diagnostics,
container contents and schemas); `a3bb446` (context-byte and portable-lineage binding).
`f4f042c` declares the synthetic legacy grading provenance; `d16c816` adds the targeted, independently reviewed publication tooling. `0a72cb4` fixes unchanged-extract revalidation after a context repair. The final evidence commit is separate.

## Delivered data and preservation

No provider calls, matcher experiments, ontology downloads or reasoner runs were made.
HermiT remains cancelled. The new revision reuses all large immutable files in:

`/Users/pedrocotovio/Library/Caches/exact-om-review/postmigration-20261011/combined`

| Artifact | Identity |
|---|---|
| New package | `package/package-definition-review-20261013.json` |
| Package ID | `sha256:55acbf149864181695c4aa19817131682ef615f307c13993172a87fd5eef322c` |
| Source publication | `publication-definition-review-20261013.json` |
| Source publication SHA-256 | `3893f7607b229fbbd3957d382ad3d27063293a642e18f15ddce8605eddfe209f` |
| Catalogue | `catalogue-definition-review-20261013.json` |
| Predecessor | `package/package-luna-20261012.json`, SHA-256 `07781b63d2acd24f2542653112e5a20d75cea30af55685b331af3ea1e9b8907b` |
| Accepted frontend tree | `luna-bundle-20261012/frontend-2d4600d`, SHA-256 `07bdf63e4954b1185a11e453f14eb597aaa34904e65039faea7fd28b7355ddb2` |
| Browser/API run backend fingerprint | `e9e95bda0d356a00080ad9638660ea2c8f78c36263a7eaf6640c1707ccd8aa30` (57 files; implementation commit `a3bb446`) |

The final startup environment uses backend commit `0a72cb4`, fingerprint
`831a0f01746b4a34fe2c63d4e60d6e47d6f76e26478a5e062f79ec13f1903588` (57 files).
Browser/API parity was run on `a3bb446`; the later source change is confined to offline
preparation replay and has its own 49-test final regression run. The full browser set was not
repeated after that offline-only change. The final source-bound environment separately passed
startup/readiness and the bounded read probe. Both environment identities are retained.

The receipt records complete file hashes, review bindings, the 17 supersessions and the final
review-environment publication identity. Only 17 small explanation files and 13 case
resources/scoped indices changed. All 1,241 predecessor artifact entries and 243 unaffected
explanation locations remain. The current 260 artifacts comprise 140 profiles and 120
comparisons; 252 remain independently reviewed and eight retain labelled fallback status.
All 291 predecessor clauses remain; 17 original-definition clauses were added.

Static comparison checks every original fact under its full typed subject, every case,
candidate identity and score, 3,443 selected evidence records over 120 presented pairs,
bridges, pair metadata, context bindings and partial-ELK components. The entire matcher/run
and candidate-pool artifacts are reused unchanged, not reduced to the study subset. The
protected predecessor inode/stat inventory also remains unchanged. Separate review confirmed
all new artifacts equal admission of its actual receipts. No shared immutable inode was
unlocked, written, chmodded or relinked.

The 12 source-limited profiles and their complete admitted category counts are listed in
`source-limit-audit.json`. `DOID_13499` additionally has a restriction involving
`RO_0004026` and `UBERON_0002115`; neither has a retained label in the admitted scope. Its
original OWL remains accessible. This is a readable-source limitation, not evidence that
there are no other axioms. Source wording and typos in quoted definitions were preserved.

## Stable integration contracts

DOID ontology identity is
`sha256:26054e89dafcade53c821c0cc1a0b00574ec59becc81f6a930fc7474e44ab500`;
the curated choice is additionally bound to context SHA-256
`d18cc163a3bcc4546029e08e347fe7d896848353fcb03df95cda3bd25660c99a`.
It applies to classes in `literal_asserted` and `structural_navigation`, not partial ELK,
other kinds, other contexts or other ontology versions. The existing root retains its actual
`entry_kind`; curation describes the entry-list choice, not an ontology relationship.

```json
{
  "selection": {
    "kind": "curated_default",
    "revision": "review-navigation/20261013",
    "assertion": "navigation_only",
    "rationale": "Curated review starting point: disease. All recorded roots remain accessible; this choice makes no claim about other entities' obsolescence."
  },
  "default_entity": {
    "ontology_version_id": "sha256:26054e89dafcade53c821c0cc1a0b00574ec59becc81f6a930fc7474e44ab500",
    "iri": "http://purl.obolibrary.org/obo/DOID_4",
    "kind": "class"
  }
}
```

The first page has one entry even when a higher limit is requested. `total_count`,
`truncated` and `next_cursor` retain the full inventory. Follow the cursor; do not infer
completion from a short page. Cursors bind selection revision, ontology/context, policy,
kind and basis. Main and scoped study endpoints have the same selection contract. OpenAPI
snapshots were regenerated and checked.

Fresh legacy support packets carry authoritative ontology/IRI/kind for each cited fact,
including two typed views of one punned label. Frozen resources retain `(fact_id, subject)`
rather than collapsing by axiom ID. Main and study reject forged support even if its packet
hash is recomputed. The original fact ID remains shared where it is genuinely shared.
An older immutable artifact without a saved typed packet cannot establish a kind from an
annotation alone; it remains readable under its historical contract, without a new typed
identity guarantee. Prepare a new revision from retained original context/request evidence
to obtain that guarantee. Never silently repair an old publication or guess from IRI spelling.

Partial ELK still means incomplete positive taxonomy evidence. Unsupported and unknown remain
separate. An absent edge/member, satisfiability or negative entailment is not established by
this partial result. No full-DL consistency or completeness claim was added.

## Verification

Final totals and exact commands are in `receipt.json` and `test-outcomes.json` beside this
report. Raw logs, private harnesses, databases and invitation-bearing test state remain outside
the repository under `backend-integration-20261013`.

- **Legacy browser set:** 19 passed, zero failed/skipped/flaky, 58.5 seconds; exact-study/1.0,
  fresh synthetic fixture, HTTPS 19644 and dedicated UTF-8 PostgreSQL on 19643. Includes
  participant journey, reload/pause/offline recovery, scope isolation, typed parent inspection,
  researcher publication/export/closure and automated viewport/accessibility checks.
- **Current Luna browser set:** 25 passed, zero failed/skipped/flaky, 393.4 seconds. Unmodified accepted frontend; normal
  CSP; retries remain zero. The complete 24-case journey is unseeded. R2-F1 multi-subject
  ambiguity, induced network failures, broad equivalents, cycle/punning and assessment
  conflicts in other tests are deliberately seeded diagnostics, not naturally occurring data.
- **Actual main/frozen/scoped API parity:** passed for all 120 pairs, 260 descriptions and
  336 citation checks. Two complete 24-case HTTPS journeys cover both counterbalanced
  explanation subsets; a duplicate allocation attempt is recorded separately. Both asserted
  and structural entry lists cover all 4,189 roots in 43 main and 43 scoped pages per basis.
  Partial-ELK unknown/unsupported distinctions, cursor binding and scope negatives passed.
- **Backend regressions:** the broad run had 152 passes, two opt-in recovery skips and one
  genuine resume failure. After `0a72cb4`, all 49 relevant checks passed, including that
  failure. Both recovery checks separately passed against actual PostgreSQL. Across these
  runs, all 172 distinct checks have a passing latest observation; this is not a single
  final-HEAD full-suite result. Preparation, context policy/admission,
  saved-response replay, typed citation negatives, scope/authorization, tutorial persistence,
  conflict/idempotency, timing, reports, exports and container reader layout are included.
- **PostgreSQL restart and backup/restore:** both v1 and v2 checks passed (10.08 seconds) in
  an owned disposable UTF-8 PostgreSQL 16.2 cluster. Actual restart and pg_dump/pg_restore into
  separate databases preserved durable rows, submissions/exports and v2 training/draft state.
  These are synthetic recovery checks, not a production-volume recovery/RPO/RTO certification.
- **Schemas:** regenerated exploration/study OpenAPI; final runtime check passed again on `0a72cb4`.
- **Frontend source/build:** no frontend edit or rebuild. Reused the independently accepted
  compiled tree. The test cache matches current source/configuration across 153 selected files;
  the prior independent review already passed 131 units, type checking and production build.

Development attempts are not counted as passes: early new-test expectation errors were
corrected; the first PostgreSQL setup used SQL_ASCII, then its retained-evidence destination
violated the test's workspace rule. UTF-8 plus the permitted evidence path resolved those
setup failures. Legacy startup additionally exposed the real context-hash and unresolved-key
problems described above. One browser command was started before its failed harness existed
and collected no tests; the subsequent 19-test run used the verified harness. The first
read probe wrongly required document CSP on JSON; all 200 reads nevertheless returned 200,
valid bodies and exact content lengths, with 210 matching server completion records and no
exception. The corrected probe asserts CSP on HTML. The later broad run also found the real extra-dispatch
resume defect; its failing receipt is retained beside the passing final regression. No product assertion or timeout was
relaxed to obtain a pass.

## Service-read incident and diagnostic limits

Original evidence: `~/Library/Caches/exact-om-dev/s2dev/final-r2f1/run1-acceptance-results.json`.
The original run began at 02:15:12 UTC; the failed profile/citation test began at 02:18:04 UTC
and timed out after about 900 seconds. It was loading `DOID_7894` (mite infestation) before
any `DOID:4110` citation click. The service then retained warnings only. Neither a correlated
server exception nor a causal network boundary was recorded. Preserve this as **undiagnosed**.
The older intermittent reload/h11 Content-Length finding (R16) also remains unreproduced and
is not closed by these checks.

`EXACT_READ_DIAGNOSTICS=/private/path.jsonl` enables pure ASGI observation. It records random
read IDs, static route templates, method/status, elapsed time, sent/declared bytes, completion,
observed disconnect and exception **type**, plus `x-exact-read-id` on observed API responses.
No raw URL, query, request/response body, cookies, invitation, credentials or participant
identity enters diagnostics. Default is off. Each worker rotates at 1 MiB with two backups;
errors/cancellation propagate and response bytes, lengths, authentication and CSP are unchanged.
`EXACT_REVIEW_READ_DIAGNOSTICS=1` assigns distinct managed-worker files.

The bounded real-browser probe makes 20 batches of four profile/discovery/comparison/context
reads, ten explicit aborts, ten reloads while reads are launched, ten stream-cancellation
attempts, and four follow-up reads after each scenario: 200 asserted normal reads per run.
Stream cancellation is a **client attempt**, not proof of an incomplete server transmission;
the observed axiom is small. Each stream/read has a finite 15-second probe deadline, and
navigation keeps its 30-second bound. These are diagnostic bounds, not altered product or
acceptance-test timeouts. The corrected normal-operation and final-code startup probes both passed: 400 asserted
normal reads, 20 completed reloads, 20 explicit aborts and 20 stream-cancellation attempts.
Each run matched 210 server completion records with no exception, incomplete transmission
or length mismatch. Intentional cancellation produced only the expected browser aborts.
This did not reproduce the original failure or an interrupted server stream.

## Startup and reproduction

Active checkout: `/Users/pedrocotovio/Developer/Repos/OM/Exact-OM`.
Use the retained Python runtime and storage guard; the former Documents checkout is a notice.

```sh
cd /Users/pedrocotovio/Developer/Repos/OM/Exact-OM
PY=/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python
ROOT=/Users/pedrocotovio/Library/Caches/exact-om-review/backend-integration-20261013
"$PY" -B -m tools.review_storage_budget check --growth-gb 1
"$PY" -B -m tools.review_environment status --root "$ROOT/environment-final"
```

The new environment serves main `https://127.0.0.1:19655/`, study
`https://127.0.0.1:19654/participate/`, PostgreSQL 19653. Its private configuration pins the
package, publication, backend and accepted frontend bytes. Use its certificate for API TLS
verification. Existing 19623–19625 services/publications/sessions were preserved.

If stopped, start only its owned services, honoring the retained reservation:

```sh
EXACT_REVIEW_READ_DIAGNOSTICS=1 "$PY" -B -c 'import resource; resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728)); resource.setrlimit(resource.RLIMIT_CORE,(0,0)); from tools.review_environment import main; main()' start --root "$ROOT/environment-final"
"$PY" -B -m tools.review_environment fresh-session --root "$ROOT/environment-final"
```

Fresh-session output contains a private development invitation: never commit, share publicly
or place it in diagnostics. `luna-harness.json` and `legacy-accepted-harness.json` are mode 0600
and are only for browser verification of the recorded runs. The Luna harness targets the
preceding environment-v2; create a new harness for a final-environment browser rerun. Environment expiry is 12 November 2026; use a new
revision after expiry, not mutation of old study sessions. Initial unused environment roots
and failed synthetic revisions are retained as diagnostic evidence, not active study datasets.

For the legacy fixture, use `tools.build_explanation_ui_fixture --output NEW_DIRECTORY`, then
`tools.serve_explanation_ui_study --fixture ... --output NEW_STUDY_DIRECTORY --database-url ...
--frontend-dir ... --tls-cert ... --tls-key ... --config PRIVATE_FILE --port ...`. Always use
a fresh disposable database and output revision. The tool refuses a populated output. It
preserves the exact-study/1.0 contract. The final researcher test closed the accepted legacy
publication, so recreate a fresh disposable revision for another full run; do not point these tests at the Luna forms/3 dataset.
The receipt records the exact 19-test selector and the separate four-file Luna acceptance set.

The quality-audit command stages proposals only:
`python -m tools.review_description_quality --package PACKAGE --output NEW_AUDIT_DIRECTORY`.
Publication requires separately authored, bound review receipts for each changed draft. The
`publish(...)` entry point takes predecessor package/publication/catalogue, audit root and a
new revision, writes only changed small artifacts and reuses existing data. It never calls a
provider. The 13 October publication used actual receipts from a separate reviewing agent.

## Storage and remaining work

Initial measured combined usage: **50,658,019,022 bytes (50.658 GB)**. Final measured usage:
**51,506,452,379 bytes (51.506 GB)** at
`2026-10-13T06:33:58.846441+00:00`; measured growth 0.848 GB.
Filesystem free space was 117.468 GB; the guard allowed the
remaining 0.05 GB handoff/commit allowance. All accounting includes repository, shared caches, other review bundles,
temporary test artifacts and databases, counting shared inodes once. The largest simultaneous
new-service reservation is 1.5 GB; foreground verification reserves 0.8 GB. The conservative
additional peak allowance for this pass is 3 GB including intermediate tiny fixtures, inactive
test clusters, runtime databases and logs. Measured snapshots are not an assertion about an
unobserved system-wide peak. No full physical real-data bundle was created.

**Frontend follow-up:** no change to the accepted typed-citation behavior is required for this
publication. Consume `selection` to label curated starting choices in the navigation UI while
preserving continuation and actual edge basis. If packetless historical artifacts must have
full citation-inspector support, eliminate the class fallback: use authoritative typed
subjects from retained resources, or show an explicitly untyped/ambiguous citation and omit
subject-dependent wording. Do not guess a class from an annotation IRI. Current Luna and new
legacy preparations already supply typed packets.

**Implementation correctness:** the requested backend corrections pass their recorded boundaries, including
independent semantic review of the 17 definition additions and the corrected resume regression.
Code and API/browser checks establish only their tested boundaries.

**Joint acceptance:** the 44 required browser checks and exhaustive prepared-data/API parity pass
against the preserved accepted frontend. Curated-navigation labelling and any packetless
historical citation support remain the explicit frontend follow-up above. The unexplained service-read incident and
historical R16 remain open, even if this pass has no reproduced error.

**Launch readiness:** not established. Actual container build/start, production deployment and
production-volume restore remain technical work. Docker's daemon was unavailable; no public
deployment was performed. Manual keyboard/screen-reader evaluation, very narrow enlarged-text
usability, participant/pilot evaluation, owner approval and ontology-admission gates remain
separate and open. Automated accessibility checks do not substitute for those activities.

No push or public deployment was performed.
