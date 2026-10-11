# Post-migration backend corrections and S2 contract handoff

**12 October 2026.** Backend C01–C03 and the backend portion of C05 are implemented.
This is a development-review publication and a backend self-verification record.
It does not supersede the independent review with a claim of independent acceptance,
and it does not establish rendered S2 acceptance or launch readiness.

The reviewed checkout was `6d9961e2aec3d08645262d91aed8c9febca203b8`.
Implementation commits: `f8131bce46578b9d0f9dd6e92945be7599d0ca5c`, `780af0c29dae37be8357f36eb178d39270e7e406`, `66a33420a144a4fdc125a581dc54057e1592a8ed`, `bc8e57cd771bc13551a90486edadcde87ce09fa3`.
The subsequent verification/handoff commit is the commit adding this file; its exact
identity can be obtained with `git log -1 --format=%H -- docs/verification/explanation-postmigration-backend-corrections.md`.
The [machine receipt](explanation-postmigration-backend-corrections/receipt.json) binds
source fingerprints, assembler/verifier hashes, schemas, artifact identities and test results.
Unrelated reviewer specifications, cleanup tools, cluster scripts and user changes were not included.

## Dispositions

| Requirement | Finding and cause | Delivered disposition |
|---|---|---|
| C01, P1, backend | Prose preparation reused a plain decision index and lost the earlier original-fact enrichment. Raw producer null qualifiers also missed canonical retained feature terms. | Combined immutable publication preserves the evidence predecessor's complete decision database and approved prose. Preparation rejects layer-dropping reuse; composition binds both predecessors, contexts, policy, renderer and ELK inputs. |
| C02, P1, backend contract | Complete-only admission could not consume the retained incomplete ELK taxonomy. | Separate, additive `elk_inferred_partial` basis with `exact-partial-hierarchy/1`, pinned native input/output admission, bounded prepared indexes and explicit unknowns. No full-ontology classification was restarted. |
| C03, P1/P2, backend | Missing referenced labels encouraged guessed descriptions; excerpt templates repeated warnings/truncated words; one-sided claims lacked display semantics. | Bound original labels and typed necessary-condition templates; word-boundary excerpts; explicit support/presentation metadata. Approved claims and original independent decisions remain unchanged. |
| C04, P1, frontend | The retained S0 frontend does not implement the negotiated review protocol and new presentation contracts. | Remains frontend S2 work; exact instructions below. |
| C05, P1/P2, both | Self-parity missed cross-publication regression and real hierarchy continuation. | Predecessor preservation, relocation, native fixtures, runtime/schema regressions and actual HTTPS/PostgreSQL parity recorded separately. Rendered/human acceptance remains open. |

## Delivered data and preservation

Use the existing root; do not create a new full physical bundle:

```text
/Users/pedrocotovio/Library/Caches/exact-om-review/postmigration-20261011/combined
```

| Identity | SHA-256 |
|---|---|
| Combined `publication.json` | `5eff10207af3ccecbd541840088b022f074df962fd35d3630854c8aa1e743b2a` |
| Combined `package/package.json` file | `1e160105df9ab9fcccbc449716066904e03eb93711c57189e896cc728d9a5e70` |
| Content-addressed package ID | `sha256:b4027297f62b41066cca3e4c529c0a28a9984c945df444491ca9a94790a9a7fb` |
| Evidence predecessor publication | `de980187339840d0f623eb8d38162bca0c08f50d32707ae691fd5a4201286dc0` |
| Reviewed-prose predecessor publication | `beb0d85c4028d6f2d567e9937346d1fce3e4cff17282d69fa5b14389c83727d2` |

Source revision: `development-review-combined-20261011`.
The full **30,374 pairs / 300 sources / 1,027,952 selected-feature records** retain their
identities, typed endpoints, scores, ranks, decisions, exact-match events and feature values.
All rows were compared across both predecessors. The resulting decision database is
byte-identical to the recovered evidence database; only its enclosing delivery manifest changes.

For the 120 presented pairs, exact per-feature/per-bridge comparisons restore
**1,079 resolved / 2,364 unresolved features** and **812 recorded / 2,517 withheld bridges**.
That recovers 778 original-fact mappings and 570 recorded bridges lost by the prose-only revision.
All 3,443 producer `graph_projection` values remain `not_recorded`.
A recorded channel comparison does not imply a drawable feature correspondence.

All **68 admitted claims in 64 independently reviewed artifacts** are preserved byte-for-byte.
The delivered 260 artifacts contain 64 independently reviewed and 196 deterministic artifacts,
with no remaining served factual fallback in this selected bundle. This is not a claim that
the failed/rejected provider outputs succeeded: the original 86 jobs, 104 rejected draft claims
and independent decisions remain retained. There were no new paid/provider requests.
Changed deterministic outputs have new packet, claim, explanation and rendering identities;
old review approval hashes were not transferred to changed text.

The catalogue, public review/About summary, portable package, frozen resources and workspace
indexes carry the combined layer identity and corrected counts. Participant summaries still
withhold researcher case catalogues. The historical `software_version` is a protocol/build
label, not a Git implementation fingerprint.

## C02: partial ELK contract

Retained native job: `14455`, from
`liscluster:/home/pgcotovio/exact-elk-inference/20261010-final-elk-7bca91/`.
The native packages are `pyelk-reasoner 0.2.1` and `pyowl-core 0.2.1`.
The new `exact-elk-partial-admission/1` receipts verify the submission manifest, runner,
input/export receipts, exact policy/scope, native execution phases, options and taxonomy hashes.
The old receipts still say `completed_incomplete` and
`eligible_for_integration_review=false`.

| Ontology | Native group nodes / direct group edges | Served named-member edges | Additional direct pairs relative to literal assertions |
|---|---:|---:|---:|
| NCIT | 211,960 / 421,430 | 246,974 | 37,573 |
| DOID | 16,375 / 35,333 | 17,224 | 4 |

Served edges are exactly the retained direct group edges expanded over admitted named members.
No ancestor closure is substituted. Native top/bottom groups remain in inference metadata;
non-admitted built-in endpoints are excluded from served named-member edges.
Six DOID SO classes are unrepresented and remain unknown; the receipt names each one.
All NCIT admitted classes are represented. There are also 161 NCIT and two DOID asserted
pairs absent from the partial taxonomy; these differences are not negative entailment.

Positive results mean returned by the pinned ELK supported projection. Directness is relative
to that returned partial taxonomy. Exported per-edge justifications are unavailable, so the
contract does not assert full-ontology entailment, whole-ontology consistency, complete
satisfiability or negative conclusions from missing edges. Consistency's returned `true`
and its `complete=false` are separate fields. NCIT datatype/datatype-definition/positive-union
and DOID exact/minimum-cardinality/positive-union limitations remain explicit.
DOID root/derived-input and unresolved original import-closure admission remain separate limits.

The pinned implementation was inspected and four bounded native fixtures were exercised:
supported subclasses/equivalence; mixed unsupported cardinality; positive union; datatype.
The latter retained positive supported results with incomplete flags. No NCIT/DOID
classification or HermiT retry was started. Existing bounded legacy navigation unit fixtures
are compatibility checks, not resumed cluster jobs.

The main API advertises per-ontology navigation; scoped capabilities expose `bases` and
`navigation[ontologyVersion].partial_inference`. The old aggregate `reasoner` field and
`reasoner_inferred` basis continue to describe complete-only inference and may still be
failed/unavailable. They must not disable an advertised partial ELK choice.

| Read | Main API suffix | Scoped workspace suffix |
|---|---|---|
| Entry points | `/ontologies/{version}/entry-points` | `/entry-points` |
| Parents, children, bounded ancestors | `/hierarchy` | `/hierarchy` |
| Equivalent members | `/ontologies/{version}/equivalents` | `/equivalents` |

Prefixes are `/api/v1` and `/api/v1/study/workspace/{scope_id}`.
Pass the exact typed entity and `basis=elk_inferred_partial`; scoped entry/equivalent routes
also take `ontology_version_id`. Hierarchy directions are `parents`, `children`, `ancestors`.
Scoped hierarchy pages allow up to 100 items; main ancestor graphs allow up to 1,000 nodes. Ancestor cursors also bind the traversal budget. Use returned cursors verbatim. They bind ontology, kind, basis, prepared artifact, policy,
query and—in study—session generation/presentation/scope. Cross-basis reuse returns
**409 `stale_cursor`**. Baseline and other-case reads return 403.
Malformed cursors use their existing 422 error contract.

A successful partial response always remains `status=partial`, including an empty result.
Distinguish unsupported kind, not-prepared/failed inference and partial empty; an empty partial
list is not proof of no parent/equivalent. Request handling reads prepared indexes only.
The relocation probe confirms no native OWL, reasoner or matcher imports on the request path.

Frozen explanation-resource hierarchy facts remain original-fact-backed. Partial navigation
is stored in the immutable, hash-bound workspace context sidecar, shared by the main/scoped
services; it is not mislabelled as a cited original axiom.

## C03: description contract and examples

The additive `exact-description-presentation/1` metadata accompanies existing
`exact-description/2` preparation/support fields. Use all of:

- `preparation_kind`, `support_status` and `support_reasons`;
- `presentation.kind`, `covered_sides`, per-claim `claim_sides`;
- `render_revision`, `review_receipt_sha256`, exact claim/citation IDs;
- `scope_note` once per description/comparison area and `relation_asserted=false`.

The presentation kinds distinguish reviewed profiles/comparisons, necessary conditions,
available facts, paired context, source-only/target-only profiles and unavailable output.
A profile's “source” role means its first artifact entity, which may be the target ontology
in the surrounding workspace. Map roles from exact artifact entities, not column position.

All 12 selected profiles without an original definition now have conservative necessary
conditions supported by their original subclass axioms and cited original parent labels.
For example, DOID:3218 is described through its recorded parent “small intestine benign
neoplasm”; DOID:3612 through “retinal disease”. These are typed summaries, not independently
reviewed medical definitions. The coverage receipt enumerates all 12 with exact citations.
Label-only synthetic negatives remain Available facts with a reason; unsupported medical
claims remain rejected.

Referenced class/property labels are limited to exact typed references in admitted axioms
(60 labels / 48 KiB), verified against the frozen context and policy, and bound in the packet.
Universal conditions do not imply existence; minimum cardinality preserves distinctness;
equivalence templates are not reduced to one-way necessity. Definition excerpts end on word
boundaries and full original text remains available in the fact inspector.

The surviving comparison in review-07 is `source_only_profile`
(`sha256:c9ed6e2359bb5ffaec32721793700d053870e9d609d998c6041e9aae34d4977c`);
review-12 is `target_only_profile`
(`sha256:189af064760204cce982ec0601b1a5b64b053b512ac7ca572516dbf84a3ea7a1`).
Display these as one-sided information, not a semantic difference. Paired excerpts remain
`paired_context`. Do not remove substrings from reviewed text in the browser.

The runtime schemas and exact positive/negative examples are linked in the receipt and
[contract examples](explanation-postmigration-backend-corrections/contract-examples.json).
Frozen resources add typed `pair_metadata[candidate_id]` and presentation metadata on claims;
historical resources remain compatible.

## Verification and its limits

- **Late live-boundary corrections:** the scoped ancestor API regression and duplicate-ID guard had three expected failures before correction; all 51 affected tests passed afterwards. The repeated full-index hashing regression failed before correction; all 76 navigation/context/scoped tests passed afterwards, including tamper rejection. These focused counts overlap earlier suites and are not added into a unique total.
- **Broad backend regressions before the late corrections:** 256 passed, one skipped, 297.48 seconds. The skipped v2 PostgreSQL backup/restore test requires a dedicated URL/PGBIN/PGDATA; it was not claimed as executed. The separate actual PostgreSQL lifecycle/restart test passed.
- **Frontend unit regressions:** 112 passed using retained Node 24.19.0, TypeScript 5.9.3 and React 19.1.0. No frontend source, assertion or dependency installation changed. Two initial test-runner dependency-resolution failures are retained; the final runner passed.
- **Contracts:** runtime schema/OpenAPI freeze check passed. Hashes are in contracts.json. [Repeatable backend checks](explanation-postmigration-backend-corrections/verification-commands.md) list the exact suite modules and schema command.
- **Real native package fixtures:** four bounded native ELK fixtures passed; their original output and implementation hashes are retained.
- **Preservation/relocation/resume:** all 120 selected pairs against both predecessors, all 30,374 complete-pool pairs and 64 reviewed artifact hashes; all native taxonomy members/edges; relocated package and frozen scopes; completed composition resume with unchanged publication/package/catalogue hashes.
- **Actual certificate-verified HTTPS/PostgreSQL:** 2 complete 24-case test journeys, 2661 checked responses, both conditions; all 120 unique presented pairs and all 260 descriptions compared across main/scoped/frozen boundaries. The real 120-second tutorial gate, deliberate rejection/recovery, five wrong retries, consultation draft clearing, completion and exports were exercised. Results, ordered hierarchy page unions, useful basis examples, negative scope/cursor checks and per-operation timing are in api-summary.json.
- **Rendered browser acceptance:** not run against an S2 build; the retained S0 frontend remains incompatible. Automated test answers and API state transitions are not a human/browser study.

Checks distinguish synthetic contract fixtures, actual retained ontology data, direct native
package fixtures, HTTPS/PostgreSQL integration and rendered acceptance. The first combined
HTTPS attempt is retained as a verifier failure: it expected 400 for a cross-basis cursor;
the service correctly returned 409. The corrected verifier requires the exact
`stale_cursor` error and does not relax the rejection.

Interrupted-response probes close an authenticated body after its first 128 bytes, then
require identical subsequent complete bytes, valid Content-Length, normal CSP and usable
session state. This does not prove browser reload causation or server cancellation behavior.
Historical R16 remains unresolved/not reproduced; no new middleware/dependency change is
claimed to fix it.

The preserved prior full entity-inventory checks are reused; the new hierarchy checks
explicitly exercise small-page parent/child continuation and compare full ordered unions
with native index rows. Useful real differences are recorded for at least three entities
in each ontology. Legacy complete-only refusal, partial-empty unknowns, unsupported types,
policy/input/scope/checksum tampering, malformed/truncated taxonomy, interrupted export,
inconsistency, cursor isolation and unrelated scope denial are covered separately.

The live run also exposed a missing scoped ancestor direction, corrected without rebuilding the data. Self-review added duplicate selected-ID rejection to composition. A 474 MB navigation rehash on every scoped request was replaced by a bounded path/inode/size/mtime/ctime digest cache. Expected hashes, stored metadata bindings and every-read stat checks still run; changed files fail closed.

Fresh process startup (including preflight and initial study admission, with existing filesystem caches) took **144.08 seconds**. This is not a clean-filesystem cold-start benchmark. Scoped basis-navigation median request time changed from **1.289 seconds** in the failed pre-fix run to **0.036 seconds** in the completed run. These are diagnostic samples with different session/cache state, not a controlled benchmark.

API request timing is recorded per operation. Whole journey duration includes a real
120-second tutorial gate per session and exhaustive diagnostic reads; it is neither human
engagement nor rendered usable-case latency. Prior cold/warm startup evidence (about
118/93 seconds) remains historical; fresh UI cold/warm/usable-case and rendering measurements
belong to the S2 joint run. The receipt includes sampled resident memory, not a claimed peak.

## Storage, reproduction and startup

Initial measured combined use was **33.474 GB**; the delivery measurement is **48.626 GB** with **164.179 GB** filesystem free. The largest recorded guard measurement was **48.626 GB**. The maximum conservative reservation across recorded checks was **67.467 GB**, below the 95 GB admission threshold and 100 GB user limit. These are measured points and peak estimates, not an exact instantaneous peak. The [storage ledger](explanation-postmigration-backend-corrections/storage.json) retains the raw-receipt locations and every estimate.

Unchanged artifacts use APFS clones with distinct private inodes, preserving portable package
paths. No protected source inode was unlocked, chmodded or modified. Before private mutation,
the copy procedure checks a distinct inode and link count one, then clears immutable flags
only on that copy. Source identity/protection is checked around each copy, and the final
receipt compares all five protected groups against the cleanup baseline.
No cleanup was repeated, no full physical bundle was duplicated, and no dependency runtime
was reinstalled. Logical APFS accounting is conservative and is not a physical-allocation claim.

The isolated review environment is:
`~/Library/Caches/exact-om-review/postmigration-20261011/environment-r3`.
It serves PostgreSQL 19423, study HTTPS 19424 and exploration HTTPS 19425 on loopback.
Test publication: `review-20261011T230854-0752049331`, SHA-256
`16578bf285162d4f1c905bfef1da97fddbbee77a0509dd8c0f07714225dc6689`.
The source publication remains separately frozen. TLS is certificate-verified and CSP is normal.
Existing environments/databases/sessions were not reset or replaced.

From `/Users/pedrocotovio/Developer/Repos/OM/Exact-OM`:

```sh
PY=/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python
REVIEW=/Users/pedrocotovio/Library/Caches/exact-om-review/postmigration-20261011
export PYTHONDONTWRITEBYTECODE=1
"$PY" -m tools.review_storage_budget check --growth-gb 2
"$PY" -m tools.review_environment status --root "$REVIEW/environment-r3"
```

Services are left available. If stopped, start with the existing explicit bounded server
reservation in `environment-r3/storage-reservation.json` and the inherited per-file limit:

```sh
"$PY" -c 'import resource; resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728)); resource.setrlimit(resource.RLIMIT_CORE,(0,0)); from tools.review_environment import main; main()' start --root "$REVIEW/environment-r3"
```

The detached service exception reserves 2 GB of accounted output. PostgreSQL WAL uses configured
128 MB maximum/32 MB minimum targets (not a hard aggregate cap). Service files have the inherited
128 MiB file limit; the storage guard still supplies the aggregate check. These are development
bounds, not an operating-system aggregate quota. Recheck growth over extended use.
Use the owner-aware `stop` action, not process-name killing. Never print or commit
`environment.json`, private keys, invitation files, cookies or participant identifiers.

For a fresh private test invitation:
`"$PY" -m tools.review_environment fresh-session --root "$REVIEW/environment-r3"`.
The earlier failed test environment was stopped through its owner-aware lifecycle with its database/sessions retained; a second prepared environment was never started. The final environment uses the same data and runtime with the corrected source fingerprint. The retained compiled frontend is S0 (hash in receipt); it intentionally fails new
review-contract negotiation. This instruction does not mean that S0 can complete this study.
After S2, prepare a **separate environment binding the new compiled frontend** with
`tools.review_environment prepare --publication "$REVIEW/combined/publication.json"
--assets "$REVIEW/combined" --package "$REVIEW/combined/package/package.json"
--frontend /absolute/path/to/new/compiled/frontend --root /absolute/path/to/new/environment`,
Use `--python "$PY"` and
`--pgbin /Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/lib/python3.12/site-packages/pgserver/pginstall/bin`,
with three checked-unused ports passed through `--postgres-port`, `--study-port` and `--exploration-port`.
Use the storage guard and bounded output reservation. Reuse these artifacts; preserve this
backend review environment as evidence instead of overwriting its frontend fingerprint.

Completed composition can be resumed without replacing the delivered publication:

```sh
"$PY" -m tools.review_storage_budget run --growth-gb 1 -- "$PY" -m tools.compose_review_publication \
  /Users/pedrocotovio/Library/Caches/exact-om-review/s1-prose-20261010/prepared-input \
  /Users/pedrocotovio/Library/Caches/exact-om-review/s1-followup-20261010/refreshed-input-v2 \
  "$REVIEW/elk" "$REVIEW/combined" --revision development-review-combined-20261011
```

The resume verifies source/dependency and completed publication/package/catalogue hashes.
A new context/policy or an additional evidence layer requires explicit admission/composition;
plain preparation refuses to discard a delivered enrichment. Relocation was tested by
renaming this root, opening prepared main APIs and all frozen scopes, then restoring the path,
without another physical copy.

To repeat the exhaustive API verifier, use a **new output directory** and the same isolated
environment; the tool creates test-only sessions and retains failures:

```sh
"$PY" -m tools.review_storage_budget check --growth-gb 2
"$PY" -m tools.review_storage_budget run --growth-gb 2 -- "$PY" -m tools.verify_composed_review \
  --environment "$REVIEW/environment-r3" --combined "$REVIEW/combined" \
  --evidence /Users/pedrocotovio/Library/Caches/exact-om-review/s1-followup-20261010/refreshed-input-v2 \
  --prose /Users/pedrocotovio/Library/Caches/exact-om-review/s1-prose-20261010/prepared-input \
  --output "$REVIEW/api-verification-new"
```

## Exact remaining frontend S2 and acceptance work

1. Implement all spec-21 S2 behavior before sending `client_contract: study-review/1`.
   Support `exact-study-forms/3`, `setup/3`, `consultation/3` and `study-methods/3`, server question/option ordering,
   optional combinable methods, conditional Other clearing and export parity. Keep Protégé
   a recommendation; familiarity must never imply per-case use. Preserve historical sessions.
2. Add the negotiated partial ELK mode alongside asserted/structural modes, using the
   per-ontology capability and exact partial metadata. Label it **ELK inferred · partial**,
   with one concise scope explanation and expandable details. Preserve independent side
   focus, typed URLs and discovered entry points; treat empty partial results as unknown.
3. Render original definitions, approved prose, typed necessary conditions, paired excerpts
   and Available facts distinctly in the shared main/tutorial/study workspace. Use exact
   citations and one-sided presentation roles; retain reviewed wording and compact provenance.
4. Render recovered original facts and the recorded bridge types faithfully. Keep unresolved
   and unprojectable evidence visible. The eight zero-evidence pairs and 12 no-definition
   profiles are enumerated in the coverage receipt for real graph/list checks; none resolves
   the **three missing sparse-source stratum examples**. A fresh read-only scan of the full 30,374-pair retained pool found no pair with one or two selected features; the smallest nonzero count is 15, while 155 pairs have zero. Zero does not prove an exported sparse example, so the quota remains open (sparse-pool-probe.json). Dense/long-label examples remain
   in the catalogue. Reveal/focus Decision trace and preserve accepted tutorial rejection/
   assessment-retry recovery without false original-fact actions.
5. Build only a bounded new frontend output, bind it to this exact data and backend source,
   and run a fresh rendered journey with no API-seeded completion: training, deliberate
   rejection and recovery, five assessment items with wrong retries, both conditions,
   changing per-case methods, completion and exports. Exercise reload, slow/interrupted
   responses, two-tab conflicts and resumption.
6. Perform the spec-23 main/study/tutorial visual comparisons: dark/light, 1280×720 and
   1440×900, 200% text, 320/375 px, keyboard operation and accessible graph list. Save real-data
   screenshots and performance evidence with exact build/publication identities.

**Implementation correctness:** backend behavior is implemented; the recorded evidence covers
self-verification and explicit limits. **Joint acceptance:** frontend S2 and the fresh rendered
journey remain required. **Launch readiness:** not established. Independent combined review,
manual accessibility/screen-reader audit, usability/pilot and study-owner approval, deployment/
restore validation, final study data approval and original ontology-admission gates remain open.
No public push, public deployment, recruitment or final matcher experiment was performed.
