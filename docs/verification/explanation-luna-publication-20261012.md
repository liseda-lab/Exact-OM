# Reviewed Luna prose publication — 12 October 2026

The authorized batch is complete and the additive publication is assembled. **251 new Luna
artifacts passed a separate claim review; nine retain their predecessor.** All 260 provider
calls completed with HTTP 200, with no repair, retry or provider fallback. The reported charge
is **$0.230666125**, below the frozen $1.052581125 conservative estimate.

The latest user reply, “Yes I approve please do that”, authorized the full 140-profile and
120-comparison batch. The earlier transfer-approval blocker is resolved. The
[preparation report](explanation-luna-bundle-20261012.md) remains historical evidence of the
state before that approval, rather than evidence that this batch had already run.

Implementation started at `0ec304b3812f1e6f59f64dd0c8d49c3fa981619d`, which contains the Luna
runner, publisher and citation/admission corrections. The frozen plan was prepared from
`90f929a8b48b6ba51caa5b74a91a540db19b9f55`. The reused compiled frontend is from
`885f5510d0358ec64410711d0b1353261346894c`. The commit adding this handoff contains the
new real-data verifier and delivery evidence. Unrelated specification, frontend, review,
storage and cluster work is excluded. Nothing was pushed or deployed publicly.

## Generation and claim review

The model was **`openai/gpt-6-luna`**, medium reasoning, through OpenRouter, with OpenAI as the
only allowed provider. All responses report exactly that model and provider. The saved batch
contains 1,246,997 input tokens and 149,622 output tokens, including reasoning tokens.
The durable ledger contains precisely 260 first attempts, all completed with status 200.
Requests contain the approved NCIT/DOID facts, original citations and frozen profile
inputs, without participant answers, grading keys, matcher scores or candidate ranks. Pair
requests deliberately reuse the predecessor's frozen profile dependencies; newly generated
Luna profiles were not silently substituted into comparison inputs.

The root Codex agent separately read every one of the **292 reviewable claims** against its
exact cited facts, typed labels and relevant missingness. **279 were approved; 13 were
rejected.** One other response failed citation-membership validation before semantic review
and was retained for inspection. These decisions are individually bound to the input,
response, ordered claim inventory and reviewer; neither schema success nor the earlier
12-case diagnostic assigned approval. The reviewer is a separate AI from the API generator
but also built the pipeline. This is **not an external, human or clinical audit**.

| Active coverage | Reviewed prose | Typed templates | Total |
|---|---:|---:|---:|
| Entity profiles | 138 | 2 | 140 |
| Pair comparisons | 114 | 6 | 120 |
| Total | 252 | 8 | 260 |

Of the 252 reviewed artifacts, 251 are new Luna outputs and one is the unchanged predecessor
comparison for Malignant Jejunal Neoplasm / jejunal cancer. There are 280 active reviewed
claims: 279 Luna claims plus that one prior approved claim. The preceding publication had
64 reviewed artifacts and 68 approved claims; all historical artifact bytes and those 68
claims remain available in the old publication and retained dependencies.

The 13 rejected claims include eight whole drafts and four partially accepted drafts (one
whole draft contains two rejected claims). Findings include an uncited *Pediculus humanus
corporis* item, moving “early-onset” from Paget disease to myopathy, a false “only the source”
contrast about reduced NK-cell numbers, unsupported scope conclusions, and name-only or
one-sided comparisons. Accurate but unhelpful statements are distinguished from factual
errors in the [individual review record](explanation-luna-publication-20261012/reviews.json).
No rejected wording was patched silently or admitted as generated prose.

The two retained profile templates are jejunal cancer (`DOID_13499`, invalid generated
citation) and inclusion body myopathy with early-onset Paget disease of bone with or without
frontotemporal dementia 3 (`DOID_0111386`, misplaced onset modifier). The six retained
comparison templates concern C110923/DOID_8886, C110923/DOID_3612, C118423/DOID_205,
C34536/DOID_3136, C123729/DOID_0111966 and C170919/DOID_3218. Templates remain visibly
identified by their existing presentation metadata; they are not counted as Luna successes.

## Exact artifacts and preservation

The generation and review root is:

```text
/Users/pedrocotovio/Library/Caches/exact-om-review/luna-bundle-20261012
```

The existing data root is reused directly:

```text
/Users/pedrocotovio/Library/Caches/exact-om-review/postmigration-20261011/combined
```

| Artifact | Identity |
|---|---|
| Frozen generation plan | `sha256:7f58bd7c31223de889baf0cb5f53ddc27c8c91dc868fd4049024122f1c95c155` |
| New package ID | `sha256:7962c960d604cdffffc9ecd326cf4d3d4ab232244cf9b31eda2887c5a6a43a2b` |
| `package/package-luna-20261012.json` file | `sha256:07781b63d2acd24f2542653112e5a20d75cea30af55685b331af3ea1e9b8907b` |
| `publication-luna-20261012.json` file | `sha256:51500ab8b54615ff938da3e70faa6816af67c6867b1b528f4d2a175cb68bff93` |
| Predecessor `package/package.json` file | `sha256:1e160105df9ab9fcccbc449716066904e03eb93711c57189e896cc728d9a5e70` |
| Predecessor `publication.json` file | `sha256:5eff10207af3ccecbd541840088b022f074df962fd35d3630854c8aa1e743b2a` |

The new catalogue is `catalogue-luna-20261012.json`; changed resources are under
`luna-20261012/resources/`, new prose under `package/luna-20261012/explanations/`, and the
small scope indexes use new identities under `workspaces/`. The
[publication receipt](explanation-luna-publication-20261012/publication.json) lists every
file hash and each predecessor-to-new-artifact disposition. Raw requests/responses,
independent review receipts, authorization and the provider ledger stay in the generation
root. The [generation receipt](explanation-luna-publication-20261012/generation.json) binds
all response files without committing credentials.

All **990 original package artifacts** retain their paths, hashes and protected file
identities; only 251 small prose artifacts are added to the package inventory. No large
file was copied, relinked, unlocked or rewritten. Old publications, databases and sessions
were not retargeted. The new manifest remains in the original package parent so existing
relative locators stay valid. The earlier tiny-archive relocation check remains applicable;
a full physical export was deliberately not allocated.

The [real-data audit](explanation-luna-publication-20261012/preservation.json) compares all
24 cases, 120 presented pairs and 3,443 selected records across old/new frozen resources and
scope indexes. It checks every named score, identity, original fact link, bridge and typed
fact view. Evidence-row and named-score order changed during deterministic SQLite assembly;
values did not. Shared axioms retain each typed focal subject rather than being collapsed
by axiom ID. Case identities, candidates, schedules, questionnaires, grading, policy and
unchanged context locators are identical.

The underlying 30,374-pair / 300-source candidate pool and 1,027,952 selected-feature records
are unchanged. The recovered 1,079 resolved / 2,364 unresolved selected records and 812
recorded / 2,517 withheld comparisons remain intact. Retained ELK artifacts and their partial,
unsupported and unknown distinctions are unchanged. No matcher or reasoner was rerun.

## Verification

**36 affected backend regressions passed** in 37.04 seconds. The earlier implementation
record contains 136 distinct tests, including an actual PostgreSQL process check; the fresh
36 overlap those tests and must not be summed as 172 distinct tests.

The complete API run passed **3,255 certificate-verified HTTPS requests** in two completed
24-case development sessions, each with 12 explanation and 12 baseline cases. Across them,
all 24 cases, 120 pairs and 260 active artifacts were checked against frozen resources,
scoped workspaces and the main API. The verifier resolved 334 distinct scoped citations and
35 referenced original expression ASTs. It checked omitted-type and unadmitted-fact tutorial
rejection, private server grading, foreign/baseline scope denials, partial hierarchy paging,
stale cross-basis cursors, unknown empty ELK results, optional per-case methods, and immutable
JSON/CSV exports. Six interrupted responses recovered with matching complete bodies. No new
Content-Length, protocol or ASGI exceptions were observed; that observation does not establish
the cause of any earlier intermittent reload stall.

The first API attempt failed in the new verifier because it expected the packet's inline
expression AST in a Fact response. The documented Fact contract returns a reference. The
verifier now follows it to the original-axiom endpoint and compares the complete AST as well
as its identity. That failed session and receipt are retained. Static verifier assumptions
about task location, evidence ordering and compound fact identities were also corrected
against the existing contracts. No production response or browser assertion was weakened.

A **fresh complete browser journey passed** in 92.15 seconds using system Chrome and the
unchanged compiled S2 frontend against the new HTTPS/PostgreSQL service with normal CSP.
It exercised setup/forms, every tutorial lesson, reload persistence, wrong assessment retries,
all 24 cases in both conditions, per-case method changes, completion and analysis/4 exports.
A deliberately corrupted tutorial submission tested rejection/recovery; successful progress
was performed through the UI rather than seeded through an API. This negative probe is not
counted as successful generation or as a separate acceptance journey. Automated accessibility
checks embedded in that journey passed; they do not close manual accessibility gates.

The [machine-readable delivery receipt](explanation-luna-publication-20261012/receipt.json)
links generation, reviews, publication lineage, real-data preservation, API, browser,
contract hashes and runtime/startup evidence. The local environment uses PostgreSQL 16.2,
Python 3.12.3, FastAPI 0.116.2, Uvicorn 0.35.0, Pydantic 2.13.4 and pyowl-core 0.2.1. Study
admission took 77.78 seconds and workspace warmup 0.44 seconds; exploration admission took
43.95 seconds. The exact runtime code fingerprint is
`752b91446f125a12af47d9c6f7c7499c3edba40ea85d5d84fd5bfa86491c8985`.

## Storage

The final shared guard measured **50,003,761,249 bytes (50.004 GB)** against the combined
100 GB limit. The generation baseline was 49,779,538,849 bytes; the combined increase was
224,222,400 bytes, including temporary tests and other local activity. The Luna generation,
review and environment directory occupies 119,324,672 bytes, and the additive publication
files occupy 41,037,824 bytes: **about 160 MB together**, including the previously frozen
plan. These figures count the existing large inputs only once in the shared total.

The conservative peak additional reservation was 1 GB, with the shared 5 GB safety margin
and 30 GB filesystem-free reserve retained. The guard monitored foreground generation,
assembly and tests. Detached local services have a recorded 1 GB output reservation and
separate bounded test sessions; this is cooperative accounting, not an operating-system
quota. The snapshot and exact counted roots are retained in
[storage.json](explanation-luna-publication-20261012/storage.json).

## Reproduce and integrate

Use the migrated checkout and retained Python; no installation or frontend rebuild is needed:

```sh
cd /Users/pedrocotovio/Developer/Repos/OM/Exact-OM
PY=/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python
LUNA=/Users/pedrocotovio/Library/Caches/exact-om-review/luna-bundle-20261012
PYTHONDONTWRITEBYTECODE=1 "$PY" -m tools.review_environment status --root "$LUNA/environment"
```

The isolated environment uses PostgreSQL **19603**, study HTTPS **19604**, and exploration
HTTPS **19605**, all loopback. Its source publication points to the new additive revision;
it creates a fresh development-review revision for test sessions. It reuses
`postmigration-20261011/frontend-s2-885f551` and the native PostgreSQL/Python runtimes. The
private `environment.json` and `browser-harness.json` contain local credentials and must not
be committed or shared. The local certificate is under `environment/tls/`.

To start a stopped environment, first run the shared storage check with a 1 GB growth
reservation, then `"$PY" -m tools.review_environment start --root "$LUNA/environment"`.
Use `fresh-session --root "$LUNA/environment"` for a new private invitation. Preserve its
database on stop/restart. The owner-aware `stop` command affects only this environment.
The handoff's machine receipt records its exact review revision and runtime hashes.

The committed `tools/verify_luna_publication.py` can run against a fresh output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 "$PY" -m tools.review_storage_budget run --growth-gb 1 -- \
  "$PY" -B -m tools.verify_luna_publication \
  --root "$LUNA" --environment "$LUNA/environment" --output "$LUNA/api-verification-next" \
  --prior-static /Users/pedrocotovio/Library/Caches/exact-om-review/postmigration-20261011/api-verification-3/static.json
```

The verifier only creates development-review sessions. It compares the two allocated
counterbalances and preserves duplicates at the first case instead of rewriting assignments.
Provider generation must not be rerun to reproduce the served publication. The publisher
can verify/resume from the frozen reviews and assembly receipt without additional paid calls.

Frontend S2 is already implemented in the reused build. This prose revision adds no wire
field or schema shape. Select the new package/publication identity, clear any old artifact
cache when changing publications, and render the returned approved wording, presentation
kind and exact citations. Existing sessions keep their old revision. Newly added typed
reference-label citations must resolve through the existing support-packet contract. No
frontend source changes are included here. The [two concrete API examples](explanation-luna-publication-20261012/examples.json)
show an approved profile and the corrected Blau comparison with their new identities.

## Acceptance boundaries

**Implementation correctness:** the new prose revision is delivered and self-verified against
real artifacts and actual service boundaries. All planned calls completed, every reviewable
claim has a bound disposition, rejected prose is excluded, and historical layers are preserved.

**Joint acceptance:** the specified API checks and current compiled-frontend browser journey
pass on the new revision. The separate AI content review is recorded. Independent joint
acceptance and the human-only checks remain open. There is no newly identified frontend
implementation blocker from this delivery; integration requires selecting the new revision.

**Launch readiness:** not established. This remains a provisional development-review bundle,
with research correctness metrics disabled. Provider authorization is not study-owner approval.

Partial ELK remains optional and incomplete; an absent edge means unknown. DOID import
scope and the missing property/individual/sparse-candidate review coverage from the backend
handoff remain open. Source definitions can be imperfect and retain dated terminology;
semantic fidelity to them does not establish clinical correctness. Manual accessibility,
usability/pilot review, study-owner approval, deployment/restore, ontology admission and
release gates have not been closed by this batch.
