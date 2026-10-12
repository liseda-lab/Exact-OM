# Postmigration frontend S2 integration — handoff, 2026-10-12

**Status: frontend S2 (spec 21 against the spec 23 C01–C04 backend) is implemented, built into a
new bounded frontend output and bound to the verified combined publication in a separate review
environment.** A fresh, unseeded browser journey passes there: forms/3 setup and background,
every tutorial lesson with a deliberate rejection and recovery, wrong assessment answers
retried, all 24 cases in both conditions with changing per-case methods, completion, and
analysis/4 exports with and without named methods. Real-data hierarchy continuation,
description kinds, graphs, Decision trace reveal and phone/200% text layouts were inspected on
the actual publication.

This is a development review build. It is **not** a claim of launch readiness,
accessibility conformance, usability, research data quality or completed human audits. Test
sessions in the review environment are development-review sessions, not research data.

## Bindings

| Identity | Value |
|---|---|
| Commits (on `dev`, not pushed) | `441cd3d` implementation · `9d36719` browser specs · `0f38a5f` browser card gutter · `129e862` sticky bars at enlarged text · `885f551` repeated comparison records · this handoff |
| Frontend source tree (`git rev-parse 885f551:explanations_visualizer`) | `a19632077d68b798ed4a15dea4114b0b25996139` |
| Compiled frontend output | `~/Library/Caches/exact-om-review/postmigration-20261011/frontend-s2-885f551` (2.3 MB, Next.js 15.5.26 static export) |
| Frontend tree hash bound by the environment | `b5511d27dabb4ba8d439580db5b4dda258493d87882ab6fcbe3dbd68fe21fc87` |
| Combined source `publication.json` | `5eff10207af3ccecbd541840088b022f074df962fd35d3630854c8aa1e743b2a` — equals the [backend handoff](explanation-postmigration-backend-corrections.md#delivered-data-and-preservation) |
| Combined `package/package.json` | `1e160105df9ab9fcccbc449716066904e03eb93711c57189e896cc728d9a5e70` — equals the backend handoff |
| Backend source fingerprint | `bb771d7fa8e40032af39771c62cadac4d4e5b343a9b9eb66348e665f088f811d` at commit `885f551` — equals the reviewed backend receipt; S2 changed no backend file |
| Review environment | `~/Library/Caches/exact-om-review/postmigration-20261011/environment-s2e` — PostgreSQL 19493, study HTTPS 19494, exploration HTTPS 19495 (loopback) |
| Environment source checkout | clean detached worktree of `885f551` at `~/Library/Caches/exact-om-dev/s2dev/source-885f551` (44 MB); the services run from it |
| Environment test publication | `review-20261012T205640-ace4cb08bd`, SHA-256 `ae6c0cf09ab6595ea37f85df1060614268fb703a98e30e169f930b71d791e980` (a test publication of the combined source; the source stays frozen) |

The data root was reused in place: no full bundle was rebuilt or copied, no generation was
rerun and no reasoner job was started. `environment-r3` (the backend review environment with
the S0 frontend) was not touched and is still running as rollback evidence.

Intermediate S2 environments were superseded and stopped through the owner-aware lifecycle,
with their databases retained:
- `environment-s2` (build of `9d36719`), `environment-s2b` (`0f38a5f`) and `environment-s2c`
  (`129e862`): superseded by the later fixes;
- `environment-s2d` (`885f551`): it bound and ran uncommitted backend edits that another
  session made in the shared checkout during this work (`exact_inspect/description_review.py`
  and `exact_inspect/study/resources.py`, fingerprint `c87ce0b9…`).

The final environment was therefore prepared from a clean worktree of the commit. The other
session's edits and its commit `90f929a` (a model-comparison study under `docs/`, `tools/` and
`tests/`) are not part of S2 and were left untouched.

## Completed work

### 1. Review contract, forms/3 and methods/3 (21 F01, 23 C04)

- The session exchange declares `client_contract: "study-review/1"`. It is declared only now
  that forms/3, setup/3, consultation/3, study-methods/3 and the partial-hierarchy contract are
  rendered. A 409 for an unsupported client is shown as "This page needs to be updated for your
  study", distinct from an unusable invitation; the private link is kept for reload.
- Historical publications keep working: forms/1, forms/2 and component variants
  (`exact-study-forms/2-<variant>`) are recognised by family. Unknown protocol versions fail
  visibly before collecting answers.
- Setup/3: the familiarity question in server order; "None of these" is exclusive; Other needs a
  name (inline error and focus). Protégé is described as a recommendation; familiarity never
  populates actual use.
- Background: server-ordered questions and options, the agreed experience bands, no "Prefer
  not to say" where forms/3 removed it, conditional Other text fields with length limits.
- Consultation/3: strict body (`consultation_version`, `methods` ⊆ protege/plain_files/
  other_method, `other_method`, optional `resource_scope`); methods are optional, combinable and
  can change in every case; copying the previous case is explicit only.
- Tutorial practice reports use the methods/3 dictionary; Other needs a name there too.
- Researcher exports request `exact-study-analysis/4` for forms/3 revisions, with an explicit,
  restricted opt-in for named Other methods; the returned manifest is checked before saving.

### 2. One workspace for the main app, the tutorial and explanation cases (21 F05–F07)

The main app, the tutorial and explanation-condition cases render the same `PairWorkspace`
(cards, comparison, Hierarchy, Evidence, Evidence graph). Baseline cases still render no
workspace and make no workspace request (asserted by the journey).

### 3. Partial ELK, entry points, independent focus, paged navigation (21 F06, 23 C02)

- Modes per ontology: Asserted, Structural, and "ELK inferred · partial" from
  `ontology.capabilities.elk_inferred_partial` (main) or
  `navigation[ontologyVersion].partial_inference` and `bases` (study scopes). A failed
  complete-only reasoner is not offered as a disabled competitor and never disables partial ELK.
- ELK details (version, unsupported constructs, "a missing link is unknown, not absent"), the
  difference from asserted parents, equivalence groups, and empty partial results worded as
  unknown. Properties fall back to Asserted with the reason.
- Discovered, paged top-level entry points; a sole entry is selected automatically.
- Parents and children page through the service's cursors; a basis change starts again from the
  first page and never reuses a cursor. Each side keeps its own focus and mode.
- The mode selector is a keyboard radiogroup (one tab stop, arrow keys); an unavailable mode's
  accessible name includes its reason.

### 4. Descriptions by what they are (21 F05, 23 C03)

Presentation follows `exact-description/2` and `exact-description-presentation/1`; nothing is
inferred from category names and approved wording is shown verbatim.

| Prepared text | Title | Tag |
|---|---|---|
| independently reviewed profile | Description | Generated · reviewed |
| typed necessary conditions | Necessary conditions | From original axioms |
| quoted definitions (`fact_extract`) | Definitions side by side / One-sided information | Original excerpts |
| fallback facts | Available facts | Original facts |

- Original definitions stay separate. Without one, a prepared description takes the meaning
  area with a note worded as the scope allows ("No original definition is included in this
  view" for partial scopes).
- Paired excerpts are laid out under the concept each quotes; one-sided text says it is not a
  difference; the scope note is shown once.
- Roles resolve from the artifact's entity identities, not column position (unit-tested with a
  reversed artifact).
- Citations keep their exact fact identities. Citations to a referenced class's label now name
  it ("Label of DOID:7505") from the artifact's own support packet instead of "cited fact 2".

### 5. Graph and list on real data (21 F08, 22 B05)

- Comparison lines come only from recorded `exact-evidence-metadata/1` bridges: channel
  comparisons between the compared concepts, feature correspondences between two drawn feature
  nodes. Nothing is inferred from coincident labels or channels.
- Undrawn and withheld records stay listed, summarised by channel with "List them".
- Metadata is read once per pair and only where a study scope declares the contract.
- A comparison recorded more than once with identical content is one line and one row,
  marked "recorded N×"; summary counts remain record counts.
- The empty state follows the evidence status: an unexported selection is not presented as
  "no features".
- Decision trace: "Open decision trace" reveals and focuses the trace and holds it in place while
  late content loads. Stages absent from the bundle's decision log read "Not recorded in this
  bundle".
- Dark palette, density and list grouping were inspected on the real dense, sparse and
  long-label pairs.

### 6. Tutorial guidance and recovery (21 F02)

Accepted S0 behaviour is preserved: adjacent task bar with Show me, Why?, acknowledged progress
and visible Back/return controls; invalid graph/context clicks submit no false original-fact
identity. Two corrections came from real-data runs:

- A refused tutorial record no longer discards valid evidence sent with it: only the
  requirements the server names are dropped and the rest is resent.
- The recovery copy (follow-up-2 P3) is presented once. During a conflict, one alert carries
  the download. Afterwards a retained copy is a quiet "kept in this browser" line, and a write
  the server refused as invalid is not archived as unsaved work.

## Defects found during self-review and fixed

Found by running the real publication rather than the synthetic fixture:

1. Quoted definitions were tagged "From original axioms" (now "Original excerpts").
2. A profile claimed "No original definition in this ontology" where the scope is partial.
3. Citation chips read "cited fact 2" for referenced labels.
4. The zero-evidence graph said "Exact recorded no selected features" for 8 pairs whose selection
   was **not exported**.
5. The recovery archive treated a server-refused tutorial record as an "earlier conflict" and
   kept a warning for the rest of the study.
6. A refused tutorial record discarded valid evidence sent with it.
7. Decision trace reveal landed off-screen when cards loaded after the scroll.
8. Component-variant form versions (`exact-study-forms/2-cmp`) were refused as unsupported.
9. Study scopes without the metadata contract received `/evidence-metadata` requests (404).
10. "About this ontology" and the top-level list ran into the browser card border.
11. At 320 px and 200% text the sticky task bar grew to the full viewport and the answer bar
    covered the rest (a focused tab was fully obscured). Sticky bars now release when taller
    than their share of the viewport.
12. Comparisons recorded twice with identical content were drawn as overlapping lines and
    listed as indistinguishable rows.

## Real-data observations (combined publication, all 120 presented pairs)

| Measure | Result |
|---|---|
| Selected evidence | 112 pairs `available`; **8 pairs `not_exported`** ("No selected evidence was exported") |
| Recorded comparisons | 812 recorded: **494 drawn** (211 channel comparisons, 283 feature correspondences), **318 listed, not drawn** (an end feature has no drawable node); **2,517 withheld**, listed only |
| Repeated records | 62 pairs repeat a recorded comparison exactly except for its ID (88 extra of 812, more among withheld): shown once as "recorded N×" (464 drawn lines, 260 undrawn rows), counts stay record counts |
| Graph size | at most 12 nodes / 17 edges per pair; 31 of 3,443 selected records have no supported graph form and stay in the list |
| Comparison presentation | 104 paired context (3 reviewed), 12 source-only (1 reviewed), 1 target-only (reviewed), 3 reviewed comparisons |
| Entity profiles (140 entities) | 56 reviewed profiles, 84 typed necessary conditions; none missing |
| Decision trace | **0 of 120** pairs carry a recorded stage event; every stage is `historical_stage_record_not_exported`; saved alignment and relation are shown |
| Hierarchy continuation | DOID_0080015: 103 asserted children (3 pages); NCIT C9305: 72 ELK children (2 pages); no duplicates in either |
| Entry points (asserted) | DOID 4,189 · NCIT 23,868; DOID's list is dominated by obsolete classes without asserted parents |

## Verification

All browser runs used system Chrome through Playwright against HTTPS services with normal CSP.

| Check | Result |
|---|---|
| Type check (`tsc --noEmit`) | passed on every commit |
| Unit (`npm run test:unit`) | **122 passed**, 0 failed (new: protocol families, partial-ELK capability, metadata contract gating, description roles/tags, refused-write archiving, repeated comparisons) |
| Production build | passed (Next.js 15.5.26 static export) |
| **Fresh journey, final environment** (`study-review-journey.spec.ts`) | **passed** — no API-seeded tutorial completion; details below |
| Real hierarchy (`review-hierarchy.spec.ts`), final environment | **2 passed** |
| Visual captures (`review-visual-s2.spec.ts`), final environment | **6 passed**, 82 captures |
| Complete suite, final environment | **68 passed**, 26 not passed, 56 skipped. All 26 are specs built on the synthetic S0 fixture (its pairs, paged-fact and property fixtures and variant publications do not exist in the review publication); each passes on the synthetic harness. Skips are opt-in captures, legacy-viewer and synthetic-only tests. |
| Complete suite, synthetic S0 harness (forms/2, same build, clean worktree) | **101 passed**, 19 failed, 30 skipped. All 19 failures are tests of the legacy exact-study/1.0 study server, which no longer starts under the current backend (see limitations); every forms/2 synthetic regression passes, including the forms/2 journey, the S0 graph, tutorial and assessment-recovery specs. |

The **fresh journey** asserts actual requests and stored data:
- the exchange body;
- the setup/3 receipt and the strict consultation/3 bodies;
- one injected 422 tutorial rejection that drops only the named record, with the next action
  saving without reload;
- reloads at lesson 2, the assessment and an unfinished consultation draft;
- wrong answers retried (one after a reload);
- both conditions;
- per-case methods (Protégé + Other, Text editor, none, Protégé + Text editor, a copied previous
  answer then edited);
- completion;
- no non-study routes, and no workspace reads from baseline cases;
- analysis/4 JSON without names (redacted) and with names (included);
- CSV/4 through the researcher page, with method codes per case matching what was entered.

Reload, interrupted requests, offline and two-tab conflicts are covered on the forms/3
environment by the existing assessment-recovery, tutorial-position (R03) and session-replacement
(R01/R02) browser tests, which now seed forms/3 correctly. Keyboard paths are covered for the
setup, the tutorial, the mode radiogroup and the trace reveal; axe (WCAG 2.2 AA rules) runs in
the journey on the tutorial, an explanation case, a baseline case and the consultation form.

Screenshots in [explanation-postmigration-frontend-integration/](explanation-postmigration-frontend-integration/)
are from the final environment (`b5511d27…` frontend, `ae6c0cf0…` test publication). The
[receipt](explanation-postmigration-frontend-integration/receipt.json) lists every capture with
its build and data identities, plus the suite results.

## Reproduction

The environment runs from the clean worktree of `885f551`, not from the shared checkout, so
another session's uncommitted backend edits cannot enter it. Run the lifecycle commands there:

```sh
PY=/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python
REVIEW=/Users/pedrocotovio/Library/Caches/exact-om-review/postmigration-20261011
cd /Users/pedrocotovio/Library/Caches/exact-om-dev/s2dev/source-885f551
export PYTHONDONTWRITEBYTECODE=1
"$PY" -m tools.review_storage_budget check --growth-gb 2
"$PY" -m tools.review_environment status --root "$REVIEW/environment-s2e"
```

If stopped, start with the bounded reservation in `environment-s2e/storage-reservation.json`:

```sh
"$PY" -c 'import resource; resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728)); resource.setrlimit(resource.RLIMIT_CORE,(0,0)); from tools.review_environment import main; main()' start --root "$REVIEW/environment-s2e"
```

For a private test invitation: `"$PY" -m tools.review_environment fresh-session --root
"$REVIEW/environment-s2e"`, then open it in a browser. The main app is
`https://127.0.0.1:19495/`. Use the owner-aware `stop`. Never print or commit
`environment.json`, keys, invitations, cookies or participant identifiers.

Browser tests need a private harness file (mode 0600, outside the repository) with `origin`,
`researcher_token`, `publication`, `study_revision` and `assets` (the combined directory), read
from `environment.json` without printing it. From `explanations_visualizer` with its locked
dependencies:

```sh
EXACT_E2E_URL=https://127.0.0.1:19495 EXACT_E2E_REVIEW_MAIN_URL=https://127.0.0.1:19495 EXACT_E2E_STUDY_V2_CONFIG=/private/path/harness.json npx playwright test e2e/study-review-journey.spec.ts e2e/review-hierarchy.spec.ts
```

Add `EXACT_SCREENSHOT_DIR=<dir>` and `e2e/review-visual-s2.spec.ts` to recapture the evidence
set; add `EXACT_E2E_SHOTS=<dir>` to keep journey screenshots. Rebuilding requires a new frontend
output and a new environment root (`tools.review_environment prepare … --frontend <new output>
--root <new root>`); the environment refuses a changed frontend.

## Storage

| Point | Combined use (decimal) |
|---|---|
| Backend delivery measurement | 48.626 GB |
| First S2 measurement (13:17 UTC) | 48.926 GB |
| Measurement after the first final build (20:29 UTC) | 49.382 GB (152.9 GB filesystem free) |
| Final measurement (21:36 UTC) | 49.966 GB (141.0 GB filesystem free) |

Every copy, build and environment preparation ran under `tools.review_storage_budget`. The
largest growth estimate admitted was 2 GB, used for each environment prepare/start.

S2-attributable use is about 0.66 GB:
- five environments: 83, 73, 88, 46 and 88 MB, databases retained;
- four frontend outputs: 2.3 MB each;
- the dev directory including the 44 MB clean worktree: 99 MB;
- the Next.js build cache: 172 MB.

A concurrent session also wrote during the interval. Use stayed below 50 GB at every measured
point, far below the 95 GB admission threshold; these are measured points, not an
instantaneous peak. Guarded writes were refused three times because another session held the
shared writer lock. Each was retried after release, never bypassed.

## Limitations and open gates

- **Human-only audits remain open:** manual keyboard and screen-reader audit, owner design
  review, real participant use. Automated axe checks and keyboard paths are not conformance.
- **Data gaps reported, not fixed:**
  - decision traces have no recorded stage events for the presented pairs;
  - 8 pairs have no exported selection;
  - no presented pair has 1–2 selected features (the sparsest has 24), so the sparse-source
    stratum stays unmet;
  - DOID's asserted entry points are dominated by obsolete classes;
  - feature-correspondence bridges whose ends are unresolved remain undrawable.
- The legacy exact-study/1.0 UI fixture no longer starts under the current backend ("Ontology
  resource lacks a complete retained category inventory"). The v1 study browser tests
  (`study.spec.ts`, `study-recovery.spec.ts`, the v1 parts of `accessibility.spec.ts`) could not
  run. The frontend's v1 code paths are unchanged.
- Specs written for the synthetic S0 fixture fail on the real publication by design, because
  their pairs and fixtures do not exist there; they pass on the synthetic harness. They are
  listed in the receipt.
- Historical R16 (intermittent reload) was not reproduced and remains open.
- Real-package distribution, deployment and launch gates remain open.
- During setup of the synthetic S0 harness a test-only signing secret of that local fixture was
  displayed in this session's tool output. It was not committed and is unrelated to the review
  environment.
