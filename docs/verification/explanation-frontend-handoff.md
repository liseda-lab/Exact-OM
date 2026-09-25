# Explanation framework frontend (F1) handoff

**Date:** 2026-09-25. **Branch:** `dev`, on top of `69dc9d6`. **Status:** `fixture_ready`, not a
passed gate. The whole frontend from [spec 08](../../specs/explanation-framework/08-frontend-implementation.md)
and specs 10–13 is implemented and was exercised end to end in a browser. That testing used a
development package: real NCIT/DOID excerpts, a synthetic Exact run and offline exact-excerpt
text. It has **not** been run against the real `backend-release` package or PostgreSQL. No
usability, benefit or accessibility-conformance claim is made.

The backend ledger still records G1 and G5 as `blocked_input`, so F1 was formally not admitted.
The work was done at the user's explicit request, which the suite README gives precedence. See
the [status ledger](explanation-framework-status.json) F1 entry and the
[backend report](explanation-backend-report.md).

## 1. What was delivered

| Area | Result |
|---|---|
| Design | A Design canvas "Exact Explain — Frontend Design" with 20 artboards (below). It is a private claude.ai artifact owned by the user: <https://claude.ai/artifact/Q99Vr9TqnzmuSymxS3inJb>. Share it before others need it. |
| Frontend | One Next.js 15 static export in [`explanations_visualizer/`](../../explanations_visualizer/README.md) for every product: exploration app, public demo, bundle library/import, participant study and researcher admin. The historical run viewer moved to `/legacy/`. |
| Backend (additive) | Three discovery routes, profile-gated static page serving with per-document CSP, study-service page serving, and a frontend build stage in the prepared and study Dockerfiles. |
| Backend fix | `verification.py` read macOS `ru_maxrss` (bytes) as kilobytes and failed the serving-memory check by a factor of 1024. Six handoff tests failed on macOS because of it. |
| Tools | [`tools/build_explanation_ui_fixture.py`](../../tools/build_explanation_ui_fixture.py) prepares the development package through the real bind-lock → prepare → portable-export path. [`tools/serve_explanation_ui_study.py`](../../tools/serve_explanation_ui_study.py) publishes and serves a synthetic three-case study locally. |
| Tests | New [`tests/explanation_frontend_serving_test.py`](../../tests/explanation_frontend_serving_test.py) and `test_frontend_discovery_routes_are_bounded_and_policy_scoped` in the preparation suite. |
| Docs | Frontend README rewritten; the [backend guide](../guides/explanation-backend.md), [study guide](../guides/explanation-study-service.md), status ledger and Render blueprint comment updated. |

Nothing was pushed, deployed, published to participants or sent to anyone.

## 2. Design

The canvas has four pages:

- **Exploration app:** Compare a candidate pair (main screen), Browse both ontologies, Evidence graph and list, Decision trace, Compare on a phone, Bundle library and import, and a sheet of empty/missing/failed states.
- **Ranking study:** Welcome and consent, Setup check, Background questions, Tutorial and practice, Ranking case in the explanation block (clickable prototype), Ranking case in the ontology-tools block, the question after each case, Final feedback, Completion, the saving/pause/return/closed-link states, and resuming on a phone.
- **Researcher admin.**
- **Interface system:** type, colours in both themes, side markers, evidence line patterns, text-origin tags, status chips and controls.

On the canvas, ontology text is real (from `specs/explanation-framework/evidence/real-entity-context-examples.json`). Scores, trace values and generated wording are marked placeholders.

System decisions carried into code (tokens in `src/styles/base.css`):

- **Type:** Newsreader for entity names and titles, IBM Plex Sans for body text, IBM Plex Mono for IRIs. Primary text is at least 16 px, and all sizes are in rem.
- **Sides:** source is slate `#2C5878` with a circle marker; target is umber `#8A4F2A` with a square marker. Every card also says "Source class", "Target object property" and so on. Warm paper ground; the dark theme has its own token set.
- **Where text comes from:** "Original · <predicate>" for ontology statements, a dashed "Generated" frame with citations for prepared text, and "Matcher record" for Exact's saved decisions. Scores are "matching scores" and never probabilities unless calibration is validated.
- **Evidence lines:** solid for an ontology assertion, dashed for structural navigation, dash-dot for a feature Exact used, and a double purple line for a cross-ontology comparison.

Where the implementation differs from the canvas:

- **Phone Compare view:** the current source and candidate are shown with "Change" controls instead of the four-tab bar. It meets the same reading-order goal and keeps state in the URL.
- **Library:** shows only what the API gives. Bundles other than the active one have no per-bundle entity counts, and there is no delete, because no route exists.
- **Practice:** uses a placeholder sandbox (§6.7).

## 3. Frontend architecture

```
src/app/            page.tsx (Compare), browse/, library/, participate/, admin/, legacy/ (old viewer), layout.tsx
src/lib/            api.ts (errors keep backend codes), types.ts, labels.ts (batched /labels cache),
                    labelSource.tsx (labels from API or from a study resource), owl.ts (Functional Syntax
                    printer + conservative constructor readings from the typed AST), iri.ts, prefs.ts,
                    urlState.ts (all navigation state in the URL), useAsync.ts, useMedia.ts (useNarrow),
                    zipManifest.ts (reads package.json from a ZIP before upload)
src/components/     common/ (Icons, Dialog with focus trap + portal, StatusText, ErrorNote)
                    shell/ (AppHeader, HelpDialog, Preferences, ExploreShell)
                    explore/ (ExploreContext, CompareView, SourceRail, candidates, ReviewPanel, EntityCard,
                              GeneratedBlock, ComparisonPanel, DecisionTrace, evidenceData, EvidenceList,
                              EvidenceGraph (Cytoscape), EntitySearch, HierarchyBrowser, BrowseView)
                    owl/ (Expression, AxiomBlock), library/LibraryView, admin/AdminApp
                    study/ (ParticipantApp, StudyChrome, Stages, CaseView, RankingPanel,
                            QuestionnaireForm, StudyExplanation)
src/study/          types.ts, session.ts (mutation queue), telemetry.ts (events + timing)
src/styles/         base.css (tokens, primitives), app.css (exploration), study.css (study, admin)
```

Rules the code follows:

- **No interpretation in the client.** It never parses OWL source, infers equivalence, calls a model or computes probabilities. The OWL printer only formats the backend's already-typed AST, and readings use the same conservative constructor words as `expression_text` ("some", "only", "at least n"), with labels substituted. No verbs are added, so "may have" predicates keep their strength.
- **Honest absence.** Every availability status gets its own wording: absent in scope ≠ not exported ≠ filtered ≠ unsupported ≠ not run ≠ failed. An empty list is never shown bare.
- **Responsiveness follows text size.** Layout breakpoints are container queries in rem on `.app-root` and `.study-root`, so the A−/A+ control (up to 200%) moves the layout to its narrow form. `useNarrow()` applies the same rule in JavaScript.
- **CSP compatibility.** The study pages' prerendered HTML contains no inline `style` attributes or `<style>` tags. React styles are applied through CSSOM, which CSP permits.

### Spec 08 feedback table and where each item is handled

| Complaint | Implementation |
|---|---|
| Overwhelming panel / hover movement | No hover effects change geometry. Details are closed by default; the legend is an explicit help dialog. `prefers-reduced-motion` is honoured. |
| Values on arrows | Matching weights are hidden behind "Show matching weights"; cardinalities stay in readings. |
| Selection order | Wide layout order is candidate list → cards → comparison → details tabs. |
| Similar blue/purple | Side markers are circle vs square, graph node shapes are rounded vs square, and line patterns differ, always with labels. |
| Ambiguous +/− | Magnifier zoom-in/zoom-out icons with accessible names, plus permanent Fit and Reset view buttons. |
| Generic "Source" | `sideTitle()` produces "Source class", "Target object property" and so on. |
| Small fonts | 16 px minimum and a persistent text-size control. Graph labels are 15 px and edge labels 13 px at the default fit. |
| Poor expansion | A separate hierarchy browser with paging. Graph expansion is explicit ("Add its parents"), cumulative, undoable and capped at 150 nodes. |
| Sparse source | "Not stated in the loaded scope" notes; nothing is invented or padded. |
| Complex relations | Plain reading plus "Show original axiom" (OWL Functional Syntax, short names or full IRIs); unsupported constructors show the original only. |
| Screen sizes | Checked at 320, 375, 1280 and 1440 px, including 200% text at 320, 375 and 1280 px, with no page-wide horizontal scroll. The graph pans inside its own region. |

## 4. Backend changes

All changes are additive, and API-only behaviour is unchanged when no export exists.

- **`GET /api/v1/runs`:** a paged list of `{run_id, revision, source_ontology_version_id, target_ontology_version_id, status, counts}`. Runs whose ontologies the policy denies are skipped. With no run: `status: not_exported`.
- **`GET /api/v1/labels?ontology_version_id=&iri=…`:** 1–100 IRIs (else 422). Items are `{entity, preferred_label}`. An undeclared IRI gives `entity: null` with status `absent_in_scope`, or `unresolved_import` when imports are unresolved. Backed by `OntologyContext.labels()`, which is policy-aware.
- **`GET /api/v1/explanations?ontology_version_id=&iri=&kind=[&task=][&counterpart_*]`:** paged summaries `{explanation_id, task, entities, grounding_status, generation_status}`. The index is built lazily once per package from explanation metadata; text stays on disk, and other-policy resources are excluded. It returns `not_requested` when nothing matches. A counterpart IRI without its ontology version is a 422.
- **`exact_inspect/frontend.py`:** page allowlists per profile:
  - `local_app`: `/`, `/browse/`, `/library/`
  - `public_demo`: `/`, `/browse/`
  - `study`: `/participate/`, `/admin/`, with `/` redirecting to `/participate/`

  Unknown pages and any unmatched `/api/…` method return a 404 JSON envelope. `_next` assets are served with path-traversal checks. HTML responses carry a per-document CSP with the exact `sha256` of each inline script (the study profile also forbids inline styles), plus `Referrer-Policy: no-referrer`. Without an export no page route is mounted, which is the previous behaviour.
- **Wiring:** `create_prepared_app(..., frontend_dir=)`, `create_study_app(..., frontend_dir=)` and `app.py` pass `resolve_frontend_dir(settings)`. `study.api.app_from_env` uses `EXACT_STUDY_FRONTEND_DIR` or the bundled `exact_inspect/static`. The study security middleware keeps a page's own CSP, and study health reports `frontend: static | api-only`.
- **Deployment:** Node 20 build stages were added to `deploy/render/exact_inspect_prepared.Dockerfile` and `exact_study.Dockerfile`. The study image copies `exact_inspect/frontend.py` and the static export only.

## 5. Verification performed

| Check | Result |
|---|---|
| Backend suites `explanation_{frontend_serving,preparation,study,read_models,study_resource,export_policy,context,http_runtime,handoff,framework,portable_run,sqlite_safety,recovery}_test.py` and `exact_inspect_test.py` | **171 passed, 1 skipped** (Python 3.12, pyowl-core 0.2.1, SQLite study mode) |
| `explanation_decisions_test.py`, `explanation_replay_test.py` | **Not run.** They import `torch` and need the full Exact environment. |
| `npx tsc --noEmit`, `next build` | Pass. The study HTML contains no inline `style=` or `<style>`. |
| black (line 100) and isort on changed Python | Pass |
| Page gating on local, demo and study | Verified with curl. Local serves `/`, `/browse/`, `/library/`, with 404 for study, admin and legacy pages. Demo 404s `/library/` and `/api/v1/bundles`. Study serves only `/participate/` and `/admin/`. |
| Compare view in the browser (1440 px, light and dark) | Cards, citations, comparison templates, generated profiles with provenance, restriction and equivalence readings, original-axiom toggles, parents; evidence list; graph (fixed layout, feature/assertion/bridge line styles); decision trace; scores. No console errors or CSP violations. |
| Browse | Combobox search with keyboard (ArrowDown/Enter), parent expansion, child paging, basis options with honest "not prepared", focus history in the URL, "Return to <compared entity>", full-context dialog. |
| Library import | A corrupt ZIP is refused with "archive is corrupt"; a valid exported bundle shows its manifest preview, uploads, validates checksums and is added to the library. |
| Responsive | 375 px and 320 px, at 100% and 200% text, and 1280 px at 200%: no page-wide horizontal scroll after the wrapping fixes. The phone reading order is selection → question → cards → comparison → details → review. |
| Study, full flow | Link exchange (fragment removed, including after the `/participate` → `/participate/` redirect); consent; setup (resource download 200 with attachment); background form (required-field errors, conditional DOID contexts, exclusive "prefer not to say", draft autosave, submit); practice; allocation; explanation case; submit; consultation; second explanation case; baseline case; final form (matrix, exclusive options, required errors); completion. |
| Study, integrity | Draft restored after reload. Coalesced drafts save only the latest. A second-tab edit gives a 409, the conflict banner shows and the screen re-syncs to the server's draft. Pause/resume. A revoked link shows "This link cannot be used" and the old cookie is refused. A fresh link pasted into an open tab is exchanged via `hashchange`. The baseline gets **403** for explanation resources of the current and other cases. Exploration routes are 404 on the study service. |
| Admin API | Progress counts, JSON and CSV exports, and 401 without a token, all checked with curl. The signed-in admin screens were **not** driven in a browser, because I did not enter the researcher token into a browser field. |

## 6. Issues found

### Backend defects and limitations (hand back to the backend owner)

1. **Per-case study resources fail when a case contains an entity and one of its asserted parents.** `study/builder.py` stores facts by `fact_id` with `setdefault`, so a shared axiom such as `SubClassOf(child, parent)` keeps only the first subject. Semantic-template claims are then re-validated against a packet whose fact belongs to "an entity outside the packet", and building fails with `ValueError: Fact belongs to an entity outside the packet`. Such cases are plausible in real candidate sets (a candidate plus its parent). **Workaround used:** one resource per source–candidate pair; the frontend merges them and keeps per-subject copies. **Fix options:** key facts by (fact, subject), or scope template re-validation to the claim's own packet.
2. **Study resources lack labels for referenced properties and fillers.** For example `NCIT:R176`, `NCIT:C102447` and the equivalence operands appear as CURIEs. The UI says "(label not in the prepared view)" rather than "no label in scope". This hurts spec 08's "faithful plain-language meaning" for participants. Suggested fix: include label facts for IRIs referenced by admitted axioms, bounded and policy-filtered.
3. **Ontology display names are `null`.** The `context-index` stage does not pass `ontology_name` to `build_context_package`, so the UI falls back to "Source ontology" / "Target ontology". Adding it changes preparation keys, so it needs a deliberate revision.
4. **No discovery routes existed** for runs, labels or explanation IDs, so the frontend could not start from a package. Added as additive routes (§4); they should enter the contract/OpenAPI docs and the typed `InspectClient`.
5. **Candidate pages are keyset-ordered by `pair_id`, not by rank.** The UI loads up to 500 candidates and sorts by the recorded `candidate_joint_rank`, then `retrieval_rank`, then score, and states the basis used.
6. **`pair-evidence` rows carry empty `display` and `semantic_terms`.** The UI derives each reading from the linked fact axioms (`/axioms/{fact_id}`).
7. **`original_syntax` is pyowl-core canonical bytes (base64), not human-readable**, and `rendering.text` is `unsupported` for top-level axioms. The UI prints OWL Functional Syntax from the typed AST.
8. **Explanation resources report `fixture_provenance: "actual provider response…"` even when an offline stub produced the text** (hard-coded in `generation.py`). The manifest's `provider`/`requested_model` still reveal it (`offline-fixture`, `fixture/exact-excerpts`), and the UI shows them under "How this was prepared".
9. **`StudyState.ontology_resources` has no role (source or target) or human name.** The UI shows `asset_id`, so study owners must name assets and setup instructions clearly.
10. **`StudyState` omits the current condition and block.** The header fetches `cases/current` to label the block; progress is `completed_cases + 1 of assigned_case_count`.
11. **No admin route lists studies.** The researcher types the study revision (remembered in `localStorage`; not secret).
12. **No route stores exploration review decisions** (`ReviewDecision`). The Compare view keeps them in browser storage and says so.
13. **The study origin must be HTTPS and cookies are `Secure`.** Local browser testing needs the loopback origin rewrite in `tools/serve_explanation_ui_study.py`; that rewrite must never be used in a deployment.
14. **Timing segments are rejected (409) once the stage has changed.** The UI closes the running segment before every step-changing save. Segments cover at most 299 s and start at `case_ready`.
15. **Fixed:** macOS peak-RSS units in `verification.py`.

### Frontend gaps and remaining work

1. **Run against the real `backend-release` package and the PostgreSQL study service.** That package is not on this machine: `data/explanation-framework/` is absent and ignored.
2. **Practice content.** The backend has only `tutorial_steps` text. The UI adds an unscored placeholder ranking sandbox, but spec 11/13 asks for simple → complex → partial → none-of-these practice cases. That needs owner-supplied practice data (a backend or content addition).
3. **Accessibility.** Screen-reader pass (VoiceOver/NVDA), a full keyboard-only walk through every screen, and colour-contrast measurement in both themes. Tree navigation uses plain lists of buttons rather than an ARIA tree, deliberately.
4. **Evidence graph.** Only hierarchy (named `SubClassOf`), some/only restrictions and literal annotations have a graph form; other facts are counted as "no graph form" and appear in the list. Bridges are drawn per channel (Exact compared this channel), not per feature pair, because the API has no pairing.
5. **Formative usability iterations, the duration pilot and the frozen evaluation** (spec 07/11). No outcome claims.
6. **Docker image builds** for the prepared and study services (Docker was not running) and the Render staging redeploy/restore rehearsal.
7. **Legacy redirect.** `/` falls back to `/legacy/` when only `/api/health` exists. This was not exercised with a real `serve --run-dir`, which needs the full Exact environment.
8. **The admin UI's signed-in screens** should be exercised in a browser by a person.

## 7. Environment notes from this session

- **Node:** Node.js was not installed. Homebrew refused because the Command Line Tools are outdated (updating needs sudo). The session used the official Node 22.23.3 darwin-x64 tarball, SHA-256 verified, from a scratch folder; nothing was installed system-wide. Install Node 20+ to rebuild.
- **Python:** the repository `.venv` hangs while importing (`python -m pip --version` never returns), most likely because files under `~/Documents` are cloud-evicted. Tests ran in a separate Python 3.12 venv with the prepared-serving requirements plus `pyowl-core==0.2.1`, `pyowl2vec-star-projector==0.2.1`, pandas, zstandard, requests, tqdm, pytest, black and isort.
- **Docker:** Docker Desktop was not running.
- **Preview launcher:** the desktop app's preview launcher could not read `~/Documents` (macOS privacy), so servers were started from the shell.
- **Tailwind:** Tailwind and PostCSS were removed from the frontend. The new app uses plain CSS; the legacy viewer never used Tailwind classes.

## 8. Reproduce

```bash
# 1. Development package (real NCIT/DOID excerpts, synthetic run, offline exact-excerpt text)
python -m tools.build_explanation_ui_fixture --output /tmp/exact-ui-fixture

# 2. Frontend
cd explanations_visualizer && npm ci && npm run build && cd ..

# 3. Exploration app (local profile with an import library)
exact-inspect serve --package /tmp/exact-ui-fixture/prepared/stages/<portable-export-id>/package.json \
  --profile local_app --library-dir /tmp/exact-library --frontend-dir explanations_visualizer/out --port 8765

# 4. Public demo boundaries (the fixture is marked development_demo)
exact-inspect serve --package <same package.json> --profile public_demo --frontend-dir explanations_visualizer/out --port 8767

# 5. Synthetic study with test links printed on start
python -m tools.serve_explanation_ui_study --fixture /tmp/exact-ui-fixture --port 8766

# 6. Tests
pytest tests/explanation_frontend_serving_test.py tests/explanation_preparation_test.py tests/explanation_study_test.py
(cd explanations_visualizer && npx tsc --noEmit)
```

`npm run dev` with `EXACT_DEV_BACKEND=http://127.0.0.1:8765` proxies `/api` for fast exploration-app work.

## 9. Suggested next steps, in order

1. Fix backend issue 1 (the study builder subject collision) and add label facts for referenced IRIs (issue 2), then rebuild the study resources.
2. Run the frontend against the real `backend-release` package and a PostgreSQL study. Repeat §5 and record the results in the ledger.
3. Accessibility audit with assistive technology, and supply practice content.
4. Build both Docker images and rehearse on Render staging per the study guide.
5. Formative usability iterations, then the duration pilot, before any frozen study.
