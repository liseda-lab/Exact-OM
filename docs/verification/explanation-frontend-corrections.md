# Explanation framework: corrective frontend handoff (specs 14–16)

**Date:** 2026-10-03. **Branch:** `dev`, on top of `8ac76ce`. **Assignment:** [15](../../specs/explanation-framework/15-frontend-corrections.md), under the programme in [14](../../specs/explanation-framework/14-corrective-programme.md).

**Status:** the frontend side of the corrective programme is implemented.
- Fixes that work on the current contract are live for `exact-study/1.0` studies and the exploration app.
- The corrected participant flows (tool-neutral setup, interactive tutorial with a server-graded check, durable per-case reports) are implemented against a **proposed** `exact-study/2.0` contract, which no backend serves yet. They are demonstrated only through a development-only synthetic service.

This is **not** a readiness claim: the backend assignment in [16](../../specs/explanation-framework/16-backend-corrections.md) (B0–B9), the joint acceptance matrix, real-package and PostgreSQL/HTTPS runs, screen-reader and usability work, and the launch gates all remain open. No participant was contacted, nothing was deployed or published, and no provider or model was called.

## 1. What changed

| Area | Result |
|---|---|
| Shared workspace (C01, C02) | One data-source interface ([`src/lib/workspace`](../../explanations_visualizer/src/lib/workspace)) behind the meaning cards, comparison, hierarchy browsers, evidence list, evidence graph and fact inspector. The exploration app, the study's explanation condition and the tutorial render the same [`PairWorkspace`](../../explanations_visualizer/src/components/workspace/PairWorkspace.tsx). The reduced study engine (`StudyExplanation.tsx`) is deleted. |
| Citations (C03) | Every citation is a button that opens the exact original record through a resolver: label citations, hidden duplicate synonyms, records outside the loaded page, shared axioms under several subjects. Unavailable records give their specific reason. Focus returns to the citation. |
| Evidence fidelity (C04, C18) | The study evidence list shows the recorded interpretation (no hard-coded "projected from an asserted fact"), offers original-axiom inspection for every record, and keeps capabilities, limitations, truncation and saved values. A literal annotation is shown as the exact `AnnotationAssertion` it encodes, labelled as reconstructed; stored typed ASTs are shown as stored. |
| Case layout (C06) | Sticky rail with candidates and the answer beside one reading column, plus a sticky answer-status line. Narrow or enlarged text puts candidates, then workspace, then answer, with a sticky answer bar. Inspection state ("Inspecting", "Viewed") is separate from rank. A queued submission is distinguished from server receipt. |
| Graph (C07, C08) | The graph is created once per compared pair. Zoom, pan, expansions and selection persist across re-renders, answer saves, tab switches, theme, text size and layout changes. The invalid `double` style is replaced by a round-dot bridge that matches the legend. "Fit all" and "Reset view" are distinct and documented. Expansion reads through the workspace and keeps typed identity. |
| Tabs (C09) | One accessible [`Tabs`](../../explanations_visualizer/src/components/common/Tabs.tsx) primitive (roving focus, arrows/Home/End, manual activation, labelled panels) used everywhere. |
| Questionnaire order (C05) | Options and matrix rows follow `option_order`/`row_order` (v2) or a versioned v1 table taken from `forms.py` (checked by [`tests/explanation_frontend_form_order_test.py`](../../tests/explanation_frontend_form_order_test.py)). A form with no declared order fails visibly instead of being sorted. |
| Tool-neutral study (C12, C13, C19) | v2 setup has no installation, opening or tool requirement. The ontology files show role, version, format, size and a full-hash copy, with the same "Ontology files" entry point in both conditions. Copy uses neutral wording (condition label, gap dialog, baseline guidance). Copying an IRI is no longer reported as external use. |
| Tutorial and check (C10, C11, C14) | Six lessons in the production workspace over synthetic resources. Actions are reported as typed requirement evidence and shown as saved only after the server acknowledges them. Five server-graded items give feedback, a route back to the lesson and unlimited retries. Allocation happens only on server-accepted completion. Progress resumes after reload. Tutorial help can be reopened during cases. |
| Per-case report (C13, C15) | v2 drafts autosave through the outbox and are restored from the server. Answers can be Yes with any combination of seven methods or No; No clears the details. Resource scope is optional and stays unknown if unanswered. Reuse of the previous case's answer happens only on request and is never preselected. The final save advances once. v1 keeps its final-only route, with a tab-local restore that says it is not saved. |
| Events (C16) | Opening a tab is no longer "hierarchy_expand". Actions map to their own types; v2-only types are sent only when the service declares them. Copying an IRI and searching are never "external use". |
| Transitions (F8) | The old Protégé-field heuristic is replaced by explicit `transition` flags. Drafts, progress saves and assessment attempts never close timing; setup submit, tutorial completion, ranking submit, final report and similar do. The outbox route allowlist includes the planned routes, and coalescing never crosses an attempt or commit. |
| Candidates (C20) | No 500 cap: "Load the next 100 candidates" continues the server cursor. Badges show the recorded rank, never a subset position relabelled as rank. |
| Library, reviews, admin (C21, C22) | The limits are stated in the UI: no copy deletion or metadata route, reviews kept only in this browser, revision entered manually. These need backend routes (B8). |
| Accessibility found during this work (C24) | Clipboard refusals now fall back to a selected IRI field. On phones at 200% text the sticky header no longer hides focused controls (WCAG 2.4.11): it becomes static on narrow layouts, and the page reserves scroll padding for sticky bars. |

## 2. Component and state map (F0, F1)

```
WorkspaceSource  (src/lib/workspace/types.ts)
 ├─ createApiSource        exploration  /api/v1/...                       (apiSource.ts)
 │                         proposed B1  /api/v1/study/workspace/{scope}/... same shapes
 └─ createResourceSource   v1 study case: frozen resources, bounded ("prepared")  (resourceSource.ts
                           tutorial: synthetic resources, complete                + resourceIndex.ts)
WorkspaceProvider(source, onAction) ─ LabelSourceContext for local labels
PairWorkspace  = header slot · EntityCard ×2 · ComparisonPanel
                 · Tabs [Hierarchy: HierarchyBrowser ×2 · Evidence: EvidenceList · Evidence graph: EvidenceGraph · shell extras]
FactInspectorProvider → FactDialog (source.fact) · citations · "Show where it appears"
Shells: CompareView (exploration; URL state; trace/scores tabs; review panel)
        CaseView    (study; CaseLayout rail; RankingPanel controller; ResourceAccess; tutorial help)
        Tutorial    (v2; LessonWorkspace = CaseLayout + PairWorkspace over the tutorial source)
```

| State | Owner | Lifetime |
|---|---|---|
| Case/presentation, condition, frozen candidate order, components | Server (`cases/current`, `state.forms`) | per presentation |
| Inspected candidate, viewed markers, open detail tab, hierarchy focus per side | Shell (`CaseView`/`LessonWorkspace` state; exploration URL `details`, `hs`/`ht`) | source focus kept across candidates; target focus per candidate; reset on case change |
| Answer, submit receipt, pending mutations | Shell + server + outbox | answer only via drafts/submit |
| Graph view and selection, open fact | `EvidenceGraph` view store keyed by workspace key and pair; `FactInspectorProvider` | until the pair or scope changes |

Cache keys include the product, package or session, presentation and resource identity. `useAsync` never shows the previous key's data, and the study source keeps its identity across autosaves.

## 3. Participant copy (F5–F7)

The exact wording is in the code. The authored synthetic tutorial material, with lessons, items, answer keys and feedback, is in [`src/study/v2/tutorialContent.ts`](../../explanations_visualizer/src/study/v2/tutorialContent.ts), for the backend to freeze in the v2 publication. The frontend never uses it as a fallback for a real publication.

- **Setup** ([`SetupV2.tsx`](../../explanations_visualizer/src/components/study/SetupV2.tsx)): the task and the two files, with an information-scope notice that names no tool. "Can you get the two files?" A problem routes to retry, Pause or contact, never a false confirmation. The programme's method-freedom paragraph appears verbatim. An optional list of familiar methods is described as committing to nothing.
- **Lessons:**
  1. The source and its candidates (initial position vs rank; inspecting is not ranking).
  2. Context (search, parent, child, return, multiple inheritance, missing information).
  3. Where each statement comes from (citation, original axiom, evidence list).
  4. The optional evidence graph (list as an equivalent; view kept while ranking).
  5. Your answer (add, move, remove, undo, keep initial order, partial check, None vs Insufficient, qualified example).
  6. The block without explanations, and your methods (copy an IRI, downloads, practice reports with two methods and with No).
- **Items:** Q1 `score_meaning`, Q2 `equivalence_scope`, Q3 `response_states`, Q4 `evidence_origin` (matching plus part B), Q5 `external_methods`. The wording and pass predicates follow the table in 14. Matching uses radio groups, never dragging.
- **Per-case report** ([`ConsultationV2.tsx`](../../explanations_visualizer/src/components/study/ConsultationV2.tsx)): "For the case you just completed, did you inspect ontology information outside this study interface?" The page states that a case is the source and its five candidates, and that the study's own panels do not count.

## 4. Backend operations needed (F0 table)

Machine-readable request, response and error examples are in [`explanation-frontend-v2-contract-examples.json`](explanation-frontend-v2-contract-examples.json).

| Control | Current (v1) | Proposed (v2, 16) | Notes |
|---|---|---|---|
| Version dispatch | `contract_version` 1.0 | 2.0 | Any other value shows a visible "not supported" page |
| Setup | `PUT /study/setup` with Protégé booleans | Same route, v2 body; draft vs `submitted` | Readiness = two acknowledgements + `resource_access=available` |
| Downloads | `asset_id`, size, hash | + `title`, `role`, `ontology_version_id`, `version_label`, `license_note`, `information_notice` | v1 roles stay "not recorded" |
| Tutorial definition | none (indices only) | `state.tutorial` (public part) | Grading keys never sent |
| Tutorial progress | none | `PUT /study/tutorial/progress` | Cumulative, add-only requirements; practice; assessment drafts; help count |
| Assessment attempt | none | `POST /study/tutorial/assessment` | Server-graded; returns feedback and lesson to revisit; replay-safe by `attempt_id` |
| Completion and allocation | setup with `completed_tutorial_steps` | `POST /study/tutorial/complete` | 422 lists outstanding items |
| Tutorial resources | none | `GET /study/resources/{tutorial asset}` for active v2 sessions | Synthetic only |
| Case workspace | frozen resources (bounded) | `GET /study/workspace/{scope}/...` mirroring exploration shapes | Frontend switches when `cases/current.workspace.scope_id` is present |
| Report draft | none | `PUT /study/cases/{id}/consultation/draft` | Never advances |
| Report commit | `PUT …/consultation` (4 codes) | Same route, 7 codes + names + `resource_scope` + `presentation_id` | No ⇒ empty methods |
| Previous report | none | `state.previous_consultation` | Explicit reuse only |
| Form order | sorted keys | `option_order`, `row_order` | Validated permutations |
| Events | v1 types | + 8 declared types (`state.telemetry.event_types`) | Tutorial-scoped events not sent until a scope is declared |
| Researcher revisions | typed by hand | Authenticated listing (B8) | |
| Library copy removal and metadata | none | B8 routes | |

## 5. Parity: exploration vs explanation condition (F1)

| Capability | Exploration | Study, v1 frozen resources (now) | Study with B1 workspace |
|---|---|---|---|
| Meaning cards, original vs generated | yes | yes (same components) | yes |
| Claim citations open records | yes | yes | yes |
| Search | full ontology | only labelled entities in the case resource (declared partial) | full admitted ontology |
| Parents / children / multiple inheritance | yes | prepared edges; children "not prepared for this study case" | yes |
| Asserted / structural / inferred basis | per ontology | only bases present in the resource | per scope |
| Original axioms | yes | yes (stored AST; literals reconstructed and labelled) | yes |
| Evidence list and graph | yes | yes; expansion only within prepared edges | yes |
| Decision trace, scores tab | yes | not shown (not admitted) | not shown unless admitted |
| Honest missingness | yes | yes, distinguishing not prepared, absent and partial | yes |

## 6. Issue dispositions (C01–C27, frontend side)

| ID | Disposition | Evidence |
|---|---|---|
| C01 | **Implemented (frontend).** Shared composition; baseline unchanged server-side. | e2e study journey (baseline has no tabs, generated text or explanation request; 403 kept); resource-index unit tests |
| C02 | **Adapter ready; data pending (B1).** Bounded resources report partial or not prepared; the API source accepts the scoped base. | `resourceIndex.test.ts` (bounded hierarchy, partial search) |
| C03 | **Implemented.** Exploration 9/9 citations (5 were dead); study 0 → 9 citations, all resolving. | e2e exploration C03 test; study journey; unit tests |
| C04 | **Implemented (frontend).** Interpretation, original-axiom action, values and capabilities kept. Backend may still enrich `display`. | e2e; unit tests |
| C05 | **Implemented (frontend);** v2 order arrays needed from the backend (B5). | `formOrder.test.ts`; Python parity test; browser check of the v1 background form; preview final form |
| C06 | **Implemented;** human usability observation still required. | 1280×720: first "Add" at y≈358, answer status always visible; narrow order verified; no overflow |
| C07 | **Implemented.** | e2e: view kept across tabs (exploration) and across a rank edit with server save (study) |
| C08 | **Implemented.** | e2e: only solid, dashed or dotted styles; no style warnings; legend glyphs match |
| C09 | **Implemented** (shared primitive). | e2e tab keyboard test; study uses the same tabs |
| C10 | **Implemented against the proposed contract;** needs B3 and frozen content. | preview e2e journey |
| C11 | **Implemented against the proposed contract;** grading must be server-side (B3). | preview e2e: wrong answer → feedback, revisit, retry; Finish blocked until all items pass |
| C12 | **Implemented for v2;** v1 sessions keep their frozen setup by design (B0/C26). | preview e2e: no install claim; drafts never advance |
| C13 | **Implemented for v2.** | preview e2e: multi-method, No, explicit reuse; unit tests |
| C14 | **Implemented against the proposed contract.** | preview e2e: reload mid-tutorial resumes lesson and saved evidence |
| C15 | **v2 implemented; v1 tab-local restore only** (no server draft route in v1). | preview e2e reload restore; browser check of v1 restore |
| C16 | **Implemented (frontend);** v2 types and tutorial scope need B7. | `workspaceEvents.test.ts` |
| C17 | Backend only (B6). | — |
| C18 | **Implemented (frontend):** qualifiers, shared axioms, label citations, availability states. | `owl.test.ts` regression fixtures; `resourceIndex.test.ts` |
| C19 | **Implemented (frontend):** equal access, metadata display; v2 metadata from B2. | e2e: "Ontology files" in both conditions |
| C20 | **Implemented (frontend):** continuation and honest rank. A rank-ordered backend cursor (B6) remains preferable. | existing capped-list e2e + manual check |
| C21 | **Disposition:** manual revision entry stated in the UI; selector needs B8 listing. Sign-out/revision privacy unchanged. | researcher e2e journey passes |
| C22 | **Disposition:** limits stated (browser-local reviews; no copy removal). Deletion and metadata need B8. | library e2e passes |
| C23 | **Preserved;** new routes integrated with the outbox, allowlist, coalescing and session header. | outbox unit tests; recovery e2e; preview offline check |
| C24 | **Partly done:** axe audits now cover real cases with evidence, per-case reports, tutorial lessons and the assessment (light, dark, phone at 200%). Focus-obscured and clipboard issues fixed. **Screen-reader and manual keyboard passes not done.** | accessibility e2e (28 audits), preview audits (4) |
| C25 | **Not done here:** real package, PostgreSQL/HTTPS and deployment were not available on this machine. | — |
| C26 | **Implemented (frontend):** version dispatch; v1 flows untouched; unsupported versions fail visibly. Backend migration and versioning (B0) pending. | browser checks for v1 and v2 |
| C27 | **This document;** earlier fixes kept (outbox, typed browse, legacy, stale search). | full e2e suite |

## 7. Verification

| Check | Result |
|---|---|
| `npm run typecheck`, `npm run build` | Pass. The study HTML has no inline styles. Production `/preview/participate/` is a 1.3 kB notice, and the export contains no preview service or grading text. |
| `npm run test:unit` (Node 24.21) | **46 passed:** outbox incl. v2 routes, form order, reports, events, lesson evidence, tutorial content and keys, resource adapter, OWL regression fixtures. |
| Backend: `explanation_frontend_form_order`, `frontend_serving`, `study`, `study_resource`, `preparation` | **79 passed, 1 skipped** (PostgreSQL restart). black and isort pass on the new test. |
| Playwright with system Chrome, all specs | **43 passed, 2 skipped** (legacy run directory not available). Exploration and public demo on the small fixture (`EXACT_E2E_SEARCH_TERM=syndrome`), library import with the exported fixture ZIP, study suites on the **SQLite harness with the loopback origin rewrite (not PostgreSQL/HTTPS)**, and the v2 preview journey under `next dev`. |
| axe | 32 audits without violations or page-wide overflow. In `accessibility.spec.ts`, 28 audits run across desktop and phone at 200% text, each in light and dark: comparison, browsing, library, study welcome, researcher sign-in, and the **study case with evidence plus its per-case report**. The preview journey adds 4: lesson, assessment, case and report. |
| Manual (in-app browser) | Citations by keyboard, tab roving, graph persistence, bridge rendering, bounded hierarchy wording, v1 order and report restore, baseline makes no explanation requests, all v2 lessons and items, offline/reconnect, narrow layout. |

**Not verified:** the real `backend-release` package, PostgreSQL and real HTTPS, Docker and Render, VoiceOver/NVDA, formative usability, the legacy real-run viewer, and v2 routes on a real backend (none exist yet).

## 8. Environment notes

- **iCloud duplicates.** The repository sits in iCloud-synced `~/Documents`, which had created `* 2` duplicate files. In the ignored `node_modules`, `.next` and `out` they broke the type check, so they were deleted. The repository `.venv` also contains `* 2` duplicates (e.g. `sentence_transformers/__init__ 2.py`). That likely explains the earlier interpreter hangs; it was left untouched.
- **Toolchain.** Node 24.21.0 came from nodejs.org (SHA-256 verified) into the session scratch folder. The scratch Python venv was rebuilt from `deploy/render/exact_inspect_prepared_requirements.txt` plus test dependencies.
- **Ports.** Servers already listening on 18765 and 18766 belonged to the 2026-10-02 review session and were left running; this work used 18865–18868 and 3123.
- **Shared build folder.** `next build` and `next dev` share `.next`. Building while the dev server runs breaks it; restart it afterwards.
- **No PostgreSQL server.** Only libpq client tools are installed here.

## 9. Reproduce

```bash
python -m tools.build_explanation_ui_fixture --output /tmp/exact-ui-fixture
(cd explanations_visualizer && npm ci && npm run typecheck && npm run test:unit && npm run build)
exact-inspect serve --package <fixture package.json> --profile local_app --library-dir /tmp/lib --frontend-dir explanations_visualizer/out --port 18865
python -m tools.serve_explanation_ui_study --fixture /tmp/exact-ui-fixture --port 18866   # SQLite, test only
(cd explanations_visualizer && EXACT_DEV_BACKEND=http://127.0.0.1:18865 npx next dev -p 3123)  # v2 preview at /preview/participate/
```

For the study e2e config, write `{origin: "http://localhost:18866", researcher_token, study_revision: "ui-fixture-study/1", publication}` to a private file and pass it as `EXACT_E2E_STUDY_CONFIG`. Then run `npm run test:e2e` with `EXACT_E2E_URL`, `EXACT_E2E_PREVIEW_URL`, `EXACT_E2E_SEARCH_TERM` and `EXACT_CHROMIUM_PATH`.

## 10. Next steps, in order

1. **Backend B0:** freeze the v2 contract from §4 and the JSON examples. Then implement B2–B5 (setup, tutorial, assessment, report drafts, order arrays), adopting or revising the synthetic tutorial material.
2. **Backend B1:** the scoped case workspace. The frontend switches by itself when `cases/current.workspace.scope_id` appears; remove the bounded-resource path for v2.
3. **Joint acceptance (F10/B9):** the same real case in both products, the real package, PostgreSQL/HTTPS, migration and legacy coexistence.
4. **Assistive technology and usability:** a screen-reader and keyboard pass over case, tutorial and report, then formative sessions and burden re-estimation (C24, C06 observation).
5. **Remaining backend items:** library copy removal and metadata, revision listing (B8), rank-ordered candidate cursor (B6), SQLite URI fix (C17).
