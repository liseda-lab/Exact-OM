# Exact Explain frontend

One static Next.js export that serves every product in the explanation framework
([specs/explanation-framework](../specs/explanation-framework/README.md), spec 08 and 10–13).
`exact-inspect` decides per deployment profile which pages exist; a client cannot switch
itself into another profile.

| Route | Product | Served by profile |
|---|---|---|
| `/` | Compare: source and target meaning cards, generated comparison, optional hierarchy, evidence list and graph, decision trace, scores | `local_app`, `public_demo` |
| `/browse/` | Two independent ontology browsers (search, parents/children paging, basis choice, full context) | `local_app`, `public_demo` |
| `/library/` | Local bundle import (manifest preview, progress, cancel, validation errors) and library selection | `local_app` only |
| `/participate/` | Ranking study: private link, consent, setup, background form, practice (v1) or interactive tutorial and five-item check (v2), scored cases in both conditions, per-case report, final form, completion | `study` only |
| `/admin/` | Researcher administration: publish, progress, invitation links, replace/revoke, exports | `study` only |
| `/legacy/` | The historical run-directory viewer, used automatically when `/` finds only the old `/api/health` backend | legacy `serve --run-dir` |

Everything reads the backend contracts in `exact_inspect` (`/api/v1/...` and
`/api/v1/study/...`); the frontend never parses OWL, calls a model or infers equivalence.
Missing, filtered, unexported, unsupported and failed states are shown with their reason.

What was built, how it was verified and the known backend issues are in the
[F1 handoff](../docs/verification/explanation-frontend-handoff.md) and its
[real-package/PostgreSQL verification](../docs/verification/explanation-frontend-e2e.md).
The 2026-10 corrective iteration (specs 14–16: shared workspace, tutorial, tool-neutral study)
is recorded in the [corrective frontend handoff](../docs/verification/explanation-frontend-corrections.md).
The integration follow-up (specs 17–19: scoped study workspace, case readiness, durable tutorial
position, analysis-3 exports) is in the
[integration frontend handoff](../docs/verification/explanation-integration-frontend-handoff.md)
and the [joint acceptance record](../docs/verification/explanation-integration-acceptance.md);
post-implementation findings R06–R09 (fact paging, study full context, retry, component-aware
readiness) are in the [follow-up handoff](../docs/verification/explanation-integration-frontend-followup.md),
and R10–R11 (attempt-bound access checks, typed navigation) in the
[second follow-up](../docs/verification/explanation-integration-frontend-followup-2.md).

exact-study/2.0 sessions run only against a study service that advertises
`integration_contract: "study-integration/1"`; otherwise the page says the service needs an
update and collects nothing. A v2 explanation case reads only its authorized
`/api/v1/study/workspace/{scope}` routes and is answerable only once the source, all five
candidates and the admitted descriptions and comparisons have loaded, validated and rendered.

## Design decisions

Tokens (both themes) live in `src/styles/base.css`; product styles in `app.css` and `study.css`.

- **Sides are never colour-only.** Source entities use a slate circle marker, target entities an
  umber square, and every card names its side and kind ("Source class", "Target object property").
- **Where text comes from is always visible.** Original ontology statements show their predicate
  (for example "Original · NCIT P97 definition"); generated text sits in a dashed "Generated" frame
  and cites the facts it rests on; Exact's saved decisions are labelled "Matcher record". Scores are
  "matching scores", never percentages or probabilities unless calibration is validated.
- **One workspace for every product.** The meaning cards, comparison, hierarchy browsers, evidence
  list, evidence graph and fact inspector read through one data-source interface
  (`src/lib/workspace`): the exploration API, a study case's frozen resources (later its scoped
  workspace routes) or the synthetic tutorial. Shells own selection, answers and navigation state.
- **Every citation opens its record.** Citations are buttons that resolve the exact original record
  (typed subject, origin, interpretation, original axiom), whether or not it is shown on the page.
- **Reading order.** Choose a candidate, read both cards and the comparison, then open optional
  details. Details stay closed by default; numeric matching weights are hidden until asked for.
  In study cases the candidates and the answer stay in a sticky rail beside the workspace.
- **No hover-driven geometry.** The evidence graph uses a fixed layout, explicit zoom/fit/reset and
  drag-to-pan; clicking selects. Line patterns carry meaning (solid assertion, dashed structural
  relation, dash-dot matcher feature, a dotted purple line for a feature kind compared across
  ontologies) and the evidence list is its accessible equivalent. The view belongs to the compared
  pair and survives re-renders, saves, tab switches and layout changes.
- **Text size and reflow.** Sizes are in rem; a persistent A−/A+ control scales the whole app to
  200%. Layout breakpoints are container queries in rem, so enlarged text gets the narrow layout
  instead of a cramped wide one. No page-wide horizontal scrolling from 320 px upward.
- **State lives in the URL or on the server.** Source, candidate, focused entities and open tab are
  URL parameters (reload, back/forward and resize preserve them). Study answers, stage and revision
  belong to the study server; the browser keeps only the latest unsent draft for resend.
- **Versioned study flows.** `exact-study/1.0` sessions keep their frozen legacy steps;
  `exact-study/2.0` uses tool-neutral setup, the interactive tutorial with server-graded items and
  durable per-case reports; any other version fails visibly. Questionnaires follow declared option
  order (v2 `option_order`/`row_order`, v1 a versioned table checked against `forms.py`).
- **Study integrity.** One ranking component for both conditions; the baseline receives no
  explanation data from the server at all. Answers start empty; "None of these" and "Insufficient
  information" are explicit; nothing auto-submits. Every mutation carries an idempotency key and the
  revision it expects, so lost acknowledgements replay safely and a second tab produces a visible
  conflict instead of a silent overwrite. The private link is exchanged by POST for an HttpOnly
  cookie and removed from the address bar immediately.
- **Security headers.** Pages are served with a per-document Content-Security-Policy whose script
  hashes are computed from the export; the study pages additionally forbid inline styles.

## Develop

Requires Node.js 20+ and a running backend. The fastest loop is `next dev` with a proxy:

```bash
npm ci
EXACT_DEV_BACKEND=http://127.0.0.1:8000 npm run dev
```

`next dev` forwards `/api/*` to `EXACT_DEV_BACKEND`. For the study pages, the backend's origin
check requires the HTTPS origin it was configured with, so test the study through a built export
served by the study service instead. For exact-study/2.0, use the synthetic HTTPS/PostgreSQL
harness described under Verification.

A realistic development package (real NCIT/DOID excerpts, a clearly synthetic run and offline
exact-excerpt generations; no network, model or provider) can be prepared with:

```bash
python -m tools.build_explanation_ui_fixture --output /tmp/exact-ui-fixture
```

It needs `pyowl-core` and the `exact` run-store dependencies. Serve the package it prints with the
built frontend:

```bash
npm run build
exact-inspect serve --package /tmp/exact-ui-fixture/prepared/stages/<id>/package.json \
  --profile local_app --library-dir /tmp/exact-library --frontend-dir explanations_visualizer/out
```

From the repository root, `make build-frontend` builds the export and copies it to
`exact_inspect/static`, where installed services find it without `--frontend-dir`.
The prepared and study Dockerfiles in `deploy/render` build the export in a Node stage.

## Check before a release

- `npx tsc --noEmit` and `npm run build`.
- `pytest tests/explanation_frontend_serving_test.py tests/explanation_preparation_test.py tests/explanation_study_test.py`.
- In a browser: 320, 390, 768, 1024, 1440 and 2560 px; 200% text; light and dark themes;
  keyboard-only use of the candidate list, ranking controls, comboboxes and dialogs.
- Study: link exchange, reload with and without the cookie, a second tab (conflict), offline edits,
  pause/resume, revoked link, baseline requests for explanation resources (must be 403).

## Verification

Use `npm run typecheck`, `npm run test:unit` (Node.js 24), and `npm run test:e2e`.
The browser suite targets running prepared/study services; see the integration verification
guide above for URLs, the synthetic HTTPS/PostgreSQL harness, and optional test inputs.
`EXACT_E2E_SEARCH_TERM` replaces the real-package search term when testing the small fixture.

The exact-study/2.0 suites (`study-v2-backend.spec.ts`, the unseeded acceptance journey, and
`study-v2-integration.spec.ts`, the seeded fault/recovery regressions) run against the real
study service:

```bash
python -m tools.build_explanation_v2_fixture /tmp/exact-v2 --navigation-size 65
python -m tools.serve_explanation_v2_e2e --fixture /tmp/exact-v2 \
  --database-url postgresql://127.0.0.1:<port>/<synthetic-db> --tls-cert cert.pem --tls-key key.pem \
  --config /tmp/exact-v2-private.json --port 18984 --frontend-dir explanations_visualizer/out
EXACT_E2E_STUDY_V2_CONFIG=/tmp/exact-v2-private.json npx playwright test e2e/study-v2-*.spec.ts
```

Rebuild the export and restart the service together: the service hashes the pages' inline
scripts for its CSP when it starts. For the J03 comparison with the main app, wrap the same
frozen contexts with `python explanations_visualizer/e2e/prepare_main_app_package.py /tmp/exact-v2
/tmp/exact-v2-main`, serve that package with `exact-inspect serve --profile local_app`, and set
`EXACT_E2E_MAIN_APP_URL`.

Add `--paged-facts 26` to the fixture build for the paging regressions: it adds a non-focal
source class, "Paged facts source", whose categories and parents exceed the first page, and an
object and a data property ("Paged object property", "Paged data property") whose named
superproperties do too. For the
main-app paging test, wrap the same contexts with `prepare_main_app_package.py …
--extra-category alternate_definitions`, serve that package, and set `EXACT_E2E_PAGED_MAIN_APP_URL`
and `EXACT_E2E_PAGED_SOURCE` (the fixture's source ontology version).

Production uses the static `out/` export served by `exact-inspect`; `next start` is not
compatible with this export mode.

