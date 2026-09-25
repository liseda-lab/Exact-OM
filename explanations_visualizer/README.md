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
| `/participate/` | Ranking study: private link, consent, setup, background form, practice, scored cases in both conditions, per-case consultation, final form, completion | `study` only |
| `/admin/` | Researcher administration: publish, progress, invitation links, replace/revoke, exports | `study` only |
| `/legacy/` | The historical run-directory viewer, used automatically when `/` finds only the old `/api/health` backend | legacy `serve --run-dir` |

Everything reads the backend contracts in `exact_inspect` (`/api/v1/...` and
`/api/v1/study/...`); the frontend never parses OWL, calls a model or infers equivalence.
Missing, filtered, unexported, unsupported and failed states are shown with their reason.

What was built, how it was verified and the known backend issues are in the
[F1 handoff](../docs/verification/explanation-frontend-handoff.md).

## Design decisions

Tokens (both themes) live in `src/styles/base.css`; product styles in `app.css` and `study.css`.

- **Sides are never colour-only.** Source entities use a slate circle marker, target entities an
  umber square, and every card names its side and kind ("Source class", "Target object property").
- **Where text comes from is always visible.** Original ontology statements show their predicate
  (for example "Original · NCIT P97 definition"); generated text sits in a dashed "Generated" frame
  and cites the facts it rests on; Exact's saved decisions are labelled "Matcher record". Scores are
  "matching scores", never percentages or probabilities unless calibration is validated.
- **Reading order.** Choose a candidate, read both cards and the comparison, then open optional
  details. Details stay closed by default; numeric matching weights are hidden until asked for.
- **No hover-driven geometry.** The evidence graph uses a fixed layout, explicit zoom/fit/reset and
  drag-to-pan; clicking selects. Line patterns carry meaning (solid assertion, dash-dot matcher
  feature, double-line cross-ontology comparison) and the evidence list is its accessible equivalent.
- **Text size and reflow.** Sizes are in rem; a persistent A−/A+ control scales the whole app to
  200%. Layout breakpoints are container queries in rem, so enlarged text gets the narrow layout
  instead of a cramped wide one. No page-wide horizontal scrolling from 320 px upward.
- **State lives in the URL or on the server.** Source, candidate, focused entities and open tab are
  URL parameters (reload, back/forward and resize preserve them). Study answers, stage and revision
  belong to the study server; the browser keeps only the latest unsent draft for resend.
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
served by the study service instead.

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
