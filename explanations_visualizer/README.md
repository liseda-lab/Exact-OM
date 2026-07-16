# Exact Inspect Frontend

This directory contains the frontend for `exact-inspect`: a
read-only graph application used to inspect user-study cases generated from an
Exact run. It is designed to be embedded in LimeSurvey through an iframe and is
served together with the optional `exact_inspect` FastAPI service.

The visualizer is not a standalone file-browser application. It serves one
fixed study run at a time and loads one source case per request through:

```text
/?source=<exact_source_iri>
```

## What The Visualizer Shows

For a given source entity, the app presents:

- the five ranked target candidates selected for the user study
- one candidate graph at a time
- explanation edges with cumulative detail levels
- bridge edges between source-side and target-side evidence
- optional one-hop ontology expansion for nodes that carry ontology provenance
- pair brief and rationale text when available

The interface is intentionally constrained for study use:

- no file picker
- no export/download workflow
- no user-editable colors
- deterministic styling across users and runs

## Runtime Contract

The frontend expects the backend to expose the study APIs at the same origin:

- `GET /api/study/source?source=<source_iri>`
- `GET /api/study/node-info?source=<source_iri>&target=<target_iri>&node_id=<node_id>`
- `GET /api/study/expand-node?source=<source_iri>&target=<target_iri>&node_id=<node_id>`

Those endpoints are implemented in the Python code under:

- [`exact_inspect/app.py`](../exact_inspect/app.py)
- [`exact_inspect/cli.py`](../exact_inspect/cli.py)

The frontend therefore assumes it is being served by the visualizer backend,
not by an isolated static web server.

## Run and bundle inputs

`exact-inspect open` reads a historical or layout-v2 Exact run through `RunReader`; it does not
require a user-study export or monolithic explanation file. A curated service bundle instead
uses:

- `analysis/user_study/study_mapping.json`
- `analysis/user_study/study_selected_records_with_rationales.json`
  or `analysis/user_study/study_selected_records.json`
- `analysis/user_study/ontology_cache.json` when precomputed ontology expansion
  is enabled

The deployed visualizer does not need the original ontology files. One-hop
expansion is read from the precomputed ontology cache bundled with the study
payload.

## Repository Layout

The active frontend surface is intentionally small:

- [`src/app/page.tsx`](./src/app/page.tsx): study page shell and sidebar state
- [`src/app/components/study/StudyGraph.tsx`](./src/app/components/study/StudyGraph.tsx): Cytoscape graph view
- [`src/app/hooks/types.ts`](./src/app/hooks/types.ts): study graph and API types
- [`src/app/hooks/graphStyles.ts`](./src/app/hooks/graphStyles.ts): deterministic node and edge styling
- [`next.config.ts`](./next.config.ts): static export configuration

`next build` exports the frontend to `out/`, and FastAPI serves that directory
alongside the study API routes.

## Local Development

### Prerequisites

- Node.js with `npm`
- the Python environment for the main Exact repository
- an Exact run directory containing user-study artifacts

### Build The Frontend

From this directory:

```bash
npm install
npm run build
```

### Serve The Visualizer

From the repository root:

```bash
poetry install --extras viz
poetry run exact-inspect serve \
  --run-dir runs/omim-ordo \
  --analysis-dir runs/omim-ordo/analysis/user_study \
  --port 8000
```

Then open:

```text
http://localhost:8000/?source=<exact_source_iri>
```

### About `npm run dev`

`npm run dev` is only useful for frontend iteration. The page fetches
same-origin `/api/study/*` endpoints, so a plain Next dev server is not enough
for end-to-end use unless you add your own proxy layer. The normal development
path is:

1. build the frontend bundle in this directory
2. serve it through `exact-inspect serve`

If you do want to point the frontend at a separately running backend during
frontend iteration, you can set:

```bash
NEXT_PUBLIC_STUDY_API_BASE_URL=http://localhost:8000
```

That makes the app request `${NEXT_PUBLIC_STUDY_API_BASE_URL}/api/study/*`
instead of same-origin `/api/study/*`.

Without that env var, the frontend also falls back to `http://localhost:8000`
when it detects a localhost frontend running on any port other than `8000`.

## Configuration

The backend can be configured either through CLI flags or environment
variables.

Required:

- `EXACT_INSPECT_RUN_DIR`

Optional:

- `EXACT_INSPECT_ANALYSIS_DIR`
- `EXACT_INSPECT_FRONTEND_DIR`
- `EXACT_INSPECT_ENABLE_ONTOLOGY_INFO`
- `EXACT_INSPECT_HOST`
- `EXACT_INSPECT_PORT`
- `EXACT_INSPECT_LOG_LEVEL`

The corresponding `EXACT_STUDY_*` names remain accepted temporarily for existing deployments
and emit a deprecation warning. New names take precedence when both are present.

Equivalent CLI:

```bash
poetry run exact-inspect serve --help
```

## Interaction Model

The current study UI supports:

- selecting one of the five ranked targets for the current source
- choosing a cumulative explanation level from 1 to 4
- filtering visible node and edge categories
- hovering nodes to highlight local structure
- clicking nodes to pin inspector details
- clicking expandable nodes to toggle one-hop additional ontology context

Important semantics:

- `ontology-extra` nodes and edges are additional ontology context, not model
  explanation evidence
- attribute nodes remain literal and are not expandable
- definitions are shown in the inspector or hover preview, not rendered as
  explanation edges

## Deployment Notes

The intended deployment model is one FastAPI service, suitable for platforms
such as Render:

1. build the frontend with `npm install && npm run build`
2. start the Python server with `exact-inspect serve`
3. point the service at a fixed run directory through environment variables

One deployment serves one study run. LimeSurvey can then embed specific source
cases by setting the `source` query parameter in the iframe URL.
