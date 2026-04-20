# Exact Study Visualizer

This directory contains the frontend for the Exact study visualizer: a
read-only graph application used to inspect user-study cases generated from an
Exact run. It is designed to be embedded in LimeSurvey through an iframe and is
served together with a lightweight FastAPI backend via
`study_visualizer_runtime`.

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

- [`study_visualizer_runtime/app.py`](../study_visualizer_runtime/app.py)
- [`study_visualizer_runtime/cli.py`](../study_visualizer_runtime/cli.py)

The frontend therefore assumes it is being served by the visualizer backend,
not by an isolated static web server.

## Required Run Artifacts

The backend loads its data from an existing Exact run directory. At minimum,
the run must provide:

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
poetry install
poetry run python -m study_visualizer_runtime.cli \
  --run-dir exp/test/Full_local_bioml_with_exp/omim-ordo \
  --analysis-dir exp/test/Full_local_bioml_with_exp/omim-ordo/analysis/user_study \
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
2. serve it through `python -m study_visualizer_runtime.cli`

## Configuration

The backend can be configured either through CLI flags or environment
variables.

Required:

- `EXACT_STUDY_RUN_DIR`

Optional:

- `EXACT_STUDY_ANALYSIS_DIR`
- `EXACT_STUDY_ENABLE_ONTOLOGY_INFO`
- `EXACT_STUDY_HOST`
- `EXACT_STUDY_PORT`
- `EXACT_STUDY_LOG_LEVEL`

Equivalent CLI:

```bash
poetry run python -m study_visualizer_runtime.cli --help
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
2. start the Python server with `python -m study_visualizer_runtime.cli`
3. point the service at a fixed run directory through environment variables

One deployment serves one study run. LimeSurvey can then embed specific source
cases by setting the `source` query parameter in the iframe URL.
