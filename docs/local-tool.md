# Exact Explain: local installation

Exact Explain opens saved ontology-matching evidence in your browser. It includes
candidate comparison, ontology browsing, evidence lists and graphs, decision
traces, score details, generated descriptions when present in the bundle, and a
local bundle library. It does not run the matcher when you open a page.

## Install

Use Python 3.11–3.12 on Linux or macOS; on Windows use WSL2. Create a virtual
environment and activate it before installing. The normal Exact dependencies
are still required; the `viz` extra adds the local web server. The matcher also
supports Python 3.10, but the viewer requires the SQLite safety controls added
in Python 3.11.

From a release wheel, which already contains the built interface:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install './exact_om-2.0.0-py3-none-any.whl[viz]'
exact-inspect serve
```

Open **http://127.0.0.1:8000/** and choose Library to import an inspection ZIP.
Node.js, a separate frontend server, an API key and a GPU are not needed for
serving an existing bundle. Fonts and scripts are served locally.

From a Git checkout, install Node.js 24 with npm, then run from the repository root:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[viz]'
python scripts/build_frontend.py
exact-inspect serve
```

The build downloads the locked npm dependencies and copies the three exported
pages into `exact_inspect/static`. Repeat it after changing the frontend. Restart
the server after changing code or replacing its static build. A release source
distribution also contains the built interface and does not require Node.js to
install; Node.js is needed only to rebuild it.

Stop the local server with **Ctrl+C**. It listens on loopback by default and has
no login system. Keep that default for personal use. To access a remote personal
installation, use an SSH tunnel rather than exposing an unauthenticated library.

## Prepare your ontologies and run

Use the same source and target ontology files that produced the saved run:

```console
exact-inspect prepare \
  --source-ontology /path/to/source.owl \
  --target-ontology /path/to/target.owl \
  --run-dir /path/to/exact-align-run \
  --output prepared/example
```

The command prints a JSON object with the absolute `package` path. Use that exact
path in the next command:

```console
exact-inspect serve --package /path/printed/by/prepare/package.json
```

Preparation builds searchable ontology context, asserted and structural hierarchy
navigation, and a read-only view of saved matcher evidence. The original run is
not modified. Omit `--run-dir` for ontology browsing without a matcher run.
Inputs and preparation stages are hashed; repeating the command with unchanged
inputs reuses completed stages. Keep preparation output on a writable disk with
space for indexes and exports. Large ontologies can take substantial time to
prepare; page requests use the prepared indexes.

`--run-dir` accepts a standard **`exact align` run directory**, including
`run_manifest.json`. Standalone `exact score`, `exact decisions`, `exact acceptance`
or `exact rank` output directories are a different format and are rejected.
They require a separate adapter before they can become inspection bundles.
Renaming a file or passing a mapping TSV does not supply the missing evidence.

For a small offline example, first run the [README demonstration](https://github.com/liseda-lab/Exact-OM#offline-demonstration),
then prepare its outputs:

```console
exact-inspect prepare --source-ontology demo/mini_src.owl \
  --target-ontology demo/mini_tgt.owl --run-dir runs/demo --output prepared/demo
```

The demonstration exercises the workflow; its evidence coverage is smaller than
that of an encoder run. A trace displays only stages actually recorded by the
producer. Preparation does not reconstruct missing historic decisions, create
reasoner results, or generate descriptions by default.

## Use the app

- **Compare:** choose a source and candidate, inspect their ontology facts, open
  original axioms, compare available generated descriptions, and inspect Evidence,
  Decision trace and Scores. Graph and list views share the same evidence records.
- **Browse ontologies:** search for an entity or start from the prepared entry
  points. Switch between hierarchy interpretations when available. Inferred views
  require a bundle with previously prepared reasoner results; opening the app
  never starts a reasoner.
- **Library:** inspect a ZIP before importing, follow or cancel the import, open
  an imported bundle, view its details, and remove an owned copy after opening
  another bundle. Removing a copy leaves the original inputs and ZIP unchanged.

Theme and text-size controls are in the header. Review decisions and preferences
are stored in the current browser, not in the bundle or an exported study dataset.
No participant tracking, invitation system or researcher administration is included.

## Export, storage and configuration

```console
exact-inspect export --package /path/to/package.json --output example.zip
exact-inspect serve --library-dir /path/to/local-library --port 8001
```

Import `example.zip` through Library on another installation. Exported bundles
include their prepared ontology context, so check ontology redistribution terms
before sharing them. The library defaults to `~/.exact-explain/library` and
remembers the selected imported bundle. `--package` explicitly opens a prepared
package at startup. Do not move or edit its contents while the server is running.

The equivalent environment settings are `EXACT_INSPECT_PACKAGE`,
`EXACT_INSPECT_LIBRARY_DIR`, `EXACT_INSPECT_FRONTEND_DIR`, `EXACT_INSPECT_HOST`,
`EXACT_INSPECT_PORT` and `EXACT_INSPECT_LOG_LEVEL`. Command-line values override
environment values. `--frontend-dir` selects an already built static export.
The read API is under `/api/v1`; `/api/v1/health` reports the open package.

## Explicit imports and optional generation

For ontology import closures, use a JSON preparation template with local files.
Paths are relative to the template. Imports are never fetched automatically:

```json
{
  "design_revision": "my-local-bundle/1",
  "ontologies": [
    {
      "name": "Source",
      "root": {"path": "source.owl"},
      "scope": "closure",
      "imports": {"https://example.org/shared": {"path": "shared.owl"}}
    },
    {"name": "Target", "root": {"path": "target.owl"}}
  ],
  "run": {
    "binding": {"path": "run/run_manifest.json"},
    "source_ontology": "Source",
    "target_ontology": "Target"
  }
}
```

```console
exact-inspect prepare --template inputs.json --output prepared/example
```

A template can also explicitly enable offline preparation of generated text:
set `profile` to an object with `name` and an OpenRouter `model` ID, and `entities`
to a list of objects with `id`, `ontology` (a name above), `iri` and `kind`.
Optional `pairs` objects name entity IDs in `source` and `target`. Configure the
provider credential through `OPENROUTER_API_KEY` in the process environment, never
in the template. This opt-in step makes provider requests and may incur charges;
normal preparation and serving do not. Generated output is checked against the
selected ontology facts. The preparation report records blocked or failed stages;
inspect it rather than treating a missing description as a successful generation.

## Troubleshooting

- **Frontend export not found:** run `python scripts/build_frontend.py` in a source
  checkout, or reinstall a release wheel containing the built pages.
- **Port already in use:** stop the other local server or choose `--port 8001`.
- **Run/ontology mismatch:** use the exact ontology versions used for that run.
- **Missing graph, description or trace stage:** check whether those records exist
  in the original run or bundle. The app does not invent unavailable evidence.
- **Invalid archive or checksum:** export again from the original prepared package;
  do not edit files inside the ZIP. A raw ontology ZIP is not an inspection bundle.
