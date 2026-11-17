# Run Exact Explain locally

The [public demo](https://exact-explain.lasige.di.fc.ul.pt/) offers ten prepared
sources, candidate comparison, ontology browsing, evidence graphs, generated
descriptions and recorded matcher information. It does not accept bundle uploads.
Some pipeline events were not retained in the study export; the demo marks those
as unrecorded rather than reconstructing them.

The local application additionally provides the Library: import prepared bundle
ZIPs, switch between bundles and remove local library entries. Serving prepared
results needs a CPU, not a GPU or an API key. Matching, ontology preparation and new
generated descriptions are separate offline operations.

## Docker: full viewer and import library

From this repository checkout:

```sh
docker compose -f deploy/local/compose.yaml up --build -d
```

Open <http://localhost:8000/library/> and import a prepared Exact inspection ZIP.
The application listens on loopback only. The `exact-library` named volume retains
imported bundles across restarts. Stop without deleting data:

```sh
docker compose -f deploy/local/compose.yaml down
```

Do not add `--volumes` unless you intend to delete your imported library. To update,
pull the desired release and repeat `up --build -d`. The image builds its frontend
and installs the lean prepared-serving requirements; it does not install the
matcher/training stack.

## Without Docker

Use Python 3.12 and Node.js 24. From the repository root:

```sh
python3.12 -m venv .venv-ui
.venv-ui/bin/python -m pip install -r deploy/render/exact_inspect_prepared_requirements.txt
(cd explanations_visualizer && npm ci && npm run build)
EXACT_UI_PYTHON="$PWD/.venv-ui/bin/python" sh deploy/local/start.sh
```

Open <http://localhost:8000/library/>. The default library is
`~/.local/share/exact-explain/library`; set `EXACT_UI_LIBRARY` to choose another
location. Stop with Ctrl+C. To open an existing unpacked package directly:

```sh
EXACT_UI_PYTHON="$PWD/.venv-ui/bin/python" sh deploy/local/start.sh \
  --package /absolute/path/to/package.json
```

The environment variable applies to that command only. No credentials are needed
to inspect an already prepared package. Do not expose a writable local library
service to the public internet.

## Prepare your own data

Install the main package with its `viz` extra (`poetry install -E viz`) for offline
preparation. Follow the [backend preparation guide](explanation-backend.md) to bind
an execution lock, prepare ontology contexts and explanations, and export a ZIP:

```sh
poetry run exact-inspect bind-lock --template bindings.json --output execution.json
poetry run exact-inspect prepare --lock execution.json --output-root prepared
poetry run exact-inspect export --package /path/to/prepared/package.json --output inspection.zip
```

The template identifies your ontology and run files and any generation profile;
see that guide for its schema. New matching or generation may require additional
compute or provider credentials; viewing existing bundles does not. Only import
prepared packages whose source and licence you trust. The server validates the
package and its checksums before making it available.
