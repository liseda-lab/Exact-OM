# WP-K — `exact-inspect`: Alignment Inspection Service

**Depends on**: WP-B (it de-Javas `analysis/study_visualizer.py` first); coordinate with WP-D
PR-D3 (delivery consolidation) on the CLI shim. **Size**: M.
**Behavior**: the Render-deployed study-visualizer service keeps working (same endpoints, env
vars honored via shims); local usage gains a simpler entry point.

## Context

The alignment/explanation viewer is currently three loosely stapled pieces:

1. `study_visualizer_runtime/` — top-level FastAPI app serving one fixed study bundle. Broken
   as a package until WP-A (imported by `exact` but not shipped), and hardcodes
   `PROJECT_ROOT = Path(__file__).resolve().parents[1]` /
   `FRONTEND_BUILD_DIR = PROJECT_ROOT / "explanations_visualizer" / "out"`
   (`study_visualizer_runtime/app.py:27-28`) — only works from a repo checkout.
2. `exact/analysis/study_visualizer.py` (1045 lines) — bundle precompute (ontology cache,
   records) + a second FastAPI surface.
3. `explanations_visualizer/` — the Next.js/Cytoscape frontend, built to static `out/`.

Its name says "user study", but it is really the tool for **inspecting any alignment run**
(reads `src2tgt.maps_local.tsv` + `full_explanations.json`), run locally or deployed as a
service. Decision (owner-confirmed): rename to **`exact-inspect`**, keep in-repo behind an
extra, per the plugin model.

## Design

### K1. Repackage

```
exact_inspect/                  # new top-level package (replaces study_visualizer_runtime/)
  __init__.py
  app.py                        # FastAPI app factory: create_app(settings) — no import-time paths
  settings.py                   # pydantic-settings: run_dir, analysis_dir, frontend_dir,
                                #   enable_ontology_info, host/port/log_level.
                                #   Env prefix EXACT_INSPECT_*; legacy EXACT_STUDY_* accepted
                                #   with a deprecation warning (Render compat).
  bundles.py                    # bundle precompute moved from exact/analysis/study_visualizer.py
  helpers.py
  cli.py                        # `exact-inspect serve` / `exact-inspect open` / `exact-inspect bundle`
  static/                       # packaged frontend build (see K2)
```

- Poetry: second `packages` entry; **`fastapi` + `uvicorn` move from main deps to the `viz`
  extra** (`pip install exact-om[viz]`) — the matcher itself never imports them. `exact_inspect`
  import-guards them with an actionable error.
- Dependency direction: `exact_inspect` → `exact` (for analysis helpers), never the reverse.
  `exact/delivery/cli/study_visualizer.py` becomes a deprecation shim console script
  (`exact-study-viz` prints "renamed to exact-inspect" and delegates); remove in 2.1.
- `exact/analysis/study_visualizer.py` contents move to `exact_inspect/bundles.py` (+ shim);
  `tools/prepare_study_visualizer_bundle.py` + `tools/run_prepare_study_visualizer_bundle_job.py`
  fold into `exact-inspect bundle` (YAML job mode preserved).

### K2. Frontend packaging

- `frontend_dir` resolution order: explicit setting/env → packaged `exact_inspect/static/` →
  repo-relative `explanations_visualizer/out` (dev fallback) → API-only mode with a clear
  banner/log (never a crash).
- Wheel builds bundle the static export: a `make build-frontend` target (npm build + copy to
  `exact_inspect/static/`) run in the release workflow; the copy is gitignored (CI artifact,
  not committed). Document that `pip install exact-om[viz]` from a **release wheel** includes
  the UI, while source installs need `make build-frontend` or the dev fallback.

### K3. Modes

1. **`exact-inspect open <run_dir>`** (local, the new default path): serves any alignment run
   directory directly — no study bundle required. Access run artifacts through the `RunReader`
   API (contracts §14): this WP ships the v1 implementation (reads
   `model/alignment/src2tgt.maps_local.tsv` + `full_explanations.json`); WP-L later swaps in
   the indexed explanation store without touching this package. Do not hardcode run-dir paths
   here. Ontology info panel enabled when the run dir (or flags) provide
   ontology paths, resolved through `exact.ontology` (Java-free after WP-B — the precomputed
   `ontology_cache.json` workaround stays supported but is no longer the only server-friendly
   option). Opens the browser.
2. **`exact-inspect serve`** (service): current Render use case — fixed bundle dir from
   settings/env, read-only, iframe-embeddable (`/?source=<iri>` preserved).
3. **`exact-inspect bundle`**: precompute a study bundle from a run dir (ex `prepare_study_
   visualizer_bundle`).

### K4. Deployment updates

`render.yaml`, `deploy/render/study_visualizer.Dockerfile`, `start_study_visualizer.sh`,
`study_visualizer_requirements.txt` → renamed/updated to `exact-inspect serve`; legacy
`EXACT_STUDY_*` env vars keep working (settings shim) so the currently deployed service does
not need same-day migration. `docs`: WP-H's User-study guide covers both modes.

## Tests

1. Existing `tests/study_visualizer_runtime_test.py` ported to `exact_inspect` (same committed
   `omim-ordo` bundle, `TestClient`) — endpoints unchanged.
2. Settings: `EXACT_STUDY_RUN_DIR` legacy env resolves with deprecation warning;
   `EXACT_INSPECT_RUN_DIR` wins when both set.
3. Frontend resolution: packaged → dev fallback → API-only banner (parametrized tmp dirs).
4. `open` mode on a fixture run dir: endpoints serve mappings/explanations without a bundle.
5. Wheel test: install `exact-om` **without** `[viz]` → `exact` works, `exact-inspect` errors
   actionably; with `[viz]` → serves.

## Out of scope

Frontend feature work (Next.js app unchanged except build packaging); write/curation
capabilities; auth.

## Acceptance criteria

1. `pip install exact-om[viz]` (release wheel) → `exact-inspect open <run_dir>` works from any
   directory — the repo-checkout requirement is gone.
2. Render deployment path builds and boots with **zero** env-var changes (legacy shims), and
   with the new names when updated.
3. Main `exact` wheel no longer depends on fastapi/uvicorn (dependency tree check in CI).
4. `git grep study_visualizer_runtime` → only deprecation shims and CHANGELOG/MIGRATION.
