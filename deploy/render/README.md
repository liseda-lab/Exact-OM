# Render Deployment

This folder contains the deployment assets for serving `exact-inspect` on Render from a
precomputed bundle.

## What gets deployed

The Render service serves:

- the static frontend built from `explanations_visualizer/`
- the FastAPI backend exposed by `exact_inspect`
- a commit-friendly study bundle under `deploy/render/study_bundles/`

## Included deployment files

- `exact_inspect.Dockerfile`: multi-stage frontend and Python service image
- `exact_inspect_requirements.txt`: minimal service dependencies
- `start_exact_inspect.sh`: runtime entry point
- `/render.yaml`: repository-root Render blueprint targeting the OMIM-ORDO bundle by default

## Preparing a bundle

Export a minimal study bundle from an existing run with:

```console
poetry run exact-inspect bundle RUN_DIR deploy/render/study_bundles/omim-ordo --overwrite
```

If you want the same Slurm workflow shape as the other Exact runners, use the
Python launcher plus the thin sbatch script:

```console
poetry run exact-inspect bundle \
  --job-config exp/user_study/omim-ordo-bundle.yaml \
  --sbatch-script deploy/sbatch/exact_inspect_bundle.sh
```

Example job configuration:

```yaml
bundle:
  run_dir: runs/omim-ordo
  bundle_dir: deploy/render/study_bundles/omim-ordo
  overwrite: true

job:
  name: prepare_omim_ordo_bundle
  logging_level: INFO
  slurm:
    sbatch_args:
      - --partition=cpu
      - --cpus-per-task=6
      - --mem=24G
      - --time=06:00:00
```

The generated bundle contains:

- `config.yaml`
- `study_bundle.json`
- `analysis/user_study/study_mapping.json`
- `analysis/user_study/study_selected_records_with_rationales.json` or fallback
- `analysis/user_study/ontology_cache.json`

`ontology_cache.json` is built offline from the source ontologies using Exact's
pure-Python ontology backend.

## Image tightening choices

The current Render image is intentionally split into stages:

- a `node` build stage for the static frontend bundle
- a slim runtime stage with only:
  - Python
  - the minimal FastAPI runtime dependencies
  - the `exact_inspect/` code
  - the `deploy/render/` assets
  - the exported frontend under `exact_inspect/static`

This avoids shipping Node, npm, the ontology-matching stack, or the full
repository into the final running container.

## Render environment variables

Required:

- `EXACT_INSPECT_RUN_DIR`

Optional:

- `EXACT_INSPECT_ANALYSIS_DIR`
- `EXACT_INSPECT_ENABLE_ONTOLOGY_INFO`
- `EXACT_INSPECT_LOG_LEVEL`
- `EXACT_INSPECT_HOST`
- `EXACT_INSPECT_PORT` (Render's injected `PORT` takes precedence)

The checked-in blueprint deliberately retains `EXACT_STUDY_*` variables so an existing Render
service boots without an environment migration. `start_exact_inspect.sh` and the settings layer
accept those legacy names with a deprecation warning; new deployments should use the names
above.

The checked-in `/render.yaml` enables ontology information because its bundle includes an
`ontology_cache.json`. Set `EXACT_INSPECT_ENABLE_ONTOLOGY_INFO=false` when deploying a bundle
without that cache or source ontology files.

## Health check

The service exposes:

- `GET /api/health`

This returns the resolved run directory, analysis directory, config path, and
the indexed source/path counts.
