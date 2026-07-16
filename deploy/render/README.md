# Render Deployment

This folder contains the deployment assets for serving the study visualizer on
Render from a precomputed bundle.

## What gets deployed

The Render service serves:

- the static frontend from `explanations_visualizer/`
- the FastAPI backend exposed by `study_visualizer_runtime`
- a commit-friendly study bundle under `deploy/render/study_bundles/`

## Included deployment files

- `study_visualizer.Dockerfile`: multi-stage image that builds the frontend separately and keeps only the lightweight Python runtime in the final layer
- `study_visualizer_requirements.txt`: minimal runtime dependencies for the visualizer service
- `start_study_visualizer.sh`: runtime entrypoint that maps Render env vars to the lightweight visualizer server
- `render.yaml`: Render blueprint targeting the OMIM-ORDO bundle by default

## Preparing a bundle

Export a minimal study bundle from an existing run with:

```bash
poetry run python tools/prepare_study_visualizer_bundle.py \
  --run-dir exp/test/Full_local_bioml_with_exp/omim-ordo \
  --bundle-dir deploy/render/study_bundles/omim-ordo \
  --overwrite
```

If you want the same Slurm workflow shape as the other Exact runners, use the
Python launcher plus the thin sbatch script:

```bash
python tools/run_prepare_study_visualizer_bundle_job.py \
  --run-config exp/user_study/omim-ordo-bundle.yaml \
  --sbatch-script deploy/sbatch/prepare_study_visualizer_bundle.sh
```

Example YAML in [`exp/user_study/omim-ordo-bundle.yaml`](/home/pgcotovio/Exact-OM/exp/user_study/omim-ordo-bundle.yaml):

```yaml
bundle:
  run_dir: /home/pgcotovio/Exact-OM/exp/test/Full_local_bioml_with_exp/omim-ordo
  bundle_dir: /home/pgcotovio/Exact-OM/deploy/render/study_bundles/omim-ordo
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
- `study-bundle.yaml`
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
  - the `study_visualizer_runtime/` code
  - the `deploy/render/` assets
  - the exported frontend under `explanations_visualizer/out`

This avoids shipping Node, npm, the ontology-matching stack, or the full
repository into the final running container.

## Render env vars

Required:

- `EXACT_STUDY_RUN_DIR`

Optional:

- `EXACT_STUDY_ANALYSIS_DIR`
- `EXACT_STUDY_ENABLE_ONTOLOGY_INFO`
- `EXACT_STUDY_LOG_LEVEL`

`start_study_visualizer.sh` also uses Render's injected `PORT`.

The checked-in `render.yaml` defaults `EXACT_STUDY_ENABLE_ONTOLOGY_INFO=false`
as the safest baseline. After regenerating the bundle with a populated
`ontology_cache.json`, you can flip that env var to `true` in Render to enable
precomputed node info and one-hop expansion.

## Health check

The service exposes:

- `GET /api/health`

This returns the resolved run directory, analysis directory, config path, and
the indexed source/path counts.
