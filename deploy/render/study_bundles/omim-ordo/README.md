# Study Bundle: omim-ordo

This directory is a minimal deployable bundle for the Exact study visualizer.
It contains only the files needed to serve the selected user-study cases and the
precomputed one-hop ontology extension layer.

## Included assets

- `config.yaml`: copied from `/home/pgcotovio/Exact-OM/exp/test/Full_local_bioml_with_exp/omim-ordo` and used to resolve the dataset class
- `study-bundle.yaml`: lightweight dataset spec for the visualizer bundle
- `analysis/user_study/study_mapping.json`: final user-study graph payload
- `analysis/user_study/study_selected_records_with_rationales.json`: selected pair records used by the UI
- `analysis/user_study/ontology_cache.json`: precomputed labels, annotations, and one-hop expansion neighborhoods for the entities used in the study
- `study_bundle.json`: metadata manifest describing the bundle contents

The visualizer can be pointed at this directory directly through `EXACT_STUDY_RUN_DIR`.
