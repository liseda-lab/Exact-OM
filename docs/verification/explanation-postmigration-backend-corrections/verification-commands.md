# Repeating backend verification

Run from the migrated repository with the retained runtime. These are the same broad-suite modules; rerunning the final source also includes the later regressions. The dedicated v2 backup/restore check remains conditional on its separate PostgreSQL configuration. Do not point it at a retained study database.

```sh
PY=/Users/pedrocotovio/Library/Caches/exact-om-dev/pyenv/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$PY" -m tools.review_storage_budget check --growth-gb 3
CHECK_DIR="$(mktemp -d /private/tmp/exact-postmigration-check-XXXXXX)"
"$PY" -m tools.review_storage_budget run --growth-gb 3 -- "$PY" -m pytest -q \
  -o cache_dir="$CHECK_DIR/cache" --basetemp "$CHECK_DIR/fixtures" \
  tests/explanation_context_test.py \
  tests/explanation_corrective_http_test.py \
  tests/explanation_description_metadata_test.py \
  tests/explanation_description_prompt_test.py \
  tests/explanation_integration_timing_test.py \
  tests/explanation_navigation_test.py \
  tests/explanation_postmigration_corrections_test.py \
  tests/explanation_review_catalogue_test.py \
  tests/explanation_review_descriptions_test.py \
  tests/explanation_review_environment_test.py \
  tests/explanation_review_forms_test.py \
  tests/explanation_review_http_verifier_test.py \
  tests/explanation_review_pair_http_test.py \
  tests/explanation_review_pair_verifier_test.py \
  tests/explanation_review_private_run_cache_test.py \
  tests/explanation_review_publication_test.py \
  tests/explanation_review_resume_test.py \
  tests/explanation_review_run_cache_test.py \
  tests/explanation_review_verifier_test.py \
  tests/explanation_study_disconnect_test.py \
  tests/explanation_study_resource_test.py \
  tests/explanation_study_v2_test.py \
  tests/explanation_study_workspace_test.py \
  tests/explanation_tutorial_entity_contract_test.py \
  tests/explanation_workspace_evidence_test.py \
  tests/explanation_workspace_pair_metadata_test.py
"$PY" -m tools.freeze_explanation_contracts --check
```

The actual certificate-verified API replay command, immutable input identities, and test environment are in the main handoff. The stored native ELK fixture receipt separates native execution from API-only diagnostics. No full NCIT/DOID classification is part of these commands.
