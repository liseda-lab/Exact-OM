# WP-A — Foundations & Hygiene

**Depends on**: nothing. **Blocks**: everything. **Size**: S (1 agent, single PR).
**Behavior-preserving**: yes — except the named bug fixes, no output may change.

## Context

Every later WP is gated on a working test harness, CI, and clean packaging. Today the repo has
stale Matcha-DL-era artifacts, tests that cannot run or aren't collected, dead code, a packaging
bug that breaks a console script on install, and lint/type tooling pointed at the wrong targets.

## Tasks

### A1. Packaging (`pyproject.toml`)

1. Fix metadata: `repository = "https://github.com/liseda-lab/Exact-OM"`; delete the dead
   `exclude = ["matcha_dl/impl/matcha/matcha/**/*"]` (line 12).
2. Package `study_visualizer_runtime`: add `{ include = "study_visualizer_runtime" }` to
   `packages`. Rationale: `exact/delivery/cli/study_visualizer.py:105` imports it, so the
   installed `exact-study-viz` script is currently broken outside the repo root.
   (Interim fix only — WP-K renames/repackages it as `exact_inspect` behind the `viz` extra;
   don't invest beyond making the current shape installable.)
3. **Torch source**: change `torch = {version = "2.7.0+cu128", source = "pytorch-gpu"}` to a
   plain PyPI `torch = "^2.7.0"` (CPU/GPU per platform default), and document the CUDA-specific
   install in README ("for cu128 wheels: `poetry source ...` / `pip install --index-url ...`").
   Rationale: the explicit cu128 source blocks CI and non-CUDA installs. Do not change the
   version floor.
4. `[tool.mypy]`: `files = ["exact", "study_visualizer_runtime"]` with a permissive baseline
   (`ignore_missing_imports = true`, `check_untyped_defs = false` initially). It currently
   type-checks only `tests/**` — almost certainly a mistake.
5. Add `[tool.pytest.ini_options]`: `testpaths = ["tests"]`,
   `python_files = ["test_*.py", "*_test.py"]`, and register markers `requires_data`, `slow`,
   `requires_llm`.
6. Add dev deps to the `prebuild` group: `import-linter`, `pytest-cov`.
7. Add console script alias `exact-eval = "exact.delivery.cli.eval:main"` next to the existing
   `bioml-eval` (same `main`; `bioml-eval` stays for compat).

### A2. Repo hygiene

1. Delete `deploy/DockerFile` and `deploy/build.sh` — both reference the nonexistent
   `matcha_dl` tree and a private `matchadl` registry image; `deploy/render/` is the real
   deployment and stays.
2. Fix `.dockerignore`: it currently excludes the `exact` package itself (only made sense for
   the study-viz image); scope it to what `deploy/render/study_visualizer.Dockerfile` needs.
3. `git rm error.log` (committed, 0 bytes) and add `error.log` to `.gitignore`.
4. `.gitignore`: drop the stale `/matcha_dl/impl/matcha/matcha/*` block; keep the new
   `/Repair Ideas/*` entry (already in the working tree — commit it).
5. Remove the tracked root `.DS_Store` if tracked (`*.DS_Store` is already ignored).

### A3. Test suite repair

1. Delete `tests/prompt/` (both files import removed modules — `exact.impl.datasets.tabular`,
   `exact.impl.models.prompt` — and error at collection; `prompt_model_test.py` even calls
   `init_jvm` at import).
2. Delete `tests/tune_api_test.py` (no test functions; imports `TuningAlignmentRunner`, which
   no longer exists).
3. `tests/single_cli_test.py`: replace the `matchadl` command with `exact`, mark
   `@pytest.mark.requires_data @pytest.mark.slow`, and drop the hardcoded GPU/heap assumptions
   (parametrize via env vars with skips).
4. Rename `tests/align_api_test_suit.py` → `tests/align_api_test.py` (currently not collected —
   matches neither `test_*.py` nor `*_test.py`), mark `requires_data`.
5. `tests/single_api_eval_test.py`: mark `requires_data` if it needs real ontologies.
6. Verify: `pytest -m "not requires_data and not slow"` runs green, CPU-only, no network.

### A4. Dead code & bug fixes (small, surgical)

1. Delete `ComponentRegistry.register_validator` (`exact/core/entities/registry/registry.py:14-30`)
   — references the never-defined `cls._type_validators`; any call would `AttributeError`; it has
   no callers.
2. Delete the empty `DirectoryEvaluationAction` stub (`exact/core/actions/evaluation.py:120`).
3. **Fix the metric-dropping bug**: `IEvaluator.__init__` (`exact/core/contracts/evaluator.py:28`)
   builds its metric set from `ComponentRegistry.list(ComponentType.METRIC)[1:]` — the `[1:]`
   silently drops the first-registered metric (currently `PrecisionMetric`; only harmless because
   `F1Metric` recomputes P/R). Replace with an explicit mapping keyed by `MetricNames`. Add a
   regression test asserting every registered metric is resolvable.
4. Replace the bare `eval()` on candidate files in
   `exact/utils/eval.py:112` (`read_candidate_mappings`) with `ast.literal_eval` — same behavior
   for well-formed files, removes an arbitrary-code-execution hazard on crafted TSVs
   (`exact/core/contracts/dataset.py:623` already uses `literal_eval`).
5. Fix the `timmings` typo (`exact/core/actions/alignment.py:392`) and the `"skyping"`/
   `"Loaded Cached Dataset"` comment typos in `dataset.py`.
6. **Audit F1 — latent train-on-test guard**: LLM-decision calibration
   (`use_llm_calibration: True`) currently fits its coefficients on `(p_llm, gold)` samples
   whose gold labels come from the merged full/test reference
   (`semantic_scorer.py:1225-1284`, collection at `pair_adaptive_scorer.py:1887`). Restrict
   sample collection to training-reference pairs (hard-error if only full-reference labels are
   available) and add a regression test. Default behavior (flag off) is unchanged.
7. **Audit F7**: align `EvaluationData.K` default (`[1]`) with the config default
   (`[1, 5, 10]`).

### A5. Renames with shims (public API only)

1. `exact/utils/paths.py` → `exact/utils/graph_search.py` (it contains best-path graph
   algorithms, not filesystem paths). Update the ~4 internal importers; leave
   `exact/utils/paths.py` as a one-line re-export module emitting `DeprecationWarning`.
2. `EvalutionRunner` → `EvaluationRunner` in `exact/delivery/api/eval.py`, re-exported from
   `exact/delivery/api/__init__.py` and `exact/__init__.py`; keep
   `EvalutionRunner = EvaluationRunner` alias with a `DeprecationWarning` on use
   (module-level `__getattr__`).

### A6. Shared-helper consolidation (mechanical dedup)

Create `exact/utils/formatting.py` and move/unify, updating all call sites:
- `format_duration` — currently 4 copies: `exact/utils/logs.py`, `impl/trainer/semantic_runner.py`,
  `impl/models/candidate_set_selector.py`, `analysis/user_study.py`.
- `_strip_code_fences` — `utils/llm_routing.py` + `impl/models/candidate_set_selector.py`.
- `_clip01`, `_safe_mean`, `_safe_div`, `_quantile` — duplicated across `pair_adaptive_scorer.py`,
  `pair_adaptive_context.py`, `candidate_set_selector.py`, `analysis/alignment_diagnostics.py`,
  `analysis/candidate_recall.py`.

Keep signatures identical to the most general existing copy; if copies differ subtly, keep both
behaviors behind parameters and note it in the PR (do not silently change numerics).

### A7. CI (`.github/workflows/ci.yml`)

- Trigger: PRs + pushes to `main`.
- Matrix: Python 3.10 and 3.12, ubuntu-latest.
- Steps: install Poetry → `poetry install --with prebuild` (CPU torch after A1.3, cache the venv)
  → `black --check exact tests tools` → `isort --check` → `flake8` → `mypy exact` →
  `lint-imports` → `pytest -m "not requires_data and not slow" --cov=exact`.
- Add `.importlinter` with the two starter contracts: `exact.core` must not import
  `exact.impl`/`exact.delivery`/`exact.analysis`; nothing imports `exact.delivery`.

### A8. Bootstrap `CHANGELOG.md` (Keep-a-Changelog format), seeded with this WP's entries under
`[Unreleased]`.

## Out of scope

Everything behavioral: timing (WP-C), Java (WP-B), big-module splits (WP-D), delivery
consolidation (WP-D). Do not edit `exact/core/contracts/dataset.py` beyond the typo fixes —
WP-B owns it.

## Acceptance criteria

1. Fresh clone → `poetry install` (no CUDA, no Java) → `pytest -m "not requires_data and not slow"`
   green; `poetry build` → install wheel in a clean venv → `exact --help`, `exact-eval --help`,
   `exact-study-viz --help` all work.
2. CI workflow green on the PR.
3. `git grep -l "matchadl\|matcha_dl"` returns nothing (outside `specs/`).
4. No collection errors from pytest; every file under `tests/` is either collected or deleted.
5. `python -W error::DeprecationWarning -c "from exact.delivery.api import EvalutionRunner"`
   raises the deprecation (shim works), and the new names import cleanly.
