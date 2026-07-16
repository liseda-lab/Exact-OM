# WP-D — Module Decomposition & Delivery Consolidation

**Depends on**: WP-A, WP-C (consumes ledger hooks). Runs parallel to WP-B — it must not touch
WP-B-owned files (`core/contracts/dataset.py`, `core/entities/ontology.py`, `impl/datasets/*`,
delivery JVM lines). **Size**: L (1 agent, 3 stacked PRs).
**Behavior-preserving**: strictly. This WP is pure refactor: same outputs, same file formats,
same registry names, same config keys. Every commit keeps the test suite green.
**Status**: Done (2026-07-16).

## Context

Five modules carry most of the iteration debt (line counts at baseline):
`impl/models/candidate_set_selector.py` **3294**, `impl/trainer/semantic_runner.py` **3055**
(one ~650-line `predict` with a duplicated completion path), `analysis/user_study.py` **2716**,
`impl/models/semantic_scorer.py` **2340**, `impl/models/pair_adaptive_scorer.py` **2171**.
Plus: actions declared as `typing.Protocol` with static `run` (namespace misuse), and CLI/API
pairs duplicating validation/assembly.

## Refactor rules

- Move code verbatim wherever possible; changes limited to imports, `self.`-plumbing, and
  extracting parameters. No logic edits, no numeric changes, no renamed config keys.
- Public entry symbols keep import paths via re-export (`from .runner import
  SemanticAlignmentRunner` in the old module path) so registry lazy-import strings and
  checkpoints keep working. Registry names (`ComponentRegistry`) must not change.
- Before each split, add **characterization tests** where coverage is thin (golden-file style on
  fixtures) so the split is provably neutral.
- After each split, run the WP-B fixture e2e smoke to confirm identical outputs.

## D1 — `impl/trainer/semantic_runner.py` → `impl/trainer/` package

```
impl/trainer/
  __init__.py          # re-exports SemanticAlignmentRunner (registry path stable)
  runner.py            # orchestration; predict() reduced to a readable skeleton
  checkpointing.py     # manifest read/write, fingerprints, restore scan, schema versioning
  audit_io.py          # audit/candidate-record JSONL+zstd shards, manifests, legacy migration
                       #   NOTE: WP-L (wave 3) replaces this module's internals with the
                       #   explanation store — keep its public surface small and boring
  overlays.py          # final overlay + post-inference checkpoint logic
  rationales.py        # generate_final_rationales_for_records + rationale checkpoint flow
```

Required internal fixes (behavior-neutral):
- `predict` currently has a fully duplicated "already-complete" fast path (~:2310-2421) mirroring
  the main-path finalization (~:2738-2842). Extract one `_finalize_run(state) -> Results` used by
  both; the two paths differ only in how `state` was produced.
- Preserve WP-C's timing hooks (`StageRecord` list, checkpoint `timing` block) exactly.
- Keep checkpoint schema version and fingerprint payloads byte-compatible —
  `tests/semantic_runner_checkpoint_test.py` (compact restore, legacy migration, fingerprint
  payload) must pass unmodified.

## D2 — `impl/models/candidate_set_selector.py` → `impl/models/selector/` package

```
impl/models/selector/
  __init__.py          # re-exports CandidateSetSelector (+ SecondPassReranker alias)
  selector.py          # IModel surface: forward(), config wiring
  features.py          # candidate-frame feature engineering
  calibration.py       # OOF validation / accept-model selection (_select_accept_model_by_* etc.)
  acceptance.py        # thresholding/cardinality acceptance logic
```

Fold `impl/models/second_pass_reranker.py` (15-line alias subclass) into
`selector/__init__.py`; keep the registered name `SecondPassReranker` (configs reference it —
the legacy shim in `ConfigModel._merge_registry_entry` rewrites `second_pass_params` to it).

## D3 — scorers

`semantic_scorer.py` (legacy) and `pair_adaptive_scorer.py` (default) share helper logic.
Extract `impl/models/scorer_common.py` for the *verbatim-identical* helpers only (embedding
cache management, fusion utilities, decision-head plumbing) — resist unifying near-duplicates
whose numerics differ. Target: each scorer file < 1500 lines with clear channel sections.
The global model-cache path (`~/.cache/exact/semantic_scorer/...`) and `cache_persist_policy`
semantics are unchanged.

## D4 — `analysis/user_study.py` → `analysis/user_study/` package

Split by the module's own phases: `selection.py` (balanced study selection), `export.py`
(mappings/records export), `taxonomy.py` (failure taxonomy), `notebook.py` (analysis notebook
generation), `__init__.py` re-exporting the CLI-facing functions. `tests/user_study_analysis_test.py`
passes unmodified.

## D5 — actions & delivery consolidation

1. `core/actions/alignment.py` / `evaluation.py`: replace the `Protocol`+`@staticmethod run`
   pattern with plain functions `run_alignment(...)` / `run_evaluation(...)`; keep
   `AlignmentAction.run`/`EvaluationAction.run` as deprecated thin aliases (external code may
   call them).
2. New `delivery/common.py`: single implementation of the duplicated CLI/API logic — input-file
   existence validation, output-dir creation, config loading, log setup, arg assembly (the JVM
   part is already gone via WP-B; rebase). `cli/align.py`, `api/align.py`, `cli/eval.py`,
   `api/eval.py` become thin: parse/accept args → `delivery/common` → action function.
   CLI flags and help text are frozen (README documents them).
3. Move `utils/llm_routing.py` → `exact/llm/routing.py` (733 lines, a subsystem not a util);
   shim module at the old path re-exporting with `DeprecationWarning`.

## D6 — deterministic tie-breaking (audit F6)

`filter_top_n_entity_mappings` / `filter_top_n_target_entity_mappings`
(`core/entities/mappings/entity.py:89-136`) use `heapq.nlargest` with key `(protected, score)` —
order among equal keys is implementation-defined. Add a deterministic secondary key (target/
source IRI). **Verification requirement**: fixture outputs byte-identical, plus one real-task
comparison showing either no metric change or only tie-order permutations (list any affected
pairs in the PR — if metrics move beyond ties, stop and flag; that would make this
result-changing and out of scope).

## D7 — layering enforcement

Extend `.importlinter` to the full contract set from `01-target-architecture.md` (core ⊥ impl,
nothing imports delivery, analysis not imported by impl, `pyhornedowl` only in
`exact.ontology.parser`, `rdflib` only in `exact.io`+`data/get_data.py`). Fix any violations
surfaced (expected: none new; report pre-existing ones and fix or whitelist with a comment).

## Stacked PRs

✋ PR-D1: trainer package split (D1). ✋ PR-D2: selector + scorers (D2, D3).
✋ PR-D3: user_study, actions/delivery, llm move, tie-break fix, import-linter (D4–D7).

## Out of scope

Anything WP-B owns (datasets/ontology/delivery-JVM); output-writer extraction from
`core/contracts/trainer.py` (WP-G does it when adding formats); behavior fixes beyond the
documented duplicate-path unification.

## Acceptance criteria

1. Full hermetic suite green after every stacked PR; checkpoint & rationale tests unmodified.
2. Fixture e2e run byte-identical outputs (alignment TSVs, `full_explanations.json`,
   `run_stats.json` minus timing fields) before vs after each PR — add
   `tools/diff_run_outputs.py` to automate this comparison.
3. A resume-from-old-checkpoint test: a checkpoint written at baseline restores correctly after
   D1 (fixture-scale artifact committed under `tests/fixtures/checkpoints/`).
4. No file in `exact/` exceeds ~1500 lines (soft; document any exception).
5. `lint-imports` green with the full contract set.

## Deviations

`exact/impl/datasets/base.py` is 1,519 lines, marginally above the approximate 1,500-line soft
ceiling; the WP-D refactor targets are below it, and this WP-B/F-owned dataset keeps its
kind-aware retrieval and cache lifecycle together.
