# WP-E — OAEI-Bio-ML-eval Integration

**Depends on**: WP-A (start immediately); final ontology-touching wiring rebases on WP-B
(coherence/ignored-class inputs). **Size**: M (1 agent, single PR + a follow-up when upstream
publishes its API). **Behavior**: additive — `evaluation.backends: ["builtin"]` default
reproduces today's outputs exactly.
**Status**: Done (2026-07-16).

## Context

Evaluation today: `EvaluationAction.run` → `Evaluator` (`exact/impl/evaluator.py`) → global
P/R/F1 (`impl/metrics/full.py`, with `use_in_alignment` filtering and train-reference
subtraction) or local Hits@k/MRR (`impl/metrics/local.py`) → `evaluation_results.csv`.

The tracks this system targets are being scored by
[OAEI-Bio-ML-eval](https://github.com/OAEI-ML/OAEI-Bio-ML-eval) — the shared library used by
both participants and organizers. **Upstream status (checked 2026-07-15): `v0.1.0.dev0`,
package skeleton** (PEP 621 + hatchling; planned modules `equivalence/` [P/R/F1, MRR, Hits@k,
reasoner-based coherence], `typed/` [Relation-Aware Typed MRR, Hierarchy-Aware Typed nDCG@10 —
"dependency-free"], `coherence/` [structural proxy]). The integration must therefore be built
against a **seam**, tolerant of upstream evolution, with the built-in evaluator remaining the
always-available fallback. Note the typed metrics coincide with BioKG-Align's leaderboard
metrics — WP-G's `typed-tsv` writer produces exactly the submission format `SrcEntity,
TgtEntity, Relation, Score` that `typed/` evaluation consumes.

## Design

### E1. Evaluator registry

- Add `ComponentType.EVALUATOR`; make `IEvaluator` self-registering like the other contracts
  (today `Evaluator` is instantiated directly). Move `exact/impl/evaluator.py` →
  `exact/impl/evaluators/builtin.py`, registered as `"builtin"` (shim at the old path).
- New `exact/impl/evaluators/bioml.py`, registered as `"bioml"` — the **only module that imports
  the upstream package**. Every upstream symbol is resolved through one internal
  `_load_bioml_api() -> BioMLApi` indirection so upstream churn is absorbed in one place.
- `EvaluationAction`/`run_evaluation` gains `backends: list[str]` (config
  `evaluation.backends`, CLI `--eval-backends builtin bioml`). Backends run in order; each
  contributes a namespaced metric dict.

### E2. Dependency

- Poetry optional extra: `oaei-bioml-eval = { git = "https://github.com/OAEI-ML/OAEI-Bio-ML-eval.git", rev = "<pinned sha>" }`
  under `[tool.poetry.extras] bioml-eval` (verify the actual distribution/import name from the
  upstream `pyproject.toml` at implementation time — do not guess). Optional because upstream is
  pre-release; selecting the `bioml` backend without it installed raises a one-line actionable
  error (`pip install "exact-om[bioml-eval]"`).
- Pin by `rev`; bumping the pin is a routine PR (adapter tests gate it).

### E3. Adapter mapping (Exact ⇄ upstream)

| Exact artifact | Upstream input |
|---|---|
| `src2tgt.maps_global.tsv` (`SrcEntity, TgtEntity, Score`) | equivalence global scoring (P/R/F1) |
| `src2tgt.maps_local.tsv` (`SrcEntity, TgtEntity, TgtCandidates`) | local ranking (MRR, Hits@k) |
| `typed-tsv` output (WP-G) | `typed/` metrics (Typed MRR, typed nDCG@10) |
| reference TSVs + `test.cands` | gold/candidate inputs, passed through unchanged (same Bio-ML formats) |
| source/target `OwlOntologySource` (WP-B) | whatever `coherence/` needs (if it needs parsed ontologies, adapt via `exact.ontology`; **never** reintroduce a Java path — if upstream's "official reasoner-based coherence" requires Java, surface it as `unavailable_local` and record the structural-proxy score instead, noting it in results) |

Until upstream lands, encode this table as `BioMLApi` (a `Protocol` with
`evaluate_equivalence(...)`, `evaluate_ranking(...)`, `evaluate_typed(...)`,
`structural_coherence(...)`), plus a `FakeBioMLApi` test double. `_load_bioml_api()` performs
capability discovery (`getattr` probing per planned module) and returns a partial API with
explicit `missing: set[str]`; requesting a missing capability produces a skipped-metric entry
(`"bioml.typed.mrr": null` + reason), never a crash. This keeps CI green at `v0.1.0.dev0` and
lights up automatically as upstream fills in.

**Upstream interface proposal**: since we co-develop the eval package, write
`specs/upstream-bioml-eval-api.md` during this WP — the exact function signatures Exact wants
(pure-function, path-or-DataFrame in, dict out, no I/O side effects) — and open it as an issue
on OAEI-Bio-ML-eval so the two evolve toward the same seam.

### E4. Results & reporting

- New canonical output `evaluation_results.json`: `{backend: {metric: value | null, ...},
  "meta": {refs, k, versions, skipped: {...reasons}}}`. `evaluation_results.csv` stays as the
  flat Metric/Value view (back-compat; namespaced keys like `bioml.equivalence.f1`).
- Timing: standalone eval runs in its own ledger session; inline eval records
  `Postprocess.Evaluation` (WP-C wiring — keep it; matching time and eval time must be
  separable in `timings.json` since OAEI reports matching time only).
- `exact-eval` CLI (alias from WP-A) gains `--eval-backends`; existing `bioml-eval` invocation
  syntax keeps working unchanged.

### E5. Reference-file provenance guard (audit F2)

Record sha256 + row counts of every reference/candidate input (`-r`, `-f`, `-c`) into
`run_stats.json` (and the eval results meta); when evaluation subtracts a train reference,
warn if its hash differs from the one the selector calibrated on in the same run dir. No
behavioral change to metrics.

## Tests

1. Builtin-path regression: existing eval outputs byte-identical with default config.
2. Adapter contract tests against `FakeBioMLApi` (full and partial capability sets; missing
   capability → skipped entry, exit code 0).
3. `requires_data`-marked integration test running `builtin` + `bioml` (real pin) on the fixture
   pair once upstream ships ≥ one metric; until then the test asserts graceful degradation
   against the real skeleton package.
4. JSON/CSV writers round-trip; CSV unchanged for builtin-only runs.

## Out of scope

Typed-relation *production* (WP-G `relation_prediction`); modifying builtin metric math;
upstream package development itself (proposal doc only).

## Acceptance criteria

1. Default runs (`backends: ["builtin"]`) byte-identical to baseline.
2. With the extra installed, `exact-eval --eval-backends builtin bioml` on the fixture pair
   produces `evaluation_results.json` with both namespaces (bioml entries possibly all
   `skipped` at current upstream state — that's a pass).
3. Without the extra installed, selecting `bioml` fails with the actionable install message;
   not selecting it never imports it.
4. `specs/upstream-bioml-eval-api.md` written and filed upstream as an issue/PR.

## Deviations

The proposal is committed, but filing it in the external upstream tracker remains project
coordination outside this repository. The initial implementation pinned the upstream
`0.1.0.dev0` skeleton and its Python-3.12 floor. The WP-M follow-up now consumes the compatible
`oaei-bioml-eval[reasoner]>=0.2,<0.3` line on Python 3.10–3.12. Its official global/local
coherence scorer receives Exact's existing source providers and therefore the same concrete
`pyowl-core` snapshots; it no longer reparses each provider's origin path or substitutes the
unfinished structural proxy. HermiT/ELK selection, timeout, and invalid-IRI policy are explicit
backend/CLI options. Publication remains coordinated with the shared packages' release gates.
