# Experiment Suite Implementation Clarifications and Agent Handoff

**Status:** authoritative implementation clarification for the experiment suite.

This file resolves the questions raised while implementing `E00`–`E26`. Read it together with
the current working-tree files in `specs/experiments/`. Where an experiment spec leaves one of
the questions below ambiguous, this file supplies the binding implementation decision. It does
not authorize changing the scientific question, promotion gate, split discipline, or reporting
contract of an experiment.

The software baseline is **Exact-OM 2.1.0 with `pyowl-core` 0.2.0**. The earlier Exact-OM
`0.2.1` wording in E00 was a typo. `R_0` is the concrete initial paper baseline; `R_n` remains
generic notation for a baseline at an arbitrary revision.

## Scope and ownership

The implementation agent owns code, configuration, fixtures, and automated validation only. The
user owns every real screen or confirm run, monitoring, retries, result inspection, candidate
selection, final design freeze, human annotation, and scientific interpretation. The agent may
run unit/integration tests with synthetic or existing repository fixtures and runner dry-runs. It
must not download research data or models, call hosted LLMs, launch cluster jobs, run real paper
experiments, or wait for/babysit those processes.

The current experiment specs are inputs to the implementation. The implementation agent must not
edit, stage, or commit anything under `specs/`. It must also leave research data, model caches,
and generated results uncommitted.

## Authoritative answers

### 1. Are the current working-tree specs authoritative and frozen with a spec-tree hash?

Yes. Treat the current `specs/experiments/*.md` tree, including this file, as authoritative for
this implementation. Record an immutable SHA-256 spec-tree hash in suite/design provenance. Hash
files in sorted POSIX-relative-path order, encoding each record as an 8-byte big-endian path
length, UTF-8 path bytes, an 8-byte big-endian content length, and the raw file bytes. A spec
change after screening invalidates confirmation until the suite is explicitly versioned and
screened again.

### 2. Is Exact-OM 0.2.1 in E00 a typo?

Yes. The required baseline is Exact-OM **2.1.0** with `pyowl-core` **0.2.0**. Do not create or
pretend there is an Exact-OM 0.2.1 release.

### 3. What is the frozen baseline called?

Use `R_0` as the concrete baseline ID for this first paper suite. Keep `R_n` only in prose and
code comments as generic revision notation. A later promoted stack becomes `R_1`, then `R_2`, and
so on.

### 4. Is a fully materialized historical B0 required now?

No. A complete B0 containing paper-data mappings, scores, metrics, and artifact hashes is not an
implementation prerequisite. Freeze `R_0` as the current production configuration with every
experimental flag disabled. Existing fixture-level replay evidence is sufficient for software
validation; the user will materialize paper-run evidence later. B0 may remain an optional
longitudinal comparator and must never be fabricated.

### 5. Must implementation cover every arm in E00–E26?

Deliver the smallest scientifically valid, runnable path for each experiment that can be
implemented with the repository's current capabilities. Implement shared primitives once and
reuse them. Optional, unavailable, or disproportionately heavy arms may remain explicitly
deferred with a machine-readable reason. Never make a fake arm that accepts configuration but
silently executes baseline behavior. E00 and the shared screen/confirm, provenance, selection,
statistics, and fail-closed machinery are mandatory.

### 6. May the public lifecycle become four stages?

No. The public and artifact lifecycle remains exactly `screen` and `confirm`. The scheduler may
order component screening, component confirmation, downstream/E17 screening, and downstream/E17
confirmation internally, but those are dependencies or subphases, not four public stage values
or separate runner implementations. There is no automatic transition from screen to confirm.

### 7. Must unavailable reporting datasets be supplied before implementation proceeds?

No. Implement and validate against currently materialized repository fixtures. Real confirmation
later uses only materialized eligible datasets named in the frozen design. Missing OMIM–ORDO,
Bio-ML, KG, or other tasks do not block code implementation and must not trigger downloads.

### 8. What happens when a spec-required dataset is unavailable?

Record the task or experiment as `deferred_unavailable` with the missing capability and expected
dataset identity. Do not substitute another task automatically and do not silently narrow the
paper claim. A deliberate scoped-down suite requires a new explicit config/design version.

### 9. Which datasets belong in the dataset lock?

Add OAEI-KG, BioKG, DISO, and E13 representation datasets only when they are actually
materialized. Every admitted real dataset needs an immutable revision/hash plus reference and
capability declarations. Test fixtures may have their own fixture identity but never count as
paper evidence. Missing real datasets remain deferred rather than blocking implementation.

### 10. What should be done for incomplete references and human adjudication?

Implement the minimal blinded sample export, annotation import/validation, adjudication merge,
and inverse-probability-weighted analysis path now. The user performs the annotation later.
Until completed adjudication is imported, precision claims on incomplete references are
descriptive or deferred; the runner must not invent labels or mark adjudication complete.

### 11. May undefined constants be chosen by the implementer?

Yes. Choose conservative, deterministic values for undefined formulas, thresholds, tie rules,
sampling constants, and small model hyperparameters. Put every choice in YAML or a hashed design
artifact, add a short rationale, and test determinism. Do not bury experimental constants in
code or tune them on reporting data.

### 12. How many promotion-primary endpoints are allowed?

Each explicit promotion decision has one primary endpoint and selects at most one candidate.
An experiment may contain multiple separately named decisions only when the specs distinguish
independent product/paper choices. Other metrics are secondary endpoints, mandatory guards, or
cost constraints; they are not an undocumented composite objective.

### 13. What is required for Holm multiplicity?

Holm-adjusted p-values plus raw paired 95% confidence intervals are acceptable. Label the
intervals as unadjusted and record the comparison family and adjustment order. Simultaneous
multiplicity-adjusted confidence intervals are not required for the MVP.

### 14. What does “reporting/test splits touched once per family” mean?

Once per **experiment family and version**. A frozen family may inspect its reporting output once
for the declared analysis. Retuning, changing arms, changing the candidate pool, or changing a
primary design creates a new version and requires a genuinely untouched reporting set; it does
not permit another look at the same test data. The rule is not one touch for the entire paper
programme and not one independent touch per dataset file.

### 15. Which LLM provider and identity rules apply?

No confirmatory LLM provider is guaranteed. Prefer a local Hugging Face model pinned to an
immutable commit when available. Hosted providers, including mutable OpenRouter aliases, are
screen-only unless the response/cache artifact records a provider-resolved immutable identity.
For confirmation, missing provider, requested and resolved model IDs/revisions, endpoint identity,
tokenizer, request time, prompt hash, decoding parameters, seed, or cache identity is a hard
failure. Never record credentials.

### 16. What replay equality/tolerance applies in E00?

Canonical source IDs, target IDs, selected mappings, and relation labels must match exactly.
Scores may differ by at most `1e-6` on deterministic CPU execution and `1e-5` on GPU execution;
aggregate metrics may differ by at most `1e-4`. Persist canonical ordering so serialization order
does not masquerade as a semantic mismatch. Any wider GPU variation must be measured and recorded,
not silently accepted.

### 17. What are the E01 stable-marriage and protected-match rules?

Use source-proposing stable marriage. Assignment component size is the number of source plus
target vertices in the connected candidate component. Protected exact matches are hard
constraints; conflicting protected exact matches are a pre-inference hard failure with the
conflicting IDs reported.

### 18. What exactly transfers in E03 `source_only`?

The donor supplies both its fitted calibrator and its fitted threshold. Apply both unchanged to
the recipient. A recipient-side unsupervised adaptation is a separate, explicitly named arm and
must not be folded into `source_only`.

### 19. What is E04's primary endpoint?

Use NIL-aware macro F1 as the single primary endpoint. NIL AUROC is secondary. Non-NIL local MRR
is a mandatory guard so improved abstention cannot conceal degraded ranking among answerable
sources.

### 20. What is the E05 development selection rule?

Compare at matched mean pool size, accepting a deviation of at most `max(1 candidate, 2%)`. A
candidate is eligible only when macro candidate recall improves by at least 0.5 percentage points
and no task loses more than 0.5 candidate-recall points. Rank eligible candidates by macro
candidate recall, then lower gold-rank p90, then lower gold-rank median, then runtime, then the
simpler configuration. Freeze at most one retrieval candidate.

### 21. Is E05 a full Cartesian sweep?

No. Use a bounded declared list: baseline, SapBERT-only, RRF-only, adaptive-k-only, and one
progressively assembled winner combination. A weighted-fusion arm is optional. Do not run the
full encoder × fusion × adaptive-k Cartesian product.

### 22. May E07 fit on source-pair labels?

Yes. Selector and calibration artifacts may be fitted from source-pair or otherwise disjoint
training data while the target pair remains label-free. The target pair must not influence the
fit. Label this `cross_pair_supervised`; retain a truly zero-shot/analytic target-label-free
control.

### 23. How are E07 permutations and stochastic samples combined?

Use two deterministic candidate-order permutations. `listwise_sc` draws three stochastic samples
per permutation, for six calls per source in the screen. For each call retain categorical
probabilities when available. Average as `0.5 * mean_categorical_probability + 0.5 *
vote_distribution`, followed by normalization. Use stable source/seed-derived permutation and
sampling seeds and deterministic target-IRI tie breaks.

### 24. How are E08 decisions structured?

Treat polarity and attribute-bank choice as two independent promotion decisions. If both produce
survivors, run and freeze the conditional 2×2 interaction. Do not encode them as one joint
winner-take-all selection.

### 25. Where do E08 allowlists and namespace rules come from?

They come from revision-pinned dataset descriptors, not code heuristics. A descriptor must name
the applicable property IRIs, polarity semantics, identifier namespaces, and normalization rules.
If the semantics are absent, the signed/polarity arm is unavailable and must fail closed or be
deferred.

### 26. What is the E09 hierarchy similarity?

Use IC-weighted Jaccard over anchor-mapped ancestor sets as the authoritative overlap:
`sum(IC(intersection)) / sum(IC(union))`. For an anchored cross-ontology ancestor pair, use the
mean of its two within-ontology normalized IC values. Lin similarity may be a labelled diagnostic,
not the primary formula.

### 27. Which E10 parameter is swept?

Sweep the scorer's LLM fusion pivot, `PairAdaptiveSemanticScorer.params.tau_LLM`, exposed through
the canonical config mapping. Do not accidentally sweep the neutral fusion pivot `tau` under a
misnamed `matching.llm.tau_llm` path.

### 28. How are E10's top three selected?

First reject configurations that fail explanation reconstruction, runtime/cost bounds, or any
per-task regression guard. Rank the rest by macro F1, local MRR, fewer LLM tokens, lower runtime,
then distance from shipped constants. Treat macro-F1 differences below 0.2 percentage points as a
tie before applying later criteria.

### 29. How are E11 and E12 phased?

Yes: first freeze the evidence-bundle result, then run the selector cross after the applicable
E15/E18/E19 artifacts are frozen. These are ordered subphases within screen/confirm, not new
public runner stages. Report each experiment once against its final frozen selector cross.

### 30. What are the E12 compatibility, literal, shuffle, and bootstrap definitions?

- Namespace compatibility is an explicit dataset-descriptor declaration.
- Identifiers compare by exact equality after the descriptor's declared normalization.
- Numerics compare only under compatible units, with relative tolerance `1e-6`.
- ISO dates compare exactly after canonical ISO normalization.
- Other free text uses the existing text encoder.
- The shuffle uses deterministic double-edge swaps within each KG, preserving predicate,
  direction, and node in/out degree; labels, types, literals, candidates, and references remain
  unchanged.
- Bootstrap independent units are weakly connected components of the source ABox graph.

### 31. What does E13 candidate parity mean?

For serialization/parity arms, realized canonical candidate rows must be identical. For the
information ablation, keep retrieval configuration, seed, and budget identical, but realized pools
may differ when added materialized facts legitimately affect retrieval. Record and compare both
pool fingerprints instead of pretending equality.

### 32. What is the minimal frozen E14 semantic contract?

Implement deterministic named-class/property transitive graph closure over the common
OWL-2-EL-like named subset already exposed by the ontology view. A bridge reasoner is optional and
may remain deferred. Use a 60-second per-task reasoning timeout; unsupported constructs or timeout
produce abstention, not guessed types. The learned head is multinomial L2 logistic regression.
Semantic anchors are exact matches plus reciprocal top-1 pairs with score at least `0.95` and
margin at least `0.10`. Represent abstention as no mapping. Keep equivalence-only fallback as a
separate explicit control.

### 33. Do E18–E23 force an E15 rerun?

No. If the original E15 pool, schema, and frozen artifacts remain compatible, append versioned
comparison columns without reopening E15 selection. A pool/schema change requires a preregistered
`E15-v2`; never overwrite or silently reinterpret E15.

### 34. How is E16 versioned and how often is target reporting inspected?

Use versioned transfer families. The first family covers artifacts available at its freeze. New
E18–E23 heads form a new transfer family rather than mutating the old one. Each target reporting
set is inspected once per frozen family/version; development data may be used for orchestration
and debugging without opening target reporting labels.

### 35. When does E17 require equal candidate pools?

Only contrasts that hold retrieval fixed must have identical candidate pools per task and seed.
Retrieval-treatment contrasts may intentionally differ, but must record their pool fingerprints
and refit every incompatible downstream artifact.

### 36. When may E17 remove a component during development?

Remove a component before the E17 freeze when `stack_minus_EXX` does any of the following on
development data: improves macro F1 by more than 0.5 points; improves local MRR by more than
`0.005`; repairs a hard guard failure; or reduces the declared cost by at least 20% while remaining
within a `-0.5` macro-F1 non-inferiority margin. Record the reason. Never use reporting data to
remove or retune a component.

### 37. Which E18 models may promote?

Under the current explanation contract, promotion is restricted to `channel_gating`. Linear
objective variants, additive GAM, and GBDT are scientific diagnostics unless a separately
versioned explanation contract is approved. They must not be silently presented as deployable.

### 38. How are E18 macro F1 and local MRR combined?

Do not combine them into a scalar. Macro F1 is primary. Local MRR is a mandatory guard with a
maximum tolerated regression of `0.005`. Report both and require explanation reconstruction.

### 39. Is E19 beta fitted?

No. Pin beta to the frozen E10 value. E19's `analytic_fitted` arm fits tau, gamma, and normalized
channel multipliers. Beta belongs to the E10 sweep so the experiments do not identify the same
constant twice.

### 40. What are the E20 preregistered defaults?

- Pool-size grid: `{10, 20, 40}`.
- Matched-mean tolerance: `max(1 candidate, 2%)`.
- Mine from the top 50 retrieved candidates.
- Keep at most 5 negatives per source and 100,000 training pairs per task.
- Exclude every known positive from negatives.
- If no safe negative policy exists, defer the arm; do not invent a PU loss.
- Evaluate in-pair supervised and leave-one-pair-out transfer as separate regimes using shared,
  disjoint artifacts.

### 41. How is the E21/E25 dependency resolved?

E25 owns gate instrumentation plus the non-learned `analytic`, `quantile`, `forced_sample`, and
`oracle` modes. E21 owns the first learned gate. E25 may later report that learned E21 gate as a
comparison, but E25 must not implement a router that E21 then depends on.

### 42. Is the learned LLM gate called `learned` or `router`?

Use `learned` consistently in configuration, artifacts, schemas, and reports. Accepting `router`
as a temporary read-only migration alias is optional; never emit it in new artifacts.

### 43. What is E22's promotion objective?

Label-efficiency/cost superiority. The selected policy must remain within a `-0.5` macro-F1
non-inferiority margin while using at least 25% fewer labelled effective units than the frozen
supervised control. Pure quality superiority may be reported but is not this decision's primary
objective.

### 44. What are E22's units, repetitions, active rule, crossover, and cost model?

- Acceptance, calibration, fusion, and ranking count unique labelled source groups, not candidate
  rows.
- Retrieval counts unique positive sources having at least one safely mined negative.
- Relation typing records a vector by relation and uses the minimum supported-relation count as
  the scalar deployment guard.
- LLM supervision counts unique sources with counterfactual/teacher decisions.
- Graph supervision counts unique labelled seed sources.
- Use three deterministic source-group subsamples per budget.
- Active selection sorts by descending uncertainty, then descending model disagreement, then
  source IRI.
- The crossover is the smallest budget whose paired 95% CI lower bound versus the full-supervised
  control exceeds `-0.5` macro-F1 points and remains so after development-only isotonic smoothing.
- Primary annotation cost is the number of effective units. Report typed tasks separately unless
  empirical time studies justify frozen cross-kind weights.

### 45. What E23 defaults are frozen?

- Inductive model: L2-regularized logistic regression over frozen graph features.
- Transductive model: two-layer mean GraphSAGE, 64-dimensional embeddings, normalized dot-product
  decoder.
- TBox retention levels: `{100%, 50%, 0%}`, using deterministic hash selection for 50%.
- Seed-count grid: `{0, 25, 50, 100, 250, 500, 1000, all}`.
- `tbox_density = named_subclass_edges / max(1, named_classes)`.
- `relation_density = non_type_object_edges / max(1, individuals)`.

Freeze seeds, optimizer settings, epochs/early stopping, and feature schema in YAML before use.

### 46. Where are E19 artifacts refitted for E23?

Refit incompatible E19-style fusion inside E23, namespaced as an E23 interaction artifact with
its own pool/feature hash. Do not overwrite E19 or call the artifact an E19 result. E17 later
confirms the combined graph-plus-fusion stack if it survives.

### 47. What are E24 `sim_fix` and `diff_absolute`?

`sim_fix` is not yet defined well enough to implement; leave it explicitly deferred and omit its
factorial unless a later spec defines it. For each side, compute unsupported mass as a normalized
mean over its available relation pool. Define
`diff_absolute = clip(0.5 * (mean_src_unsupported + mean_tgt_unsupported), 0, 1)` and
`s_diff = 1 - diff_absolute`. When both pools are empty, the channel is inactive rather than
neutral.

### 48. How are E25 quantiles and ties handled?

Fit quantiles separately per development task and entity kind. Sort deterministically by
`(-U, source_iri, target_iri)` and route exactly `ceil(p * N)` rows. Persist the selected count,
cutoff value, boundary IDs, and fitted threshold artifact so confirmation never refits on
reporting data.

### 49. Does the E25 oracle call a real LLM?

No. It is an analytical, always-correct ceiling derived from the reference and never invokes or
mixes a real LLM response. Mark it oracle-only and prevent it from entering selection or product
configuration.

### 50. Is E26 Stage 2 a five-arm decomposition or an eight-cell factorial?

Use exactly the five named arms: `full`, `constant_q`, `no_sharpening`, `suppression_only`, and
`uniform`. Do not generate an eight-cell factorial. Stage 2 is diagnostic and has no promotion
candidate.

### 51. What are E26 entropy and encoder-agreement definitions?

For fewer than two labels, raw entropy is `0`, normalized entropy quality is `0`, and
`quality_defined=false`. Use the configured model IDs
`cambridgeltl/SapBERT-from-PubMedBERT-fulltext` and `BAAI/bge-large-en-v1.5`, each pinned to an
immutable Hugging Face commit recorded in the model lock; never invent a commit or use `main` for
confirmation. Per-pair agreement is `1 - abs(s_sapbert - s_bge)`. Kendall agreement between
candidate rankings is descriptive only.

## Prompt for the implementation agent

Copy the prompt below into the implementation agent. It assumes the current branch already
contains partial experiment-runner commits and asks the agent to audit and finish them, not start
again.

```text
Role: You are the lead implementation agent finishing Exact-OM's lean paper experiment suite.

Goal: Finish the existing implementation so it satisfies the current files in
specs/experiments/, with specs/experiments/IMPLEMENTATION-CLARIFICATIONS.md resolving every
ambiguity. Produce lean, accurate, production-quality research code compatible with Exact-OM
2.1.0 and pyowl-core 0.2.0. Reuse the implementation already committed; do not redesign it.

Success criteria:
- tools/run_experiment.py and exact/experiments/* provide one strict screen/confirm workflow.
- E00 shared orchestration, provenance, split guards, deterministic selection, resume safety,
  paired reporting, and fail-closed behavior are complete.
- Each E01-E26 config either has its smallest scientifically valid runnable path or is explicitly
  deferred for a genuine unavailable/optional dependency. No configured arm silently behaves as
  the baseline.
- R_0 is the concrete baseline; R_n is generic prose notation only.
- Experimental switches default off and the normal production path remains unchanged.
- E17 implements the bounded rolling/stack/leave-one-out/declared-interaction contract.
- Targeted tests, fixture integration tests, and a suite dry-run pass.

Authority and scope:
1. Read all current specs/experiments/*.md before changing code. The clarification file is binding
   where an older spec is ambiguous.
2. Inspect the existing commits and working tree first. Preserve all unrelated and user-owned
   changes. Do not reset, rewrite, or duplicate completed work.
3. Implement code, experiment YAML, fixtures, and tests only. NEVER edit, stage, or commit specs/,
   exp/results/, research datasets, downloaded models, caches, or generated paper results.
4. Do not run real screen or confirm experiments, download data/models, call hosted LLMs, submit
   cluster jobs, monitor jobs, retry long jobs, choose winners from real data, freeze a real final
   selection, perform human annotation, or interpret paper results. The user will do all of that.
5. You may run unit/integration tests using synthetic or existing repository fixtures and
   --dry-run. Do not use reporting references in tests to tune behavior.

Implementation approach:
- Audit the partial implementation against the specs and clarification, then make only the
  missing changes.
- Prefer existing Exact-OM execution, config, metric, and artifact code. Keep schemas strict and
  deterministic. Fail clearly for unavailable datasets/models/capabilities and fake/no-op arms.
- Keep the public lifecycle screen|confirm. Internal dependency phases are allowed, but never
  auto-run confirm after screen.
- Put experimental constants and selection rules in hashed YAML/design artifacts, not hidden code.
- Keep orchestration local and bounded: --jobs for independent CPU cells; serialize GPU/LLM work
  per declared resource key. Do not add a dashboard, workflow database, sbatch generator, or full
  benchmark framework.

Parallel work:
- Use at most three subagents when work divides cleanly. Give each exclusive file ownership and a
  bounded task; parallelize independent primitives/tests, not overlapping edits.
- The lead owns integration, runner/schema changes, shared configs, E17, final validation, and all
  commits. Subagents do not commit. Review every subagent change before staging it.

Git and validation:
- Use explicit staging and small coherent Conventional Commits, for example
  feat(experiments): ..., fix(experiments): ..., and test(experiments): ....
- Never use git add -A or commit pre-existing spec changes. Verify the staged file list before
  every commit. Do not amend or rewrite existing commits unless explicitly asked.
- Run the narrowest relevant tests after each coherent change, then the experiment test group and
  a paper-suite --dry-run. Run broader tests only when the touched production path warrants it.
- If a required external resource is absent, keep the path fail-closed and mark it
  deferred_unavailable; do not fabricate evidence or substitute a dataset.

Stop rules:
- Stop when the code/config/test acceptance criteria above pass and no in-scope implementation
  gap remains. Do not start the experiments.
- If a missing choice would change scientific intent, report the exact blocker instead of making
  a new research decision.
- In the final response, list implemented behavior, remaining explicit deferrals, validation
  commands/results, commits created, and the exact manual dry-run/screen/confirm commands for the
  user. Be concise and do not include experimental conclusions.
```

## Commands reserved for the user

The implementation agent may use the first command only because it is a dry-run. The user runs
the real stages later.

```bash
# Implementation smoke check: resolve and print only; no experiment execution.
.venv/bin/python tools/run_experiment.py \
  --suite exp/experiments/paper-suite.yaml \
  --stage screen \
  --output-root exp/results \
  --jobs 4 \
  --dry-run

# User-run development screen.
.venv/bin/python tools/run_experiment.py \
  --suite exp/experiments/paper-suite.yaml \
  --stage screen \
  --output-root exp/results \
  --jobs 4 \
  --resume

# User-run confirmation after manually reviewing and freezing the screen selection/design.
.venv/bin/python tools/run_experiment.py \
  --suite exp/experiments/paper-suite.yaml \
  --stage confirm \
  --selection-record exp/results/exact-om-paper-v1/screen/selection.json \
  --output-root exp/results \
  --jobs 4 \
  --resume
```

Confirmation must remain impossible until the selection/design record, dataset/model locks, and
reporting matrix are complete and consistent. The runner stopping for an unmet prerequisite is a
successful guard, not permission for the implementation agent to supply or run the missing work.
