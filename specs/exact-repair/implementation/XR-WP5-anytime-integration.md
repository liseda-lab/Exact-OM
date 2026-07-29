# XR-WP5 — Anytime solving, Exact-OM integration, and review artifacts

**Status:** proposed<br>
**Depends on:** XR-WP2; XR-WP3 for rich states; XR-WP4 only for learned modes; WP-M release gates or
an explicitly supported shared-stack version<br>
**Unlocks:** G5a/G5b candidate, XR-E06–XR-E09<br>
**Research boundary:** supplies systems studied by RQ7–RQ10. Performance engineering and UI
availability do not themselves demonstrate scalability, robustness, transfer, or review benefit.

## Objective

Turn the exact research kernel into a bounded, inspectable, opt-in post-processing stage. Add valid
anytime bounds/incumbents, conservative component acceleration, caching, public CLI/API integration,
artifact inspection, and certificate-derived explanations without weakening fail-closed semantics.

## Anytime engine

Support budgets for wall time, memory, master time, oracle time, explanation time, and reasoner-call
count. At every interruptible boundary, persist only:

- the last completely verified safe incumbent;
- a solver-provided valid upper bound for the frozen master space when available;
- objective/gap definitions and timestamp;
- pending/unverified candidate state only as trace data, never as output.

`SAFE_WITH_GAP` requires both verified incumbent and valid comparable bound. If no valid bound is
available, report the verified incumbent as `SAFE_FALLBACK` with `gap=null` and the mandatory
no-optimality-claim warning; if that incumbent is all-off, also emit the information-loss warning.

## Conflict graph and decomposition

Discovered conflicts define a state-literal hypergraph. Conservative acceleration:

1. solve currently disconnected components for a warm start;
2. combine component assignments under global governance constraints;
3. verify the whole integration;
4. merge components linked by any new cross-component conflict;
5. continue until exact/anytime termination.

Component objectives/bounds may be composed only when all cross-component hard constraints and
policy interactions are absent or accounted for. A whole-KB check is mandatory. Decomposition is
never used to declare a component-wise solution globally safe without that check.

## Caching and incremental behavior

Versioned caches may contain core classification/baseline, compiled state modules, entailment
queries, explanations, shrink results, solver conflicts, and local modules. Keys include all
semantic inputs: snapshot logical/import fingerprints, policy/query compiler, state compiler and
inventory, reasoner/backend, and relevant settings.

Incremental reasoning/solving must be differential-tested against canonical rebuild. Cache hits and
misses are recorded. A stale/ambiguous cache is invalidated; unsafe deserialization/conversion is
not attempted.

Reference mode disables acceleration and rebuilds every assignment. XR-E07 compares optimized and
reference results on overlapping tractable cases before using optimized measurements.

## Exact-OM integration

### Standalone first

Finalize `exact repair` and `RepairRunner` over captured alignment artifacts. This remains the
benchmark path.

### Opt-in pipeline stage

Add an explicit `repair.enabled=true` post-processor after global alignment selection and before
final alignment serialization/evaluation. It receives the live shared snapshots/providers and a
captured immutable mapping/evidence view. It does not rerun retrieval/scoring and does not mutate
the provisional artifact.

Write both provisional and repaired alignments. Evaluators require an explicit artifact choice;
published matcher scores are not silently replaced by repaired results. Repair time receives its
own timing-ledger phase and is excluded/included transparently in task totals.

When repair is disabled, all config, API, CLI, output, timing, and import behavior remains
compatible. The default stays false through XR-1 unless G6 is separately approved.

## Output formats

Simple selected states serialize through existing relation-aware writers. Guarded/qualified states
that cannot be represented faithfully in OAEI RDF/TSV require:

- a canonical OWL axiom artifact as authoritative output;
- a JSONL mapping-state view with full semantics;
- an optional lossy simple projection marked `lossy=true` with omitted-component counts.

Never serialize a guarded state as plain equivalence/subsumption without a loss marker. Evaluation
uses the authoritative state/axiom form for semantic metrics.

## Explanations and review

Generate human-facing explanations solely from certificate facts:

- provisional and selected states plus changed components;
- violation and responsible state conflict that rejected a higher-utility alternative;
- affected classes/queries and immutable support summary;
- selected state's evidence/utility/uncertainty;
- nearest higher-utility rejected and nearest safe alternative from counterfactual solves;
- safety/optimality distinction and fallback/profile warnings.

Model-generated prose may summarize these fields but is labelled and cannot add logical premises.
The structured certificate remains authoritative. Add `RunReader`/`exact-inspect` repair views only
after schemas are stable; historical matching explanations remain distinct.

Counterfactual solves freeze other choices according to a recorded intervention definition. They
are independently verified and cached; “why not” text cannot be inferred from utility alone.

## Search prioritization

Optional L3 models may order violation checks, explanation requests, component expansion, or master
warm starts. They cannot alter constraints, skip final enabled queries, or change status semantics.
XR-E07 compares them with deterministic order under identical frozen candidate spaces/budgets.

## Operational failure handling

- Process/worker termination closes reasoners and finalizes only atomic verified artifacts.
- Memory budgets use explicit monitored limits; OOM without an incumbent returns fallback/unknown
  according to completed verification.
- Model/network dependencies are not required for deterministic hand-coded runs.
- Import acquisition remains pinned/offline according to the snapshot manifest.
- A production coverage requirement that rejects all-off is a governance constraint. If no safe
  assignment meets it, report no governed solution; never return an unsafe alignment.
- Retention/GC treats repair certificates, selected states, final verification, and repaired
  alignment as protected artifacts.

## Tests and performance gates

- forced budget at every loop boundary returns only verified incumbent/fallback;
- bound/gap arithmetic agrees with exhaustive optima;
- component warm-start plus merge equals global reference optimum on tractable cases;
- injected cross-component conflict triggers merge and whole-KB recheck;
- incremental/cache on/off results and certificates are semantically identical;
- repair-disabled end-to-end output parity;
- standalone and integrated runs consume identical captured mappings/snapshots and yield identical
  repair artifacts;
- lossy writer flags are mandatory and accurate;
- counterfactual explanations replay and never cite absent conflict evidence;
- historical run/inspect paths remain readable;
- scale benchmarks record, rather than waive, the shared-stack WP-M performance blockers.

## Acceptance criteria

XR-WP5 is done when:

1. anytime interruption always preserves a verified incumbent and valid/null gap semantics;
2. decomposition/caching match reference mode on all tractable overlap cases;
3. standalone and opt-in integrated paths are semantically identical for the same captured input;
4. default-disabled behavior is unchanged;
5. simple and complex state outputs are faithful or explicitly lossy;
6. certificate-derived structured explanations/counterfactuals replay;
7. run manifests/timing/retention include repair artifacts without corrupting historical readers;
8. XR-E06–XR-E09 runners can execute from immutable configs and captured inputs.

## Experiment handoff

- XR-E06 tests transfer, not merely whether an unseen pair runs.
- XR-E07 maps the exact/anytime frontier and L3 efficiency (RQ8/H7).
- XR-E08 tests fail-closed operational robustness (RQ9/H8 safety portion).
- XR-E09 evaluates expert review benefit and requires separate ethics/governance approval.

G5a may release the base-state stage as opt-in; rich-state families and learned modes additionally
require G5b. G6/default promotion remains a separate decision.
