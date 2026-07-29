# XR-WP2 — Complete-or-fail-closed exact repair engine

**Status:** proposed<br>
**Depends on:** XR-WP1; shared-core axiom construction/integration support; the pyELK/pyHermiT
complete reasoning and justification stack (assumed available, verified by the audit below); one
exact optimization backend<br>
**Unlocks:** G2, XR-WP3, XR-E01, XR-E03, XR-E08<br>
**Research boundary:** implements the mechanism studied by RQ1, RQ4, and RQ9. Acceptance tests do
not establish benchmark safety or comparative superiority.

## Objective

Implement an end-to-end class-repair engine for base states using a lazy reasoner-constrained
master problem. It must return only verified safe assignments, prove candidate-relative optimality
in exact mode, and retain a verified all-off fallback when the immutable core is admissible.

## Capability dependencies

Upstream pyELK and pyHermiT are committed to complete reasoning and full justification support at
parity with the original Java reasoners; the audit's job is to verify conformance and measure
throughput at representative scales, not to establish whether those capabilities exist. Before
implementation, run it against the installed compatible `pyowl-core`, pyELK, and pyHermiT APIs.
The audit must verify, measure, or explicitly block:

- constructing an integration view from the exact existing source/target snapshots plus trusted
  and selected generated axioms without reparsing paths;
- consistency, named-class satisfiability/coherence, and finite entailment queries under Direct
  Semantics for declared profiles;
- complete result/unknown distinction;
- sufficient axiom explanation/proof support or a correct black-box deletion-check path;
- rebuild-mode verification and lifecycle cleanup;
- canonical mapping between generated state modules and reasoner axioms;
- measured incremental-versus-rebuild classification cost at representative fixture and pilot
  scales;
- measured justification generation/enumeration cost, configured limits, and timeout behavior;
- batch classification/query support and throughput, including whether coherence can be obtained
  without per-class satisfiability calls.

Package names are not capabilities. A verification failure is an upstream regression: record it in
a versioned upstream contract/fixture and block the affected path until it is fixed. Do not fall
back to asserted hierarchy and call the result safe. The capability/throughput ADR and G2 decision record the measurements, usable limits,
and any profile or benchmark scale that remains unsupported.

## In scope

- `RepairReasoner` capability negotiation and shared-snapshot adapters;
- immutable-core baseline checks and policy query evaluation;
- `RepairSolver`, exhaustive reference backend, and one selected exact backend;
- one-hot state variables, governance constraints, integer/lexicographic utility, bounds/gaps;
- counterexample-guided orchestration and sufficient state-conflict generation;
- module-restricted checks implementing `../03-module-soundness.md`, including its required
  fixtures; module results verify incumbents only for the policy families that contract grants;
- optional deterministic eager conflict seeding (for example, mapped disjointness paths) validated
  by the same replay rules as lazy cuts;
- exact research mode plus deterministic safe fallback;
- final rebuild-mode verification and authorizing certificates;
- standalone `exact repair` and replay commands behind an optional repair extra;
- base-state logical, exhaustive, differential, fault-injection, and certificate tests.

## Out of scope

- guards, qualifications, endpoint replacements, learning, component decomposition, and production
  anytime performance;
- a full conservative-extension decision procedure;
- property/individual mapping repair;
- enabling repair in `exact align` or changing matching defaults.

## Policy release 1

The first complete path supports:

1. immutable-core consistency;
2. relative named-class coherence over a frozen monitored signature;
3. an optional finite list of forbidden entailments only if the selected adapter declares complete
   query and explanation support.

Compute and record baseline unsatisfiable classes and already-entailed forbidden queries before
adding mutable states. If the core fails a non-relative policy condition, return `UNSAFE_CORE`.
Trusted pinned mappings included in the core are reported in explanations but cannot be revised.

The monitored class signature defaults to named classes in both resolved import closures. Any
sampling or restriction is a different, explicitly named policy—not “full coherence.”
Relative coherence must use a complete classification or equivalent batch unsatisfiable-set
operation and compare the result with the recorded baseline set. Its budget is per policy family;
per-query budgets are reserved for the explicit finite, capped forbidden-entailment list.

## Solver selection

At the start of XR-WP2, compare candidate exact backends on deterministic tiny pseudo-Boolean
fixtures. Record an ADR covering licensing, Python support, incremental no-goods, integer objective
bounds, deterministic settings, packaging, and timeout behavior. Select one product backend while
retaining the exhaustive enumerator as the semantic oracle for small tests.

Required constraints:

- exactly one state per revision object;
- pinned/governance requirements;
- each conflict `C`: sum of its state literals is at most `|C|-1`;
- optional cardinality constraints only when explicitly configured and recorded;
- lexicographic tie-break: maximize semantic utility, then minimize complexity, then changed
  objects, then canonical assignment ID.

Backend objective/bound direction and scaling receive dedicated tests. A feasible/timeout response
without a valid bound cannot produce an optimality gap.

## Oracle and explanation behavior

The oracle returns `SAFE`, `UNSAFE(violations)`, or `UNKNOWN(reason)`. Check cheap complete queries
first, but persist results for every enabled policy family on the accepted final assignment.

For each unsafe assignment, process every violation the oracle reports in that iteration — batch
cut extraction, since one classification typically reveals many violations at once — and for each
violation:

1. obtain a proof/justification or run black-box deletion checks over immutable axioms and selected
   state modules;
2. project mutable occurrences to selected state literals;
3. add any conditional state literal required by the query;
4. verify the projected modules are sufficient for the same violation;
5. optionally shrink at state-module granularity;
6. persist explanation, sufficiency result, and no-good before adding it to the solver.

Minimality is not required; sufficiency is. A conflict containing no mutable state after a baseline
pass is an invariant failure/unknown, not an empty no-good. An unexplained unsafe candidate causes
a compatible retry or `ORACLE_UNKNOWN`; it is never accepted and never yields a guessed cut.

Black-box deletion checks and shrinking operate at state-module granularity, so their cost is
bounded by the number of selected non-off states, not by axiom count. Multiple justifications for
one violation may each yield a distinct valid cut.

## Exact orchestration requirements

- Verify all-off once before search and retain it as the initial incumbent.
- Never reason over a solver assignment that violates one-hot/governance validation.
- A safe master optimum with a proven bound may return `OPTIMAL_SAFE` immediately after final
  rebuild verification.
- A safe nonoptimal candidate is retained but exact mode continues until its budget stops the run;
  termination routes it to `SAFE_WITH_GAP` when a valid bound is retained and `SAFE_FALLBACK`
  otherwise.
- Each `UNSAFE` iteration must persist at least one valid cut violated by the current candidate, so
  the next solve makes strict progress. Where explanations succeed, persist one cut per distinct
  reported violation; iteration counts and cuts per iteration are recorded for the throughput
  audit. A projected conflict that is already present or fails to
  exclude the current candidate is an invariant failure handled as `ORACLE_UNKNOWN`, never a
  re-solve with unchanged constraints.
- Final verification reconstructs from the frozen state inventory, not an incremental reasoner
  object that may contain stale axioms.
- Interruptions finalize only a previously verified incumbent and atomically mark trace truncation.

## Failure matrix

| Failure | Required result |
|---|---|
| Core inconsistent/forbidden | `UNSAFE_CORE`; diagnostic certificate, no repaired alignment |
| Unsupported profile/query | reject affected states or whole run; no safe status without a complete compatible path |
| Reasoner timeout/exception | compatible retry; otherwise verified incumbent with retained valid bound as `SAFE_WITH_GAP`, verified incumbent without one as `SAFE_FALLBACK`, or `ORACLE_UNKNOWN` when no verified incumbent exists |
| Unsafe assignment without sufficient conflict | retry explanation path; otherwise `ORACLE_UNKNOWN` |
| Solver timeout | verified incumbent with a valid bound as `SAFE_WITH_GAP`; verified incumbent without one as `SAFE_FALLBACK`; otherwise `ORACLE_UNKNOWN` |
| Master remains infeasible after governance constraints are removed | internal invariant error/`INVALID_INPUT`, never `UNSAFE_CORE` |
| Master is infeasible only with explicit coverage/pinning governance constraints | `NO_GOVERNED_SOLUTION`; record the responsible constraints and do not publish an unsafe alignment |
| Objective overflow/NaN | validation failure and no solve |
| Model unavailable | deterministic hand-coded utility; record degraded quality mode |
| Certificate/replay write failure | do not publish an authorizing result |

On master infeasibility, re-solve—or perform the equivalent static feasibility check—with explicit
governance constraints removed while retaining one-hot constraints and valid reasoner-derived
conflicts. If that model is feasible, return `NO_GOVERNED_SOLUTION` and identify the responsible
governance constraints. If it is still infeasible, return the invariant error/`INVALID_INPUT`.
One-hot selection plus mandatory off states for every mutable revision object guarantees
feasibility in the governance-free model after pinned trusted axioms have passed the core baseline,
so this check is cheap and decisive.

## Verification

### Exhaustive equivalence

For generated small instances, enumerate every complete assignment, classify it with the reference
oracle, and compare the engine result to the maximum safe integer utility. Assert:

- returned assignment is safe;
- exact objective equals exhaustive optimum;
- every accumulated no-good is satisfied by every enumerated safe assignment;
- every rejected assignment is excluded and never returned again;
- all-off is safe exactly when the recorded baseline is admissible.

### Logical regression

Run supported-profile cases from XR-WP1's catalogue, including irrelevant axioms, multiple
explanations, trusted mappings, baseline unsatisfiability, and forbidden entailments. Unsupported
DL mechanisms must return an unknown/unsupported status until a complete path exists.

### Differential and replay tests

- the required tests of `../03-module-soundness.md`, including the L4 witness fixture proving a
  module-level `SAFE` never verifies a coherence incumbent;
- incremental versus rebuild assignment checks;
- two complete adapters/reasoners where profile overlap makes this meaningful;
- source/target shared-snapshot fingerprints before and after each run;
- certificate replay without the solver or utility model;
- worker-wire versus in-process semantic equality if worker isolation is supported.

### Fault injection

Inject timeouts/exceptions after baseline, master solve, unsafe result, explanation, safe incumbent,
and final verification. No injected path may serialize `safe=true` without an authorizing final
report.

## Acceptance criteria

XR-WP2 is done when:

1. the capability/throughput audit and solver ADR are committed, and the G2 record accepts their
   measured classification, batch-query, and explanation limits;
2. one declared complete profile/policy path runs end to end without Java or reparsing;
3. exhaustive small-instance utility and no-good validity tests pass;
4. all supported logical regressions and final replays agree;
5. every forced unknown/timeout/unsupported case follows the failure matrix;
6. exact status requires zero gap and safe statuses contain only verified assignments;
7. repair-disabled alignment outputs remain unchanged;
8. XR-E01 can consume immutable certificates and replay reports.

Any unavailable complete explanation/reasoner capability narrows the accepted profile or blocks G2;
it may not be papered over with an incomplete diagnostic path.

## Experiment handoff

- XR-E01 tests RQ1/H1 beyond fixtures.
- XR-E03 uses the exhaustive and exact solver seams to compare joint and sequential choices.
- XR-E08 injects governed failures and imperfect-core/reference conditions.

This WP does not support RQ2 claims because base states have not been compared against deletion on
frozen benchmarks.
