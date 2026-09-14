# Native ontology optimization plan

**Implementation and applicable T1–T4 validation complete; see [IMPLEMENTATION.md](IMPLEMENTATION.md). G0 remains separate.**

Implement the opportunities in the [native stack audit](../experiments/NATIVE-STACK-AUDIT.md)
while preserving intended algorithm behavior. The deliverable is a smaller amount of work
for the same inputs, settings and semantic results, with Python serving as the interface to
native ontology-scale routines. The package specifications define acceptance; the implementation record identifies completed
checks and outstanding gates. No long scientific experiment is authorized by this suite.

## Authority and scope

The user's instructions control. For this optimization work, [CONTRACT.md](CONTRACT.md)
defines invariants; [VALIDATION.md](VALIDATION.md) defines acceptance; package files define
requirements. The audit is evidence, not an implementation contract. Existing upstream OWL,
projection-profile, completeness and lifecycle specifications remain authoritative for
semantics. A conflict or discovered baseline bug requires a separate recorded resolution;
performance does not justify changing the intended answer.

This is a follow-up to the existing shared-stack migration. It does not replace the
[0.2 migration release gates](../pyowl-core-0.2/README.md), change experiment hypotheses,
or reopen unrelated work. N00–N11 below are local identifiers, separate from historical
Exact WP-N. Native code belongs in the four upstream repositories, with small public-API
integration changes in Exact. No second OWL model, parser, projector or reasoner belongs in
Exact; no generic cache framework, orchestrator or benchmark service is required.

Read CONTRACT and VALIDATION first, then the relevant package spec:

- [CORE.md](CORE.md): native validation, ROOT selection, annotation and typed indexes.
- [PROJECTOR.md](PROJECTOR.md): annotation membership and native canonical edge handling.
- [REASONERS.md](REASONERS.md): pyELK queries, pyHermiT hierarchy, query state and updates.

## Work packages

Current statuses are recorded in IMPLEMENTATION.md. A dependency means evidence or an interface is required before
integration; independent code work may proceed against the agreed contract.

| ID | Work | Owner repository | Depends on | Required completion |
| --- | --- | --- | --- | --- |
| N00 | Freeze contracts, baseline identities and bounded fixtures | Exact + package test owners | — | Gates and evidence setup in VALIDATION |
| N01 | One native class-membership index per projection selection | projector | N00 | Exact parity; no full-node scan per annotation |
| N02 | Native validation and opt-in native execution requirement | core + all consumers | N00 | Native admission, safe ownership and actual path evidence |
| N03 | Proven ROOT/CLOSURE reuse or native selection mapping | core + projector | N02 interface | No duplicate table on proven equivalent scopes; exact general behavior |
| N04 | Native filtered annotations and compact typed indexes | core + Exact | N00, N02 interface | Selected rows only cross into Python; exact exclusions/hierarchy |
| N05 | Task-aware pyELK queries and shared, bounded query state | pyELK | N00; N02 for strict conformance | Named-query reuse, stage demand, bounded complex state |
| N06 | Indexed pyHermiT hierarchy adjacency | pyHermiT | N00 | Identical graph/query results; degree-proportional neighbor access |
| N07 | Shared permanent pyHermiT program and query-local state | pyHermiT | N00; N02 for strict conformance | Query parity/isolation; no full base per witness |
| N08 | Eligible encoded incremental updates | pyHermiT | N07 | Conditional: explicit eligibility and native rebuild otherwise |
| N09 | Native canonical sorting/deduplication and bounded output | projector | N01; N02/N03 integration | Required for canonical-profile strict conformance; advanced output APIs conditional |
| N10 | Remaining native canonical/compiler pass acceleration | core/consumers | N01–N04 profile | Conditional: demonstrated residual cost, exact canonical contract |
| N11 | Compatible packages and Exact integration | Exact + package maintainers | Applicable N01–N09; N00 gates | Installed-wheel parity, measured preprocessing and accurate readiness |

N08 and N10 require an applicability/performance decision with retained evidence. Defer them
explicitly when their triggers are absent or cost is immaterial; do not build speculative
infrastructure or call deferred work validated. N09's minimal native canonical implementation
is required for Exact's canonical output profile even if richer packed-output APIs are deferred.

## Parallel implementation waves

Use at most three implementation agents plus an integration/review owner on this machine.
Assign one writer per repository or mutually agreed file set; consumer changes sharing an API
must wait for its contract rather than invent independent encodings.

| Wave | Worker A | Worker B | Worker C | Integration/review owner |
| --- | --- | --- | --- | --- |
| 0 | Projector fixtures/counters | Core contract and ownership cases | Reasoner query/isolation fixtures | N00 baseline inventory and API agreement |
| 1 | N01 | N02 core portion | N06, then N05 in its separate repo | Review localized changes; package test evidence |
| 2 | N03 projector portion + N09 | N03 core portion, then N04 | N05 completion, then N07 | Consumer N02 conformance and cross-package fixtures |
| 3 | N11 projector/Exact integration | N10 only if gate triggers; otherwise review | N08 only if applicable; otherwise review | Candidate wheels, one bounded real-input gate, handoff |

N07 needs its own implementation review; it must not be squeezed into an adjacency patch.
Reassign completed workers to concrete outstanding tests/reviews, not duplicate audits.
Do not keep agents or monitoring loops alive merely to appear busy. Build and benchmark
resource limits in VALIDATION apply across workers, not independently per agent.

## N11 integration requirements

1. Build candidate wheels in isolated environments at recorded commits; test installed
   artifacts, not a mixture of source checkout imports and an old native extension. Keep
   the current Exact environment and saved baseline artifacts usable for comparison.
2. Publish a compatibility matrix of tested core/consumer package versions and encoded,
   model and validation-capability versions. Prefer additive capabilities over a needless
   schema break. Select actual release numbers using existing repository conventions.
3. Exact requests the strict pipeline only when every used path supports it; it must fail
   before expensive processing on an incompatible combination. Partial optimization commits
   can be evaluated without being represented as full native-pipeline compliance.
4. Replace the existing private projector bridge only after equivalent public guarantees
   and installed-wheel tests pass. Keep exclusion policy, source populations, ontology/import
   content and matching configuration fixed. Use the new core annotation/index API directly.
5. Keep scientific and execution revisions separate. Bind artifacts to relevant content,
   semantics and implementation identity using the existing recovery contract. Preserve old
   measurements and charges; never relabel stale timings as measurements of optimized code.
6. Run the applicable validation gates, record results and update readiness. Passing ontology
   preprocessing is not passing G0, and does not launch the long experiment campaign.

## Commits and definition of done

Each implementation change gets a scoped commit in its owning repository, with problem,
behavior-preservation argument, relevant tests and measured evidence. Separate localized
algorithm changes, shared interface changes and consumer adoption so regressions are
bisectable. Keep commit hashes in the integration record; publishing packages/defaults is
separate from developing and testing local candidates. Avoid unrelated refactors and
permanent runtime shadow implementations.

A work package is done only when its requirements, relevant native-path checks and output
parity pass, the intended repeated work is demonstrably removed, and the evidence is saved.
A preprocessing milestone requires N01, applicable N02, N03, N04, N09 and N11 evidence.
Full-stack completion additionally requires N05–N07 and reasoner conformance, with explicit
N08/N10 dispositions. Neither milestone may be inferred from a backend label or a timer
alone. No numerical speedup or completion date is promised before measurement.
