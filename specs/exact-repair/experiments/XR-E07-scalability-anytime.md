# XR-E07 — Exactness frontier, anytime utility, and search acceleration

**Status:** planned<br>
**Depends on:** XR-E00; XR-WP5; XR-E01 safety gate<br>
**Question/hypothesis:** RQ8/H7<br>
**Co-primary outcomes:** time/probability to zero gap and verified-incumbent regret over time

## Authorized claim

Characterize—not overgeneralize—the exact and anytime operating frontier for frozen software,
hardware, profile, policy, and candidate budgets. Test whether conflict-component structure/state
count explain exact runtime better than raw ontology size.

## Systems and ablations

1. global reference rebuild mode;
2. incremental solver with rebuild reasoner;
3. plus reasoner/query/explanation caches;
4. plus component warm starts and dynamic merging;
5. deterministic oracle/conflict ordering;
6. learned search prioritization, if implemented;
7. exact mode;
8. anytime budgets at preregistered wall/reasoner-call cutoffs.

All systems use identical frozen states, utilities, policies, and complete final verification. Cache
arms distinguish cold and warm runs; cold is primary.

## Factorial instance matrix

Sample/construct instances across:

- ontology axiom/import size and OWL profile;
- revision-object count and states per object;
- discovered conflict count, size, overlap, component count, and largest component;
- guard depth/constructor count and replacement budget;
- explanation density/multiplicity;
- provisional alignment error/correlation rate.

Include full public/captured outputs and controlled synthetic scale series. Exact optimum labels are
available for tractable prefixes/components to quantify anytime regret.

## Measurements

- end-to-end and phase wall/CPU time;
- time to first verified non-all-off incumbent;
- time to each gap threshold and zero gap;
- incumbent external utility/regret and solver objective over time;
- reasoner/classification/explanation/master calls and time;
- conflicts, duplicate/subsumed conflicts, sizes, and shrink benefit;
- peak RSS and artifact/certificate size;
- cache hit rates and invalidations;
- status/fallback/unknown rates.

Time-series snapshots must not force extra reasoning that changes the algorithm materially; use
solver/orchestrator events already produced or a frozen low-overhead trace policy.

## H7 model comparison

Before test execution, define competing predictive models:

- raw ontology size variables only;
- revision/state counts only;
- conflict-component topology only;
- combined structural model.

Compare held-out predictive fit using preregistered criteria and account for censored timeouts.
Avoid interpreting correlation as causal. H7 is supported only if topology/state predictors add the
G4 minimum predictive value over ontology size.

## Correctness guard

Every optimized/cached/decomposed result on the tractable overlap must match reference mode in
safety and exact objective/assignment set (allowing recorded equal optima). Any mismatch invalidates
the affected performance arm until fixed. Whole-KB final verification remains included in reported
time.

## Decision rule

Report a frontier surface and budget-quality curves. Anytime is operationally acceptable only if it
returns a verified incumbent within the G4 latency budget, exposes a valid/null gap honestly, and
meets the frozen semantic-regret target on the selected deployment matrix. Exact mode claims stop
at observed zero-gap coverage.

## Non-claims

- No asymptotic complexity improvement is inferred from empirical slopes.
- Hardware-specific throughput is not a universal runtime promise.
- Learned ordering may improve time but cannot support semantic-quality or safety claims by itself.
- Excluding loading/final verification from end-to-end time is prohibited.
