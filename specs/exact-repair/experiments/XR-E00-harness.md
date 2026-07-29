# XR-E00 — Harness, benchmark registry, and preregistration freeze

**Status:** planned<br>
**Depends on:** XR-WP1 schemas; completed Exact-OM dataset/run-artifact infrastructure<br>
**Bounds:** infrastructure for RQ1–RQ10; supports no substantive hypothesis by itself

## Purpose

Create the immutable input, split, baseline, query-basis, runner, replay, and analysis foundation
used by every later study. The output is a frozen protocol manifest, not a performance conclusion.

## Deliverables

1. `tools/repair/` experiment runner that consumes captured provisional alignment/evidence artifacts;
2. dataset registry entries with release/license/checksum/import-closure metadata;
3. mechanism-aware corruption generator and trace schema;
4. logical mechanism suite manifest;
5. relation-aware and positive-unlabeled evaluation adapters;
6. weighted semantic query-basis builder with provenance and train-only/pilot-only fitting rules;
7. baseline adapters and comparability reports;
8. split/leakage auditor;
9. repair certificate batch replay and eligibility aggregator;
10. immutable protocol/amendment manifest and report generator;
11. external-validity report comparing corruption operators and mechanism frequencies with an
    error taxonomy derived from real captured matcher outputs, published with the G4 freeze.

## Frozen inputs

Pin eligible releases/subsets from Bio-ML, Conference, Anatomy, Complex, DISO-OAEI, and captured
outputs from Exact-OM plus other matchers where permitted. Record why any proposal-named family is
unavailable or non-comparable. “Latest” is prohibited.

For every ontology task, capture source/target snapshots/import manifests, references, candidates,
Exact-OM output/evidence, matcher config/model, and provenance hashes. Repair baselines all consume
this capture.

## Query basis

Define query families before test repair:

- mapping relation consequences;
- selected source/target hierarchy entailments near mapped entities;
- benchmark competency questions where available;
- finite native subsumptions used for preservation but not necessarily policy;
- clean-integration consequences for synthetic cases.

Weights, deduplication, unsupported-query behavior, and reference-derived query construction are
fixed using training/pilot material. Record queries already true in the immutable core so repair
cannot take credit for them without an explicit relative definition.

## Baseline qualification

For every external repair system, record install/version, supported syntax/relation types, timeout,
conversion steps, and policy-verification coverage. Produce one of:

- comparable for the declared outcome;
- comparable only on a named subset/projection;
- descriptive only;
- unavailable with reason.

Lossy conversions cannot enter a semantic comparison without a marked sensitivity analysis.

## Pilot

Use only training/pilot pairs to fix:

- candidate/state budgets and guard depths used by confirmatory variants;
- wall/reasoner budgets and hardware tiers;
- query weights and primary RQ2 outcome choice;
- numerical minimum-effect/go-no-go thresholds;
- sample counts/seeds and feasible external baselines;
- statistical models, multiplicity families, and exclusions.

The zero false-safe requirement and fail-closed status rules are not tuneable.

## Harness invariants

- input capture is read-only and content-addressed;
- rerunning the same config/seed gives identical state inventory and selected assignment for
  deterministic arms;
- repair method is the only changed factor in paired contrasts;
- no test group appears in training/calibration/pilot manifests;
- every safe row has a certificate and replay report;
- aggregate eligibility counts reconcile with raw run statuses;
- unknown/timeout/fallback runs are never silently dropped;
- result tables regenerate from long-form artifacts.

## Acceptance criteria

XR-E00 is complete when:

1. two independent developers can materialize or verify every distributable frozen input from its
   manifest;
2. a smoke matrix runs at least two datasets, two baselines, and one corruption through capture,
   repair, replay, evaluation, and report generation;
3. deterministic reruns are byte-stable except declared timing fields;
4. intentional split overlap and artifact mutation are rejected;
5. baseline comparability and all unavailable data/system decisions are recorded;
6. the corruption external-validity report documents taxonomy coverage, distribution mismatches,
   and resulting limits on synthetic claims;
7. pilot outputs produce a signed/hashed G4 protocol freeze before confirmatory test access.

## Non-claims

Harness correctness does not show safety, semantic gain, generalization, speed, or expert benefit.
Smoke results must not appear in final effect estimates.
