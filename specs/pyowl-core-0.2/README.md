# Exact-OM pyOWL 0.2 migration suite

**Target Exact release:** `2.1.0`
**Required native releases:** published `0.2.1`, supported through `>=0.2.1,<0.3`
**Dependency status:** adopted in Exact-OM; see the [published stack contract](../native-stack.md).
Migration and release acceptance remain governed by the gates below.

## Authority and reading order

This directory defines the migration from Exact-OM `2.0.0` to the published pyOWL 0.2
API/model contracts and Exact-OM `2.1.0`. The current native package baseline is `0.2.1`.

Implementation agents must read only this suite for migration scope and acceptance:

1. `README.md` — scope and decisions;
2. `00-legacy-compliance-closure.md` — preserved prior features and known audit defects;
3. `01-version-contract.md` — dependency and persisted-data contract;
4. `02-implementation-plan.md` — ordered code changes;
5. `03-verification.md` — focused tests and the single NCIT–DOID gate; and
6. `04-release-checklist.md` — final release procedure.

The older `WP-M-shared-owl-stack.md`, `WP-N-native-view-handoff.md`, and `03-performance.md`
remain unchanged historical design records. They are not prerequisites for this migration, and
their former scale/performance release gates do not apply. If an old specification conflicts with
this suite, this suite controls the Exact-OM `2.1.0` migration.

Exact-OM `2.1.0` and the native dependency release `0.2.1` have independent version numbers.
Core API `(0, 2)` and model/encoded schema `2` remain the compatibility contract.

## Goal

Release a behaviorally correct, package-tested Exact-OM version that:

- installs only published compatible pyOWL 0.2 packages;
- uses the public core 0.2 API, model-schema, wire, and encoded-view contracts;
- safely invalidates schema-1 ontology-derived caches;
- keeps one ontology owner and never reparses paths for projector/reasoner consumers;
- works in base, visualization, reasoning, and Bio-ML evaluation installations;
- remains Java-free; and
- makes no new performance claim.

This is a compatibility and release-engineering migration. It must not change matching methods,
configuration defaults, scoring, evaluation definitions, repair behavior, or experimental
features.

## Deliberately simple release gate

Normal hermetic CI, packaging, and documentation checks remain mandatory. There is exactly one
external-data acceptance run: the frozen OAEI Bio-ML NCIT–DOID pair.

NCIT–DOID is selected instead of Conference because it already has content-addressed inputs and
classified projection semantics in this repository, and it exercises the loader, structural
model, encoded native handoff, projection, cache, and ownership invariants at realistic scale.
It is used as a correctness and bounded-execution smoke, not as a benchmark:

- no comparison with the old Exact 2.0 wall time;
- no 25% regression threshold;
- no pinned-runner RSS threshold;
- no GO, additional Bio-ML pair, licensed workflow, CUDA, or hosted-LLM run; and
- no full benchmark cross-product.

Wall time and RSS may be recorded for diagnosis. They do not block release unless the run fails
to finish within the ordinary job timeout, exhausts the runner, reparses an ontology, or creates a
second ontology-sized Exact representation.

## Work packages

| ID | Deliverable | Acceptance owner |
|---|---|---|
| V0 | Prior feature compliance and audit-defect closure | `00-legacy-compliance-closure.md` |
| V1 | Released dependency and schema contract | `01-version-contract.md` |
| V2 | Descriptor negotiation and consumer handoff | `02-implementation-plan.md` |
| V3 | Cache invalidation and provenance | `02-implementation-plan.md` |
| V4 | Focused schema-2 tests | `03-verification.md` |
| V5 | One NCIT–DOID acceptance record | `03-verification.md` |
| V6 | Wheel/sdist, docs, manifest, and `2.1.0` release | `04-release-checklist.md` |

## Definition of done

The migration is complete only when:

1. every preserved feature row and applicable audit defect in
   `00-legacy-compliance-closure.md` is closed;
2. the lock resolves to published compatible packages in `>=0.2.1,<0.3`;
3. all normal CI and focused schema-2 tests pass on Python 3.10–3.12;
4. base, `viz`, and `reasoning` distribution smoke tests pass without Java;
5. schema-1 ontology caches are rejected and rebuilt rather than converted;
6. the single NCIT–DOID acceptance record passes every invariant in `03-verification.md`;
7. documentation, changelog, SBOM, and `release/core-compatibility.json` describe the tested
   package set; and
8. Exact-OM is versioned `2.1.0` with `performance_claim` remaining false.

## Out of scope

- Exact Repair;
- all work under `specs/experiments/`;
- new matching or evaluation methodology;
- performance optimization or performance claims;
- rewriting external parser/projector/reasoner implementations inside Exact;
- decoding or persisting core encoded structural buffers; and
- supporting schema-1 ontology caches after the upgrade.
