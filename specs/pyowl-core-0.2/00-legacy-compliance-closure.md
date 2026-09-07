# 00 — Legacy compliance closure

## 1. Purpose and authority

This document closes the remaining engineering-compliance work inherited from the completed
Exact-OM overhaul before the pyOWL 0.2 migration is released. It is self-contained: an
implementation agent does not need to reopen the older WP-A–WP-N specifications to determine
release scope.

This document covers the production system and its release engineering. It excludes all work
under `specs/experiments/` and `specs/exact-repair/`.

The previous specifications remain historical records. Their functional and behavioral
requirements are preserved unless §4 explicitly supersedes a release-evidence requirement. A
requirement is not waived merely because its old test is missing or broken.

## 2. Baseline conclusion

At the start of this migration:

- the production features described by WP-A through WP-L are implemented;
- the shared OWL ownership migration described by WP-M M0–M4 is implemented;
- the production-path portion of WP-N is implemented but must be rebound to core schema 2;
- no missing matching, evaluation, format, track, configuration, inspection, or run-artifact
  feature was identified in the audit; and
- the remaining compliance work consists of code-quality failures, non-hermetic tests,
  incomplete distribution proof, schema-2 compatibility changes, and final release evidence.

Implementation must re-run the checks below against the current worktree. This conclusion is not
permission to ignore a newly discovered regression.

## 3. Preserved feature-compliance matrix

Every row is release-blocking. The “closure evidence” may be an existing test, a corrected test,
or a small new regression test. Do not restore deleted legacy implementations merely to satisfy a
historical test.

| Area | Behavior that must remain true | Minimum closure evidence |
|---|---|---|
| Foundations and packaging | Python 3.10–3.12 install; CPU-only base; CLI entry points; clean pytest collection; declared extras remain isolated | quality matrix plus wheel/sdist clean-install checks |
| Java-free ontology backend | no JVM/JPype/mOWL/OWLAPI/DeepOnto/private parser runtime; one ontology load and one retained core owner; fixture semantic parity | forbidden-import scan, ontology parity tests, ownership instrumentation |
| Timing ledger | concurrent or resumed runs do not overwrite each other's timing state; stage totals and estimates remain attributable to the correct configuration/run | timing-ledger unit and interruption/resume tests |
| Module and import architecture | public delivery/core boundaries and import-linter contracts hold; no duplicate ontology engine returns under `exact/` | import-linter plus duplicate/private implementation scans |
| Evaluation | built-in evaluation and optional Bio-ML evaluation install independently and produce the documented global/local metrics for supported entity kinds | evaluator unit/integration tests and optional-extra smoke |
| Entity kinds | class, object-property, data-property, annotation-property, and named-individual flows preserve defaults and typed output behavior | entity-kind candidate, mapping, filtering, and evaluation tests |
| Formats and KG sources | supported OWL/RDF/CSV-KG inputs and TSV/OAEI-RDF/typed-TSV/JSON outputs round-trip with deterministic ordering | source conformance and format round-trip tests |
| Documentation | strict documentation build, generated API/config/CLI references, and internal links are valid | `mkdocs build --strict` and linkchecker |
| Dataset and track retrieval | declarative descriptors, integrity hashes, offline/cache behavior, and licensed-data diagnostics remain deterministic and actionable | hermetic provider/materialization tests; no network release gate |
| Config schema v2 | v2 validation and v1 migration preserve old defaults for one compatibility release; resolved config fingerprints remain stable | config migration/golden tests and CLI smoke |
| Exact Inspect | base install does not acquire service dependencies; `viz` installs the service/static frontend; legacy command shim remains actionable | isolated base/viz wheel smokes and frontend-resolution tests |
| Run artifacts | layout-v2 manifests, explanation storage, interruption recovery, and legacy-v1 read compatibility remain intact | artifact acceptance, interrupted-run, export, and legacy-reader tests |
| Shared OWL consumers | projector and optional reasoners receive the retained public core owner rather than a path; workers use verified core wire/mmap without parsing OWL | focused schema-2 handoff tests in `03-verification.md` |

### Clarifications

- The old module-size criterion applies to files under `exact/`. A large module under
  `exact_inspect/` is a maintainability concern but not, by itself, a failure of that criterion.
- Historical output artifacts remain readable where the compatibility policy promises read
  support. Ontology-derived schema-1 caches are different: they must be rejected and rebuilt.
- Existing behavior-preserving tolerances remain valid. The migration must not loosen semantic
  tolerances to hide a core 0.2 difference.

## 4. Explicitly superseded requirements

Only the following old release-evidence requirements are replaced for Exact-OM `2.1.0`:

1. the fixture benchmark's 25% timing comparison is informational rather than release-blocking;
2. the old Exact 2.0 versus shared-stack 25% wall-time comparison is not a release gate;
3. a pinned-runner comparative RSS baseline is not required;
4. Conference, GO, multiple Bio-ML pairs, and the largest licensed workflow are replaced by the
   single content-addressed NCIT–DOID correctness run in `03-verification.md`; and
5. the exhaustive native-consumer performance cross-product is deferred to experiments or
   upstream performance work.

These changes waive only comparative performance evidence. They do not waive:

- deterministic semantic results;
- exactly one load per ontology;
- retained in-process identity;
- zero worker-side OWL parses;
- absence of a second ontology-sized Exact representation;
- Python/native functional parity on supported production paths;
- safe cache invalidation;
- supported Python/package installation; or
- Java-free and private-import boundaries.

Exact-OM `2.1.0` must keep `performance_claim: false`. Any later performance claim needs separate
evidence and does not retroactively block this compatibility release.

## 5. Known open compliance defects

The audit found the following concrete defects. Each must be reproduced against the current
worktree and closed if still present.

### 5.1 Formatting

`isort --check-only` reported ordering differences in:

- `exact/impl/datasets/base.py`;
- `exact/impl/datasets/contextgraph.py`; and
- `exact/core/actions/alignment.py`.

Apply mechanical import sorting without unrelated rewrites.

### 5.2 Typing

`mypy` reported:

- missing PyYAML type information in `exact/tracks/descriptor.py`,
  `exact/io/sources/csv_kg.py`, `exact_inspect/bundles.py`, and `exact_inspect/cli.py`; and
- an `Any` return from `_public_positive_int` in `exact/ontology/reasoning.py`.

Add `types-PyYAML` to the development/prebuild dependency group and lock it. After validating the
integer value, use typed narrowing or `cast(int, value)` rather than weakening mypy.

### 5.3 Pytest collection

The local `benchmarks/` directory can be shadowed by an installed top-level `benchmarks` package.
Make the repository import deterministic, for example by adding `benchmarks/__init__.py`, and
prove full test collection succeeds from a clean environment.

### 5.4 Exact Inspect test isolation

Two tests assume the packaged frontend is absent, while an existing
`explanations_visualizer/out` directory correctly changes production frontend resolution. Isolate
the tests by controlling or mocking frontend resolution. Do not change the production precedence
merely to make the test environment pass.

### 5.5 Reasoner cleanup test

The drift-cleanup test currently mocks every non-HermiT distribution as `0.1.0.dev0`, allowing a
core mismatch to fail before HermiT construction and cleanup. Mock only the intended pyHermiT
mismatch, delegate actual metadata versions for the rest of the stack, and assert cleanup after
the intended failure point.

### 5.6 Distribution coverage

The current packaging workflow builds on Python 3.12 and clean-installs primarily the wheel. Close
the declared support claim by distributing wheel/sdist installation and base/viz/reasoning smoke
coverage across Python 3.10–3.12 as specified in `04-release-checklist.md`.

### 5.7 Core 0.2 migration defects

The old dependency constraints and lock select the 0.1 stack. The projection adapter combines a
dynamic encoded schema with the v1 descriptor digest, and schema-1 cache/provenance expectations
remain in tests and manifests. Close these defects exactly as specified in
`01-version-contract.md` and `02-implementation-plan.md`.

## 6. Required closure procedure

1. Run the normal CI commands before dependency migration and record genuine baseline failures.
2. Fix the defects in §5 without weakening functional assertions.
3. Run the complete hermetic test selection successfully on Python 3.10–3.12.
4. Complete the core 0.2 migration and focused tests in this suite.
5. Run the distribution, documentation, and NCIT–DOID gates.
6. Review the matrix in §3 row by row and link each row to passing test/job evidence in the release
   handoff.
7. Search the final tree for stale 0.1 constraints, schema-1 runtime expectations, Java/private
   imports, and old performance claims.

## 7. Compliance certificate

The release handoff must include a small `release/evidence/legacy-compliance-closure.json` with:

- Exact commit and version;
- Python versions tested;
- lock and distribution hashes;
- one pass/fail entry for every matrix row in §3;
- links or stable job/test identifiers for the supporting evidence;
- confirmation that every applicable defect in §5 is closed;
- the list of superseded requirements from §4;
- `experiments_in_scope: false`;
- `exact_repair_in_scope: false`; and
- `performance_claim: false`.

The certificate summarizes evidence; it must not duplicate test logs, contain credentials or
machine-local paths, or claim that excluded experiments were executed.

Compliance is complete when every preserved matrix row passes, every still-applicable audit defect
is closed, the only waived requirements are those explicitly listed in §4, and all other files in
this migration suite satisfy their definitions of done.
