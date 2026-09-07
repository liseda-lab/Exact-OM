# 02 — Ordered implementation plan

Follow this order. Each phase must leave the repository testable; do not combine the migration
with matching-methodology or performance work.

## Phase 0 — establish a clean engineering baseline

Run the normal non-data CI suite before changing dependencies. Fix genuine formatting, typing,
collection, and hermetic-test failures; do not weaken those gates or update semantic goldens to
hide failures.

At the time this suite was authored, the audit had identified these likely baseline cleanups:

- isort ordering in `exact/impl/datasets/base.py`,
  `exact/impl/datasets/contextgraph.py`, and `exact/core/actions/alignment.py`;
- missing `types-PyYAML` typing support;
- typed narrowing of `_public_positive_int` in `exact/ontology/reasoning.py`;
- local `benchmarks` package shadowing during pytest collection;
- Exact Inspect tests depending on an existing built frontend; and
- a reasoner drift-cleanup test that mocks the core version before reaching the intended HermiT
  mismatch.

Reconfirm each issue against the current worktree before editing. These are starting observations,
not permission to change working behavior.

## Phase 1 — update and lock the released stack

1. Apply the dependency ranges from `01-version-contract.md` to `pyproject.toml`.
2. Regenerate `poetry.lock` without unrelated upgrades where the resolver permits.
3. Verify that the selected packages are published releases and all resolve to the same core 0.2
   minor line.
4. Install and import the base, `viz`, `reasoning`, and Bio-ML evaluation dependency sets.
5. Record the exact resolved versions for the compatibility manifest.

Do not edit Exact's version yet.

## Phase 2 — correct the encoded-view contract

In `exact/ontology/projection.py`:

1. continue reading the schema name/version from public core attributes;
2. replace the schema-1-specific descriptor constant with
   `pyowl_core.EncodedStructuralView.DESCRIPTOR_SHA256`;
3. validate that the descriptor has the public expected type/length before comparison;
4. compare it with projector/reasoner public diagnostics; and
5. preserve the retained source/provider identity passed to the consumer.

Do not request an encoded view merely to validate compatibility. Do not import native modules,
decode buffers, stage structural rows, or add a path-based fallback.

Review `exact/ontology/reasoning.py`, `exact/ontology/provenance.py`, and
`exact/ontology/_reasoner_worker.py` for schema-1 assumptions. Change only core model/wire/encoded
contract expectations; leave independent projector/reasoner compiler schemas unchanged.

## Phase 3 — invalidate caches and update provenance

1. Increment Exact's ontology backend and ontology-derived cache versions in
   `exact/impl/datasets/base.py` and any other owning cache module.
2. Add the core 0.2 model/encoded contract to cache keys where it is not already included.
3. Ensure an old cache produces one actionable “rebuild required” result rather than conversion,
   fallback unpickling, or a secondary parse path.
4. Preserve read-only access to completed historical run artifacts.
5. Update run provenance and `release/core-compatibility.json` to the contract in
   `01-version-contract.md`.

The compatibility-manifest document schema may remain version 1 if its JSON shape is unchanged.
The values inside it must describe core model/encoded schema 2 and published package revisions.

## Phase 4 — rebind and add focused tests

Update schema-1 expectations in the existing ontology tests. Prefer public constants over copying
numeric values into many tests. The likely affected modules are:

- `tests/ontology_stack_provenance_test.py`;
- `tests/reasoner_adapters_test.py`;
- `tests/shared_owl_stack_test.py`;
- `tests/owl_stack_scale_test.py`;
- `tests/ontology_compatibility_test.py`; and
- `tests/ontology_public_handoff_test.py`.

Add the focused cases enumerated in `03-verification.md`. Keep fixtures small and committed except
for the one NCIT–DOID release run.

## Phase 5 — simplify CI and package verification

The compatibility release must not be blocked by the repository's historical timing reference.
Change the fixture benchmark job to one of these non-blocking forms:

- run `benchmarks/bench.py` without `--check-reference` and upload its JSON; or
- move it to a manually triggered/informational workflow.

Do not delete the benchmark runner or historical evidence. Experiments may continue to use them.

Keep normal quality/unit jobs blocking. Expand distribution smoke coverage so one built wheel and
one sdist are clean-installed across the declared Python 3.10–3.12 matrix, divided across CI jobs
as needed. Explicitly exercise:

- base installation and CLI entry points;
- absence of Java and optional reasoner/visualization dependencies from base;
- pure-Python projection;
- published native projection on a supported platform;
- `viz` installation and CLI entry points; and
- `reasoning` installation with ELK and HermiT, without a JVM.

## Phase 6 — documentation and release metadata

Update installation, migration, troubleshooting, architecture, dependency extras, API examples,
benchmark policy, changelog, and SBOM expectations. Remove stale 0.1 constraints and any claim
that Exact owns an OWL parser/projector implementation.

Run the single NCIT–DOID acceptance in `03-verification.md`. Only after it and all release checks
pass:

1. write the final compatibility/evidence records;
2. set Exact's version to `2.1.0`;
3. rebuild the final wheel and sdist from the tagged candidate; and
4. repeat the clean-install smoke against those exact artifacts.

## Rollback

Rollback restores the prior dependency constraints and lock together. Delete schema-2 caches and
rebuild under the restored line. Never convert caches in either direction. Completed immutable run
artifacts remain historical records.
