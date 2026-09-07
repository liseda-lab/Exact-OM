# 04 — Release checklist

This checklist is the only release handoff for the pyOWL 0.2 migration. Complete it in order and
record the exact commit and artifact hashes used.

## 1. Source and dependency state

- [ ] `pyproject.toml` uses the ranges from `01-version-contract.md`.
- [ ] `poetry.lock` contains only published compatible 0.2 stack releases.
- [ ] Exact uses `EncodedStructuralView.DESCRIPTOR_SHA256`, with no schema-1 descriptor constant.
- [ ] No Exact runtime code imports core/projector/reasoner implementation modules.
- [ ] No Java, JPype, mOWL, DeepOnto, OWLAPI, ROBOT, or private parser runtime has returned.
- [ ] Schema-1 ontology-derived caches are rejected and rebuilt.
- [ ] Exact-owned cache and backend versions are incremented.

## 2. Blocking automated checks

- [ ] Formatting, linting, typing, import boundaries, and docstring checks pass.
- [ ] Hermetic tests pass on Python 3.10, 3.11, and 3.12.
- [ ] Focused core 0.2 tests from `03-verification.md` pass.
- [ ] Strict documentation build and internal link check pass.
- [ ] The historical timing-reference benchmark is removed from blocking CI or made
      informational.

Do not mark a failing quality or compatibility test as non-blocking. Only performance timing
comparisons are excluded from release acceptance.

## 3. Distribution checks

- [ ] Build exactly one wheel and one sdist from the release candidate.
- [ ] The wheel and sdist contain the same `release/core-compatibility.json`.
- [ ] The wheel contains the Exact Inspect frontend assets.
- [ ] Base metadata requires core/projector and does not require visualization/reasoner packages.
- [ ] Base clean-install works on the declared Python 3.10–3.12 matrix.
- [ ] Sdist clean-install works without Java or Cargo and retains pure-Python projection.
- [ ] A supported platform wheel selects and exercises native projection.
- [ ] `[viz]` clean-install exposes `exact-inspect` and `exact-study-viz`.
- [ ] `[reasoning]` clean-install exercises ELK and HermiT without a JVM.
- [ ] `pip check` passes after each installation state.
- [ ] The generated SBOM includes core/projector and excludes Java/private parser runtimes from
      base.

The matrix may split these checks across jobs. It does not need to rebuild every artifact for
every Python/extras combination.

## 4. Single external-data gate

- [ ] NCIT and DOID input sizes/digests match `03-verification.md`.
- [ ] The required encoded-native command completes.
- [ ] Every invariant and semantic assertion in the NCIT–DOID acceptance section passes.
- [ ] `release/evidence/pyowl-core-0.2-ncit-doid.json` contains package/schema/backend identities,
      input/output digests, ownership/counter assertions, cache state, and diagnostic timings.
- [ ] The evidence contains no ontology bytes, credentials, or machine-local paths.

No other external-data or performance run is required.

## 5. Documentation and compatibility record

- [ ] Every preserved feature row and applicable defect in
      `00-legacy-compliance-closure.md` is closed.
- [ ] `release/evidence/legacy-compliance-closure.json` records the required closure certificate.
- [ ] Installation and troubleshooting docs name the 0.2 dependency ranges.
- [ ] Migration docs explain schema-1 cache invalidation and anonymous-individual identity changes.
- [ ] Architecture/API docs describe one retained core owner and public consumer handoff.
- [ ] Changelog describes the compatibility migration without making a performance claim.
- [ ] `release/core-compatibility.json` names the exact tested package versions and core schema-2
      descriptor.
- [ ] `performance_claim` is `false`.
- [ ] Every documented command is checked on Python 3.10 and 3.12.

## 6. Version and final artifact

- [ ] Set Exact-OM version to `2.1.0` only after sections 1–5 pass.
- [ ] Rebuild wheel/sdist from the final candidate.
- [ ] Repeat base and relevant extras smoke tests against those exact files.
- [ ] Record Exact commit, tag, wheel SHA-256, sdist SHA-256, lock hash, and NCIT–DOID evidence hash.
- [ ] Confirm the tag workflow uploads the same verified artifacts rather than rebuilding from a
      different source state.

## 7. Stop conditions

Do not release when any of these occurs:

- a schema-1 cache is accepted or converted;
- a consumer reparses from an OWL path;
- the core descriptor is hard-coded or mismatched;
- projector/native outputs have an unexplained semantic difference;
- NCIT–DOID produces unexplained residual edges;
- Exact creates a second ontology-sized structural representation;
- a base installation requires Java, a reasoner, or visualization dependencies;
- a supported Python version fails normal CI or clean install; or
- final package versions differ from the compatibility/evidence record.

A timing regression by itself is not a stop condition for this release. Record it as a follow-up
issue and leave performance claims disabled.
