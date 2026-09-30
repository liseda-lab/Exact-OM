# XR-E00 — Harness, identities and accounting

**Target:** XR-2.1, 30 September 2026. **Question:** Prerequisite. Requirements for new runs; archived XR-2 results remain separate.

## Data

All semantic/regression fixtures, preserved pilot artefacts and small captured real inputs. Real input availability is explicit.

## Comparisons

Materialisation and v3 loaders versus independent tiny fixtures; prepared versus direct candidate paths; identical baseline reports and pair selection across arms. Compare clean execution with injected startup, cache, process and late-query failures.

## Measurements

Input/split/schema hashes; requested and effective language; baseline and per-query costs; unique case results versus attempts/reuse; all failure statuses and deadline/cleanup behaviour.

## Acceptance and interpretation

Pass MIG-01–03, MIG-11, MIG-13–15 and schema migration before attributing experimental gains. No second parser, secrets in artefacts, silent legacy checkpoint loading or partial positive acceptance. The reference tests are necessary narrow checks, not enough to qualify live backends.

Shared splits, resources, failure accounting and statistics follow [02](../02-experimental-protocol.md). Dependency gates are in [08](../08-experiment-matrix-and-handoff.md); code changes and tests are in [12](../12-implementation-migration.md). Resolve every numerical setting in a new [run manifest](../protocol/README.md) before execution.
