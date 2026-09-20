# Native stack implementation record

Exact-OM now consumes the [published native stack](../native-stack.md): `pyowl-core`,
`pyowl2vec-star-projector`, `pyelk-reasoner` and `pyhermit` are locked to `0.2.1`.
Use the published distributions and current public capability checks for integration.

## Scope and recorded gates

The original implementation and applicable T1–T4 validation completed on 14 September 2026.
The following dispositions describe that implementation milestone; its full evidence is
preserved in the [historical implementation record](../../docs/archive/native-stack/IMPLEMENTATION.md).
Publication does not rerun those checks or change their artifact identities.

| Work | State | Implemented mechanism / remaining evidence |
| --- | --- | --- |
| N00 | validated | Baseline commits and immutable NCIT/DOID artifacts retained |
| N01 | validated | Native exact-IRI membership index, one per selected projection |
| N02 | validated | Core receipts, consumer preflight and native metadata/results; final coherent installed-wheel checks pass |
| N03 | validated | Proven one-document ROOT reuse; general scope selection remains explicit |
| N04 | validated | Native annotation postings, class/property features, selected typed rows and domain/range indexes; installed Exact parity |
| N05 | validated | ELK task-aware stages, shared query base, bounded LRU and native taxonomy adjacency |
| N06 | validated | HermiT native parent/child adjacency with degree-proportional enumeration |
| N07 | validated | Native assertion deltas share immutable rules, joins, role/datatype/blocking state; isolated bounded batches and rebuild parity |
| N08 | deferred_with_reason | Exact's current static hierarchy adapters make no recurring committed update calls |
| N09 | validated | Native canonical sort/dedup/spill/merge with bounded working storage |
| N10 | deferred_with_reason | Residual loading/projection costs measured; no specific additional canonical/compiler pass selected |
| N11 | validated | Final installed integration and NCIT–DOID T4 pass, including 128 exact ordered feature rows |

The package specifications and [shared contract](CONTRACT.md) continue to define semantic,
ownership, capability and resource requirements. The archive retains the tested compatibility
boundaries, deferred-work rationale, native-query constraints and recovery accounting.

## Completed T4 result

The recorded T4 gate passed exact comparison of 64 ordered feature rows for NCIT and 64 for
DOID: 128 rows with zero mismatches. Input/import identities, effective axiom counts,
signatures, canonical edges, exclusions, labels and entity selections also matched the
saved structural baseline. The [dated result and measurements](../../docs/archive/native-stack/IMPLEMENTATION.md#completed-t4-result)
remain evidence for the artifacts tested at that time, not a benchmark of the published release.

T4 preprocessing does not pass G0 or authorize the long scientific campaign. Current experiment
readiness belongs to [PREPARATION-STATUS.md](../experiments/PREPARATION-STATUS.md).
