# Exact-Repair implementation validation

Validation date: 19 September 2026. These are component-conformance and small integration results, not benchmark or model-quality claims. The initial implementation was `37db8f5`; `e033026` adds direct grammar, retrieval and supervised runtime paths; the following training/study commit completes the protocol tools described below.

## Local optimized ontology stack

Repair validation used isolated copies of the latest local source checkouts and their built native extensions, without modifying the checkouts or the matching model's environment:

| Component | Local revision |
| --- | --- |
| pyowl-core | `5fd93c8839318f0ccc1afff4d3ec537509c72e87` |
| pyowl2vec-star-projector | `a0676012c5f6437c304b4ed04bc2b41b6a083da1` |
| pyHermiT | `bbf31c26ce7ae03bd092f04bcb8955466b1fe3e0` |
| pyELK | `bce95f8552ed8396315c9a9e4ddce184ad95596f` |

The temporary overlay is `/tmp/exact-repair-local-stack`; its `manifest.json` records source/native hashes. Study runtime manifests also hash the ontology implementations actually imported. Python 3.12, PySAT 1.8.dev24, PySDD 1.0.6, torch 2.7.0 and torch-geometric 2.6.1 supplied the other components. No package version was substituted for capability checks.

## Executed checks

| Check | Result |
| --- | --- |
| `PYTHONPATH=/tmp/exact-repair-local-stack .venv/bin/python -m pytest tests/repair_*_test.py -q` | 166 passed |
| Finite XR-2 specification reference | 37 passed |
| Pilot and smoke protocol validation | Passed; static only |
| Targeted mypy | 28 modules passed |
| Black, isort, flake8 | Passed |
| Repository import contracts | 5 kept, 0 broken |

## Conformance scope

- Every action family, mirrored directions, canonical expressions, activated non-vacuity, duplicate occurrences, unsupported inputs, pending alternatives, signed/pairwise MaxSAT coefficients and safety/optimality replay.
- Direct typed-slot grammar versus independent finite enumeration, exact mixture normalizers/posteriors, summed alias probabilities and gradients, side restrictions and canonical emitted-expression bounds.
- Killable native compilation and immutable circuit transport, nested worker cleanup, partially received pipe frames, crashes, timeouts, sampled worker-tree RSS caps and explicit proposal fallback.
- Standalone and sequential preparation, full matching feature channels, nested matcher alternatives, bounded retrieval and graph omissions, evaluator-label exclusion.
- Connected structural corpus variation, protocol family/split counts, typed partial teachers, local real-pair preparation, decoded checkpoint selection and generated-pool coverage.
- Matched generation controls, captured study inventories/objectives, declared release/split validation, all-scheduled status accounting, partial references, grouped paired effects and multiplicity correction.

The finite XR-2 specification reference passed 37 tests. Pilot and smoke protocol files passed static validation; this did not execute either campaign. Repair-targeted mypy, Black, isort, flake8 and repository import contracts are checked separately from matching-stack work.

## Executed small integrations

- A saved model through the standalone repair CLI, with the selected full OWL bundle independently safety-replayed.
- Compiler-free bounded enumeration and rejection versus a conditioned product on one tiny captured input; all three completed.
- Two generated cases with six assignments each, one training epoch, saved versioned checkpoint and actual training/development provenance. All teacher assignments were decided; common-inventory decoded development regret was zero and useful generated-candidate coverage was complete. A novel generated selected bundle was correctly reported unknown because it was absent from the teacher cache. These numbers describe that fixture only.
- Local `.ofn` source/target preparation with explicit clean/observed alignment supervision and whole-ontology holdout rejection.

No full-scale generated campaign, Conference/Bio-ML benchmark, or real-data adaptation training was run. The tools for preparing and scheduling those studies are implemented. Their runtime, transfer quality and comparative efficiency remain empirical questions.

The in-memory neural API snapshots model tensors before entering its proposal worker; this setup is measured against its budget but is not itself a killable native operation. The checkpoint CLI performs loading inside supervision. Memory enforcement samples Linux worker-tree RSS, can overshoot between samples, and excludes parent inputs and accelerator allocations. Neither interface claims an operating-system-wide hard memory quota.
