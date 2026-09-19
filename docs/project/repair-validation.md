# Exact-Repair implementation validation

Implementation commit: `37db8f5`.

Local validation on 19 September 2026 used Python 3.12, pyowl-core 0.2.0, pyHermiT 0.2.0, pyelk-reasoner 0.2.0, PySAT 1.8.dev24, PySDD 1.0.6, torch 2.7.0 and torch-geometric 2.6.1. These are component-conformance results, not benchmark or model-quality claims.

| Check | Result |
| --- | --- |
| `pytest tests/repair_*_test.py -q` | 106 passed |
| XR-2 finite specification reference | 37 passed |
| Pilot and smoke protocol validation | Both passed; static validation only |
| Available CLI, run artifacts, alignment IO, public OWL boundary and workflow regressions | 25 passed, 1 skipped |
| Mypy over all repair modules, delivery CLI and repair tools | Passed |
| Black, isort, flake8 over repair changes | Passed |
| Import-linter repository architecture contracts | 5 kept, 0 broken |
| Dependency lock validation | Passed; existing package versions unchanged |

The repair tests include every action family, mirrored templates, all five specified semantic examples against both qualified reasoners, activated-expression non-vacuity, duplicate origins, unsupported constructs, exact signed/pairwise objective comparisons, pending assignments, bounds, independent replay, process crashes/timeouts, circuit probabilities/gradients, typed teacher masks, grouped splits, source/target roles, rich matching evidence and standalone/sequential handoff.

Two additional tiny integration checks ran in temporary directories:

- One synthetic captured alignment through deletion, directional and greedy study controls. All three returned verified repairs; exact controls certified their finite-pool optimum and greedy retained an honest gap.
- HGT training on two generated training parents and one development parent: 10 assignments decided, seven complete feasible labels, two epochs, with a saved development-selected checkpoint and teacher/cache provenance.

No full-scale generated campaign, Conference/Bio-ML benchmark, or real-data adaptation was run. The current generator is a fixture-based, mirrored structural corpus with coherent and retrieval-miss controls. The circuit implementation is the bounded-enumeration arm. Broad corpus variation, adaptation campaigns and scale/model-quality evaluation require separately scheduled work.

## Existing environment limitations

A broader selection of legacy tests produced 22 failures, 34 passes and one skip. Failures concern missing local test fixtures and the installed shared ontology/projector stack lacking native feature/projection methods required by existing matching code. Representative failures reproduce on an unchanged `HEAD` checkout. The repair path uses the qualified shared snapshot/reasoner interfaces directly and its tests pass.

Repository-wide mypy reports 12 errors in unchanged `exact/ontology/native_projection.py` and `exact/ontology/store.py`, involving the same missing installed native APIs. The repair modules and tools pass their targeted type check. This validation does not claim the entire repository suite is green in this environment.

The system Poetry 2.0.0 resolver stalled while locking the optional graph dependencies. A temporary Poetry 2.4.3 installation generated and checked the lock successfully; the user's global tooling was not replaced.
