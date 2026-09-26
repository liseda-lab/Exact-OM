# Exact-Repair implementation validation

## 26 September 2026: published-stack and recovery qualification

The current installed stack passed 309 combined repair, finite-reference,
supervisor and notification tests. A subsequent focused run passed 88 tests after
the final budget, STOP-file and type-check fixes; these counts overlap. The six
changed runtime modules pass targeted mypy, formatting, lint and diff checks.
Strict pilot/smoke protocol validation passes with the original hashes and counts.

The environment uses pyowl-core/pyhermit/pyelk-reasoner 0.2.1, PySAT 1.8.dev24,
PySDD 1.0.6, torch-geometric 2.6.1 and Torch 2.7.0+cu128. Actual HGT and matched
R-GCN forward/backward paths passed on the RTX 5090 and RTX 2080 Ti. A Slurm probe
confirmed both GPUs are accessible in allocation 14387 while retaining its
interactive step.

Recovery checks cover exact optimizer/RNG continuation after an interrupted epoch,
changed-input/setting/split rejection, no study-budget replenishment, lost-work
reservations, and failed-worker completion/cost records. A real preparation CLI
fixture generated all 96 smoke cases and labelled an explicit one-case-per-split
conformance subset; resumption reused identical labels and preserved 18.863 seconds
of accumulated label work. Its partial development teacher remains partial.

The review found missing durable training/preparation checkpoints and reset study
budgets, now addressed in the repair runners. The shared supervisor adds only
continued observations during intervention and optional independent-batch admission.
The [batch plan and launch documentation](repair-campaign.md) record the handoff.
These are implementation-conformance results, not a full smoke campaign, trained
model-quality result, or completed Conference/Bio-ML benchmark.

Validation date: 19 September 2026. These are component-conformance and small integration results, not benchmark or model-quality claims. The initial implementation was `37db8f5`; `e033026` adds direct grammar, retrieval and supervised runtime paths; the following training/study commit completes the protocol tools described below.

## Published ontology stack and historical evidence

Current repair execution uses the [published native stack](https://github.com/liseda-lab/Exact-OM/blob/main/specs/native-stack.md):
`pyowl-core`, `pyowl2vec-star-projector`, `pyhermit` and `pyelk-reasoner` are locked to `0.2.1`;
the reasoners remain optional. Qualify actual capabilities for each selected operation.

The checks below ran before adoption of the published packages. The complete
[19 September validation record](../archive/native-stack/repair-validation.md) preserves
the original dependency identities, commands and environment. These counts describe those
runs; they have not been relabelled as validation of the published `0.2.1` artifacts.

## Executed checks

| Check | Result |
| --- | --- |
| Repair component tests (historical environment) | 166 passed |
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
