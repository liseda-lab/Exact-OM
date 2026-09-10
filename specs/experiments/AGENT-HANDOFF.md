# Agent handoff: implement and execute v2

Use this document when assigning the implementation/execution task. It supersedes the previous
instruction that real runs were reserved for the user. This specification edit itself does not
launch experiments or claim the missing implementation is finished.

## Assignment

Implement the missing v2 experiment capabilities in Exact-OM and run a bounded broad search on
one RTX 5090 node with 64 GB or 128 GB RAM, using OpenRouter for all generative LLM work.
Use the campaign time/token limits and the user's broad OpenRouter spending discretion;
record projected and actual cost without requesting another arbitrary monetary cap. Focus each family on its appropriate
development case, then validate selected findings more broadly at the defined gates.
The user confirms OAEI and BioKG-Align data are available: locate and lock them, including
replacing stale descriptors, rather than deferring them from old repository notes.

Read README, RUN-PLAN, CHECKPOINT-RECOVERY, IMPLEMENTATION-STATUS, shared clarifications,
campaign-v2.yaml, and each experiment you implement. Inspect the current checkout first;
the status inventory was audited at 655f599 and must be reconciled with subsequent work.
Preserve unrelated changes and all previous experiment directories.

## Ordered work

1. Resolve the target node/runtime, dataset paths and capabilities, OpenRouter configuration,
   model/provider identities, and budget. Record missing bindings precisely; never invent data,
   revisions, prices, credentials, or results. Route all generative roles through OpenRouter.
2. Implement the v2 schema/planner in the existing runner. Materialize executable configs from
   the blueprint, replacing legacy broad matrices. Separate scientific design, numerical
   artifact compatibility, and attempt provenance. No second matcher or workflow platform.
3. Fix global/local dispatch and the five reviewed control/numerical defects. Add stage
   checkpoints, compatible reuse from prior directories, repair impact plans, result-set
   lineage, and budget admission. Make the NCIT–DOID vertical path executable before widening.
4. Complete fitting/LLM/grouped execution and feature-specific minimum arms in the status
   inventory. Update per-arm screen/confirm readiness with code and test evidence. A missing
   optional large model does not block a valid smaller family screen.
5. Run targeted tests, schema/matrix validation, and a dry-run. Perform a small real development
   interruption and relocation replay, then G0 throughput/feasibility. These are operational
   checks, not a substitute for the experiments.
6. Execute the predeclared focused screens across all applicable families, apply budget/selection
   rules, run broader development gates, mechanically freeze G4, then execute E17's final study.
   No repeated permission is needed for already-authorized steps within the supplied budgets.
7. On interruption, resume. On a bug, produce and apply the dependency-based repair plan and
   recompute affected paired results; retain valid upstream work and old attempts.
8. Report per-family outcomes, selected stack, all omissions/reasons, per-task quality/cost,
   test exposure, repair history, current result-set IDs, and exact continuation commands.

No model training, threshold, fallback, task selection, or feature choice may use final labels.
An exploratory amendment after exposure remains exploratory until fresh evidence is available.
Do not ask the user to approve a winner that the frozen rule can select mechanically.
Ask only for genuinely missing resource/budget information or a choice outside the approved
scientific scope. Do not contact annotators or send external messages without authorization.

## Implementation acceptance

- One true global run and one local-ranking run on the primary pair, with explicit stage traces.
- Per-arm/capability readiness; no silent no-op/fallback or fake fitted artifact.
- Standalone and pipeline-level LLM outcomes with complete OpenRouter request/cost ledger.
- Exact pair/final-decision reconstruction and consistent source denominators.
- Frozen pools, source-group splits, safe negatives, independent final-role enforcement.
- All CHECKPOINT-RECOVERY interruption/repair/relocation cases pass.
- Budget forecast includes the final panel, all controls, and repair reserve.
- Current-result aggregation refuses stale/mixed attempts and reports every incomplete cell.

Implementation work can update readiness/status evidence and executable manifests; it cannot
silently amend scientific selection rules. Put proposed research amendments in a separate design
revision before affected results. Use existing pipeline/config/metric code and narrow meaningful
tests. Do not create a dashboard, scheduler cluster, or exhaustive factorial.

## Planned command surface

The following commands are a v2 implementation target, not flags claimed to exist today.
The existing tools/run_experiment.py must support them and document their exact resolved paths.
Retain screen/confirm as the only public research stages.

~~~bash
python tools/run_experiment.py --campaign /path/to/campaign.lock.yaml \
  --stage screen --output-root /path/to/results --dry-run

python tools/run_experiment.py --campaign /path/to/campaign.lock.yaml \
  --stage screen --output-root /path/to/results --resume

python tools/run_experiment.py --campaign /path/to/campaign.lock.yaml \
  --stage confirm --selection-record /path/to/frozen-selection.json \
  --output-root /path/to/results --resume

python tools/run_experiment.py --campaign /path/to/campaign.lock.yaml \
  --stage screen --output-root /path/to/new-results \
  --resume-from /path/to/old-results --repair-record /path/to/repair.json \
  --reuse-plan-only
~~~

An approved implement-and-run assignment may sequence the frozen confirm command automatically
after G4 passes. The selection remains explicit and immutable. A STOP file or user interruption
must pause safely, with no loss of already-verified expensive work.
