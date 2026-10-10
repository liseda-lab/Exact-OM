# Implementation review and post-G4 verification

The implementation is **partially accepted, not scientifically qualified or
deployed**. On 2026-10-10 the user authorized review and supervisor-managed checks
on liseda-03 after G4. This authorizes the verification queue below, not the full
corrected scientific campaign. The previous
[implementation evidence](THROUGHPUT-IMPLEMENTATION-20261009.md) remains historical.

## Review findings and corrections

1. **Frozen-recipe admission:** a deployment could change its threshold, retrieval
   or evidence configuration while keeping the selected-config binding and a
   self-consistent inference manifest. Admission now reconstructs the expected
   deployment from the selected recipe and verified runtime fits. Only the
   declared case, mode, seed, public populations and manifest substitutions are
   admitted. Regression tests cover three changed recipes and legitimate fitted
   global/local deployments. Commit `6b33909a`.
2. **False successful parity command:** the comparator wrote a failed receipt but
   exited zero. It now exits nonzero on mismatch and rejects empty comparisons.
   The supervised worker also inspects the receipt, independently of exit status.
3. **Qualification isolation:** an accidental hosted code path could modify a
   request ledger before the socket-level network block fired. The candidate
   harness now denies hosted generation before key, ledger or transport access.
   Tests inject both client-owned and inherited ledger paths. Workers additionally
   isolate inherited experiment state, caches, temporary files and fixture ledgers.
   These safeguards and the queue tool are in commit `d205cdee`.

Independent review found no additional default-path scoring/cache/native evidence
regression in its inspected scope; 167 focused CPU tests passed. The deployment
review suite passed 40 tests, and the combined supervisor/queue/parity/guard/
throughput suite passed 106 tests. These suites overlap; their counts are not a
unique test total. No GPU measurement or paid request was made during this review.

## Finite GPU checks

Artifacts are under
`data/experiments-v2/throughput-review-20261010-01/`. The candidate is the detached
worktree `candidate-source/` at `d205cdeed08505ea21422b96a626bd691df2c358`.
`gpu-checks-01/queue-proposal.json` contains three prepared launch descriptors:

| Queue ID | Pair | Work |
| --- | --- | --- |
| `gpu-checks-01-H0` | NCIT–DOID | Reference prefix parity; complete bounded cold/warm/replay measurement |
| `gpu-checks-01-H1` | SNOMED–FMA | Same checks on its independently frozen public workload |
| `gpu-checks-01-H2` | SNOMED–NCIT | Same checks; fixture templates only, not a frozen scientific recipe |

Each check waits for successful `G4-run-once-followup` completion, following its
registered recovery lineage. The single GPU and six-CPU resource envelope
serializes execution; one case's failure does not erase the other queued cases.
G4 and its current source are unchanged. The workers use detached numeric Slurm
steps inside allocation 14372, six CPUs and 112 GiB RAM, with no wall-time kill
timer. They require an RTX 4090 receipt, matching source/input/precision/step
identities, at least 5,000 unique computed pairs, three chunks, 30 seconds, and
nonempty evidence parity. A measured throughput miss is reported separately from
failed parity; neither grants scientific promotion. Existing storage guards and
intervention handling remain active. A duplicate invocation preserves old terminal
receipts instead of replacing a successful exit with a failed duplicate's exit.

The reference is the default unbatched scorer from `5cd906b2`, whose recorded 2060
prefix matched the original `0eb61e8f` scorer. Its new isolated copy adds only an
observer field reporting actual precision. File inventories bind this overlay;
`reference-observer-review.json` identifies it explicitly. No old source snapshot
was modified. The new comparisons run reference and candidate on the **same
4090**; old-GPU values cannot substitute for them.

The existing workloads contain 6,028 / 6,092 / 6,026 unique candidate pairs before
exact prefiltering. Original complete queries and source groups remain intact;
private reference labels are absent. H2 fixture templates are a recorded union of
already generated H0/H1 templates, with H1 precedence for the one collision.
Both compared paths use identical templates. This is explicitly qualification
evidence and cannot be promoted into the actual fitted H2 scientific recipe.

At review, no implementation-check worker remained active to adopt. Allocation
14452 had already been released; other work on liseda-05 remains untouched.
Supervisor 14372.269 and G4 14372.270 were alive. Four baseline cells had completed,
two core D0 cells had failed, and core D1 global was active. The existing supervisor
retains responsibility for G4 recovery; this is not an eight-cell completion.

## Required supervisor continuation

`T09-fitted-checks-and-admission-prepare-20261010` is a preparation task after G4
and the three GPU checks. It must prepare real follow-up workers and update their
dependency links, rather than mark scientific acceptance from binary fixtures:

- Reconcile G4's failed cells using the reviewed fusion replay correction and
  supplied-local-pool retrieval fix; preserve completed work and paid accounting.
  Obtain fair deployment costs before the corrected mechanical selection/fitting
  freeze. Do not open H2 validation before that freeze.
- Extend verification to actual frozen fitted recipes, including grouped or
  exemplar decisions where selected, fitted selectors and complete mapping
  semantics. The current binary fixture harness cannot establish these.
- Complete the counted whole-program forecast implementation and measurements.
  `forecast_scoring_successor.py` currently produces conditional top-20
  sensitivities and unknown whole-program totals; supplying more GPU receipts
  alone does not turn it into final admission. Include retrieval, fitting, hosted
  work, assignment, all physical storage roots and protected spending limits.
- T04 batching already failed evidence parity and remains disabled. It needs an
  implementation correction plus independent proof before scientific use. Do not
  repeatedly rerun the same rejected batching to claim progress toward 200.
- Prepare corrected launch artifacts only after the remaining gates pass. The
  historical 63-cell E17 schedule must not be dispatched. The two legacy pending
  E17 rows are held for replacement by the corrected design and explicit rollout
  admission; the current review authorizes checks only.

The supervisor can handle correctness fixes, checked recovery and the remaining
verification preparation within existing policy. It must request intervention
for an actual scientific decision, allowance change or unresolved repeated error.
It may not waive parity, change precision/coverage, raise caps or invent missing
receipts. Keep rationales off and all G4/E17 usage in the same spending family.

`publication.json` in the review artifact directory records whether the queue was
published and binds the before/after registry and unchanged supervisor policy.
Current queue/health files, not this dated snapshot, determine live status.
