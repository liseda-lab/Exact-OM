# Batch implementation handoff

**2026-10-10 current continuation:** read the
[throughput implementation review](THROUGHPUT-REVIEW-20261010.md) and the live
registry's `throughput_review_20261010` handoff before acting on the historical
schedule below. Post-G4 RTX 4090 checks are authorized; the old full E17 schedule
is held for the corrected design and remaining scientific/resource gates.
Preserve the active G4 worker and its recovery lineage. Binary fixture checks do
not qualify actual fitted recipes, and experimental batching remains disabled.

**2026-09-25. E09 resubmitted; subsequent batches require measured admission.**
Six days is a soft target on liseda-03 only. Continue to use detached Slurm steps inside
allocation 14372; retain the interactive shell. The current E09 continuation uses its own
frozen source and configuration. Prospective changes do not mutate that running experiment.

The complete D1 qualification and D0 control passed on September 24. Its continuation
stopped before starting E09 because the forecast extrapolated fixed loading time per pair
and then charged loading again. The [forecast repair](../../data/experiments-v2/e09-forecast-repair-01/repair.json)
uses authenticated stage timings: setup once per arm, the slowest observed scoring rate,
and output/evaluation costs, retaining the 1.5 safety factor. E09's allowance is now
**9.896 hours**, previously 52.125 hours; budget limits and cumulative costs are unchanged.
This is a conservative forecast, not a deadline or measured four-arm result.

E09 was resubmitted at **01:41 UTC on September 25**, Slurm step **14372.5**, tmux
`exact-e09-14372-forecastfix`. The qualification is reused, with numerical snapshot
`75c4d7a` unchanged. The [new preparation](../../data/experiments-v2/prepared-campaign-13/)
passed its admission-only preflight before submission; the failed attempt is preserved.
These notes record submission state; consult the live files below for subsequent progress.

## Reviewable preparation

[prepared-campaign-12](../../data/experiments-v2/prepared-campaign-12/) contains the current
recipe declarations, original case/model/baseline bindings, preserved E13 OWL/CSV parity
inputs, and a [preparation summary](../../data/experiments-v2/prepared-campaign-12/preparation-summary.json).
It reads configuration and receipt metadata only; preparation starts no jobs, loads no
ontologies or models, reads no private answer contents, and makes no hosted requests.

The prospective inventory has **192 remaining declared cells: 121 development, 8 G4 and
63 class-final**. The previous planning inventory had 187. The five extra declarations are
one repeated E08 deduplication control and the four required E24 D0 controls. E10, E14 and
E23 are split into dependency-correct steps without adding unique settings. The E08 control
is reusable only after matching actual numerical identities. Conditional/inapplicable rows
remain visible. Expanded screens and selected feature-track submissions are still separate.

The 22 completed E05/E06/E26/E02 cells and G0 evidence are preserved as historical receipts.
They are not claims that changed code has identical predictions. Verify historical decision
imports during admission; do not rerun completed science just because this inventory is new.
E09 imports the completed qualification's cumulative accounting from
`data/experiments-v2/mechanism-screen-02/runtime/exact-om-focused-v2/budget.json` into
`data/experiments-v2/mechanism-screen-03/runtime/exact-om-focused-v2/budget.json`.
Later work must inherit the latest completed account; no budget is reset or increased.

## Implemented behavior

| Area | Implementation and boundary |
| --- | --- |
| E01/E10/E21 orchestration | E01 holds scores/thresholds/cardinality fixed while changing extraction. E10 selects six analytic settings before its two acceptance recipes consume the frozen winner. E21 consumes the selected E07 judge and verifies training-only teacher inputs. |
| Numerical reuse | Checksummed, bounded caches reuse compatible pair/channel outputs and training score tables; selector/extraction changes can reuse upstream work, while evidence, model, role and numerical changes invalidate it. This is not a measured full-scale speedup or a claim that different tau values are equivalent. |
| E08 annotations | Explicit property descriptors carry category, namespace, normalization, exclusivity and provenance. Declared equivalent facts are deduplicated before truncation while preserving language/datatype and all property origins. Missing metadata and arbitrary xref mismatches never become contradictions. |
| E24 missingness | Supported/contradicted/unobserved states; one-sided absence has zero contradiction authority. Hash-pinned deletion, duplication, zero-support insertion, equivalent serialization and reversal inventories reuse frozen supports without extra encoders/hosted calls. D0 controls accompany D1. Asymmetric directions are explicit and reverse together with the sides. |
| E13 representation | Matched OWL/CSV parity retains base evidence. Native Souffle enrichment uses the exact anchor-free raw graph with empty auxiliary property relations; inputs, code, engine, outputs and provenance are bound and recoverable. This is a declared structural fragment, not full OWL or full legacy materialization. |
| E14 typing | Required graph/learned typing paths report relation-macro full-pipeline metrics separately from oracle-pair diagnostics. The native OWL bridge is a separate optional comparison and cannot establish consistency for a CSV graph. |
| E23 structure | Three matched class ablation pairs use D1 at 0/50/100% hierarchy removal. The natural K0 comparison remains separate. Labels, populations and retrieval stay fixed within each pair; positive-unlabelled cases cannot invent supervised graph negatives. |
| Public inference/submission | Full global populations come from native ontology signatures. Local preparation retains each original query and puts repeated-source queries in distinct inference shards. The selected recipe applies immutable fitted heads under a same-pair deployment manifest; all references and training inputs are removed, fitting/evaluation are forbidden, and missing heads are rejected. Exports retain original membership and reject partial/mismatched inventories. |

The current string-only CPU microbenchmark measured median **24.9 ms without cache**,
**59.8 ms on replay**, and **108.8 ms on first write**. Cache overhead dominates this cheap
fixture; it establishes no neural-inference speedup. Measure representative encoder and
training workloads before counting saved time or enabling reuse indiscriminately.

Controlled E24 channel diagnostics have an explicit scope; a changed channel alone is not a
corrected alignment. End-to-end decision replay is restricted to its validated frozen
label-free/fixed-threshold path, with actual fusion, uncertainty and extraction recomputed;
unknown labels remain unknown in correction/harm reporting. Unsupported learned-head or
hosted-decision paths must not silently use that replay.

## Explicit current inapplicabilities

- **E08-identifiers:** no curated namespace/exclusivity descriptor is bound. Its repeated
  deduplication control and signed arm remain declared; the three ordinary E08 arms can proceed.
- **E24-asymmetric:** D0/D1 equivalence cases do not establish a directional interpretation.
  An optional `e24_typed_diagnostic: {case: T0, relation: '<'}` binding needs a justified case
  and direction, not an automatic promotion.
- **E14-bridge:** current T0 is CSV. A separately bound development OWL class case uses a
  paired graph-entailment control on that same population; its extra control must be counted.
- **K0 supervised E23 arms:** current positive-unlabelled data do not supply confirmed
  training negatives. No replacement loss or absent-pair negatives are introduced.

These dispositions are not empirical null results. Native implementations and required
branches remain subject to their own input and resource admission checks.

## Native E13 preparation

The executable is
`data/tools/souffle-2.5/usr/bin/souffle`. Run the following only in an admitted detached CPU
Slurm step. Each command prepares an anchor-free public raw view and executes native closure;
`--prepare-only` freezes the program without starting closure. Nothing below has been launched
by the prospective campaign preparation.

```bash
.venv/bin/python tools/materialize_biokg.py \
  --package /home/pgcotovio/BioKG-Align/build/release/BioKG-Align-v0.2.0-rc3-bbb08d188e48 \
  --ontology NCIT --output data/experiments-v2/e13-native-enrichment-01/NCIT \
  --souffle data/tools/souffle-2.5/usr/bin/souffle
```

Repeat for DOID, then bind each `materialized/enriched` directory to the corresponding
`csv_materialized` arm input and its paired `raw` view to `csv_raw`. All released `facts.dl`
rows are excluded and counted: alignment anchors are not imported, and auxiliary property
axioms have no verified per-ontology ownership. Souffle evaluates the frozen positive rule
program; Python only prepares and serializes facts. A complete native receipt survives
conversion interruption; changed requests or damaged native outputs fail verification.

## Admission and monitoring handoff

1. Finish current E09 qualification/comparison; preserve completed checkpoints and costs.
2. Freeze the committed implementation and verify case/model/input bindings. Measure the
   actual remaining training recipes and representative numerical-cache parity/throughput.
3. Admit eligible steps separately, respecting the dependency graph. Upstream fitted heads,
   donor artifacts, chosen judge and training-only teacher responses are execution outputs,
   not placeholder files. E13 materialization is a separately measured preparation job.
4. Freeze G4 before full reporting inference; prepare complete public populations/query
   shards and track-specific exports from those frozen recipes. Private answers do not tune
   methods. Submission files are local artifacts; organizer-side scoring is separate.

Run one heavy GPU worker, retaining two CPU workers and four hosted requests unless a measured
amendment changes them. Rationales stay off and OWL loading/projection stay native. Apply the
existing 1.5 forecast factor and cumulative envelopes; 144 hours is neither a timeout nor a
new admission cap. Code/fixture checks support implementation review; real-scale costs and
scientific outcomes remain to be measured during the batches.

Public-format validation exercised all 16,144 original Bio-ML query rows; synthetic smoke
outputs were discarded. The receipt is
[bioml-public-format.json](../../data/experiments-v2/public-inference-validation-01/bioml-public-format.json).
This validates the interface, not prediction quality.

## Current monitoring

The E09 launcher owns the four comparison arms: current, ancestor IC, sibling context and
hierarchy removed. All six qualification phases have passed. The interactive allocation
remains intact and each experimental submission uses a numeric Slurm step within it.

```bash
tail -f data/experiments-v2/prepared-campaign-13/launcher.log
```

Structured launcher status is in `data/experiments-v2/prepared-campaign-13/advance-status.json`;
scientific progress is under `data/experiments-v2/mechanism-screen-03/runtime/exact-om-focused-v2/screen`.
`launcher-exit-code` is written when the Slurm step exits. A nonzero exit or blocked status
requires checking the recorded reason; it is not a completed scientific result.


## Automatic next batch

**Slurm step 14372.6**, tmux `exact-next-batch-14372`, was submitted at **01:53 UTC on
September 25**. Its verified initial state is `waiting_for_verified_E09`. The
[queue handoff](../../data/experiments-v2/next-batch-01/HANDOFF.md) describes the frozen
source (`f9fa3bf`), input bindings, measured admission and recovery boundaries.

After E09 completes, the queue inherits its final cumulative ledger and verified caches,
runs one full 300-source D0 control to qualify the updated source, and admits **five E01
extraction arms followed by six E10 analytic settings**. The E01 arms share label-free
scoring with selector and decision gate off; extraction alone varies. The core provisional
allowance is 9.869 hours plus 0.897 hours for qualification; actual comparison admission
uses the slower old/new D0 control and the unchanged 1.5 safety factor. No setup-per-pair
extrapolation or assumed cache speedup is used. E08 and other families retain explicit
pending-input/measurement dispositions and do not block these 11 cells.

The queue uses one GPU worker at a time, retains budget limits and historical spending,
and permits cached verbalization responses only. It detects terminal E09 failure and
records the reason instead of waiting indefinitely. Whole-family admission still precedes
execution. Neither a forecast nor a queued step establishes a completed scientific result.

```bash
tail -f data/experiments-v2/next-batch-01/launcher.log
```

Monitor [queue status](../../data/experiments-v2/next-batch-01/status.json) and
[submission receipt](../../data/experiments-v2/next-batch-01/submission.json). The top-level
queue intentionally rejects overwriting an existing runtime; after interruption use the
saved family campaign/runtime and recorded accounting, as described in its handoff.

The forecast repair passed 30 tests; E01 preparation/campaign checks passed 69 tests.
The final queue preflight resolved all 11 configurations against the frozen source and
verified historical imports. Separate mocked checks covered worker budget caps, failed
preparation, accounting finalization and cooperative signals without running models.

## Hourly supervision

The [hourly supervisor](SUPERVISION.md) checks the registered experiment steps without model
calls when healthy. Confirmed failures and a completed queue needing its next batch can invoke
Codex using the current ChatGPT subscription login, preserving allocation, experiment budgets,
frozen sources and compatible checkpoints. Monitor
`data/experiments-v2/hourly-supervisor-01/status.json`; its deployment receipt records the
actual Slurm step. Pause supervision before manual experiment repairs and verify no repair
agent is active. The supervisor is local to this node and does not survive allocation loss.

## Validation and source identity

Implementation commit: `35909e7`. The final combined suite passed **512 tests**
(34 focused/integration modules, 109.85 seconds). Mypy checked 284 modules successfully;
black, isort, flake8, all five import contracts, the public-API docstring gate and the
tracked Java-free runtime check passed. CPU test threads were capped at two and CUDA hidden.

The [validation receipt](../../data/experiments-v2/implementation-validation-20260924/validation.json)
binds the committed numerical code, test log and prospective lock. It is an implementation
qualification record, not measured full-scale batch admission. Remaining runtime work is
materialization, fitting, measured resource admission and the declared experiments.


## E01 cardinality amendment queued — 2026-09-25

The approved amendment adds an unrestricted threshold baseline alongside five
explicit one-to-one comparisons, with legacy assignment diagnostic only.
Collision-free lexical exact anchors remain protected; conflicting anchors retain
their scores and compete consistently. Threshold, sample, ontology scope and saved
pair scores are unchanged. Selected policies carry their cardinality settings.

Slurm step **14372.13** was verified live at 11:57 UTC, waiting for E10 step
**14372.12** to finish and close its cumulative ledger. The six E01 arms then replay
frozen scores without ontology/model calls or new experimental hosted tokens;
rationales remain off. Frozen implementation commit: `bef200d`; merge: `5f3d3bf`.
Focused regression/integration checks and actual-artifact preflight passed.
The [E01 handoff](../../data/experiments-v2/e01-cardinality-01/HANDOFF.md) records
submission, validation, accounting, monitoring and recovery details. The supervisor
registry includes the queued step. No amended E01 scientific completion is claimed.


## E10/E01 completed and supervision updated — 2026-09-25

E10's six saved cells were finalized by step **14372.14** after repairing imported
selection bindings across schema defaults (`bf61182`). The signed scientific
selection remained `screened_out`; no predictions were recomputed. E01 restarted
as step **14372.15**, completed all six amended variants at **20:06 UTC**, and selected
`mutual_best` on its declared development sample. These are development decisions,
not final-track performance or submission claims.

The [E10 recovery](../../data/experiments-v2/e10-finalization-01/HANDOFF.md) and
[E01 handoff](../../data/experiments-v2/e01-cardinality-02/HANDOFF.md) retain frozen
sources, exact artifact validation, cumulative budgets and completion evidence.

The updated supervisor has **no daily repair limit**, checks every five minutes,
and bounds repeated unresolved errors across replacement launches. It can continue
independent work when another family needs a decision. Email intervention alerts use
the connected Gmail account; the one-time delivery test has a verified Gmail receipt.
The user should give decisions in Codex, because email replies are not monitored.
See [supervision](SUPERVISION.md) and the runtime deployment receipt for its current
step, source, registry, configuration and preserved intervention history.

Validation: 176 finalization regression tests, 14 E01 queue/report checks, and
96 combined supervisor/notification tests passed. The final Gmail parser and receipt
recovery check passed 13 focused tests after testing real delivery. Counts overlap;
no unique total is implied. Lint/format checks passed for the supervisor changes.

## E14 candidate-first validation authorized — 2026-10-05

The user authorized PR #2 candidate `d8d7a2f` validation before other pending work, after
active E21. Prepare a fresh isolated environment with a real optimized candidate wheel;
verify installed-wheel resource/identity cases and tiny native admission/bridge with all
three resource controls before full ontology work. The planned detached Slurm chain is
`E14-pr2-native-admission-20261005-01` (full prepared NCIT, original DOID/imports), followed
by `E14-pr2-bridge-validation-20261005-01` (unchanged 300-source, seed-17 public-development
diagnostic and full training anchors). Reuse verified preparation, preserve checkpoints,
and inherit the latest cumulative account at dispatch; keep host/storage guards enabled.

The 11 other pending batches gain the bridge-validation prerequisite without losing their
existing dependencies. Held actual `E14-bridge-run-once-followup` remains unchanged. A passed
bridge restores ordinary queue eligibility automatically; a failed gate stays for bounded
repair or an explicit user decision. This scoped amendment overrides prior frozen supervisor
instructions about unpublished packages and independent work only for this validation.
It grants no merge, publication, production integration or automatic scientific promotion.

Preparation and execution evidence belong under
`data/experiments-v2/e14-pr2-validation-20261005-01/`. **This is preparation until
`publication.json` records actual queue publication.** Consult its handoff and terminal
receipts for subsequent state; no full-input pass, queue submission or release readiness is
claimed here. Existing E21 work, completed main E14 results and failed-attempt costs remain
preserved. The exact protocol is in [E14](E14-typed-relations.md).

### E14 candidate validation queued — 2026-10-05

The above chain is now queued: its `publication.json`, `queue-verification.json` and
`post-publication-check.json` establish actual publication and priority, with E21 and the
existing supervisor still live. Both workers use frozen Exact `b463e45c`, the isolated
optimized PR2 wheel, matching explicit memory/work/index options, and the reviewed storage
guard. The bridge's candidate-only query-evidence check prevents unsupported/error
abstentions from counting as a successful validation. Its protocol remains 300 sources,
935 query pairs and 3,680 public training anchors; no hosted calls or rationales.

Local verification: 15 installed-wheel resource cases and tiny native/checkpoint checks;
36 Exact diagnostic/gate tests. Full ontology admission and bridge execution have not yet
started. The GitHub matrix's missing-native-core prerequisite is being repaired separately;
local fixture success does not establish release readiness. Monitor the two queued roots
named above after E21 finishes. No long validation monitoring or supervisor restart is needed.

CI follow-up: pyHermiT PR2 commit `e666050` repairs only the test prerequisite and separates
its mandatory 15 native resource cases from the original compatibility phase. Package source
and build metadata remain identical to the queued `d8d7a2f` candidate. Local checks passed:
21 packaging/helper tests on Python 3.12, nine helper tests on 3.10, the 15 installed resource
cases and a fresh native-core receipt probe. The actual missing-native source-build path
and hosted matrix still await CI verification; no publication/merge was performed.
Evidence: `e14-pr2-validation-20261005-01/evidence/ci-review.json`, also bound by the registry's
priority-validation record. Existing candidate wheel, environment and receipts are unchanged.
