# Batch implementation handoff

**2026-09-24. Implementation amendment; subsequent batches are not yet admitted.**
Six days is a soft target on liseda-03 only. Continue to use detached Slurm steps inside
allocation 14372; retain the interactive shell. The current E09 continuation uses its own
frozen source and configuration. Prospective changes do not mutate that running experiment.

At handoff, the active continuation is **Slurm step 14372.3**, tmux
`exact-d1-production-14372-prefixfix`. Cold300, warm300, completed-replay and ancestor-prefix
are retained; sibling-prefix is continuing. The [operational repair](../../data/experiments-v2/d1-prefix-repair-01/repair.json)
separates verified checkpoint blobs from later timing files, retaining historical costs and
numerical snapshot `75c4d7a`. Monitor the [live status](../../data/experiments-v2/d1-production-probe-01/status.json);
these notes are a handoff snapshot, not a promise that the job remains running indefinitely.

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
Existing cumulative accounting remains at
`data/experiments-v2/mechanism-screen-02/runtime/exact-om-focused-v2/budget.json`.
No budget is reset or increased by this preparation.

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

At the implementation handoff, Slurm step **14372.3** is running the **D0-current-control**
qualification. Cold300, warm300, completed replay, ancestor prefix and sibling prefix have
passed. Its launcher advances to the four E09 arms once qualification and admission pass.
The interactive allocation remains intact.

```bash
tail -f data/experiments-v2/d1-production-probe-01/launcher-14372-prefixfix.log
```

Structured status is in `data/experiments-v2/d1-production-probe-01/status.json`; inspect
its `status`, `phase` and `worker_pid` fields. The detached session is
`exact-d1-production-14372-prefixfix`. Implementation work stops after final validation;
subsequent experiment monitoring belongs to the user.

## Validation and source identity

Implementation commit: `35909e7`. The final combined suite passed **512 tests**
(34 focused/integration modules, 109.85 seconds). Mypy checked 284 modules successfully;
black, isort, flake8, all five import contracts, the public-API docstring gate and the
tracked Java-free runtime check passed. CPU test threads were capped at two and CUDA hidden.

The [validation receipt](../../data/experiments-v2/implementation-validation-20260924/validation.json)
binds the committed numerical code, test log and prospective lock. It is an implementation
qualification record, not measured full-scale batch admission. Remaining runtime work is
materialization, fitting, measured resource admission and the declared experiments.
