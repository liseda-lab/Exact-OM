# XR-2.1 development smoke on liseda-t2

The user authorized planning and starting this campaign on 1 October 2026, after
qualification of allocation 14408: one RTX 2060 SUPER (8 GiB), eight CPU threads
and 61,952 MiB of allocated RAM. The campaign directory is
`data/exact-repair-xr21-t2-20261001`. Its `plan.json`, `handoff.json`, resolved
protocols and immutable batch receipts are the executable record. This is an
exploratory development smoke, not a confirmatory result or a production repair.

## Frozen initial design

Generate 16 cases with split seed 20261001, four editable objects each, and the
directional-strengthening and complementary-interaction families. Each family
has two training parents, one development parent and one test parent, with one
corruption and one clean control per parent: 8/4/4 cases. Preserve siblings and
the evidence cutoff. Do not label test cases during preparation or training.
Test parents are reserved for evaluation after checkpoint selection is frozen.

Run HGT, matched R-GCN and no-graph controls, each with unary-only and pairwise
benefit, at seed 13. Keep width 128, three graph layers, four attention heads,
dropout 0.1, batch size eight, learning rate 0.001 and all frozen loss weights.
Each smoke arm permits ten epochs and development evaluation every five epochs;
the selection criterion remains generated-pool verified quality/effort with the
declared coverage requirement. This epoch count establishes operation and yields
exploratory comparisons; it does not claim converged training. Seeds 37/73 and a
larger cohort belong to a subsequent development profile based on measured cost.

Keep grammar stage 0 at menus 8/4/8, depth/constructor limits 1/1, candidate cap
64 and 32 draws/object. Retain the declared later expansion stages, but do not
claim that this stage-0 training executed them. Preserve all protected families,
the exact selector, query completeness and bounded unknowns. Collection keeps
three rounds with two training cases per round and 128 scheduled plan attempts
per case: 32 each of utility, diversity, quartet members and proposals.

LLM semantic labels are disabled; external API expenditure is zero. Production
matcher inputs stay deferred until the user explicitly releases them. Real
ontology preparation remains permitted but is outside this first finite batch
series. Large Bio-ML experiments and all XR-E00–XR-E09 completion claims remain
outside the scope of this smoke.

## Batches and gates

1. **B00:** native/current-generation qualification, single-GPU model checks,
   corpus materialization and exact train/development labels. Store every label
   outcome and require the declared complete development supervision before
   scheduling its dependent training arms.
2. **B01–B06:** the six model arms. Freeze each protocol independently. Reuse
   labels only after checking that all non-architecture dependencies match.
   Retain the original cache hashes, source preparation, label time and CPU cost.
   Each model has independent optimizer/RNG/preparation checkpoints and a cold,
   separate compiler cache; no warm start from the historical XR-2 campaign.
3. **B07:** independent symbolic/uniform/deletion controls, prepared from the
   qualified cases. Common-inventory comparisons are explicitly diagnostic.
4. **B08:** generated-pool and small circuit/decoder comparisons after compatible
   models exist, using identical language, proof conditions, budgets and cache
   policy across controls. Freeze the schedule before opening held-out results.
5. **B09:** machine-readable cumulative report, including all scheduled arms,
   unavailable/unknown results and costs. Report process completion, verified
   repair status and scientific conclusions separately. The supervisor prepares
   the evaluation/report batches autonomously from the recorded gates.

Only one GPU training job runs at a time. Experiment admission permits six CPU
threads, one GPU and 49,152 MiB RAM in total. A model worker requests four threads
and 32,768 MiB; an eligible independent CPU worker can use two threads and
16,384 MiB. The supervisor uses one thread and 8,192 MiB; the remainder preserves
interactive access. Every launch is a detached step in allocation 14408. Other
allocations and the paused historical campaign are outside this controller.

## Resource accounting and recovery

Per-case limits are 300 seconds wall, 600 CPU seconds and 8,192 MiB; compilation
uses the existing 20-second per-call/60-second aggregate limits. Native query
limits remain 30 seconds. Per-model stage allowances are 600 seconds for corpus,
3,600 for labels and 14,400 for training. The per-model cumulative worker cap is
21,600 seconds, including replacements; the protocol's inner whole-run cap is
19,800 seconds. The preparation job has 7,200 cumulative worker seconds.

The initial series receives a **48-hour cumulative worker-time slice**: two hours
for B00, up to 36 hours across six model jobs, and ten hours for controls,
evaluation, report preparation and their compatible retries. These are execution
ceilings, not runtime predictions. All attempts count; no limit resets on repair.
The slice remains below the previously authorized research budget's remaining
allowance. Its ledger carries forward all historical XR-2 pilot attempts and
costs, and separately pins the historical smoke ledger for the combined report.
Neither historical ledger is rewritten. A protocol or budget amendment must be
recorded explicitly; it cannot disguise a failed or incomplete scientific arm.

The existing supervisor checks health every 300 seconds without model calls. Its
prepared-job dispatcher can admit an eligible frozen model job without Codex.
Only a confirmed failure or genuinely unprepared next batch invokes
`gpt-6-astra` with `xhigh` reasoning and the existing ChatGPT login. No daily
repair limit applies; the same incident retains its two-attempt limit across
replacement jobs. Keep logical IDs, dependency/replacement links and costs.

Routine experiment preparation, code fixes, tests, commits, compatible recovery
and next-batch submission are already authorized. Email pgcotovio@gmail.com
through the existing connected Gmail transport only for an actionable decision
or intervention, while continuing independent eligible work. STOP/PAUSE markers
for this new campaign govern this series; old campaign stops remain intact.

The launch session stops after verifying the first batch and supervisor startup.
The supervisor owns subsequent progress and publishes the final scope accounting.

## B08 held-out evaluation adapter

`tools.repair.evaluate_campaign` freezes the six development-selected model
identities and all four reserved cases before querying any test consequence.
Unavailable models keep their four primary rows. Native v3 checkpoint freezing,
shared effective preparation and interaction selection, exact repair, and the
original symbolic target evaluator provide the primary generated-pool results.
Inference uses the same CPU worker profile across available arms; model fitting
and checkpoint selection are closed. Each primary row retains the 300-second,
600-CPU-second and 8,192-MiB case ceilings, including startup and transport.
Generation receives 60 seconds; selection retains 60 seconds of the remaining
case allowance for independent labels and five seconds for orchestration.

The small circuit diagnostic schedules grammar-only and ontology-informed
circuits against the existing bounded semantic expression enumerator, separately
for every object in every reserved case. The latter is an exhaustive finite
expression decoder, not an implemented incremental learned decoder. It uses the
same stage-0 menus, proof-supported conditions and uniform derivation sampling.
Each object/method row has 60 seconds and 32 draws, so each method has at most
240 seconds per four-object case. Compiler cold and warm attempts retain their
actual cache telemetry. This diagnostic does not establish repair quality or
learning efficiency. Every timeout, error, partial family and unvisited row
remains in the schedule denominator.

The batch has a 25,200-second cumulative worker ceiling and a 24,900-second
inner allowance. Results and per-row budget receipts are dependency-bound and
reused without repeating completed evaluation. Both labels and full repair/pool
evidence are saved behind small artifact receipts to avoid transport truncation.
B09 remains responsible for cumulative campaign costs and finite-scope closure
once B08 has an actual completion receipt.

## B09 cumulative reporting and finite-scope closure

`tools.repair.report_campaign` binds completed worker outputs, the frozen B07/B08
schedules, selected-model evidence, terminal HGT accounting and every attempt.
It preserves 24 primary model/test rows, 48 circuit diagnostic rows and 80
common-inventory control rows. B07's 20 reserved-test controls remain explicitly
deferred; its unavailable score-based controls and unknown logical outcomes
remain in the denominator. No missing observation becomes zero or success.

The report separates model process completion, training, verified repair rows,
finite-scope accounting and scientific conclusions. G0–G2 and XR-E00–XR-E09
completion are not inferred. Means over known observations are named as subset
means; an all-scheduled semantic mean is null if any value is unavailable.
Selector-local first-repair time remains distinct from the full row's elapsed
resources. This smoke does not establish learned efficiency or superiority.

The report worker uses the existing 3,600-second allowance. The batch runner
settles its cost before the reporting closure command can publish final costs
and completion. The closure checks all active completion receipts and rejects
pending work or outstanding reservations. Historical pilot and retained smoke
costs are included once, failed attempts remain charged, and the original
172,800-second incremental ceiling remains intact. Final closure artifacts live
separately from the worker's receipt-bound report, preserving that output hash.
After closure the registry marks remaining scope terminal and monitoring idle;
larger cohorts, additional seeds, real/production inputs and wider grammar
remain explicitly deferred.

## HGT report recovery authorized on 2026-10-03

The user's request, “Can we repair and run the HGT armas then?”, authorizes
reopening the two terminal HGT arms. Their ten optimization epochs and original
development selections are already complete. Attempt 003 failed while returning
an oversized report through the worker channel. `--resume-report-transport`
admits only the pinned predecessor with unchanged repair dependencies, settings,
input/split identities and a complete selection. It finalizes saved work using
the existing report artifact transport; it performs no further optimization or
development selection. Pairwise retains epoch 5; unary retains epoch 10.

The machine-readable amendment, checkpoint backups, actual-checkpoint preflight,
tests and launch descriptors are in
`artifacts/hgt-report-recovery-20261003` under the campaign directory. Preserve
all old terminal receipts, previous repair attempts and the first scope report
as historical evidence. The explicit authorization starts one recovery episode
for report transport, with at most two unsuccessful attempts at the same error
across replacement descendants. It does not expand any resource budget. All
existing logical model, training-stage and campaign ledgers remain authoritative.

Attempt 004 for each HGT arm uses its original logical work directory, frozen
protocol and four-thread, one-GPU, 32768-MiB profile. The existing dispatcher
runs them sequentially. Shared supervisor source and pinned instructions stay
unchanged. Its supplied registry and current handoff identify the authorized
reopening and remaining work.

After both model receipts exist, prepare an evaluation addendum for the eight
previously unavailable HGT primary rows using the original four held-out cases,
original frozen inference settings and these already selected model weights.
Validate dependency compatibility and retain the sixteen completed non-HGT rows
and forty-eight model-independent circuit rows with their original schedule
hashes and provenance. Do not relabel old rows with a new schedule hash or rerun
completed evaluation unnecessarily. The addendum and its merged report must
retain all original denominators and unavailable outcomes. Use the original
evaluation logical ID and cumulative 25200-second cap, and the original report
logical ID and 3600-second cap. Publish a new report revision without modifying
the completed first report or its receipts. B07's twenty reserved-test control
rows and all larger/deferred studies remain deferred. Routine preparation and
small tested adapter changes for this addendum are already authorized; do not
request approval for them. If one arm becomes terminal again, account for it and
continue the independent eligible arm and reporting.

`tools.repair.evaluation_addendum` freezes the recovered selections before
reading prior outcomes for provenance. It requires identical serialized cases,
protocols, schedule rows and inference dependencies. The evaluation runner
retains each reused row's original schedule hash, receipt, proof and cost record;
only previously unavailable recovered HGT rows execute. The original evaluation
stage ledger remains unchanged, and its spent time is deducted from the new
revision's inner allowance. The batch runner additionally enforces the unchanged
logical-job cumulative cap across all attempts. Attempt 003 writes the merged
72-row evaluation to `work/xr21-t2-evaluation/addendum-003/evaluation-report.json`,
preserving the original evaluation and report outputs. Report revision
preparation must consume that relative output from its actual completion receipt.
