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
