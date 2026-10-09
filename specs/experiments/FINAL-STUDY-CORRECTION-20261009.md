# Final-study correction: full submissions and bounded diagnostics

**Design:** `exact-om-final-mixed-scale-20261009-v1`

**Status:** `specified_not_applied`

**Recorded:** 2026-10-09

**Execution authorization:** none conferred by this document.

This is a prospective scientific-design correction, not a claim that the live campaign has
changed. The user requested a smaller computational workload that retains full global and
local submissions and the research questions. The correction separates full-population
comparisons from bounded developmental diagnostics. It changes the scale of some evidence
and therefore narrows the claims that that evidence can support; it does not disguise a
bounded run as a full evaluation.

The machine-readable companion is the prospective planning record
[throughput-final-plan-20261009.yaml](throughput-final-plan-20261009.yaml). The implementation
work and correctness/performance gates are in
[SCORING-THROUGHPUT-200.md](SCORING-THROUGHPUT-200.md). Neither document is a campaign lock or
permission to rewrite the supervisor queue. Until a reviewed implementation and admission
record apply this design, the existing frozen recipes remain the authoritative execution
history. Record the eventual source revision, design hash and queue amendment separately.

## 1. Why the design changes

The historical final declaration expands to **63 logical cells**: 54 Exact-OM cells
(three arms × three pairs × two task modes × three seeds), plus nine LogMap cells
(three pairs × three seeds). G4's eight comparisons are additional development work.
Identical deterministic LogMap executions never constitute independent seed replications.

The corrected design preserves full-population baseline-versus-selected evidence on every
class pair and in both task modes. It measures stochastic sensitivity and most mechanism
questions on fixed public development cohorts. The full H0 supervision control remains
because otherwise a gain from the selected stack could be confused with access to target
training labels. It does not provide a full-population supervision contrast on H1 or H2.

Previously published configurations, results, costs and the 63-cell runtime declaration
are immutable historical artifacts. A new design identifier, not an in-place edit of an
old lock, distinguishes this correction. The reason for the change is computational scope,
not observed private test performance. Preserve the record of which original full-scale
seed and attribution claims are no longer made.

## 2. Full-population primary matrix

| Case | Ontology pair | Exact-OM arms | Modes | Seed | Logical cells |
| --- | --- | --- | --- | --- | ---: |
| H0 | NCIT–DOID | frozen v2 baseline; G4-selected stack | global; local | 17 | 4 |
| H1 | SNOMED–FMA | frozen v2 baseline; G4-selected stack | global; local | 17 | 4 |
| H2 | SNOMED–NCIT | frozen v2 baseline; G4-selected stack | global; local | 17 | 4 |
| H0 supervision control | NCIT–DOID | frozen target-label-free counterpart | global; local | 17 | 2, if needed |

The base matrix is **12 logical Exact-OM cells**. The expected matrix is **14** when the
selected H0 stack uses target training labels. If it does not, record the exact identity
that makes the label-free control redundant; do not create a fictitious additional result.
Likewise, identical baseline/selected predictions may share physical execution, but retain
both named logical roles and disclose their identity.

The H0 label-free counterpart uses the same declared component family and downstream
semantics, with all target-label-dependent fitting removed or replaced by its frozen
label-free policy. This may require refitting descendants; relabeling supervised weights
is not a control. Record component-specific supervision as `in_pair_supervised`,
`target_label_free` or `cross_pair_transfer`. Donor-label transfer is not an unsupervised
claim. The selected stack and the counterpart must have immutable recipes before final
prediction or outcome exposure.

**LogMap contributes three unique full global runs**, one on each H0/H1/H2 pair, under a
pinned implementation and declared configuration. Reuse an existing same-task output only
after verifying ontology versions, full populations, allowed supervision and provenance.
LogMap is deterministic evidence, not three independent seeds. Do not fabricate a local
ranking comparator by filling sparse global mappings with invented candidate scores. The
paired published-matcher competitiveness claim is global unless a separately specified
valid local interface exists; no additional matcher tournament is introduced here.

Thus the expected full execution design has **14 Exact-OM logical cells and three unique
LogMap global cells**, subject to exact-identity reuse. Bounded diagnostics below are
counted separately. Fitting and shared preparation are real work even though they are not
additional scored cells.

### Complete inputs and outputs

- Global runs use the entire frozen eligible source-ontology population and full eligible
  target population. No development source cap, local-query source list or local test pool
  may restrict global retrieval, scoring eligibility or alignment coverage.
- Local runs retain every original public query row, including repeated source IRIs,
  original query identities and all supplied candidates. Query-dependent computation
  remains query-dependent. Deduplicating a source is not permission to collapse its queries.
- The currently verified public local inventories contain 1,819 NCIT–DOID, 3,414
  SNOMED–FMA and 10,911 SNOMED–NCIT rows: **16,144 original queries per method**. Revalidate
  these against the pinned input hashes at admission. A disagreement is an input-binding
  issue, not permission to truncate or silently replace the population.
- All official global/local outputs are generated from reference-free inference
  configurations with `run_eval=False`. Export global Alignment RDF and complete positional
  local rankings using the strict public-input validators. Unscored rows or candidates are
  incomplete outputs, never zeros or fabricated ranks.
- Preserve native ontology loading/projection, selected numerical semantics, exact score
  reconstruction and rationales-off behavior. External inspection or human annotation is
  not a prerequisite introduced by this correction.

## 3. Bounded stochastic diagnostics: exact cohort recipe

These are developmental sensitivity measurements, not partial official submissions.
Membership is determined without prediction scores, private references or choosing easy
sources. Freeze each source list, original local-query membership and input checksum before
executing its new comparisons. Source membership is the same for all arms and seeds.

| Diagnostic cohort | Membership | Maximum source groups | Exposure rule |
| --- | --- | ---: | --- |
| D0-public-valid | All 619 sources in the already bound NCIT–DOID eligible validation universe | 619, all available | Existing public development exposure retained |
| D1-public-valid | Exact 1,000-source SNOMED–FMA membership already frozen for G4 | 1,000 | Reuse G4 membership, never resample to improve scores |
| D_H2_valid | Public SNOMED–NCIT validation eligible universe, separately prepared and deterministically capped as below | min(1,000, available) | Only after the corrected G4 selection and fitting recipes are frozen |

D0 uses the existing eligible-source file, not a newly reconstructed set of only known
positive sources. D1 uses the existing G4 membership artifact and hash. If their recorded
counts do not verify, resolve the discrepancy in a binding record before execution; do not
invent replacement entities. These sizes count source groups, not candidate pairs or
original local query rows.

The public H2 validation files have been located under
`data/experiments-v2/bioml-primary/SNOMED-NCIT/`: `local.valid.cands.tsv` has 3,614 query rows
(SHA-256 `c3e7f3e5ff2e4c986e73ae30785bba3a3d81d2c3bb741c39ca3af075815a1f09`), and
`refs_equiv/valid.tsv` has 2,864 reference rows
(SHA-256 `78fab8470d94856c5f5fb3a3b1018d87bb1a4c55241cd6b8a1787c13e9267617`).
This establishes availability, not a prepared or admitted diagnostic. The frozen H2
reporting descriptor has no validation binding and must remain a reporting descriptor.
Prepare a distinct **D_H2_valid** descriptor after G4 freeze; do not overload the existing
optional D2 role, which means OMIM–ORDO, or relabel the H2 test pool as validation.

For this diagnostic, verify the public validation descriptor, ontology revision, source
universe, original local queries and public label permissions. Materialize stable original
query IDs and gold-stripped candidate inputs; keep labels and any implicit gold-position
information out of inference features. Bind a public eligible validation source population
independently of positive reference rows and retain the full global target universe. Sort eligible source
IRIs by the hexadecimal SHA-256 of the UTF-8 string
`exact-om-final-mixed-scale-20261009-v1/H2-public-valid/` followed by the exact IRI, breaking
hash ties by IRI, then take the first min(1,000, N). Record N, selected count and list hash.
If fewer than 1,000 exist, use every available group. If the public validation inputs cannot
be bound to an admissible source/query role, record `unavailable_public_validation` for this diagnostic
only; do not substitute private test answers, fabricate D2/OMIM–ORDO, or block the full
reference-free H2 submissions. This document does not assert that the D_H2_valid binding is ready.

For each selected source, global diagnostics retain the full target search universe and
normal retrieval policy. Local diagnostics retain every original validation query row for
that selected source and every candidate in each row. Record sources without local queries
as mode-specific inapplicability, not failed predictions; local results use their actual
query denominator. Do not pad a local cohort to the global source count or remove sources
based on gold availability after seeing outcomes. Publish the actual denominator per mode.

### Arms, seeds and counts

Each available cohort compares the **frozen baseline, frozen selected stack and frozen
target-label-free counterpart**, in both global and local modes, with seeds **17, 29, 43**
for genuinely stochastic dependencies. This is at most **54 logical diagnostic cells**
(three cohorts × three arms × two modes × three seeds). It is 36 if H2 public validation
cannot be bound. Scientific identity can reduce physical execution further; counts must
not imply independent replications where none occurred.

- Reuse G4 seed-17 artifacts only when cohort, ontology bytes, query rows/pools, model,
  training groups, fitted weights, configuration, evidence and relevant code identities
  agree. A changed fitting recipe or training cap prevents this reuse.
- Repeat stochastic fitting, declared prompt-order sampling and their dependent outputs
  under each real seed. Reuse seed-independent deterministic artifacts once. Hosted
  response-cache replay is the same observed response, not a new independent model draw.
- Fixed diagnostic cohorts do not imply bounded training unless the frozen recipe says
  so. Record training group counts and donor/target roles separately from reporting counts.
- Use paired source/query results and report per-seed values, spread and paired intervals.
  Three seeds provide a limited sensitivity estimate; do not oversell the precision of a
  variance estimate from three replicates. A bounded global source set also changes the
  competitors seen by joint assignment/cardinality rules. Its score is a result for that
  bounded problem, not automatically an unbiased estimate of full global alignment.
- Extra full-population seed outputs are optional derived artifacts only when all required
  compatible computation is already available and export is materially free. They neither
  admit new scoring/fitting/paid calls nor automatically expand the mandatory matrix.
  Record their provenance and deterministic identity before calling them replications.

Public H2 validation outcomes may be reported after freeze but must not select components,
hyperparameters, thresholds, seeds or a replacement cohort. If they are inspected, the
H2 claim becomes **a pair excluded from configuration selection, with post-freeze public
validation diagnostics**, not an entirely unobserved pair. H2 shares ontologies with D0/D1
in any event, so it is not unseen-ontology or unseen-domain generalization.

## 4. Mechanism questions and missing combined controls

Completed E00–E26 screens, recoveries, negative results and diagnostic replays remain valid
within their recorded scope; do not repeat them for qualification or nicer bookkeeping.
Only a missing comparison of the combined selected stack warrants new bounded mechanism
work. An existing isolated-component result is not automatically a combined-stack result.

Before new diagnostic outcomes, bind any remaining combined controls to a finite manifest
on **D0-public-valid only, seed 17, both modes**, with the fixed membership above. The cap is
four named component removals and two predeclared 2×2 interaction boundaries, as inherited
from the earlier programme. The proposed removal classes are calibration; learned
ranking/acceptance; structured-fact LLM judgment; and hierarchy/graph evidence, each only
when present in the selected stack. The interaction boundaries are calibration × learned
acceptance, and LLM fusion × learned ranking. Bind exact switches, weights and descendant
refits; a prose class is not an executable treatment.

For absent components or already answered, exactly compatible contrasts, record
inapplicability or reuse rather than replacing them with a new hypothesis. All four cells
of an admitted 2×2 must exist; shared controls and compatible removals are executed once.
With the selected stack already available, the absolute maximum is ten additional unique
configurations (four removals plus three per interaction), hence **20 additional logical
mode-specific cells before overlap/reuse**. This is a ceiling, not a requirement to fill it
or an automatic queue admission. If all retained combined-system objectives already have
sufficient compatible evidence, the count is **zero additional cells**. An interaction is
one declared two-factor question with all four corners, not permission to treat every
parameter as an independent research axis. Publish the exact claim-to-overlay manifest and
actual new-work count before admission or final test exposure. No new treatment is chosen
because a final score was disappointing.

These bounded contrasts provide development-only mechanism evidence. They do not establish
full-population causal attribution, confirmatory effects on all three pairs, or the original
full-scale component/interaction claims. Applying a completed result to a combined stack
without its required comparison is reported as unresolved, not assumed successful.

## 5. Objective-to-evidence and claim ledger

| Objective / family | Evidence retained or produced | Permitted claim and limit |
| --- | --- | --- |
| E00 operational harness and reconstruction | Existing native/runtime/checkpoint evidence; reconstruction on final predictions | Executable and numerically reconstructable computation; not semantic correctness of mappings |
| E01 extraction/cardinality | Completed paired extraction screen; frozen extractor in full outputs | Development effect under declared cardinality, not universal one-to-one ontology semantics |
| E02 exact anchors | Completed anchor controls and nulls | Recorded developmental anchor effect; no extra full benchmark repeat |
| E03 calibration | Completed calibration screen; fixed selected fits and bounded seed sensitivity | Development calibration benefit and final stack performance; no unrestricted cross-task calibration claim |
| E04 NIL/abstention | Completed historical benchmark-NIL, pool-miss and listwise diagnostics | Candidate-pool NIL under its original versions; no verified ontology-wide NIL or current-track NIL labels |
| E05 retrieval | Completed matched retrieval controls; final source/candidate coverage | Known-positive retrieval recall where public labels permit; final test recall awaits organizer evidence |
| E06 string channel | Completed string-channel screen, including retained control | Development result only unless part of an explicitly admitted combined contrast |
| E07 LLM evidence/judgment | Completed brief/fact/listwise comparison; selected evidence in final stack | Evidence for the tested judge/context configuration; no claim that all larger models or contexts fail |
| E08 attribute polarity | Completed main screen; identifier branch remains approved deferred | Main development result; no empirical result for unavailable justified identifier metadata |
| E09 hierarchy | Completed hierarchy screen and selected policy | Development hierarchy effect; bounded combined removal if actually needed |
| E10 fusion/selector | Completed fitted and analytic controls | Effects under declared features and training; interaction claims restricted to admitted bounded controls |
| E11 property equivalence | Existing property-specific evidence | Property development capability only; class final outputs do not provide held-out property validation |
| E12 instance equivalence | Completed labels/context and retrieval controls | Instance/KG development findings on the actual case, not universal instance alignment quality |
| E13 representation/enrichment | Completed same-information and enrichment comparisons | Recorded parity/differences and enrichment result; no inferred benefit from merely using a reasoner |
| E14 relation typing | Completed typed-relation main screen | Three-relation case-specific evidence; optional expensive full-OWL bridge remains approved deferred |
| E15 label-free selection | Completed label-free/supervised screen; full H0 control; bounded other cohorts | Full H0 supervision contrast; other-pair supervision effects are bounded development evidence |
| E16 cross-pair transfer | Completed donor-transfer screen; frozen supervision labels | Transfer under its recorded donor/recipient setup; a trained final arm must not be called label-free |
| E17 stack integration | Full baseline/selected outputs on H0/H1/H2, both modes; organizer scores later | Full population, fixed-seed stack comparison; no full-population seed-variance claim |
| E18 supervised reranking | Completed reranker comparison; bounded seed fits and final selected recipe | Tested reranking effect and bounded stochastic sensitivity |
| E19 supervised fusion | Completed analytic/fitted/learned comparison | Recorded developmental fusion effects; preserve unsuccessful alternatives |
| E20 learned retrieval/encoder | Completed cross-encoder/contrastive controls | Tested accuracy/cost tradeoff; fresh implementation timing must include actual encoder work |
| E21 LLM adaptation | Completed frozen-judge, student, exemplar and routing controls | Development evidence, including negative results; no rerun of expensive failed-to-improve variants |
| E22 label efficiency | Completed budget curve and policy comparison | Sensitivity to recorded effective training groups; no universal optimal-label-count conclusion |
| E23 graph supervision | Completed natural/shuffled/richness controls | Case-specific effects with shuffled-control caveat; class results cannot replace KG held-out evidence |
| E24 difference degeneracy | Completed contrastive/perturbation diagnostics | Diagnostic behavior retained; diagnostic-only replays excluded from deployment inference timing |
| E25 routing/judge/trust | Completed forced, actual-response and oracle replays | Observed corrections/harms and hypothetical diagnostic headroom; oracle never enters deployed selection |
| E26 quality proxy | Completed proxy-validity screen | Recorded evidence about proxy reliability; heuristic quality is not a calibrated truth guarantee |
| Published comparator | Three compatible full global LogMap outputs | Paired global competitiveness after official scores; no fabricated local LogMap ranking claim |
| Cost, memory, storage and robustness | Prospective fair phase measurements, cumulative ledgers and recovery receipts | Measured costs with cold/shared/replay scope explicit; no speedup inferred from implementation alone |

The previously approved E08 identifier and E14 full-OWL bridge deferrals are unchanged.
Existing inapplicable variants remain explicitly inapplicable. This correction does not add
new feature-specific final datasets or claim that the class panel confirms properties,
instances, natural NIL or typed relations. If any such held-out claim is retained elsewhere,
its missing matched reporting evidence must be listed as unresolved rather than silently
asserted from these class submissions. No manual dataset construction or annotation is
introduced.

## 6. Correct G4's cost comparison before final freeze

The inspected selected/core configuration enables `diff.enabled`, formulation `off`,
`dump_components` and `controlled_perturbations`. The difference-channel path performs ten
synthetic diagnostic replays per pair; the baseline does not enable the same extras. These
E24 diagnostics do not feed the natural prediction. Their wall-time contribution has not
been isolated, so the observed runtime ratio cannot be treated as a fair deployment-cost
comparison. G4's inference-cost guard (at most 1.2× baseline) can consequently affect
selection for an implementation-diagnostic reason.

1. Preserve all existing G4 predictions, quality outcomes, raw timings, attempts and spend.
   Do not discard valid quality results because their cost measurement is contaminated.
2. Before private final exposure, freeze a common deployment timing boundary for both
   arms. It includes every operation required by natural predictions, selected evidence,
   genuine explanation/reconstruction output, hosted calls and retries. Exclude only
   explicitly diagnostic perturbation/replay exports that are not needed for deployment.
   Record the switches and verify that removing them leaves natural outputs unchanged.
3. Obtain a prospective matched cost receipt using the admitted development population
   and the same cold/shared-cache definitions, hardware and resource limits. Use isolated
   stage measurements or a minimally sufficient replay; do not run a new qualification
   matrix or repeat paid requests merely to measure their cached replay time.
4. Separate original service-generation cost, cache replay, preprocessing, inference,
   diagnostics, output writing and recovery. Preserve historical wall time as actual
   operational cost. Never subtract an estimated diagnostic duration from an old total or
   present warm cached inference as a cold end-to-end measurement.
5. Re-evaluate only the predeclared mechanical development cost/quality rule with its
   corrected valid cost evidence. If the winning stack changes, issue a new G4 selection
   and freeze record before final predictions/outcome exposure. Keep the prior decision
   and explanation. If final outcomes have already been exposed, stop this selection
   amendment and report the affected claim; do not retune on the same final outcomes.

A receipt must say which cost guard it can actually support. If deployment throughput can
be measured but paid-service or cold costs remain unknown, mark those parts unavailable
instead of declaring the entire cost guard passed. The common measurement definition and
the rule for cache reuse are fixed before the corrected comparison is inspected.

## 7. Exposure, metrics and publication boundaries

Training uses only the admitted train labels and the already frozen fitting procedure.
Public development labels support the diagnostic roles above. Public final ontologies and
query/candidate inventories permit transductive inference under the frozen policy, but
never supply optimization labels. Keep role-specific manifests and a read/exposure ledger;
do not bind private references to the worker, supervisor, logs or metrics API.

The **primary endpoint remains paired task-macro global class-equivalence F1** over
H0/H1/H2 baseline versus selected, using the respective organizer scoring conventions.
The H0 full label-free comparison is a prespecified supervision-control analysis. **Local
MRR is a secondary endpoint**, with per-task ranking measures, coverage and costs retained.
Do not change the primary endpoint because a secondary looks better. Known-incomplete
references retain their official raw convention; absent mappings are not independently
verified false positives. No adjusted semantic-precision claim is made without the required
independent annotation evidence.

Before organizer scoring, available final evidence is coverage, structural validity,
predictions, reconstruction, resource use and costs. Test F1, P/R, MRR, test candidate
recall and test error attribution are not locally available merely because compute is
finished. Do not substitute development metrics in a table labeled final results.

Freeze and checksum all primary output files before any official result is inspected.
Keep secondary output variants sealed from outcome-driven selection. Submission to the
respective organizer/CodaBench service is a distinct handoff from local computation and
requires its authorized upload action and current accepted formats. Queue completion is
not submission completion; submission completion is not organizer scoring completion.

Where source-level official paired outcomes can be obtained, apply the declared paired
bootstrap and multiplicity controls. With only aggregate leaderboard scores, report the
aggregate differences and inability to calculate the intended paired intervals; do not
invent source-level outcomes or borrow bounded development intervals for final claims.
Three tasks and one full-scale seed do not support an unseen-domain or full-population
stochastic-robustness claim. Bounded seed results remain in a separate evidence table.

## 8. Reuse, throughput and resource safeguards

The implementation compiles logical cells into shared computation keyed by all relevant
scientific inputs. Native ontologies, deterministic encodings, candidate-independent pair
features and compatible exact requests may be shared. Fitted heads, pool-dependent
features, listwise prompts, source routing, NIL and global extraction retain their actual
dependencies. Changed features/pools invalidate dependent fits. Shared work retains one
cost receipt with all consumers, not a zero-cost illusion or duplicated independent trials.

Reuse the existing public-input preparer, strict exporters and frozen-inference entry point;
integrate them rather than introduce a parallel execution stack. Reconcile the G4 frozen
runner's native import/options and public-population fixes with the implementation revision.
Freeze the actual runtime fitted-selector/head artifact paths and identities: a saved
`config.yaml` alone does not prove the fitted weights were injected. Admission tests must
establish same-pair artifact compatibility and that full inference reads neither reference
files nor training data; any necessary fitting happens earlier under its separate permitted
training role.

Enforce the separation required by each role: fitting/OOF and disjoint development
cohorts must retain their training exclusions. The complete global deployment population
will include some entities seen in training; that overlap is permitted under the explicit
frozen same-pair application contract and must not truncate the ontology. It does not make
those training examples held-out evidence or change the organizer's evaluation denominator.
Test both the permitted frozen-deployment overlap and rejected impermissible fitting or
evaluation overlap.

Follow [SCORING-THROUGHPUT-200.md](SCORING-THROUGHPUT-200.md) for implementation acceptance.
The workload forecast must count remaining unique source/pair batches, stochastic fits,
hosted tokens, I/O and exports rather than multiplying a historical whole-cell duration by
the number of logical cells. Measure sustained end-to-end progress from useful admitted
work, including writing and recovery overhead. No numerical optimization may change
candidate membership, thresholds, model, evidence context, scores beyond frozen tolerance,
ties or selected mappings merely to satisfy a speed target.

One RTX 4090 node remains the capacity assumption. Use one heavy accelerator worker,
bounded CPU overlap, chunked resume and quota-aware artifact retention. Preserve cumulative
spend, current approved alerts/pauses and request guards; this scientific amendment neither
raises the 25M family/200M campaign allowances nor resets prior consumption. Runtime
estimates are forecasts, not deadlines that kill in-progress work. The companion's
200-cold-unique-pairs/second implementation target is unmeasured, and does not include
hosted waiting, fitting or preparation by assumption. A roughly one-week to ten-day
planning target is conditional on measured end-to-end throughput and admitted unique work,
not a guaranteed completion date or a reason to silently shrink populations. The available
800 GiB dataset mount is durable capacity, not evidence of fast local scratch; extending
storage onto it requires the same quota/growth guards and explicit artifact placement.

## 9. Acceptance and handoff

Progress is tracked through distinct states:

1. **Specified:** this document and the planning inventory define the mixed-scale study.
2. **Implemented and fixture-verified:** code/tests establish coverage, reuse identities,
   seed handling, supervision isolation, diagnostics separation, export and recovery.
3. **Bound and ready:** exact cohort/fit manifests, final public inventories, corrected G4
   cost/selection freeze, measured forecast and source/resource admission are reviewed.
4. **Applied:** a new immutable runtime and queue amendment replace only unstarted final
   work. Existing G4/completed results and accounting retain their original identities.
5. **Compute complete:** all admitted full outputs, bounded diagnostics and required cost,
   reconstruction and coverage receipts exist; unresolved evidence is listed explicitly.
6. **Submitted and scored:** authorized submissions and organizer results are separately
   recorded, with claims limited to the actual available paired evidence.

The eventual handoff must include the logical-to-physical cell map; all input/cohort/query
hashes; supervised and label-free fit lineage; each seed's actual stochastic dependencies;
reuse and invalidation decisions; missing/inapplicable diagnostic bindings; retained E08/E14
deferrals; corrected G4 cost receipt and selection; whole-program cumulative spend; output
checksums and public validators; measured time/storage forecast; and the exact remaining
organizer/manual actions. No status advances solely because a document, helper test,
zero-call replay or queue descriptor exists.

**Current disposition: specified, not applied.** No launch, resubmission, live configuration
mutation, completed-run invalidation or spending-allowance change follows from this file.
