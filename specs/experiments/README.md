# Exact-OM experiment programme v2

**Design revision: 2026-09-10 (E04 benchmark-NIL scope). G0 passed 2026-09-18; the first bounded retrieval wave is submitted.**
See [PREPARATION-STATUS.md](PREPARATION-STATUS.md) for current code, input and operational evidence.
[FIRST-SCREEN.md](FIRST-SCREEN.md) records the submitted E05 scope and batch handoff.
[G0-VALIDATION.md](G0-VALIDATION.md) records the detached validation handoff;
[NATIVE-PREPROCESSING.md](NATIVE-PREPROCESSING.md) describes the native-only repair and bounded
model-free measurement workflow;
[the native optimization plan](../native-optimization/README.md) specifies behavior-preserving
upstream performance work and its correctness/resource gates;
[LABELS-AND-SUBMISSIONS.md](LABELS-AND-SUBMISSIONS.md) records the private-label boundary and
reference-free final submission formats.
This revision implements the accepted 2026-09-07 review and the user's single-node, 2–3 week
execution constraint. It supersedes the v1 exhaustive matrices, per-component reporting
requirements, whole-run invalidation rules, and implementation-only handoff. No v2 empirical
result is implied by the design specifications.

Read in order:

1. [RUN-PLAN.md](RUN-PLAN.md): scope, task roles, budgets, gates, selection, and final study.
2. [PREPARATION-STATUS.md](PREPARATION-STATUS.md): current implementation, validation, local
   inputs and remaining execution prerequisites. [IMPLEMENTATION-STATUS.md](IMPLEMENTATION-STATUS.md)
   preserves the historical pre-implementation audit.
3. [CHECKPOINT-RECOVERY.md](CHECKPOINT-RECOVERY.md): interruption, relocation, bug fixes,
   artifact compatibility, and statistical validity.
4. [IMPLEMENTATION-CLARIFICATIONS.md](IMPLEMENTATION-CLARIFICATIONS.md): precise method contracts.
5. The relevant E00–E26 files: bounded treatments and acceptance criteria.
6. [AGENT-HANDOFF.md](AGENT-HANDOFF.md): implement, validate, then execute the campaign.

[campaign-v2.yaml](campaign-v2.yaml) is the machine-readable **design blueprint**, not an
executable run declaration. `tools/prepare_experiment_campaign.py` binds it to inspected
readiness and materialized inputs, then the shared runner materializes strict run declarations. Existing exp/experiments/E*/exp.yaml matrices
are explicitly blocked legacy scaffolds pending that migration. The previous production
baseline R_0 remains historical evidence; never rewrite its source/configuration hashes.

## Authority and change control

The user's current instructions take precedence. Within this suite, RUN-PLAN governs scope,
budget, and data roles; CHECKPOINT-RECOVERY governs reuse; IMPLEMENTATION-CLARIFICATIONS governs
shared numerical semantics; an E-file governs its method-specific treatments. A conflict must
be resolved before affected execution, without blocking unrelated valid work. Status records
describe observed implementation and do not amend scientific requirements.

A scientific design version changes when its question, candidate selection rule, task roles,
primary endpoint, treatment, or reporting population changes. A runtime repair creates a new
execution revision and impact record, not automatically a new scientific experiment. Editorial
edits and unrelated source changes are provenance, not a reason to discard valid computation.
Both scientific designs and completed attempts remain immutable and recoverable.

## Objectives and bounded claims

The programme separates decision quality, entity kind, representation, relation semantics, and
supervision. Its core claim is competitive, auditable biomedical class-equivalence matching.
Property/instance/KG, typed-relation, transfer, and human-review claims require their own
eligible data and controls. An adapter or passing fixture test is not evidence of matching quality.
Exact-Repair remains a separate proposed system; no matching result establishes logical safety.

The experiment must identify where errors arise: candidate loss, in-pool ranking, acceptance/NIL,
exact anchors, target collisions, and relation typing. Report candidate recall before matching
quality. Separate ontology-wide NIL, no answer in the candidate pool, and insufficient evidence.

An exact explanation reconstructs recorded computation. It must cover pair fusion, fitted
ranking/acceptance, calibration, thresholds, exact-match decisions, and cardinality competitors.
Generated prose and pretrained knowledge are not ontology assertions or logical certificates.
Quality q is an evidence heuristic until experiments establish predictive reliability.

## Focused execution

Most development uses NCIT–DOID train/validation, initially 300 source groups and one seed.
Only promising or scientifically necessary contrasts expand to at most 1,000 development
sources. Small, explicitly development-only checks on contrasting pairs occur at two gates.
There is one frozen final study, normally on three class pairs, rather than a full benchmark
confirmation after each of 26 component screens.

Every applicable E00–E26 family receives a small focused screen. E11/E12/E23 use real
property/instance/KG tasks; E14 uses typed BioKG-Align data; E04 uses explicitly scoped historical Bio-LLM benchmark NIL.
Public missing data can be downloaded from official OAEI/subtrack sites. The local BioKG-Align
active release is now bound; its private test answers remain excluded. Resolve paths and capability metadata before assigning cases. Every family gets a status and reason; a genuine capability or budget
limit is explicit, and is not a negative empirical result.

The normal execution budget is 336 node-hours; a predeclared expanded profile permits 504.
These include retries and reserve. They are caps, not runtime forecasts. The first throughput
probe must show that the remaining frozen work fits the selected node. Implementation time before
readiness acceptance is separate. See RUN-PLAN for protected final-study and repair reserves.

## Shared experimental discipline

- Generated LLM rationales are disabled for G0 and all experiment runs. They are not
  evaluated by this programme. Production defaults and recorded decision evidence remain
  unchanged. A future rationale study must explicitly set `generate_rationales: true` in
  its experiment declaration or campaign lock (`--generate-rationales` for G0). The
  resolved configuration and design hashes record this choice; existing artifacts remain
  immutable.
- Keep one runner and one pipeline. Public stages remain screen and confirm; discovery,
  expansion, sentinel checks, and freeze are internal screen phases. Do not build another matcher.
- Execution mode is explicit: global alignment versus supplied-pool local ranking. Candidate
  files do not determine mode. A global run may use a frozen unlabeled candidate pool.
- Freeze an eligible source universe independently of predictions. In particular, source caps,
  NIL exclusions, or exact matches must not silently change denominators between arms.
- Generate train, development, and reporting pools separately. Fit on training labels; use
  development only for the declared selection/early-stop procedure; never tune on final labels.
- Record per-component in_pair_supervised, target_label_free, cross_pair_transfer, or
  oracle_diagnostic status. Explicit label-free ignores target training labels even if present.
- Missing reference entries are negatives only under an explicit completeness/negative policy.
  Use confirmed negatives or positive-unlabelled methods otherwise; unknown remains unknown.
- Hash materialized input bytes, fitted artifacts, model/tokenizer revisions, relevant numerical
  configuration, prompts, and component dependencies. A model name or mutable alias is not a lock.
- CPU work can overlap within one node; admit only one heavy accelerator job by default.
  Count encoding, training, LLM preparation/judgment/rationale, retries, and evaluation costs.
- Generative posthoc rationales are off by default unless explicitly requested. Preserve matching
  decisions, scores and exact numerical explanation traces; record costs of any opted-in narratives.
- Reuse semantically identical deterministic artifacts once; do not count copies as independent
  seeds. Repeat stochastic training/prompt-order procedures with paired seeds at confirmation.
- Retain removals and nulls. Screen results are developmental; only frozen final contrasts
  establish confirmatory claims. Never choose an E17 stack from earlier final-test wins.

## Completion

The implementing agent first delivers a tested NCIT–DOID vertical path with stage checkpoints,
repair/reuse planning, and complete source-level outputs. It then fills bounded experiment
capabilities by priority, updating readiness evidence. It must not claim implementation complete
from schemas or fail-closed stubs alone.

When assigned execution, the agent may run the approved bounded campaign and mechanically freeze
predeclared selections after gates pass; repeated permission is not required. All generative LLM roles use OpenRouter under the user's broad spending discretion; record
cost forecasts/actuals and honor phase request/token limits. Missing credentials or file paths
are specific input-resolution blocks, not a reason to declare the available datasets unavailable. A final release/default change is a separate action from running the research.
