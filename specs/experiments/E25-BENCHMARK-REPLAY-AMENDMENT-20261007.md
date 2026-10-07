# E25 cached benchmark replay amendment

Approved by the user on 2026-10-07. This changes the scope of two pending E25 development
diagnostics. Implementation verification and queue publication are recorded below;
scientific results require the new workers' completion reports.

## Shared population and evidence

Use the existing forced-response producer's 300 development sources, including all unjudged
sources, with up to 200 actually judged sources. Freeze its candidate pairs, base scores,
displayed intervention support, observed response probabilities, threshold, acceptance and
cardinality rules. Every arm evaluates the same population; uninvoked candidates retain their
base behavior. Do not enlarge the judged support or fabricate missing responses.

The earlier main E25 decision-off result covers 200 sources. It is not a valid paired control
for this 300-source replay. Construct the off replay from the same forced producer's frozen
pre-LLM evidence and retain all 300 sources in the denominator.
The off arm uses a replay artifact with zero selected interventions, so it validates the
original base scores and uncertainties while leaving them unchanged. A successful replay
must also retain the complete source and candidate inventory, including protected exact
pairs and unjudged or empty sources; per-batch score checks alone are insufficient.

## Approved comparisons

1. **Benchmark-relative oracles:** compare paired decision-off, routing from observed cached
   responses, and hypothetical reference-perfect answers on the same fixed intervention
   support and budget. The declared development reference defines benchmark agreement for
   this diagnostic. An absent reference pair may count as a benchmark nonmatch here, but
   does not become a confirmed semantic negative or ontology-NIL label. Report benchmark
   corrections/harms and actual downstream extraction separately from source-local routing
   targets. A reference-perfect intervention is hypothetical; neither routing construction
   establishes a global optimum or completeness of the reference.
2. **Binary-judge trust:** compare only the shipped `beta*U` mixture and constant weight `0.5`
   using identical observed responses and downstream rules. `source_first` requires a real
   comparative choice, which this binary judge does not provide. Record it as inapplicable;
   do not invent a source choice, convert it to an empirical null or silently substitute a
   different judge.

The historical true-label oracle remains available and remains the default for genuinely
complete references or explicitly confirmed candidate labels. Benchmark-relative replay is
an explicit alternative scope, recorded in its artifacts and reports. It must not silently
weaken the default label checks or reuse an artifact with a different label scope.

## Boundaries and recovery

Both comparisons make **zero new hosted calls**, including setup, teacher, brief and rationale
roles. They require no new data, manual annotation, private gold or generated rationales.
They use development labels only as declared diagnostic outcomes, never as training targets,
deployable routing identities or final product-selection evidence. This amendment changes no
other experiment's scientific metrics and makes no new assertion about training-pool usage.

Preserve the original failed run, original receipts and cumulative charges. Retain valid
cached responses and checkpoints with their provenance; use new, scope-bound replay artifacts
and an explicit lineage to the frozen producer. Changed replay semantics invalidate consuming
descendants, not the original evidence. Tests must verify population/support identity,
scope separation, no-call execution and rejection of incompatible cached artifacts before
the runtime readiness ledger or queue marks the amended comparisons ready.

## Implementation and verification

The isolated implementation adds an explicit `benchmark_reference` outcome scope to the
oracle builder and runtime validator; the default remains `verified_labels`. The E25
materializer binds the producer configuration, development reference, evaluator and full
source universe, rejects task overrides that change frozen scientific settings, and creates
the paired off arm from that same producer. Hypothetical probabilities use only the observed
intervention support. A shared hosted cache-only guard permits exact saved-response replay
and rejects cache misses before planning or sending a new paid request.

The final worker suite passed **175 tests in 74.04 seconds**, covering replay scope, population
and support preservation, configuration bindings, runtime validation, legacy campaign
signatures, completion checks, accounting and hosted cache-only behavior. After cherry-picking
onto `dev`, 57 focused integration checks passed (an earlier integration pass covered 86).
Mypy passed on the five changed library modules; Flake8 and whitespace checks passed.
Execution source is frozen at `3de671a044a5d7127f9aeebd483fc90b7b87e631`, preserving the
producer-compatible worker lineage. Main retains its subsequent unrelated changes.
This records implementation verification, not completion of the amended experiments.

The benchmark routing target is source-local and is not a global-F1 optimum. This scoped
implementation requires the frozen builtin evaluator and fails closed on ignored reference
pairs or training/null-reference pairs within the replay source population. The binary
judge's `source_first` arm remains explicitly inapplicable; the trust comparison contains
only the shipped and constant-weight arms.

## Operational handoff

On 2026-10-07 the existing supervisor queue received
`E25-oracles-benchmark-reference-20261007-01` and `E25-trust-binary-replay-20261007-01`.
Both approval holds are resolved. They wait for available resources behind E13 native
preparation in allocation `14372`; E13 and the supervisor were not stopped or restarted.
Queue publication does not mean the five scientific cells have run.

The final actual-data preflight verified every arm against the original **300 sources and
6,000 candidate pairs** (5,842 scored and 158 protected exact pairs). It also checked the
effective configurations and every descriptor binding. No new hosted calls were made.
The off control has zero interventions; the four cached-response treatments preserve their
200-source intervention allowance. `source_first` remains inapplicable.

Receipts are under `data/experiments-v2/e25-benchmark-amendment-20261007-01/`:
`verification-v2.json`, `prepared-v2/prepared.json`, `prepared-v2/world-validation.json`,
`publication.json` and `HANDOFF.md`. Original failed runs, their costs and the earlier metadata
preflight failure remain preserved. Workers resolve the latest cumulative account at dispatch.
Existing storage and spending controls remain unchanged. The ordinary pipeline may perform
local scoring when shared caches cannot supply it; the zero-new-call constraint is enforced
at the hosted transport boundary, independently of cache availability.

## Subsequent output-only recovery

Both approved comparisons finished their primary and selector scoring but failed in
candidate-audit hashing when the replay policy contained a `Path`. Use the same JSON-safe
policy already written to the source audit; no scoring or selection semantics change.
All five latest saved checkpoints contain 5,842 primary pairs and complete selector rows.
The strict importer preserves numerical rows and relocates only identical gate-file paths
and dependent checkpoint fingerprints. Its paired population and compatibility checks remain
in force. See the [repair and deployment handoff](E25-AUDIT-RECOVERY-20261007.md).
