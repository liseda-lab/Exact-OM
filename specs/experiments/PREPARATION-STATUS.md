# Implementation preparation status

The implementation/preparation pass is complete, with the optional extensions and execution
prerequisites below still explicit. No long experimental campaign has been started, no final
labels have been used for selection, and no empirical hypothesis result is claimed.

The implementation uses the existing screen/confirm runner, source-grouped fitting,
per-stage content-addressed artifacts, immutable attempts, dependency-scoped repair plans,
and node/request/token admission. Bounded declarations cover E00–E26;
[implementation-evidence.yaml](implementation-evidence.yaml) links each family to inspected
code and tests. Optional treatments retain explicit prerequisites; a fixture test does not
authorize a real screen.

## Local preparation

All generated files below are intentionally outside Git, relative to the repository root.

| Input | Prepared location / status |
| --- | --- |
| Verified DOID import closure | `data/experiments-v2/ontology-normalization/doid-611355c44553/`; original v2026-05-30 plus 15 pinned imports; explicit declaration derivatives, strict complete closure passes |
| Public Bio-ML ontologies, split pools and references | `data/experiments-v2/bioml-primary/`; immutable upstream revision `c454644a334ab43754bc1070c0fb7fdd56a90a1d` |
| Locally converted SNOMED | `/home/pgcotovio/OAEI-Bio-ML-SNOMED-CT/ontologies/`; converted file bytes separately hashed |
| Confirmed training labels | Task-specific `prepared/train.confirmed.candidates.tsv`; [documented safe-label contract](BIOML-TRAINING-LABELS.md) |
| Bio-LLM 2024 benchmark NIL | `data/experiments-v2/biollm-nil/`; N0 source-stratified 60/20/20 research split, N1 separate reporting population; original ontology/pool hashes pinned |
| DISO submission inputs | `data/experiments-v2/diso/`; public ontology/pool data retained; no public answer labels |
| OAEI-KG | `data/experiments-v2/oaei_kg/`; Starwars–SWG development and Starwars–SWTOR heldout inputs materialized; byte identities pinned |
| Conference property case | `data/experiments-v2/property-case/case-lock.json`; P0 CMT–confOf, prospective P1 Conference–edas |
| Matched OWL/CSV representation views | P0 `source-matched-csv` and `target-matched-csv`; normalized evidence parity verified |
| BioKG-Align local release | `data/experiments-v2/biokg-release/v0.2.0-rc3-bbb08d188e48/`; active release pointer verified, public train/valid and test input bindings only; private test answers unopened |
| Baseline/model locks | `data/experiments-v2/locks/`; corrected `R_v2` with historical `R_0` preserved |
| Local case bindings | `data/experiments-v2/local-bindings.yaml`; paths, hashes, roles, negative policy and exposure reasons |
| Published comparator | `data/experiments-v2/comparator/logmap/`; official July 2021 release, selected before current quality outcomes |

P0 has six positive property mappings in total, including both property kinds. It supports
a small label-free feature check; it does not support a robust property-generalization or
supervised-property claim. Its research split and capability selection exposures are recorded.
K0 uses a source-grouped 60/20/20 development split of 1,096 reference-bearing instance
sources (657 train, 219 valid, 220 internal check). This scope does not establish full-KG
precision or natural-NIL performance. K0 and K1 share the Starwars source ontology; that
exposure is recorded. K1 gold values have not been opened.

E04 uses explicit historical benchmark NIL, not verified ontology-wide absence. N0's 60/20/20
source split is balanced by mapped/unmatched status. Its original NCIT and DOID inputs pass
strict loading. N1 retains all 100 SNOMED–FMA sources for reporting; FMA's native loader fails
on OWL axiom reification, and a 90-second strict Python check timed out without a result.
N1 remains unadmitted until loader compatibility is resolved. Frozen NIL transfer is implemented
through the existing immutable donor manifest, with no recipient fitting; actual reporting also
requires a selected N0 artifact and an explicit reporting declaration. `heldout_case` metadata
does not itself schedule a reporting run. These small public research splits do not support
blind official-test or nonbiomedical-generalization claims.

E16 initially binds the compatible D1 recipient. The prepared legacy OMIM–ORDO D2 case
remains unadmitted until a safe training-negative pool is available; the two-recipient
transfer mechanism is fixture-tested. No shared-versus-disjoint transfer result is claimed.

Public Bio-ML final reference labels are organizer-held. Per the user's updated instruction,
final inference will produce reference-free submission mappings after development selection.
Private test labels are not required for submission generation and must not enter optimization.
See [LABELS-AND-SUBMISSIONS.md](LABELS-AND-SUBMISSIONS.md) for label derivation and output contracts.

## Operational evidence

The four-source NCIT–DOID recovery proof is at
`data/experiments-v2/operational-recovery-proof-v5/proof.json`. Actual pinned MiniLM encoders
ran on CUDA. Interrupted plus resumed encoding performed 90 + 182 rows, equal to the 272-row
uninterrupted run; the completed replay encoded zero rows. The 16 confidence payloads and
mapping bytes matched after relocation. This is a real ontology slice with a limited evidence
neighborhood, not a full-graph throughput or 300-source vertical acceptance result.

The pinned LogMap executable completed a three-class synthetic operational check through
the shared evaluator in approximately 1.35 seconds, with approximately 219 MiB observed
peak memory. This validates the executable/adapter interface only. Raw RDF and normalized
predictions remain in `data/experiments-v2/comparator/operational-probe/`.

The original configured key file returned HTTP 401; those attempts remain recorded. With the
user-updated repository `api_key` file, `hosted-profile/probe-04/result.json` passes binary and
listwise first-token probability checks, completed-cache replay and relocated replay without
credentials (two completed requests, 56 tokens). Use `--api-key-file api_key` for the validation
tools; the frozen baseline's older credential locator is unchanged. No key value is logged.

Historical G0 attempt 03 failed in the native signature-count index; the facade now uses
public typed enumeration. Attempt 04 completed cold64, then failed registering a disabled
evaluation stage. That bookkeeping is fixed; attempt 05 adopted its verified outputs and
original timing, but its cumulative feasibility gate blocked further work. Those costs and
outcomes remain unchanged.

Native T4 passed [128 exact ordered NCIT–DOID feature rows](../native-optimization/IMPLEMENTATION.md#completed-t4-result).
G0-06 subsequently completed cold64 (848.92 s), warm64 (581.41 s), hosted20 (756.06 s) and
fit64 (1,749.54 s), then stopped `blocked_budget`. Its time/request/token forecasts exceeded
the declared limits; actual usage was 3,936.42 s, 522 hosted requests, 377,208 tokens and
USD 0.09027675. No global/local 300-source or interruption/replay validation ran. Its original
report, native identity and measurements remain under `data/experiments-v2/g0-validation-06/`.

On 2026-09-15, the user authorized G0-07 with no wall-time limit and finite headroom of
100,000 hosted requests/32 million tokens; 56 GiB RAM, two numerical CPU threads, one GPU
worker, STOP and unknown-delivery protections remain. Rationales are off by default for all
experiments unless explicitly requested; matching and numerical explanation traces are
unchanged. The non-rationale subtotal of G0-06 was 133 requests/19,779 tokens/USD 0.00401265,
including its one validation-only probe; this is not a newly measured rationale-free run.

G0-07 reused the verified cold/warm/fit probes and passed hosted20 with rationales off.
Global300 completed 5,842 scored candidate pairs and wrote 300 mappings, then failed in the
source-decision audit writer: renaming both dataset `Scores` and `Score` to `S_final` created
duplicate columns. This was an output-writing failure, not resource exhaustion. The failed
attempt, completed checkpoint, original measurements and historical hosted charges remain
under `data/experiments-v2/g0-validation-07/`.

The writer repair (`1ed45c3`) preserves the existing exact-score precedence and validates
entity aliases. All 23 focused export/replay tests pass. A label-free replay of the actual saved
G0 artifacts reproduced the failure and verified the repair: 300 source records, 6,000 audit
candidates (158 protected exact), unchanged final scores and the same 300 emitted mappings,
without model calls. Hashed evidence is in `data/experiments-v2/g0-audit-repair-01/`.

G0-08 recovered the verified global300 checkpoint, completed audit export and builtin
evaluation with zero new encoded texts/hosted requests, then failed in posthoc E00 attribution.
The public development reference uses `<=`/`>=`, whose canonical `<`/`>` meanings are already
specified; the diagnostic parser rejected those aliases. The evaluator-only repair uses the
shared relation normalizer, preserving identities, directionality, scoring and saved mappings.
Actual saved-output attribution/evaluation/replay checks now pass with zero score or metric
drift; evidence is in `data/experiments-v2/g0-postprocessing-repair-01/`.

Under the user's 2026-09-17 repair/rerun authorization, G0-09 recovery is configured with
`/tmp/exact-native-candidate/bin/python`, tmux `exact-g0-09`, output
`data/experiments-v2/g0-validation-09/`; configuration does not itself establish launch.
It adopts verified probes and the complete content-addressed global300 extraction artifact,
then reruns reporting without a model worker before the remaining checks. Prior attempts,
historical charges and original cold/warm/fit timings remain intact. Rationales remain off,
with no wall deadline and the unchanged 100,000-request/32-million-token allowance. G0 is
not yet complete, and the long scientific campaign remains unadmitted.
See [G0-VALIDATION.md](G0-VALIDATION.md) for monitor/STOP commands and retained history;
`status.json` and `report.json` remain authoritative for progress and completion.

## Validation and remaining readiness work

Focused regression suites cover numerical controls, grouped fitting, safe training labels,
source populations, explanation reconstruction, budget admission, stage recovery, representation
parity, and published-output evaluation. The strict documentation build and all five import
boundary contracts pass. The preparation baseline passed the complete tracked offline regression suite: **779 tests,
9 deselected**; subsequent changes use focused regression checks. The final recipe change also passes its focused eight-test preparation suite. Black, isort and flake8
pass for all 347 tracked Python files in the quality directories; mypy passes 218 source files.
Priority public API docstrings pass at 83.5%, and both runtime import guards pass. Unrelated
pre-existing untracked user scripts were excluded from these checks.

The Bio-LLM converter passes six focused tests; benchmark NIL integration passes 46, and
strict NIL transfer passes 13. Blueprint preparation passes seven tests, and the offline import
adapter plus existing IO sources pass 20. The declaration-repair helper passes seven tests.
These overlapping focused suites are not a new full-suite test count. Current scoped static
checks, strict documentation build and five import boundary contracts pass.

Historical seven-repeat fixture checks failed six timing thresholds. Repeating the same
command at unchanged pre-implementation revision
`655f599e714e13d592f702f326ca5a36f6b50b2f`, in the same environment, failed the same six thresholds.
Before the native preprocessing repair, recorded hierarchy medians were 0.11196 s / 0.11076 s
baseline and closure medians were 0.11837 s / 0.11703 s. Those profiles located substantial
hierarchy cost in native-storage index/canonicalization, while projection reported Python.
These measurements remain historical evidence; thresholds were not relaxed.

The subsequent repair requires native loading and encoded-native projection without scalar
fallback, lazy graph labels and typed axiom partitions. Native T4 now establishes the bounded
full-input preprocessing and exact feature-parity result documented above; it does not pass
G0's encoder, fitting, hosted or 300-source operational checks.
[NATIVE-PREPROCESSING.md](NATIVE-PREPROCESSING.md) records the model-free workflow and
compatibility limits. Old cold64 timings and cumulative budget charges remain unchanged.

Before admitting real screens, complete G0's prescribed full-input memory/throughput probe,
300-source vertical acceptance and cost forecast, validate the hosted credential/profile,
and validate each feature case's actual labels/artifacts. E04 now uses explicitly scoped
Bio-LLM benchmark NIL; verified ontology-wide NIL remains unavailable. K0 graph fitting needs confirmed negatives. BioKG typed positives are now bound;
Datalog consequence materialization and family-specific artifacts still need validation. The optional E14 bridge-reasoner executor remains unimplemented and explicitly
planned; it is outside the implemented graph/learned typing controls. E24's asymmetric arm
needs a directional typed diagnostic input, which D1 equivalence labels do not provide.
The planner keeps these separate from fixture readiness. See [NODE-SETUP.md](NODE-SETUP.md) for the observed CUDA stack and concurrency
limits. Final request/token and node-hour reserves are protected by runtime admission.

The final local declaration snapshot is
`data/experiments-v2/prepared-campaign-06/campaign.lock.yaml`, with stage declarations under
`runtime/declarations/screen/` and the read-only inventory in `readiness-plan.json`.
All 27 families are represented by 37 steps: 35 screen declarations and two final declarations.
The input-hash-verified screen plan has 142 arm rows, all awaiting admission; its missing
forecasts are not zero-cost estimates. Earlier preparation revisions remain unchanged. Revision 06 also snapshots its blueprint bytes,
so subsequent workspace design edits cannot alter it.
E20 uses one pinned MiniLM contrastive recipe
and one pinned BGE-reranker-large pair scorer (at most three epochs/1,000 steps, batch 4 ×
accumulation 8); actual fitting and resource admission remain pending. Conditional E05/E20
combinations are not scheduled in the initial screen.

## Preparation commands

These commands create declarations and plans; they do not run experimental cells. Use a new
output directory for each immutable preparation revision. These commands reuse the existing
verified baseline/model locks and local bindings.

```console
.venv/bin/python tools/prepare_experiment_campaign.py \
  --base-config data/experiments-v2/locks/R_v2.config.yaml \
  --bindings data/experiments-v2/local-bindings.yaml \
  --output data/experiments-v2/new-campaign
.venv/bin/python tools/run_experiment.py \
  --campaign data/experiments-v2/new-campaign/campaign.lock.yaml \
  --stage screen --materialize-only \
  --output-root data/experiments-v2/new-campaign/runtime
```

The current authorization includes bounded G0 execution and its explicit recovery budget
amendment. Starting the long screen or confirmation campaign remains outside this handoff.
