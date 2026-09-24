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

G0-09 completed the repaired reporting and global interruption/resume/cache sequence, then
failed an overly strict byte comparison. All 300 mappings and metrics match; six scores differ
by at most `3.33e-16`, within the already specified pinned-GPU `1e-5` tolerance. The resumed
output and completed-cache replay are byte-identical. Attempt 09 took 2,111.19 seconds of new
active work and added no hosted requests.

G0-10 adopted the verified global stages and completed local300 successfully in 5,231.10
seconds. The node rebooted unexpectedly while publishing the final local completed-cache
replay, leaving nine completed measurements and all local extraction/evaluation artifacts
intact. A second unexpected reboot occurred without a G0 model job running. The available
journal excerpts and empty persistent crash directories do not identify the reset cause.

G0-11 launched at 15:32 UTC on 2026-09-18 to finish only that cache replay with
`--resume-interrupted-from`,
using `data/experiments-v2/runtime/native-candidate/bin/python`. The four validated native
code fingerprints match attempt 10. The continuation verifies saved measurements and artifacts,
blocks model workers and hosted clients, and requires input, extraction and evaluation cache
reuse together. It supplies no API key and does not repeat scoring. All 74 helper and 40
launcher/replay focused checks pass. Prior files remain intact, and all 4,405 requests,
2,726,711 tokens and USD 0.4454526 remain charged. Missing interrupted elapsed time stays
explicitly unknown; cumulative active time is reported as a lower bound.

G0-11 **passed at 15:36 UTC on 2026-09-18**, with exit code 0. Finalization took 247.70 seconds
of launcher wall time and added zero model workers, hosted requests or tokens. All four
replay checks passed; completed-cache mappings are byte-identical with zero score/metric
drift. The final report is `data/experiments-v2/g0-validation-11/report.json`. This completes
bounded operational validation; family-specific forecasts, unmeasured cases and the sampled
local reevaluation issue still require review. Rationales remain off. No long scientific
campaign has started or been admitted.
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

G0's bounded full-input memory/throughput probe, 300-source operational acceptance and
hosted capability checks passed. Wider admission still requires case-specific forecasts
and validation of each feature case's actual labels/artifacts. E04 now uses explicitly scoped
Bio-LLM benchmark NIL; verified ontology-wide NIL remains unavailable. K0 graph fitting needs confirmed negatives. BioKG typed positives are now bound;
Datalog consequence materialization and family-specific artifacts still need validation. The optional E14 bridge-reasoner executor remains unimplemented and explicitly
planned; it is outside the implemented graph/learned typing controls. E24's asymmetric arm
needs a directional typed diagnostic input, which D1 equivalence labels do not provide.
The planner keeps these separate from fixture readiness. See [NODE-SETUP.md](NODE-SETUP.md) for the observed CUDA stack and concurrency
limits. Final request/token and node-hour reserves are protected by runtime admission.

The full pre-admission declaration snapshot is
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

On 2026-09-18 the user authorized the first bounded retrieval wave after the sampled
evaluator repair. Revision 07 admits only E05's four label-free arms and imports verified
G0 operational acceptance without new E00 cells. At the user's request, pending batch job
**14217** was cancelled before execution. The wave started at 17:59 UTC inside interactive
allocation **14212**, as step **14212.2** in detached tmux session `exact-screen-01`, preserving
the shell and VS Code access. The conservative forecast is 8.03 hours, with zero hosted calls
and no wall-time cutoff. The wave failed at 18:48 UTC: all four cells reached an inherited
LLM relation-template path omitted from the zero-hosted-call preflight and received HTTP 401.
Rationales remained off; no E05 cell completed. The interactive allocation remains alive.
Later comparisons and final reporting remain gated.
See [FIRST-SCREEN.md](FIRST-SCREEN.md) for scope, validation, limits and monitoring commands.

Revision 08 corrects E05 relation-verbaliser admission and supplies the updated key explicitly.
The retry under `first-screen-02/` preserves scientific configurations and uses verified G0
templates, saved datasets and encoder vectors through the normal recovery checks. Prior failure
usage and the successful 135-token authentication probe remain charged. The revised conservative
allowance is 8.28 hours; rationales remain off. The retry started at 19:58 UTC in Slurm
step **14212.3**, detached in tmux `exact-screen-02` within the existing interactive allocation.
`launch-status.json` records current launcher state.

The 19:58 retry was interrupted by the node reboot around 20:24 UTC; step 14212.3 closed
at 20:25:45. It saved 512/5,842 baseline pairs, with every checkpoint output verified,
and made no new hosted requests. The user authorized continuation under allocation 14220.
`first-screen-03/` uses the same revision 08 campaign and combines that scored boundary with
the other arms' prepared caches. SQLite recovery operates on copies, preserving prior files.
See the current handoff in [FIRST-SCREEN.md](FIRST-SCREEN.md).
The continuation started at **22:39 UTC** as step **14220.1**, in detached tmux session
`exact-screen-03`. Source/native pins passed; the launcher preserves the interactive shell.

All four E05 cells completed by 00:30 UTC on 2026-09-19. Selection then stopped on missing
candidate recall: cached datasets omitted the development reference and raw retrieval ranks.
The node did not reboot. The committed repair restores scoped references, persists raw ranks
in future caches, and rejects incomplete audited caches before scoring. Sixty-three focused
tests pass. `first-screen-04/` launched at 10:56 UTC on 2026-09-19 for retrieval-only reconstruction plus saved-prediction
evaluation; exact candidate fingerprints must reproduce and no scientific settings change.
Original prediction artifacts and all previous charges remain intact. See FIRST-SCREEN.md
for the current interactive-step handoff.


## Published native runtime, 2026-09-20

The active interpreter for new experiment work is `.venv/bin/python`, with all four
native ontology packages installed from PyPI at 0.2.1. Published native wheel hashes
and installed-code fingerprints are recorded in
`data/experiments-v2/pypi-transition-01/`; 100 native integration tests passed with
zero skips. The previous `runtime/native-candidate` environment is retained only
for historical execution identities. Do not mutate frozen prior launch bindings or
reuse prepared artifacts across package identities without verified compatibility.

E02 finalized successfully and retains the hard-anchor baseline. D1 SNOMED native
preparation completed. FMA normalization repaired 90 annotation links across 54 classes
without changing the original file. The full XML structural audit and 17 regression tests
passed; this does not establish full native before/after semantic parity.
`data/experiments-v2/fma-reification-repair-01/receipt.json` binds the derivative and audit.
The target-only probe under `d1-native-probe-02/` started at **13:24:15 UTC** on
2026-09-20 in Slurm step **14234.14**, detached in tmux `exact-d1-native-02`, using PyPI
0.2.1, and completed successfully at **13:39:30 UTC** (914.85s, 3.32GiB peak RAM).
Full native loading, projection and 300 target feature queries passed. The separate
`fma-reification-repair-01/native-acceptance.json` binds this completion without changing the
original repair receipt. Production candidate/recovery and hierarchy qualification remain
for E09; the native-input gate in the earlier readiness audit is now satisfied.
[MECHANISM-SCREEN.md](MECHANISM-SCREEN.md) records the repair and validation limits.


## E09 continuation submitted, 2026-09-20

The E09 queue started at **18:56 UTC** in Slurm step **14234.16**, detached in
tmux `exact-d1-production-01` within the existing interactive allocation.
It uses isolated source commit `75c4d7a` and the published native 0.2.1 environment;
112 focused hierarchy/recovery/reporting checks passed before submission.

`data/experiments-v2/d1-production-probe-01/` first runs cold/warm D1 production
preparation, completed-cache replay, ancestor/sibling checkpoint measurements, and
a separate D0 current-control evaluation. If measured resource admission passes,
`prepared-campaign-11/advance.py` continues automatically into the four E09 D1
arms under `mechanism-screen-03/`. The D0 control does not enter D1 winner selection.
Completed E02/E05/E06/E26 decisions are retained. Hosted template verbalization is
preserved, with decision calls and rationales off. There is no wall-time deadline;
a cooperative 56 GiB process-tree RSS guard preserves checkpoints.

Monitoring and the submission receipt are in
`data/experiments-v2/prepared-campaign-11/HANDOFF.md` and
`data/experiments-v2/d1-production-probe-01/submission.json`.

On 2026-09-20 the node reboot interrupted step **14234.16** during ontology loading,
before a scored-pair checkpoint or prepared dataset was published. Its 325-second
Slurm interval was conservatively charged as interrupted, with zero new hosted
requests/tokens. Original launch bindings, logs and accounting are preserved under
`d1-production-probe-01/interrupted-14234-16/`. The unchanged experiment queue was
resubmitted at **19:10 UTC** as step **14250.1**, detached in the same tmux session
within the user's replacement interactive allocation. Only operational allocation
and submission bindings changed; standard recovery retains the artifact store and
shared caches while restarting unfinished cold preparation.


## E09 relocation to liseda-03, 2026-09-24

The same qualification/continuation queue resumed at **16:49 UTC** as Slurm step
**14372.1**, detached in `exact-d1-production-14372`, inside the existing interactive
allocation. The new node has an RTX 4090 (24 GiB), 125 GiB host RAM and six allocated
CPUs. Interactive shell step 14372.0 remains active. The original scientific source
snapshot `75c4d7a`, all four native 0.2.1 fingerprints, and immutable inputs verified.
Current uncommitted application work is excluded through the existing isolated snapshot.

The previous step 14250.1 was confirmed CANCELLED by Slurm, with a 480-second interval.
That interval is charged as interrupted, with zero additional hosted requests/tokens;
original launch files and run metadata are retained in
`d1-production-probe-01/interrupted-14250-1/`. Only input artifacts were committed;
there was no scored checkpoint or prepared D1 dataset, so cold preparation restarts.
The 22 completed E02/E05/E06/E26 cells and decisions are retained without rerunning them.

Qualification now measures the replacement node; the existing measured RAM/VRAM and
budget checks still gate automatic advancement into the four E09 treatments. GPU-specific
embedding-cache identities prevent reusing incompatible 5090 vectors. Rationales and
matching LLM calls remain off, hosted relation templates remain enabled, and private test
references remain excluded. There is no wall-time deadline; the 56 GiB cooperative RSS
guard remains in place. The new launcher log is
`data/experiments-v2/d1-production-probe-01/launcher-14372.log`; `status.json` in that
directory reports the active qualification phase. Relocation evidence is in
`data/experiments-v2/d1-relocation-14372/`.
