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
| Public Bio-ML ontologies, split pools and references | `data/experiments-v2/bioml-primary/`; immutable upstream revision `c454644a334ab43754bc1070c0fb7fdd56a90a1d` |
| Locally converted SNOMED | `/home/pgcotovio/OAEI-Bio-ML-SNOMED-CT/ontologies/`; converted file bytes separately hashed |
| Confirmed training labels | Task-specific `prepared/train.confirmed.candidates.tsv`; [documented safe-label contract](BIOML-TRAINING-LABELS.md) |
| DISO natural-NIL candidate cases | `data/experiments-v2/diso/`; ontology/pool data bound, expert NIL truth still unresolved |
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

A bounded G0 validation job was launched separately after preparation.
See [G0-VALIDATION.md](G0-VALIDATION.md) for the detached monitor, stop command, limits and
retained attempt history. Its `status.json`/`report.json` are authoritative for completion.

## Validation and remaining readiness work

Focused regression suites cover numerical controls, grouped fitting, safe training labels,
source populations, explanation reconstruction, budget admission, stage recovery, representation
parity, and published-output evaluation. The strict documentation build and all five import
boundary contracts pass. The complete tracked offline regression suite passes: **779 tests,
9 deselected**. The final recipe change also passes its focused eight-test preparation suite. Black, isort and flake8
pass for all 347 tracked Python files in the quality directories; mypy passes 218 source files.
Priority public API docstrings pass at 83.5%, and both runtime import guards pass. Unrelated
pre-existing untracked user scripts were excluded from these checks.

The seven-repeat fixture benchmark check fails six historical timing thresholds. Repeating the
same command at unchanged pre-implementation revision
`655f599e714e13d592f702f326ca5a36f6b50b2f`, in the same environment, fails the same six thresholds.
For example hierarchy medians are 0.11196 s current / 0.11076 s baseline and closure medians
0.11837 s / 0.11703 s. Profiling locates most hierarchy time in the installed pyowl native-storage
index/canonicalization path; the projector also reports its Python backend. Thresholds were not
relaxed. Full-ontology resource estimates must use this node's measured backend behavior.

Before admitting real screens, complete G0's prescribed full-input memory/throughput probe,
300-source vertical acceptance and cost forecast, validate the hosted credential/profile,
and resolve each feature case's actual labels/artifacts. Natural-NIL cases need expert source
status labels; K0 graph fitting needs confirmed negatives. BioKG typed positives are now bound;
Datalog consequence materialization and family-specific artifacts still need validation. The optional E14 bridge-reasoner executor remains unimplemented and explicitly
planned; it is outside the implemented graph/learned typing controls. E24's asymmetric arm
needs a directional typed diagnostic input, which D1 equivalence labels do not provide.
The planner keeps these separate from fixture readiness. See [NODE-SETUP.md](NODE-SETUP.md) for the observed CUDA stack and concurrency
limits. Final request/token and node-hour reserves are protected by runtime admission.

The final local declaration snapshot is
`data/experiments-v2/prepared-campaign-05/campaign.lock.yaml`, with stage declarations under
`runtime/declarations/screen/` and the read-only inventory in `readiness-plan.json`.
All 27 families are represented by 37 steps: 35 screen declarations and two final declarations.
The input-hash-verified screen plan has 142 arm rows, all awaiting admission; its missing
forecasts are not zero-cost estimates. Earlier preparation revisions remain immutable.
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

The current task authorizes implementation and preparation only. Starting the long screen or
confirmation campaign is outside this handoff.
