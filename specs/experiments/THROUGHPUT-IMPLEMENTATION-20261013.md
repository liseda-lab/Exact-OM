# G4 scoring optimization implementation and verification

Status: corrective cache fixes implemented and regression verified; conservative
execution passes bounded offline parity. Full throughput admission and checkpoint
migration remain open. Scientific execution is paused and requires a new user
approval, even though the passing checks below are complete.

This follow-up implements the October 12 review in
[the throughput contract](SCORING-THROUGHPUT-200.md). It preserves the earlier
implementation, completed scientific evidence and paid accounting. The 200 cold
unique pairs/second target and the one-day full-dataset objective remain
measurement requirements, not established results or wall-time kill limits.

## Execution hold

G4 exited after committing 9,728 of 12,047 examples in the active core D0 global
cell at 2026-10-12 23:46:38 UTC. Its wrapper recorded exit code 1 because the
checkpoint stop raises `KeyboardInterrupt`; this is an intentional stop, not an
incident authorizing automatic repair or restart. No forced signal was needed.
Allocation 14372, its interactive shell, and the supervisor were preserved.

The runtime `STOP` and supervisor `PAUSE` remain present. The latter explicitly
requires fresh user approval before G4 or any other scientific experiment may
restart. Isolated, unpaid optimization tests are the only authorized GPU work.
Old post-G4 queue dependencies are not rewritten as successful completion.

Operational receipts:

- `data/experiments-v2/g4-optimization-pause-20261012-01/`
- `data/experiments-v2/g4-optimization-implementation-20261012-01/`
- `data/experiments-v2/hourly-supervisor-01/PAUSE`

## Fitted-artifact validation

`LocalRetrievalArtifact` still hashes all admitted model and manifest bytes when
resolved. Later checks reuse that verified digest only while the complete file
inventory agrees: paths, device, inode, size, modification time and change time.
Replacements, additions, deletions, symbolic links and same-size edits invalidate
the inventory. A process change or explicit forced verification hashes again.
The content digest format and scientific identity are unchanged.

Scorer entry and return validate mandatory fitted inputs, including full cache
hits, hosted-enabled scoring, disabled caches and gradient-enabled calls. Nested
channel lookups share the outer validation boundary and identity scope. Scientific
identity failures propagate; only failures of disposable cache storage may fall
back to computation. Training-cache discovery follows the same distinction.

The actual G4 fitted cross-encoder measurement recorded one admission hash of
2,261,814,184 artifact bytes in 1.35665 seconds (2,261,858,876 process-level logical
read bytes including observation overhead), followed by 100 validations
in 0.17034 seconds total with no additional model-content reads. Its expected
content SHA remained unchanged. The tiny-real-artifact regression additionally
counts file opens and bytes across 1,024 hosted channel lookups; it does not
substitute a mocked fingerprint counter for file-read verification.

Evidence:
`data/experiments-v2/g4-optimization-implementation-20261012-01/retrieval-artifact-validation-measurement.json`.
This establishes removal of repeated hashing, not end-to-end throughput.

Encoder weight digests are also memoized separately from tokenizer/namespace
identities. The existing parameter/buffer signature and process ID control that
memo; changing a tokenizer or namespace rebuilds its full identity without copying
unchanged weights from the GPU again. Final identity payloads and digests are
unchanged. Tests count real tensor CPU/NumPy byte conversions and reconstruct the
legacy digest independently; mutations, replacements and process changes rehash,
while unversioned inference tensors continue to disable reuse. This correction is
independent of the default-off tokenizer normalization below. It was committed
after the frozen GPU comparison and performs no numerical scoring operation.

Resident and persistent encoder lookups now share the identity checked during
that same `_encode_with_cache` invocation. Public direct cache calls still
validate independently, and every subsequent encoder invocation checks again.
The private helper does not retain a trusted identity across calls. A bounded
CPU benchmark of 400 warm durable lookups counted 800 versus 400 identity checks
and median elapsed times of 0.7891 versus 0.4237 seconds. This isolates lookup
overhead and is not a GPU scoring speed claim. Regressions cover cold, durable,
partial and resident hits, mutation, training, unpinned models and verified legacy
row migration. Editing `scorer_common.py` changes its whole-file cache source SHA;
old rows require the existing reviewed source compatibility binding, not an
unconditional identity override. Numerical `_encode_texts`/`_pool` stay unchanged.
Evidence: `data/experiments-v2/g4-optimization-implementation-20261012-01/encoder-identity-reuse-cpu.json`.

## Batching and fitted-recipe checks

Scheduling, encoder regrouping, matrix kernels and reduction placement now have
independent controls. The conservative path retains original encoder call order,
padding, normalization shapes and device reductions; packed transfers combine
only completed values. Channel-cache hits bypass block preparation and a second
cache lookup. Direct helper/cache entry points enforce qualification admission.
A score tolerance cannot excuse changed facts, citations, prompts, routes or
rankings. Unqualified batching remains unavailable to scientific execution.

The first RTX 4090 comparison used an unfitted H0 prefix, not the actual G4
selector. Strict versus legacy matched all 198 ordinary records exactly in cold,
warm and replay modes: maximum absolute error zero, no discrete differences.
Cold scoring took 43.94 versus 43.74 seconds, about 4.5 pairs/second for both.
Native preparation took about 590 seconds separately. Cold feature gathering
took about 22.7 seconds and warm gathering less than a millisecond; the former
must not be extrapolated as a per-entity steady-state cost.

The nominal warm pass still encoded 3,810 texts in 884 calls. Investigation found
that tokenizer backend serialization includes mutable padding/truncation state,
which incorrectly changes the persistent encoder identity after ordinary calls.
The opt-in correction strips only those transient fields for the known encoder
implementation that explicitly sets them. Semantic components, special tokens,
sides, vocabulary, weights, precision and execution identity remain bound; custom
encoder implementations keep their backend settings. The larger GPU check below
**failed**: changing this identity changes which first-computed FP16 vectors are
reused. It remains default-off, qualification-only and scientifically inadmissible.
Stable serialization alone is insufficient to establish numerical equivalence.

Durable prefix receipts and their SHA inventory:
`data/experiments-v2/g4-optimization-implementation-20261012-01/strict-prefix-198-01/`.
They preserve the measured evidence and source binding, not caches or model
copies. This prefix predates the final admission-guard/tokenizer fixes and cannot
qualify the final fitted recipe or the 200 pairs/second target.

`tools/fitted_scoring_qualification.py` now prepares a label-free qualification
panel from committed G4 inference work. It verifies the saved fitted fusion,
selector, calibrator and retrieval bindings; excludes fitted training sources;
retains complete global candidate groups and exact anchors; and restores the
feature columns used by the real selector. The workload does not read private
test labels or retrain any model. The only supported extraction recipe is checked
explicitly rather than approximated.

The finite actual-fitted job uses frozen source `fe6b5803` in Slurm step
`14372.278`, with 600 candidates before exact prefiltering across three complete
chunks. It checks cold, warm and partial caches, followed by retained-native
causal ablations of the independent execution switches. API keys are cleared,
network/hosted generation is blocked, and deterministic synthetic replies can
record the prompt interface. This particular fitted panel made zero hosted or
synthetic calls, so its request comparison is vacuous; it does compare the
constructed evidence packets and brief text. Fitted selector scores, evidence, thresholds, ranks,
protected anchors and bounded-cohort mappings are compared. Synthetic replies
cannot establish hosted-model quality or replace historical paid predictions.

The source inventory and launch recipe are content-bound; a nonblocking owner
lock prevents duplicate workers. Caches use local `/tmp`, 1 GiB store caps,
physical tree guards and 64 GiB free-space reserves. Durable receipts are separate
from caches. The worker stops after this finite qualification; it cannot restart
scientific work.

Actual fitted receipts:
`data/experiments-v2/g4-optimization-implementation-20261012-01/fitted-g4-600-01/`.
The worker completed all requested numerical phases and causal ablations, then
exited with code 1 at the failed candidate parity gate. This was a detected
correctness failure, not a timeout or resource crash. Independent comparisons
cover all three modes, including those after the first rejected worker check.

| Path | Cold numerical seconds | Warm numerical seconds | Partial numerical seconds | Correctness |
| --- | ---: | ---: | ---: | --- |
| Legacy control with artifact correction | 85.96 | 30.60 | 15.31 | Reference |
| Conservative plus stable tokenizer identity | 74.33 | 16.26 | 10.79 | Failed all modes |
| Conservative, original tokenizer identity, natural diagnostics | 50.17 | 24.92 | 10.00 | All 600 ordinary records exactly equal in all modes |

Each mode returns 600 rows. Partial mode computes only 200 fresh pairs; its
throughput numerator must not be 600. The first two rows include their own initial
feature gathering. The causal third row reuses already materialized native and
feature state, so these are not three interchangeable cold-pipeline measurements.
Native preparation takes about 589–590 seconds separately; fitted
selector/extraction takes about two seconds per mode. Both primary arms already
contain the artifact fix, so their ratio does not measure the older repeated-hash
bottleneck. No measured path establishes 200 cold unique pairs/second.

The rejected primary candidate changed 2,284 numerical fields and 32 discrete
comparisons in cold mode, including evidence text/citation anchors and rankings;
warm and partial modes each changed 4,107 numerical and 61 discrete comparisons.
Fitted selector scores and pair ranks also changed, although final mapping
membership happened to agree. The tokenizer-only ablation exactly reproduces
this candidate in all three modes, isolating the cause. Exact ordinary comparison
of conservative batching against the legacy control instead has maximum absolute
error zero across 143,444 floating fields in each mode.

The final CPU-only regression in step `14372.280` also passed cold, warm and
partial modes. Before applying the saved selector/calibrator to conservative
outputs, it reconstructed the complete baseline fitted decisions and selector
evidence canonically byte-for-byte from retained public candidate metadata and
full-precision scalar outputs. Captured pre-selector scores were checked exactly.
The conservative replay then reproduced all 600 fitted rows and decisions
byte-for-byte, with identical calibration, mappings, policy and anchors; ordinary
selector evidence had zero numerical or discrete differences. It performed no
native processing, encoder execution, training or API calls. This is bounded
saved-input fitted parity, not a new scientific result or a throughput timing.
Receipt: `fitted-g4-600-01/cpu-fitted-replay/summary.json` beneath the evidence root.

Additional causal cold checks retain native/features and compare all 600 ordinary
records; they do not independently qualify the fitted selector or warm caches:

| Single change from conservative execution | Seconds | Ordinary parity |
| --- | ---: | --- |
| Staged scheduling only | 49.39 | Exact pass |
| `bmm` with legacy scheduling and single-job matrices | 50.12 | Exact pass; not staged multi-pair qualification |
| Encoder regrouping only | 49.58 | Failed; 99 discrete differences |
| Staged scheduling and encoder regrouping | 18.77 | Failed; 205 discrete differences |
| CPU arithmetic | 52.55 | Exact on this panel; earlier H0 failure remains disqualifying |
| Numerical cache disabled | 46.00 | Exact pass |

These controls show why the fastest path is not promoted. They also do not
support a database lockup explanation for this measured workload: disabling
numerical storage yields a modest improvement, while profiling identifies encoder
execution and repeated identity inspection as the principal remaining costs.
The instrumented cache-disabled run takes 72.26 seconds; its cumulative
`encoder_identity` time is 33.41 seconds across 8,263 calls, and encoder forward
time is 28.97 seconds across 2,091 calls. Profile times overlap and cannot be summed
as independent phases or substituted for the uninstrumented 46.00-second timing.

An additional finite encoder-only CUDA-graph feasibility test completed in step
`14372.279` after explicit user approval. All ten synthetic cases (five fixed
batch/sequence shapes for each installed encoder) matched stock eager execution
bitwise, including changed token IDs and attention masks: maximum error zero.
Context calls were 2.22–3.64 times faster and label calls 3.03–4.92 times faster;
graph capture cost about 0.18–0.22 seconds per shape. The pinned attention-mask
branch and SDPA/precision settings were preserved. Timing includes input copies
and output cloning, but excludes tokenization and the CPU mask predicate.

This probe is not integrated into the scorer. It establishes a possible next
optimization that preserves batch shapes, not ordinary/fitted pipeline parity,
200 pairs/second or safe residency for all natural shapes. A runtime implementation
would need model-local capture without a shared mutable helper, bounded graph
memory, model/configuration invalidation, unsupported-model fallback and whole
pipeline verification. No speculative graph runtime was installed. The probe
read no datasets and made no API calls. Its measured receipt is
`data/experiments-v2/g4-optimization-implementation-20261012-01/cuda-graph-fixed-branch-01/completion-summary.json`.

The ordinary-evidence comparator rejects nonfinite outputs and invalid
tolerances. Its explicit `--ordinary-only` mode removes only
`experiment_diagnostics.difference.controlled_perturbations` and
`experiment_diagnostics.difference_replay`; natural evidence and all other
diagnostics remain compared. Strict comparison remains the default.
Fitted comparison also verifies complete ordinary/selector record populations
against the hashed workload, including each exact-prefilter count. Identically
truncated results, extra unreported rows, missing or overwritten query identities,
and mappings outside the admitted cohort/anchors cannot establish parity.

## Verification and remaining gates

The integrated regression suite on `64d52bce` passed **426 tests** across 23 files,
covering artifact mutation/read counts, cache failures and hits, real tokenizer
semantics, shape-sensitive batching, fitted qualification, prompts/selector
traces, recovery and native projection. Its 11 warnings concern fixture entity
kinds, filesystem mtime fallback and an existing fork test. JUnit:
`data/experiments-v2/g4-optimization-implementation-20261012-01/cpu-regressions-final-64d52bce.xml`.
All five import architecture contracts passed. Earlier focused test counts are
overlapping evidence and are not added to this total.
Scoped mypy passed for the five changed artifact/runtime/cache/measurement
modules (`--follow-imports=silent`). Full-repository mypy still reports 25 errors
in nine other files, including the study interface, repair learning, extraction,
run store and campaign code. Those unrelated paths were not modified to make this
optimization check appear green. Full output is retained as
`data/experiments-v2/g4-optimization-implementation-20261012-01/mypy-full-fe6b5803.txt`.

The stable-tokenizer/encoder-regrouping candidate failed GPU correctness.
Conservative ordinary and fitted scoring pass the 600-pair panel; wider stress
coverage, final fitted-recipe admission and the full-pipeline forecast remain
separate gates. This is below the contract's 5,000-pair throughput qualification.
No fixture decisions may be reused as scientific predictions. No throughput
receipt removes the manual execution hold.

All three finite validation steps have exited. The CUDA-graph probe and final
CPU replay initially encountered automatic approval-review rejections; both ran
only after the user explicitly approved their bounded scope. Their approval
receipts are retained alongside the results. The scientific supervisor still
monitors under `PAUSE`, without authorization to dispatch or repair G4.

## Checkpoint continuation boundary

Do not start a generic `--resume-from` against current `dev`. The recovery
extraction identity binds source files, so it changes with these fixes. Current
training code also derives a different fitting directory from the older frozen
G4 runtime; simply copying fitting files can cause unnecessary rescoring or
refitting before inference restoration. Relocating fitted paths additionally
changes the primary model fingerprint.

A future approved G4 continuation needs a narrow compatibility import that
verifies the ordered 9,728-example prefix, all saved output/fitting checksums,
source and configuration deltas, and the explicit fitted-state admission. Existing
E14 checkpoint-import and account-copy mechanisms provide reusable pieces. The
old G4 helper rejects inference checkpoints and must not be run unchanged. Keep
original artifacts and the cumulative SQLite paid ledger; do not mark stopped
G4 complete or discard earlier charges. Preparing/running that migration is a
separate, still-open restart gate, not authority to release the current hold.

## Implementation commits

- `ad907a61`: mandatory artifact verification with inventory-based digest reuse.
- `31298438`, `58d8c693`: conservative batching, packed transfers, cache-hit
  planning and direct-path admission guards.
- `c9695134`: isolated stable tokenizer identity and cached-channel admission.
- `bc8e9bb0`: independent, versioned encoder weight digest reuse with identical
  resulting identities.
- `64d52bce`: reuse one checked identity within the resident/persistent lookup;
  retain independent public and subsequent-invocation validation.
- `0c3fd039`, `2f077248`: actual fitted offline qualification and execution-mode
  receipts.
- `14145b87`, `d4a8d999`, `dcfa57bc`, `fe6b5803`: numerical/discrete, fitted,
  prompt and complete-workload parity checks.
- `3869eb84`, `4d33c11e`, `ba2ab993`: scoped typing/style corrections.

Unrelated concurrent repair work was preserved. No commits were pushed.
