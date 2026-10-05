# E14 — Equivalence versus subsumption on typed BioKG data

**v2 specification, 2026-09-09. Implementation still required.**
This file replaces the v1 matrix for this family. [RUN-PLAN](RUN-PLAN.md),
[shared clarifications](IMPLEMENTATION-CLARIFICATIONS.md), and
[checkpoint recovery](CHECKPOINT-RECOVERY.md) are binding. A passing helper test does not
establish an executable experiment or a performance result.

## Existing implementation and missing work

Inspected baseline: 655f599e714e13d592f702f326ca5a36f6b50b2f.

**Already implemented:** Heuristic/graph relation primitives and typed writers exist; trained typer, full bridge integration and coherence audit are incomplete.

**Agent must implement:** Replace stale BioKG stub with the available data descriptor; grouped typed fitting, leave-query-bridge-out semantics, oracle-pair/full-pipeline evaluation and supported-profile reasoning audit.

**Inputs/bindings to resolve:** The user states BioKG-Align is available; resolve its actual paths/revision/typed split. Do not retain the old biokg_not_published deferral.
The user confirms the OAEI/BioKG data are available. Resolve paths, revisions and capabilities;
do not perpetuate an old unavailable flag without checking the supplied data.

## Focus and dependencies

Primary focused case: **T0: available BioKG-Align pair with =,<,> labels; distinct T1 final**.
Resource envelope: **extensions** in RUN-PLAN.
Prerequisites/consumed outputs: **E00, E13, typed_pool_freeze**. A pool-freeze or selected-head dependency
accepts the declared baseline output when no candidate wins; optional research must not deadlock
other families. Kind-specific pool freezes do not change the already-frozen class pool.

Start with 300 development source groups (all eligible if fewer), seed 17, except a stated smaller LLM limit. Expand at most two non-control survivors to 1,000 nested development groups. Every applicable family receives a focused initial screen; widen cases or models only at scheduled gates. Eligibility/power is recorded per kind/relation.

## Question, treatments, and implementation contract

Choose one actual typed pair for broad method screening. Run oracle-pair typing separately from end-to-end pair detection. No calibration or rule may use T1 outcomes. Exclude the queried mapping as its own bridge. Cycles/SCC collapse and logical unsatisfiability have different metrics.

At most **6 distinct treatment configurations/cells as specified below** before any explicitly declared source expansion. This is a bounded sequential design, not a Cartesian product. Shared deterministic controls are computed once.

- all_equivalent: required lower control.
- hierarchy_heuristic: current control.
- graph_entailment: relation paths conditional on declared anchors.
- learned_three_way: bounded multinomial relation head.
- semantic_then_learned: selected hybrid with explicit conflict abstention.
- bridge_parity: optional supported-OWL reasoner comparison.

All generative roles use OpenRouter. Local non-generative encoders/heads use the single RTX 5090.
Separate target-label-free, in-pair supervised and transferred results. A named diagnostic may
use development reference information only under its explicit oracle/diagnostic role.

## Validation and selection

Primary outcome/guard: **Relation-macro F1 with oracle-pair and full-pipeline results separately.**
Use the family rule plus RUN-PLAN's frozen selection, practical-effect, reconstruction and cost
criteria. A screen chooses what to evaluate next; it does not establish a reporting-set claim.
Report all controls, negative results, corrections/harms where relevant, and inapplicable or
budget-deferred cells. Never suppress a difficult kind or source group from the denominator.

Broader validation happens on the designated development sentinel after a promising focused
screen, then only in E17's frozen final panel for the claims selected at G4. Do not run a full
OAEI confirmation for every treatment. A feature-specific claim needs its matching held-out
case; NCIT–DOID cannot substitute for property, instance, natural-NIL or typed-relation labels.
No individual experiment uses final outcomes to qualify its component for E17.

## Acceptance and recovery

- < means source-subsumed-by-target consistently in every adapter/writer.
- A mapping cannot be independently justified solely by inserting itself.
- All relation labels have real counts/provenance; unresolved label completeness remains explicit.
- Reasoner timeout/unsupported profile is unknown, not safe or proven equivalent.
- The optional native known-pair comparison follows the [approved public-input amendment](PUBLIC-INPUT-AMENDMENT-20261003.md).
  Keep original NCIT/DOID sources unchanged; a separately prepared reasoning input may remove
  only the verified unsupported datatype declarations and annotation-property ranges while
  preserving full logical content and import identity. Native XML preparation must fail closed
  on other datatype uses. Small correctness fixtures and fresh full strict native admission
  are prerequisites for comparison execution; no native package policy change is required.
  Report prepared-input reasoning distinctly from admission of the unmodified original.

### Published reasoner integration — 2026-10-04

The user authorized installation of published pyHermiT after merging its resource-limit PR.
Version 0.2.2 is installed in a new isolated E14 environment; existing workers and packages
remain unchanged. Its explicit `max_native_symbol_index_bytes` must be passed to admission
and bridge execution: omission still means 64 MiB. Exact now binds this setting to cache and
admission identities, checks configuration compatibility before native loading, and preserves
all resource settings when applying loader overrides.

**Full admission remains blocked.** The published release omits the earlier compiler work
and memory fixes from the E14 candidate: `max_compile_work` is unsupported, with independent
2-billion-work and 512 MiB compiler/profile defaults. The candidate receipt records prior
NCIT failure at the work limit. Do not drop that option or repeat full admission with a known
incompatible release. A corrective upstream PR must preserve published default behavior,
strict reasoning and existing constructor/cache/pickle compatibility. After its publication,
prepare a fresh bound admission and dependent bridge comparison with identical input and
resource identities. Retain the completed main E14 treatments and all failed-attempt costs.

Local integration verification: 99 focused tests pass with published 0.2.2 and 99 with the
unchanged main 0.2.1 environment. This establishes integration/fixture behavior, not full
NCIT admission or a bridge result. Evidence is in
`data/experiments-v2/e14-pyhermit-20261004-01/`.
Another 41 adjacent tests pass. Tiny real native admission and bridge fixtures fail at a
one-byte index limit and succeed at an explicit 128 MiB allowance with strict reasoning.

The corrective upstream branch `fix/native-compilation-resource-limits` is pushed at
`e074de753f49364a4ce0185ede8b5aeeaa0fb79d`. Its native/compatibility checks and a tiny Exact
adapter/bridge fixture using the combined intended allowances pass. GitHub rejected PR
creation with an integration-permission 403; the owner must open/review the comparison and
publish a corrected version after CI. Hosted CI and Distribution matrix started on push.
The supervisor registry records this publication prerequisite; it retains the old comparison
descriptor as ineligible and preserves the completed main experiment and failed history.
See the local `HANDOFF.md` and `integration.json` in the evidence directory for the branch,
verification report, CI links, exact owner action and next admission/queue steps.

### Publication-readiness review — 2026-10-05

PR [#2](https://github.com/OAEI-ML/pyHermiT/pull/2) is now open. Two independent source
reviews found the known memory, work and index repairs together, with no further proven
package implementation defect in the E14 path. Both hosted workflows passed `e074de7`,
including six native platforms and eight ABI3 checks. A verification gap was then closed
on the same PR in `d8d7a2f`: installed native wheel suites now include 15 focused resource,
index and deferred-identity cases. Those cases and 10 runner-contract tests pass locally;
the updated head must pass hosted checks separately. No new version was published.

Hold publication pending full-input candidate validation. First install a fresh candidate
wheel and run a tiny strict native admission/bridge with the intended resource settings:
config construction and the generic native probe alone cannot detect a stale extension's
missing optional capabilities. Then validate full prepared NCIT, original DOID/imports and
the actual training-anchor composed world. Separate admission does not prove combined-world
consistency, endpoint satisfiability or entailment. Preserve all host/storage protections;
compiler byte accounting is not a total-RSS limit. The historical recipe/frozen runtime
must be replaced because it lacks the explicit symbol-index allowance.

This review did not queue or start full-input validation. It needs the heavy lane after E21;
active experiments and completed main E14 results remain unchanged. Detailed review and
continuation: `data/experiments-v2/e14-pr2-review-20261005-01/HANDOFF.md` and `review.json`.

### User-authorized candidate-first validation — 2026-10-05

The user now authorizes queuing PR #2 candidate `d8d7a2f` ahead of all other pending work,
after active `E21-binary-recovery-01` finishes. Install a real optimized candidate wheel in
a fresh isolated environment; do not use a source-path override or change production
environments. Before full inputs, require the installed-wheel resource/identity cases and
a tiny strict native admission/bridge with the combined intended memory, work and index
allowances. Preserve native package identities and the approved metadata-only preparation.

The planned chain is `E14-pr2-native-admission-20261005-01` (full prepared NCIT and original
DOID/imports), then `E14-pr2-bridge-validation-20261005-01` (the existing full 300-source,
seed-17 diagnostic with all public training anchors and per-query checkpoints). Use the
existing prepared diagnostic runner, public train/valid roles, no hosted calls or rationales,
and detached numeric Slurm steps. Retain host/storage guards, no imposed time deadline and
the latest cumulative accounting, copied only at dispatch.

The candidate bridge recipe sets `require_native_query_validation: true`. Before comparison
completion, its exact query inventory must contain successful composed-world consistency,
endpoint-satisfiability checks and both directional answers for every satisfiable pair;
at least one pair must exercise directional queries. `not_entailed` and genuinely
unsatisfiable endpoints remain valid observations, not fabricated relations. Identity-checked
completed query checkpoints satisfy this requirement on resume without compiling again.
Missing/unsupported query evidence, inconsistent worlds or an empty query inventory produce
a failed `native-validation.json` and keep the gate closed. Ordinary diagnostics retain
their existing reporting behavior when this option is absent or false.

Append the bridge-validation dependency to the 11 other pending batches, preserving their
original dependencies; leave held `E14-bridge-run-once-followup` unchanged. Successful bridge
completion releases ordinary queue eligibility automatically. Failure keeps the gate in
place for bounded repair or an explicit user decision; do not bypass it as optional work.
This scoped authorization supersedes the earlier unpublished-package and independent-work
instructions only for this candidate validation. It authorizes no merge, package publication,
production integration or automatic promotion of candidate outputs into scientific results.

Operational root: `data/experiments-v2/e14-pr2-validation-20261005-01/`.
**Preparation only until its `publication.json` records actual queue publication.** That
receipt establishes scheduling, not successful admission, completed validation or release
readiness; those require their own execution evidence.

Durable boundaries: **Typed train features/heads, bridge queries, typed predictions.**
All changed inputs/semantics invalidate their consuming descendants; preserve valid upstream
artifacts. Store completed source/request/fold IDs and attempt lineage. Tests must demonstrate
this family's checkpoint/repair boundary, not merely mirror a formula. Mark screen-ready and
confirm-ready separately in the runtime readiness ledger, with evidence, once these checks pass.

## Deliverable

Produce a result record with actual treatment/configuration, case/role, supervision, artifact
IDs, source counts, metrics/cost, controls, uncertainty, decision and reason. A legitimate null,
removal, or inapplicability is a deliverable; an unimplemented arm is not an empirical null.
Resolve this family's question into numbered research questions and a primary endpoint in the
executable design before its screen; answer each as supported, not supported or inconclusive
with evidence. RUN-PLAN section 7 governs incomplete references and claim limitations.
