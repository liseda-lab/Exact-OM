# E24 — Contrastive-Channel Degeneracy: Why the Only Penalising Signal Never Fires

**Motivation** (review-response WP1/WP6, `Paper/specs/review-response/`): the contrastive channel
is the system's only mechanism able to push a pair below the neutrality pivot, and it is inert on
every task measured so far.

| Measurement | OMIM–ORDO (300 src) | SNOMED–FMA Body (300 src) |
|---|---:|---:|
| pairs with `sigma_diff < 1e-6` | 100.0% | 96.7% |
| pairs with `q_diff > 0` | — | 58.2% |
| of those, still at `\|s_diff − τ\| < 1e-6` | — | **94.4%** |
| ΔMRR from removing the channel entirely | 0.000 | not measured |

FMA has the densest relational graph in the benchmark, so this is not simply "sparse pools". The
channel *finds evidence* on the majority of FMA pairs and still returns exactly the neutral score,
which the informative deviation then multiplies to zero. That is a degeneracy in the confidence
map $s_{\mathrm{diff}} = 1 - \tfrac12(c_x + c_y)$, not a tuning problem — a channel whose output is
a constant carries no information regardless of its weight.

This experiment is diagnostic first and corrective second. It is a prerequisite for any claim that
the system can penalise conflicting evidence, and therefore for the precision story on
same-family confounders.

## Research questions

- **RQ24.1**: Why does $s_{\mathrm{diff}}$ concentrate at $\tau$? Decompose the pivot mass into
  (a) empty pool on one or both sides, (b) $c_x + c_y \approx 1$ arising from the normalisation,
  (c) explicit neutral fallback in code, and (d) genuinely balanced unsupported mass.
- **RQ24.2**: On pairs where the channel is active, does $s_{\mathrm{diff}}$ separate correct from
  incorrect pairs at all, measured per source rather than over the pooled table?
- **RQ24.3**: Does an unnormalised or asymmetric formulation of unsupported mass produce a
  non-degenerate distribution without inverting the channel's meaning?
- **RQ24.4**: Does a working contrastive channel improve precision on same-family confounders
  specifically, which is the case it was designed for and the case the paper's running example
  illustrates?
- **RQ24.5**: Does the relational-similarity channel share the same failure mode? It draws on the
  same object-relation pools and is inactive on 100% (OMIM) and 96.7% (FMA) of pairs.

**Hypotheses**: (a) the dominant pivot mass is structural — empty-pool fallback plus a
normalisation that forces $c_x + c_y \to 1$ whenever both sides are comparably unsupported —
rather than genuine balance; (b) an unnormalised formulation, scoring absolute unsupported IC mass
against a pool-size-aware baseline, yields a distribution with usable spread; (c) the corrected
channel improves precision on confounder-heavy sources by 1–3 F1 points on identifier-poor tasks,
with negligible effect where lexical evidence already dominates; (d) relational similarity fails
for a related but distinct reason (pool intersection is empty rather than balanced) and needs its
own fix.

## Change

1. **Diagnostic instrumentation** (no behaviour change): extend the channel dump with
   `n_triples_src`, `n_triples_tgt`, `unsupported_mass_src`, `unsupported_mass_tgt`, `c_x`, `c_y`,
   and a categorical `diff_pivot_reason` recording which of RQ24.1's branches produced the value.
   Default off, as with the existing `channel_dump`.
2. **Alternative confidence maps**, config-selected under `channels.diff.formulation`:
   - `normalised` (current): $1 - \tfrac12(c_x + c_y)$;
   - `absolute`: unsupported IC mass scaled by a pool-size baseline, so a pair with two well-matched
     triples is distinguishable from a pair with none;
   - `asymmetric`: penalise unsupported mass on the *target* side only, on the argument that a
     target asserting facts absent from the source is the confounder signature;
   - `off`: explicit control.
3. **Empty-pool handling** made explicit rather than implicit: a channel with no evidence must
   report inactive, not neutral, so that "no evidence" and "balanced evidence" stop being the same
   number.

## Arms & sweep

| Arm | Purpose |
|---|---|
| `R_n` baseline | current normalised formulation |
| `diff_off` | control; isolates what the channel currently contributes (expected: nothing) |
| `diff_absolute` | primary corrective arm |
| `diff_asymmetric` | secondary corrective arm |
| `diff_absolute` × `sim_fix` | factorial with RQ24.5's relational fix, only if both are individually non-null |

Diagnostic stage runs on the frozen 300-source OMIM–ORDO and SNOMED–FMA subsets. Confirmatory
stage runs the full eligible task set with ≥3 seeds.

## Validation

Eligible tasks: all five Bio-ML pairs plus Anatomy, since the hypothesis concerns relational
density and Anatomy is relationally rich with weak lexical overlap. Entity kind: classes.
Supervision label: `target_label_free` — no arm here reads target labels.

Primary metric is macro F1 on the global task. **A confounder-restricted secondary slice is
mandatory and pre-registered**: sources whose candidate pool contains at least one non-reference
target sharing a label token with the reference target. This is the population the channel exists
to serve, and a whole-task metric will dilute the effect to invisibility.

Report alongside: the activity fraction per arm (a corrective arm that is still inert on 90% of
pairs has not been fixed), the $s_{\mathrm{diff}}$ distribution, and per-source discrimination AUC.

## Promotion decision rule

Primary comparison: `diff_absolute` against `R_n`, endpoint macro F1, paired bootstrap CI
excluding zero on ≥3 seeds. **Pre-registered override**: because the channel is currently inert,
criterion 2's task-regression bound is applied to the confounder slice as well as globally; an arm
that improves the confounder slice while holding whole-task F1 within non-inferiority
($-0.3$ F1) is promotion-eligible, since the channel's purpose is precision on a minority of hard
sources rather than average-case gain.

If every corrective arm is null, the deliverable is the RQ24.1 decomposition plus a documented
decision either to remove the channel or to retain it with an honest statement of when it fires.
Removing a channel that demonstrably contributes nothing is a legitimate promotion outcome and
simplifies the explanation surface.

## Effort & risks

Size: M. Diagnostic stage is cheap (re-analysis plus one instrumented run per subset). Corrective
arms are one config flag each.

Risks: (a) the asymmetric formulation could encode a directional assumption that fails on
subsumption-typed references — gate it behind E14's relation typing before promotion; (b) making
empty pools report inactive changes $\sum \omega_k$ normalisation on affected pairs, so the
explanation invariant must be re-verified (promotion criterion 4); (c) a channel that starts
firing changes $U$, so any E04/E07/E21/E25 comparison running alongside must pin its fusion arm.

## Results note

*(appended after running; must answer RQ24.1–RQ24.5 with `supported`, `not supported`, or
`inconclusive`, and include the power declaration and both gate outcomes)*
