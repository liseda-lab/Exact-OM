# E25 — LLM Gate Viability: Does the Branch Ever Fire, and Does It Help When It Does?

**Motivation** (review-response WP1/WP3/WP4 and the earlier `omim-ordo-val` comparison): the
uncertainty-gated LLM branch is a headline design element that, on all evidence collected so far,
never activates and does not improve results when it does.

| Measurement | Result |
|---|---|
| LLM invocations, reported OMIM–ORDO ranking run | 0 / 30,404 pairs |
| LLM invocations, SNOMED–FMA 300-source run | 0 / 32,724 pairs |
| Sources routed to the LLM, all 8 global selector settings (WP3) | 0 |
| `tau=0.6` sweep arm | 4 pairs gated out of 30,404 |
| Ranking MRR with vs without `p_LLM` (`omim-ordo-val`) | 0.810 with, **0.826 without** |

Two distinct failures are entangled here and must be separated before either E07 (listwise
arbitration) or E21 (supervised LLM) can be interpreted. Both of those specs presuppose a
population of gated pairs; if that population is empty or worthless, their results are about a
mechanism that does not engage. **E25 is therefore a prerequisite for E07 and E21**, not a
parallel line of work.

The failure is not simply that `tau_LLM` is set too high. $U = \max(U_{\mathrm{ind}},
U_{\mathrm{dis}})$ reaches $\tau_{\mathrm{LLM}} = 0.5$ only when the mixed score sits within 0.25
of the pivot or two well-supported channels disagree by a wide margin, and the measured score
distribution is strongly bimodal away from the pivot. Lowering the threshold alone would gate
pairs the system is already confident about, which is the opposite of the design intent.

## Research questions

- **RQ25.1**: What is the joint distribution of $U_{\mathrm{ind}}$ and $U_{\mathrm{dis}}$, and what
  threshold would gate a target fraction (1%, 5%, 10%) of pairs? Is either term ever the binding
  one in practice?
- **RQ25.2**: On pairs the system currently gets **wrong**, is $U$ elevated relative to pairs it
  gets right? That is: is $U$ a useful error detector at all, independent of any threshold?
- **RQ25.3**: When the LLM is invoked on a forced sample (gate disabled, fixed budget), what is its
  standalone accuracy, and does mixing it at weight $w_i = \beta U$ improve or degrade the pair
  decision relative to the unmixed score?
- **RQ25.4**: What is the oracle ceiling — if an oracle chose which pairs to route and always
  answered correctly, how much macro F1 is available? This bounds every possible gating policy.
- **RQ25.5**: Is a learned router over the channel evidence better than the analytic $U$ at
  selecting pairs where LLM arbitration changes the outcome correctly?

**Hypotheses**: (a) $U$ is near-zero for the overwhelming majority of pairs because the lexical
channel dominates the mixture and is rarely near the pivot; (b) $U$ is only weakly elevated on
errors, so it is a poor error detector and no threshold on it recovers a useful population;
(c) forced LLM arbitration is accurate on lexically ambiguous same-family confounders and harmful
on pairs where structural evidence already decides, which is why unconditional mixing lowers MRR;
(d) the oracle ceiling is small in absolute terms (< 2 F1) on lexically rich tasks and larger on
identifier-poor ones; (e) a learned router beats analytic $U$ but not by enough to justify the
cost unless (d) is large.

## Change

1. **Gate instrumentation**: log $U_{\mathrm{ind}}$, $U_{\mathrm{dis}}$, $U$, and the gate outcome
   for every scored pair, independent of whether the LLM is called. Currently only invocations are
   counted, which makes the gate distribution unobservable.
2. **`llm.gate.mode`**: `analytic` (current), `quantile` (route the top-$p$ fraction by $U$, giving
   a controllable budget), `forced_sample` (route a fixed random sample, for RQ25.3), `oracle`
   (route pairs the reference says are errors — diagnostic only, never deployable), and `router`
   (learned, for RQ25.5).
3. **Decoupled mixing**: allow the LLM's contribution weight to be set independently of the gating
   statistic, so "when to ask" and "how much to trust the answer" can be measured separately. They
   are currently tied through $U$.

## Arms & sweep

| Arm | Supervision label | Purpose |
|---|---|---|
| `R_n` baseline | `target_label_free` | current analytic gate (expected: ~0 invocations) |
| `llm_off` | `target_label_free` | explicit control |
| `quantile_p` ∈ {0.01, 0.05, 0.10} | `target_label_free` | controllable budget; the practical candidate |
| `forced_sample` | `target_label_free` | RQ25.3 standalone accuracy |
| `oracle_route` | `oracle_diagnostic` | RQ25.4 ceiling; never promotable |
| `router` | `in_pair_supervised` | RQ25.5; consumes E21's machinery if E21 has run |

## Validation

Eligible tasks: all five Bio-ML pairs, plus at least one identifier-poor track (Anatomy or
Conference) where lexical evidence is weaker and the gate has more reason to fire. Classes.
Primary metric macro F1 (global) and macro MRR (local).

**Cost reporting is a first-class endpoint here, not a footnote**: call count, token count, and
wall time per arm, since the design's stated justification is selective use. An arm that improves
quality by routing 10% of pairs is a different product than one routing 0.1%.

Mandatory secondary slice, pre-registered: pairs where the top-2 candidates are same-family
confounders, since RQ25.3's hypothesis is that this is the only population where the LLM helps.

## Promotion decision rule

Primary comparison: the best `quantile_p` arm against `R_n`, endpoint macro F1, paired bootstrap CI
excluding zero, ≥3 seeds, with LLM token cost reported beside it.

**This experiment has an explicit removal outcome.** If RQ25.2 shows $U$ does not discriminate
errors and RQ25.4's oracle ceiling is below 1 F1 point, the recommended promotion is to **remove
the LLM branch from the default configuration** and document it as an optional component. That is a
positive result: it removes a hosted-model dependency, a reproducibility risk, an unbounded cost,
and the one component of the explanation that carries no fidelity guarantee. The paper's
auditability claim is strictly stronger without a branch whose contribution cannot be traced to
ontology evidence.

If the branch is retained, promotion requires that the gated population be non-empty by
construction (`quantile` mode) rather than incidental, so that its behaviour is measurable in
future runs rather than silently zero.

## Effort & risks

Size: M. Instrumentation is small; `forced_sample` and `oracle_route` need a bounded LLM budget
declared before running.

Risks: (a) hosted-model drift confounds any comparison spanning weeks — pin a dated snapshot and
cache responses for the reporting runs; (b) the oracle arm must never leak into a promotion path,
enforced by the existing `oracle_diagnostic` label; (c) if E24 succeeds and the contrastive channel
begins firing, $U_{\mathrm{dis}}$ changes distribution, so E24 and E25 must not have reporting runs
in flight against different fusion arms.

## Results note

*(appended after running; must answer RQ25.1–RQ25.5 and state explicitly whether the branch is
retained, re-gated, or removed)*
