# E26 — Quality-Proxy Validity: Does $q_k$ Measure Anything About Correctness?

**Motivation** (review-response WP1/WP5/WP6): the informative deviation
$\sigma_k = q_k \cdot |s_k - \tau|^{\gamma}$ is the system's central mechanism, and $q_k$ is the
term that distinguishes it from fixed-weight fusion. The weighting demonstrably works — uniform
structural weights cost 0.077 MRR — but the *interpretation* of $q_k$ as channel reliability is
unsupported by direct measurement.

| Test (both 300-source subsets) | Result |
|---|---|
| per-source discrimination, attributes (OMIM) | Spearman $-0.015$, CI $[-0.158, 0.134]$, $n{=}204$ |
| per-source discrimination, attributes (FMA) | $-0.025$, CI $[-0.134, 0.086]$, $n{=}300$ |
| per-source discrimination, hierarchy (FMA) | $0.092$, CI $[-0.028, 0.199]$, $n{=}300$ |
| $q_{\mathrm{lex}}$ | constant at 1 on every scored pair; untestable |
| attribute-bin positive rate (OMIM), low→high quality decile | 0.0135 → 0.0023 (**falls**) |

Two things follow. First, no channel shows a quality–correctness relationship that survives a
discrimination test, so the paper now describes $q_k$ only as a within-channel evidence measure.
Second, and more actionable: $q_{\mathrm{lex}}$ is **constant**, which means the lexical channel —
the one carrying most of the decision — contributes no quality signal at all. Its
$\sigma_{\mathrm{lex}}$ is driven entirely by $|s_{\mathrm{lex}} - \tau|^{\gamma}$. Whatever the
weighting is achieving, it is not achieved through lexical quality.

A methodological caveat is recorded here because it will otherwise be rediscovered: on candidate
tables that are ~99% negative, a Brier score computed within a quality bin reduces to
approximately $\overline{s_k}^2$ and tracks mean confidence rather than calibration (predicted
0.4995 against observed 0.4994 on the lowest OMIM hierarchy bin). Bin-wise Brier is not a valid
calibration test on this data and must not be used as one.

## Research questions

- **RQ26.1**: Why is $q_{\mathrm{lex}}$ degenerate? The margin
  $(z_1 - z_2)/\max(1 - z_2, \varepsilon)$ should vary; determine whether it saturates because
  $z_2$ is near 1, because label sets are small, or because of the $\varepsilon$ floor.
- **RQ26.2**: Does a non-degenerate lexical quality — margin over more than two labels, entropy of
  the top-$m$ similarity distribution, or agreement across encoders — predict lexical-channel
  correctness per source?
- **RQ26.3**: Do any of the current quality terms (coverage, strength, specificity, stability)
  individually predict correctness, or is the null result uniform across the components?
- **RQ26.4**: Is a directly supervised reliability estimate — predicting per-channel correctness
  from channel features on training pairs — better than the analytic $q_k$, and does substituting
  it into $\sigma_k$ improve end-to-end quality?
- **RQ26.5**: Given the 0.077 MRR uniform-weights gap, how much of that gap survives if $q_k$ is
  replaced by a constant while retaining the $|s_k - \tau|^{\gamma}$ term and the
  inactive-channel suppression? This decomposes the mechanism into its three parts.

**RQ26.5 is the scientifically important question.** The current evidence is consistent with the
entire benefit coming from suppressing inactive channels and sharpening confident ones, with $q_k$
contributing nothing. If so, the mechanism should be described that way — which is both more
honest and simpler to defend.

**Hypotheses**: (a) $q_{\mathrm{lex}}$ saturates because biomedical label sets are small and the
top-2 margin is near-maximal whenever any synonym matches; (b) an entropy-based lexical quality is
non-degenerate but still only weakly predictive; (c) most of the 0.077 gap survives constant-$q$
ablation, attributing the benefit to suppression and sharpening rather than to quality; (d) a
supervised reliability head beats analytic $q_k$ on in-pair supervised tracks but is not
label-free deployable, making it a diagnostic ceiling rather than a default.

## Change

1. **Quality-term dump**: extend the channel dump with each quality *component* (coverage,
   strength, specificity, stability) separately, not only the aggregate $q_k$. RQ26.3 is currently
   unanswerable because only the mean is persisted.
2. **`channels.lex.quality`**: `margin` (current), `entropy` over top-$m$ label similarities,
   `encoder_agreement` across SapBERT and BGE, `constant` (control).
3. **`fusion.sigma.mode`**: `full` (current), `constant_q` (RQ26.5 — sets $q_k = 1$ for all active
   channels while retaining inactivity suppression), `no_sharpening` ($\gamma = 0$),
   `suppression_only` (both).
4. **Supervised reliability head** (RQ26.4), reusing E19's fitting infrastructure, predicting
   per-channel correctness rather than fusion weights directly.

## Arms & sweep

Stage 1 (diagnostic, re-analysis plus instrumented runs on the frozen subsets): RQ26.1, RQ26.3.
Stage 2 (mechanism decomposition, the primary contribution): `full` / `constant_q` /
`no_sharpening` / `suppression_only` / `uniform`, giving the full factorial over the three
components of $\sigma_k$.
Stage 3 (improvement): lexical quality variants, then the supervised head as an
`in_pair_supervised` arm with its `target_label_free` comparator in the same table.

## Validation

Eligible tasks: all five Bio-ML pairs plus Anatomy and Conference, since quality behaviour on
lexically rich biomedical labels may not generalise. Classes. Primary metric macro MRR (local) for
the mechanism decomposition, macro F1 (global) for any promotion arm.

Reliability claims use **per-source discrimination AUC** as the primary test, which is insensitive
to class balance. Bin-wise Brier is reported only as a descriptive companion with the confound
stated. Positive rate per quality bin is reported alongside every calibration figure so base-rate
drift stays visible.

## Promotion decision rule

Stage 2 has **no promotion arm** — its deliverable is the mechanism decomposition table, and it is
a required output regardless of outcome. If `suppression_only` recovers most of the uniform gap,
the finding is that the mechanism is simpler than claimed, and the documentation and paper
description change accordingly. That is a result, not a failure.

Stage 3 promotion: best lexical-quality variant against `R_n`, endpoint macro F1, standard gates.
The supervised head promotes only as the `supervised` resolution of the fusion component under the
README's regime-specific rule, never replacing the label-free path.

## Effort & risks

Size: M. Stage 1 is re-analysis of dumps already collected plus a component-level dump flag.
Stage 2 is four config arms and no new modelling.

Risks: (a) changing $q_{\mathrm{lex}}$ changes $w_c$ and hence the importance decomposition, so
promotion criterion 4 must be re-verified; (b) an entropy-based quality needs a defined behaviour
for single-label entities, which is common for properties and instances — specify before running,
and check the E11/E12 slices; (c) results are conditional on the candidate-pool fingerprint,
since quality terms depending on coverage move with pool size.

## Results note

*(appended after running; must answer RQ26.1–RQ26.5, and must report the Stage 2 decomposition
table whether or not any arm is promoted)*
