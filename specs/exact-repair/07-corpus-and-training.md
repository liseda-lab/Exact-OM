# XR-2 corpus, symbolic teacher and learning

## 1. What is supervised

A reasoner can establish policy feasibility. It cannot identify intended meaning from coherence alone. Start with an explicit symbolic teacher over generated intended theories, desired/unwanted probes and edit costs. No learned reward model or large expert-labelled corpus is required to begin.

The student learns useful proposals and approximate benefit from observable inputs. The teacher is a declared simulator objective, not universal domain truth.

## 2. Generated examples

Construct a clean intended pair and alignment, verify the relevant consistency/coherence/non-vacuity, then corrupt selected observable mappings or ontology axioms. Build intended queries before seeing repair outputs.

Required mechanism families:
- directional strengthening;
- subclass-expression specialisation on each side;
- necessary-condition/complex-equivalence recovery on each side;
- endpoint confusion;
- disjointness revision;
- superclass conjunct removal;
- ontology subclass specialisation/generalisation;
- domain/range typing;
- existential filler generalisation;
- interacting conflicts and redundant/complementary benefit;
- mixed compositions.

Mirror ontology roles; vary path length, branch count, explanation overlap, irrelevant structure, expression depth, score calibration and evidence quality. Include coherent controls, ambiguity and cases whose useful candidate is missing from normal retrieval. A rich intended alignment can be projected to a simple input for a declared corruption; do not expose the intended complex expression as a hidden feature.

Generate observed labels/descriptions and noisy matcher-like evidence independently of correctness flags. Counterbalance provenance: human-authored does not always mean correct, imported/generated does not always mean wrong. Keep corruption traces, latent intended vocabulary and teacher labels evaluator-only.

Partition clean structural parents before generating renamed/corrupted siblings. Hold out composition/template families as well as instances. An identical observable case with different hidden intended labels is unidentifiable; preserve ambiguity or request informative evidence, not a forced arbitrary label.

Use the fixtures in [06](06-first-experiment-implementation.md). The teacher scores all comparable repaired consequences, not merely reversal of the planted edit.

## 3. Typed symbolic teacher

Partition the query universe into desired Q+, unwanted Q− and unspecified. Synthetic non-entailment is not automatically a real-world negative; Q− requires a declared construction/intention. Public reference absence is unspecified.

For a consistent feasible T:
~~~text
ok_T(C ⊑ D) =
  entails(T,C ⊑ D) AND Sat_T(C)

ok_T(C AND D ⊑ bottom) =
  entails(T,C AND D ⊑ bottom) AND Sat_T(C) AND Sat_T(D)

ok_T(C ⊑ EXISTS r.D) =
  entails(T,C ⊑ EXISTS r.D) AND Sat_T(C)
~~~
Desired disjointness requires individually satisfiable operands, not a satisfiable intersection. A complex subclass inclusion uses its whole subclass expression. Property/instance probes use their own typed conditions; role-use non-vacuity may require Sat(∃r.top), not the named-class rule.

Record unwanted raw entailment separately; emptying its antecedent does not earn restoration credit. Inconsistent candidates are infeasible and receive no benefit through explosion. Unknown queries remain unknown, not false.

Normalise within nonempty semantic families:
~~~text
B*(R) =
  sum_f gamma_f mean_(q∈Q+_f) ok_T_R(q)
  - sum_g delta_g mean_(q∈Q−_g) entails(T_R,q)

U*_u(R) = B*(R) - u^T sum_i f_i(a_i)
~~~
Freeze gamma/delta and query basis independently of predictions. Hard policy is separate; soft reward cannot compensate for a hard failure.

Five equally weighted paper-fixture probes are AcceptedSubmission_s ⊑ Accepted_t, Accepted_t ⊑ Paper_s, Accepted_t ⊑ E, Rejected_s ⊑ Rejected_t, and Accepted_t disjoint Rejected_t. In the no-invited-exception case, deletion preserves two, specialisation with reverse subsumption four, and complex equivalence five. Illustrative benefit/cost/utility are .40/.10/.30, .80/.05/.75 and 1/.07/.93. Costs are examples, not observed user judgments.

## 4. Teacher caches and partial labels

Small generated problems permit whole-case enumeration. Freeze the candidate universe and record each assignment's feasibility, typed query outcomes, benefit and cost-independent semantic vector. Reuse vectors across profile weights.

Exact optimum/regret and the exact normalised teacher distribution require the entire relevant space to be decided. Exceeding the label-product or deadline cap marks the cache incomplete. Do not assume discovered conflict components are independent.

Verified complete-query outcomes from a partial cache can still train declared regression/comparison losses. They cannot claim an exact optimum, exact state marginal or globally normalised teacher distribution. Unknown assignment/query labels are masked with coverage counts. An infeasible singleton is not proof that a candidate is unusable jointly with an ontology edit.

No label method assumes all-off is feasible. Anchors are actual verified feasible repairs. Caches include input, patch, policy, query, inventory and backend hashes.

## 5. Proposal supervision

For an exhaustively known finite feasible set F:
~~~text
rho_u(R) = exp(U*_u(R)/tau) / sum_(R'∈F) exp(U*_u(R')/tau)
t_i(a) = sum_(R∈F) rho_u(R) 1[a_i=a]
L_prop = -sum_i sum_a t_i(a) log p_theta(z_ia | c_i,u,K_i)
~~~
Retain ties. These marginal proposals need not form feasible combinations; joint selection remains necessary. Include the exact circuit log-normaliser in likelihood training. Candidate likelihood sums encodings unless unique canonical encoding is guaranteed.

Accepted-set mass alone can collapse onto one easy expression. Use the teacher distribution/ranking and evaluate useful coverage. Reference terms outside the observed menu are retrieval misses, not permission to inject answers.

## 6. Benefit learning

Train the shared HGT encoder/readouts and completed-candidate value head against full-repair B*, comparing unary and bounded pairwise models. A robust regression on B_hat(R)−B_hat(R0) versus B*(R)−B*(R0), for a verified anchor R0, preserves scale without requiring an arbitrary offset. Add a pairwise ranking loss for useful distinctions.

Do not assign each candidate the best total repair utility among repairs containing it and then sum those targets. That double-counts contributions from other objects. A singleton marginal target is only a restricted diagnostic baseline, not the main joint teacher.

Explicit user costs are subtracted once outside semantic benefit. Keep scale identifiable relative to costs. Proposal probability is not benefit and not a correctness probability.

Use generated pretraining, then a separately reported adaptation arm on training-side Conference/biomedical structure and permitted public supervision. Generated-only transfer remains a control. Freeze model selection on development data and use grouped held-out tests.

## 7. Preferences

Begin with declared nonnegative edit-cost weights. Human ontology edits have a greater default increment than otherwise comparable automatic edits. Learn a small regularised nonnegative profile vector from pairwise complete-repair choices when feedback exists:
~~~text
Pr(R preferred to R') =
  sigmoid(B_hat(R)-B_hat(R') - u^T(F(R)-F(R')))
~~~
Freeze the shared semantic model initially for this comparison. Preference changes costs; factual clarification changes evidence/queries; a prohibition changes hard eligibility. Update profiles between frozen solves.

Simulated profiles are labelled simulations. Real user preference performance requires collected feedback and held-out comparisons; no expert availability is assumed.

## 8. Concrete optimisation protocol

One training example is a repair case, not an isolated conflict node. It contains the observed graph/evidence, frozen candidate bundles, profile, feasibility/query masks and complete-repair teacher labels. The hidden intended theory stays in the teacher store.

For a case with a verified feasible anchor R0, define d_hat(R)=B_hat(R)−B_hat(R0) and d_star(R)=B*(R)−B*(R0). The pilot losses are:
~~~text
L_value = mean_R SmoothL1_beta(d_hat(R) - d_star(R))
L_rank  = mean_(R,R') softplus(
              -sign(B*(R)-B*(R')) * (B_hat(R)-B_hat(R')) / t_rank)
L       = lambda_value L_value + lambda_rank L_rank + lambda_prop L_prop
~~~
Exclude equal-benefit pairs from L_rank; it learns semantic benefit ordering. Proposal targets use the full teacher utility including the profile costs. Decoded validation also subtracts those costs exactly once. The pilot uses beta=1, t_rank=1, loss weights 1/.2/1 and at most 64 sampled verified repair pairs per case per step. An exact target remains relative to its frozen teacher candidate universe, not the entire bounded grammar.

Select R0 from actually verified feasible assignments, preferring the lowest declared edit cost and then a stable candidate-ID order. For a partial cache, regression/ranking can use only pairs whose entire required soft-query vectors are decided. An unknown query is not silently omitted from its family denominator to fabricate a complete scalar score. Component-level partially observed targets require a separately named loss and coverage report. Cases with no usable anchor provide no anchored value loss.

Each minibatch:
1. Encodes observed graphs and object contexts.
2. Computes proposal logits and exact circuit log-normalisers.
3. Encodes labelled complete candidates and computes unary/optional pair factors.
4. Forms complete-repair benefit sums, masked value/ranking losses, and proposal loss only for complete teacher distributions.
5. Backpropagates through graph layers, readouts, candidate heads and differentiable circuit evaluations. Grammar topology and symbolic labels are fixed.
6. Applies AdamW, gradient clipping and the recorded development-only checkpoint criterion.

The MaxSAT solver does not need to be differentiable. Decode on development cases periodically to measure actual repair regret and coverage. Freeze the selected model, profile and coefficients before every evaluated optimisation run. Use the same case/split/seed and comparable optimisation budgets for neural controls.

Start with generated pretraining, then a separately identified adaptation run using training-side real-structure cases and allowed real supervision. Warm-start model weights only; do not bring cached test labels or test-selected checkpoints into adaptation. Numeric optimisation settings are in the versioned protocol.

## 9. Training acceptance

Require typed disjointness/vacuity/unknown tests, zero credit for inconsistent theories, nonadditive utility fixtures, no hidden-label access, proper masking of incomplete teacher caches, accurate normalised circuit likelihood, a feasible-anchor strategy, and grouped splits. Report requested versus produced/labelled cases, label compute, convergence and sensitivity. Pilot counts/settings in protocol/pilot.json are a bounded exploratory run, not an adequacy claim for GNN training.
