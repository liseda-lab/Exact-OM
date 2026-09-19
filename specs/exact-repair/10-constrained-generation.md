# XR-2 grammar and probabilistic circuit layer

## 1. Role

The layer samples finite well-formed replacement components for one editable object. It is not a circuit for all coherent ontology alignments. It adapts Semantic Probabilistic Layers to expression/template choices, conditioned by the graph model. Cite the prior circuit and SPL work in [references.md](references.md).

## 2. Finite language and canonical encoding

For finite menus C_i and R_i:
~~~text
E ::= C | E AND E | EXISTS r.E
C ∈ C_i; r ∈ R_i
~~~
Fix depth, node/constructor count and template bounds. Root depth is zero; a named leaf has no constructor cost; existential/intersection add one plus child costs. Flatten, deduplicate and sort intersections, then give them a fixed binary association. Bounds apply to the canonical tree.

Slots have constructor, class choice, property choice and unused markers. Boolean one-hot fields encode categorical choices; separate variables encode template and retained directions. Force all inactive slots/irrelevant fields to a unique unused value.

The template domain includes elementary keep/delete/direction/endpoint alternatives as well as complex templates when applicable. An elementary branch can leave all expression slots inactive. This gives every proposal-supervised candidate an encoding. Deterministically inserting elementary states into the optimiser's pool remains mandatory; their availability never depends on drawing them. If an implementation uses a complex-only generator, it must explicitly restrict and renormalise its proposal targets to the encoded candidate subset and report that as a different arm.

K_i enforces one-hotness, active-child structure, depth/size limits, type, ontology-side/template restrictions, canonical ordering, and only genuinely justified fixed constraints. Use one designated encoding per identical canonical emitted bundle and activation. If several encodings are retained, training/scoring must sum their probability for a candidate. Do not pretend syntactic canonicalisation identifies every logical equivalence.

Generated symbols come from observed finite menus. Arbitrary newly invented domain concepts are not supported. Private definitional probes used by verification are separate.

Do not permanently mask an expression using an editable axiom unless that dependence is modelled. For example, an original range may be revised by another object. Joint conflicts belong in the master/verification loop.

## 3. Compile and condition

Use an existing compiler to represent K_i by a deterministic decomposable Boolean circuit, with smoothing where weighted counting needs it. AND children have disjoint variable scopes; OR branches have disjoint satisfying assignments. Circuit nodes are logical gates/literals, not ontology graph nodes.

All neural parameters are computed before sampling:
~~~text
q_m(z | c_i,u) = product_j Bernoulli(z_j; sigmoid(logit_mj))
q(z) = sum_m pi_m q_m(z), with pi = softmax(component_logits)
Z_m = sum_(z:K_i(z)) q_m(z)
Z = sum_m pi_m Z_m
p(z | c_i,u,K_i) = 1[K_i(z)] q(z) / Z
~~~
M=1 is the independence control; a small mixture (pilot M=4) captures limited property/filler/template correlations. This is not an arbitrary autoregressive model with a claimed tractable global normaliser.

Assign literal weights p_mj and 1−p_mj. Multiply at decomposable ANDs and sum at deterministic ORs to compute Z_m. Sample component m using pi_m Z_m/Z, then use its weighted branch probabilities. Using pi_m without the conditioning correction is wrong.

Compilation may be exponential. Once compiled, evaluation of all component normalisers is O(M|circuit|). Exact sampling and normalisation do not guarantee efficient exact mixture MAP decoding. Track compilation size/time, cache keyed by K_i/variable order/compiler, duplicate rate and useful coverage.

Z=0 yields an empty constrained proposal space, not a valid probability distribution or a reason to relax a hard constraint. Elementary candidates remain separately available. Numerical underflow requires stable arithmetic or an explicit numerical failure.

## 4. Example

For E = ∃hasDecision.Acceptance and target S ≡ T, one sample chooses an existential root, property hasDecision, filler Acceptance, and the template:
~~~text
{ S AND E SubClassOf T, T SubClassOf S }
~~~
A different template adds T SubClassOf E and yields T ≡ S AND E. The candidate/value model sees the complete bundle, including that extra condition. The circuit asserts grammar/template validity; the verifier checks whether it contradicts, for example, an invited-paper subclass.

## 5. Candidate budget and coverage

Always supply applicable elementary states independently of sampling. Deduplicate sampled canonical bundles, preserve provenance and record all attempted/retained counts. Sampling need not exhaust the bounded grammar.

Measure vocabulary retrieval recall, useful proposal coverage given retrieval, and selection quality given a pool separately. Oracle vocabulary/candidate injection is a labelled evaluator-only comparison and never a deployment input.

Alternatives are bounded enumeration and standard masked grammar decoding. Compare total budgets, not just accepted sample counts. If circuits do not improve useful coverage enough to offset compilation, the experiment may favour the simpler alternative.

## 6. Required conformance

On tiny finite grammars enumerate all Boolean assignments and compare:
1. satisfying assignments versus valid canonical trees/templates;
2. unused-field uniqueness and permutation deduplication;
3. brute-force Z and mixture posterior weights versus circuit evaluation;
4. total probability one and zero probability outside K_i;
5. sampled support and distribution, with statistical rather than impossible exact-frequency assertions;
6. gradient/likelihood consistency including log Z;
7. independent versus correlated property/filler examples;
8. invalid global ontology combinations despite grammar validity;
9. editable-axiom dependencies and circuit cache invalidation.

The compiler and tensor library are trusted dependencies to qualify, not new local reimplementations.
