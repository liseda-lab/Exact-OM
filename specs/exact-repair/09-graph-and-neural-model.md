# XR-2 graph construction and neural architecture

## 1. Observable graph

Build G = (V,E,node_type,edge_role) from asserted ontology snapshots, the complete provisional alignment, bounded initial diagnosis and available evidence. Node types include classes, object/data properties, constructors, ontology axioms, mappings, discovered explanations, and evidence. Include individuals/literals and typed assertion roles when relevant; do not discard ABox constraints from verification.

A statement is distinct from its endpoints and syntactic expression. Example edges:
- inclusion axiom → subclass / superclass;
- existential constructor → property / filler;
- intersection → unordered operands;
- mapping → source / target;
- explanation → each supporting statement and witnessed query;
- evidence → the entity/statement it supports, with evidence kind.
Reverse message edges have separate relation types.

One explanation node represents a discovered support plus witnessed violation. There is neither one required node per mapping pair nor exactly one per unsatisfiable class. A witness without support is marked explanation-missing.

Initial features: type, ontology side, syntax, text embedding, matcher/score and score-missing flag, provenance/authorship or unknown, edit eligibility, available diagnostic counts and evidence reliability. Natural-language claims are soft evidence, not entailments. Represent punning by (IRI, kind).

Forbidden features: hidden clean theory, corruption trace/position, reference membership, teacher optimum, split IDs and explanations discovered only after the current decision. Confidence normalisation is fitted on training data only. Report constant/missing-score and unseen-matcher controls.

## 2. Context and retrieval

Retrieve finite class/property menus using ontology labels/definitions, matcher alternatives, structural neighbourhoods and discovered explanation membership. Introduce the retrieved symbols' observable descriptions into graph memory before encoding that round.

Large ontologies use bounded context. Preserve the complete support of any included explanation, then budget extra neighbourhood expansion. Record omitted nodes/supports. The reasoning input remains independent of this neural sampling. Vocabulary retrieval misses are measured separately from proposal misses.

## 3. Encoder

Main model: a standard Heterogeneous Graph Transformer (HGT); control: R-GCN with identical downstream attention readouts. Do not write a new message-passing framework. HGT is justified by typed attention, not claimed logical reasoning.

A schematic attention head:
~~~text
q_v = W_Q[type(v)] h_v
k_u = W_K[type(u)] h_u
e_uvr = q_v^T W_A[r] k_u / sqrt(d_head)
alpha_uvr = softmax over typed incoming neighbours of v
m_v = sum_(u,r) alpha_uvr W_M[r] W_V[type(u)] h_u
~~~
Combine heads with type-specific output projection, residual/normalisation and feed-forward layers. Retain H ∈ R^(N×d), one contextual row per node.

Exploratory starting values: d=128, three layers, four attention heads, dropout 0.1. They are settings to measure, not established optima. The same context and training budgets apply to encoder comparisons.

## 4. Shared target readout and proposal heads

For editable statement i:
~~~text
q_i = MLP_target(h_i || role-labelled argument embeddings)
r_i = Attention(q_i, H[M_i])
c_i = MLP_context(q_i || r_i || observable_features_i)
~~~
M_i contains relevant statements, explanations and soft evidence. It can read individual supporting axioms directly. Multi-argument/unordered axiom operands need typed/permutation-invariant aggregation, not an arbitrary ordering feature.

Shared proposal heads consume c_i, eligible templates, retrieved menu embeddings and optionally the cost profile u. They output mixture weights and Boolean-choice logits for [10](10-constrained-generation.md). Entity selectors share parameters across IRIs, for example:
~~~text
logit(i,slot,C) = (W_slot c_i)^T W_class h_C + bias_slot
~~~
No per-ontology IRI output layer is required. All logits for one circuit invocation are computed before sampling.

These are shared post-encoder prediction functions run per editable object. Explanation nodes do not each own a separate trained head. Attention heads inside HGT are distinct from these output heads.

## 5. Candidate encoding and value head

After proposal, encode the complete emitted axiom bundle into e_ia. Named entities use their contextual embeddings; existential expressions combine property and filler; intersections aggregate canonical operands; axiom/bundle composition preserves argument roles, relation, directions and active obligations.

~~~text
q_ia = MLP_candidate(c_i || e_ia)
r_ia = Attention(q_ia, H[M_i ∪ M_a])
b_ia = MLP_value(c_i || e_ia || r_ia || candidate_features_ia)
~~~
M_a includes context for newly introduced symbols. Distinct specialised and complex-equivalence candidates receive distinct representations. Output b_ia is a contribution to teacher-scaled semantic benefit, not proposal probability, entailment probability, or a probability that an axiom is wrong.

Explicit cost uᵀf_ia is subtracted outside the value head once. Cost-only user profiles share the benefit model. A different desired competency-query task is a changed input/target, not just a different cost profile.

## 6. Interaction head

Compare unary benefit sum_i b_ia_i with:
~~~text
B_hat(R) = sum_i b_ia_i + sum_(i,j in P) b_(ia_i,ja_j)
~~~
Choose a finite pair set P from observable shared supports or affected declared queries. A shared pair head reads both candidate encodings and their joint context; freeze all coefficients before selection.

Required counterexample: U00=0, U10=9, U01=9, U11=8 when either edit restores the same benefit 10 at cost 1. Additive scores cannot fit the mixed difference −10. A pair coefficient −10 resolves this table, not all higher-order OWL interactions.

A whole-repair reranker is an optional control with explicitly limited search; do not claim that MaxSAT optimises an arbitrary neural reranker unless its objective is encoded exactly.

## 7. Training and acceptance

Train proposal likelihood and full-repair value losses from [07](07-corpus-and-training.md). Test unseen vocabulary, permutation of symmetric operands, missing evidence, shared conflicts and ontology-edit provenance. Match readouts when comparing HGT/R-GCN so attention and encoder effects are not confounded.

Acceptance includes shape/interface checks, shared parameter use, no hidden-label access, candidate-specific value sensitivity, cost subtraction once, and the interaction counterexample. Benchmark improvements are separate evidence.
