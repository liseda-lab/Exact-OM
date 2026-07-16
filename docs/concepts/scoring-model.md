# Scoring model

The default pair-adaptive scorer keeps lexical evidence primary and separates structural
signals before fusion. This avoids treating every graph edge as equally informative.

1. Candidate-specific label variants produce `s_label` and label quality.
2. Pair-adaptive context selects hierarchy, supported similarity, distinctive difference, and
   attribute evidence.
3. Each channel yields a score, quality, reliability weight, importance, and contribution.
4. The structural channels combine into `S_struct`; confidence-aware fusion combines it with
   lexical evidence into `S_base`.
5. Ambiguous/disagreeing pairs may receive LLM arbitration, producing `p_llm` and `S_final`.
6. An optional selector calibrates source-local acceptance, abstention, and target conflicts.

| Channel | Main record fields |
| --- | --- |
| Lexical | `s_label`, `q_label`, `I_label`, selected labels. |
| Hierarchy | `s_hier`, family scores, hierarchy attributions. |
| Similarity | `s_sim`, supported cross-side triples. |
| Difference | `s_diff`, distinctive or unsupported triples. |
| Attribute | `s_attr`, definitions/synonyms/xrefs/literals. |
| LLM | `p_llm`, `I_llm`, backend use, decision, rationale. |

The output explanation is an audit record, not a post-hoc prose-only explanation: it retains
the values and evidence used by the decision. Final selector/rationale changes are overlaid
crash-safely and compacted back into the indexed record.

All hard-coded blend/cap constants from the 1.x audit are exposed under their owning v2 config
sections with behavior-preserving defaults. See the generated configuration reference and the
more detailed [pair-adaptive scoring notes](../pair_adaptive_scoring_slides.md).
