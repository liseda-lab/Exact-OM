# Pair-Adaptive Scoring: Practical Walkthrough

This file is written in slide-style Markdown so it can be reviewed directly in
Git or converted later into a presentation.

---

# 1. What Changed

## Legacy mode: `SemanticScorer`

- Build one cached context subgraph per entity.
- Join that context into one text block per side.
- Compare `lexical` vs one `context` score.
- Explanations are mostly `lexical / context / llm`.

## Current default: `PairAdaptiveSemanticScorer`

- Keep lexical matching.
- Replace one joined context with four pair-adaptive structural channels:
  - hierarchy
  - non-hierarchical similarity
  - difference
  - auxiliary attributes
- Fuse `lexical` with an aggregated `structural` score.
- Compute ambiguity from both:
  - indecision near the middle
  - disagreement between lexical and structural evidence

---

# 2. High-Level Pipeline

For one source-target pair:

1. Score labels lexically.
2. Pull cached per-entity evidence pools:
   - labels
   - hierarchy bundles
   - non-hierarchical object triples
   - attributes / projected literals
3. Build pair-specific channel evidence.
4. Compute:
   - `s_label`
   - `s_hier`
   - `s_sim`
   - `s_diff`
   - `s_attr`
5. Aggregate structural channels into `S_struct`.
6. Fuse `s_label` with `S_struct` into `S_base`.
7. Compute ambiguity `U`.
8. If ambiguity is high enough, ask the LLM.
9. Produce final score `S_final`.

---

# 3. What Does Neutral Mean?

The neutral anchor is:

- `tau = 0.5`

Interpretation:

- above `0.5`: evidence leans toward a match
- below `0.5`: evidence leans toward a non-match
- exactly `0.5`: the channel is effectively saying "I learned nothing useful"

This matters because empty or weak channels are designed to fall back to `0.5`
instead of inventing evidence.

---

# 4. Label Channel

## What it uses

- all source labels
- all target labels
- lexical encoder

## How it works

- compare every source label against every target label
- keep the best label pair
- that best similarity becomes `s_label`

## Confidence

- `q_label` is high when the best label pair is clearly better than the
  second-best pair
- `q_label` is low when many label pairs look similarly plausible

## Practical example

Source labels:

- `myocardial infarction`
- `heart attack`

Target labels:

- `myocardial infarction`

Result:

- `s_label` very high
- `q_label` high because one label pair clearly dominates

---

# 5. Hierarchy Channel

## What it uses

- ontology-native hierarchy facts, not the projection guesswork
- configured families such as:
  - `is_a`
  - `part_of`
  - `has_part`

## How depth works

- depth is controlled by `hierarchy_max_depth`
- each family keeps at most `max_hierarchy_triples_per_family` triples per
  entity

## How score is built

For each hierarchy family:

- compare hierarchy target concepts lexically
- select the best supporting hierarchy triples on each side
- verbalize them
- combine:
  - `strength`: how well the selected hierarchy targets line up
  - `embedding`: similarity of the verbalized hierarchy snippets

Then:

- `s_family = 0.5 * embedding + 0.5 * strength`

## Confidence

Family confidence combines:

- coverage
- strength
- specificity from depth

Practical intuition:

- shallow, clearly aligned parents give stronger confidence than vague,
  deep, noisy hierarchy evidence

---

# 6. Similarity Channel

## What it uses

- non-hierarchical object-property triples only
- from the projected neighborhood around each entity

## Important parameter

- `n_hops` still matters here

Current default behavior:

- build the raw projected neighborhood within `n_hops`
- drop hierarchy relations
- drop literal tails
- rank triples by edge information content
- keep at most `max_object_triples`

## How pair-specific similarity is found

For each source triple and target triple:

- compare relation labels
- compare neighbor / tail labels
- average them

So a triple pair supports similarity when:

- the relations look similar
- and the linked concepts look similar

## Practical example

Source:

- `disease has_location heart`
- `disease associated_with inflammation`

Target:

- `disorder located_in heart`
- `disorder associated_with inflammatory response`

These triples support `sim` even if they are common rather than rare.

---

# 7. Difference Channel

## What it uses

- the same non-hierarchical object-triple pool as `sim`

## Key idea

The difference channel does not search elsewhere. It asks:

- which informative triples on one side have no good counterpart on the other?

For each triple:

- unsupported mass =
  `edge_IC * (1 - best_support_to_other_side)`

So a triple becomes strong difference evidence when:

- it is informative in its own ontology
- and the other ontology does not support it

## How it is selected

- rank unsupported triples by unsupported mass
- keep at most `max_diff_triples`
- cap relation repetition to avoid one relation flooding the channel

## Final score

- `conflict` is the average unsupported mass
- `s_diff = 1 - conflict`

Interpretation:

- high conflict means low compatibility
- therefore low `s_diff`

---

# 8. Attribute Channel

## What it uses

- ontology annotations
- projected literals when `projection_include_literals = True`

## Important behavior

Attributes do not need to match only other attributes.

An attribute can support the pair by matching:

- labels
- hierarchy evidence
- similarity evidence
- other attributes

## Why this helps

Some ontology-specific information appears only as:

- a definition
- an xref
- a note
- a synonym-like annotation

That may still strongly support the pair even if the other ontology does not
store equivalent text in the same property.

## Practical example

Source attribute:

- `definition: necrosis of heart muscle caused by ischemia`

Target side bank may contain:

- label: `myocardial infarction`
- hierarchy sentence: `myocardial infarction is a kind of ischemic heart disease`

That can raise `attr` even without direct attribute-to-attribute symmetry.

---

# 9. How Confidence Works

Each channel outputs:

- a score `s_k`
- a quality `q_k`

`q_k` is not "probability of correctness".

It is closer to:

- how trustworthy
- how complete
- how stable
- how non-noisy

that channel looks for this pair.

Examples:

- `q_label`: margin between best and second-best label pair
- `q_hier_family`: mean of coverage, strength, and specificity
- `q_sim`: mean of coverage, strength, and stability
- `q_diff`: mean of coverage, strength, and stability
- `q_attr`: mean of coverage, attribute informativeness, and stability

If a channel is empty, it returns:

- score = `0.5`
- quality = `0`

So empty channels automatically stop influencing the result.

---

# 10. Structural Aggregation

The structural channels are:

- hierarchy families
- `sim`
- `diff`
- `attr`

Each one gets a signal mass:

- `sigma_k = q_k * |s_k - tau|^gamma`

Interpretation:

- if a channel is weak or neutral, it gets little weight
- if a channel is strong and far from `0.5`, it gets more weight

Then:

- `S_struct` = weighted average of structural channel scores
- `Q_struct` = weighted average of structural channel qualities

Practical meaning:

- `S_struct` is the structural opinion
- `Q_struct` is how much we trust that structural opinion

---

# 11. Fusion, Ambiguity, and LLM Gating

## Lexical vs structural fusion

- `s_label` represents lexical evidence
- `S_struct` represents structural evidence

If both are active:

- `S_base = (1 - w_struct) * s_label + w_struct * S_struct`

where `w_struct` depends on how strong and reliable the structural side is.

## Ambiguity has two parts

### `U_ind`

High when `S_base` is near `0.5`

This catches:

- "I am unsure"

### `U_dis`

High when `s_label` and `S_struct` disagree and both are reliable

This catches:

- "lexical says yes, structure says no"
- "lexical says no, structure says yes"

## Final ambiguity

- `U = max(U_ind, U_dis)`

## LLM usage

If `U >= tau_LLM`, the LLM is allowed to influence the score.

---

# 12. Practical Comparison: Current vs Legacy

## Legacy mode

Question asked:

- "What is the best single context block around this entity?"

Main controls:

- `n_hops`
- `context_method`
- `best_path_method`
- `context_hop_penalty`
- token-budget heuristics

Main comparison:

- lexical vs one joined context

## Current mode

Question asked:

- "What evidence supports hierarchy similarity?"
- "What non-hierarchical triples support similarity?"
- "What informative triples remain unmatched?"
- "What auxiliary text supports the pair?"

Main controls:

- `hierarchy_max_depth`
- `max_hierarchy_triples_per_family`
- `n_hops`
- `max_object_triples`
- `max_diff_triples`
- `max_attr_items`

Main comparison:

- lexical vs aggregated structural evidence

---

# 13. Explanation Differences

## Current default explanations

Top level:

- lexical contribution
- structural contribution
- LLM contribution

Structural layer:

- hierarchy
- similarity
- difference
- attribute

Also exposed:

- `S_struct`
- `S_base`
- `S_final`
- `Q_struct`
- `U`
- `U_ind`
- `U_dis`
- `I_label`, `I_struct`, `I_llm`
- `I_hier`, `I_sim`, `I_diff`, `I_attr`

Triple / item level:

- selected hierarchy triples
- selected similarity triples
- selected difference triples
- selected attributes
- support / specificity / unsupported mass / importance

## Legacy explanations

Top level:

- lexical
- joined context
- LLM

There is no structural breakdown into four channels.

---

# 14. What To Tune First

If `sim` is too shallow:

- increase `n_hops`
- possibly increase `max_object_triples`

If `diff` is too weak:

- increase `n_hops`
- increase `max_diff_triples`
- check whether edge IC is suppressing useful relations too much

If hierarchy is too thin:

- increase `hierarchy_max_depth`
- expand `hierarchical_relation_families`

If attributes are too weak:

- keep `projection_include_literals = True`
- increase `max_attr_items`
- add custom `attribute_property_weights`

---

# 15. One-Sentence Summary

Legacy mode asked for one good context per entity.

Current mode asks for the best pair-specific evidence split into:

- what makes the pair similar lexically
- what makes it similar structurally
- what makes it different
- what auxiliary text supports the match

and then only calls the LLM when the fused evidence is still ambiguous.
