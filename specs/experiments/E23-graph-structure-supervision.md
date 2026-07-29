# E23 — Supervised Graph-Structure Matching for TBox-Poor Inputs

**Motivation** (audit obs. 23; extends obs. 2, 12, 13): the evidence model is TBox-shaped. The
hierarchy channel reads depth-2 ancestors with `1/(d+1)` specificity, the attribute channel reads
annotations, and the lexical channel carries most of the weight. On a deep, well-labelled
biomedical ontology that is the right design. On a plain knowledge graph — OAEI-KG instance
tasks, BioKG/KG-Align CSV inputs, any source whose "class hierarchy" is one shallow type level or
absent — the hierarchy channel degenerates toward its `τ` default and the matcher falls back to
labels, exactly where labels are least reliable.

But those inputs are not structurally poor; they are structurally *different*. What they lack in
subsumption depth they carry in relational density: many typed edges per entity, repeated
relational patterns, and neighborhoods that identify an entity far more precisely than its name
does. That is the regime the knowledge-graph entity-alignment literature targets, and its
methods are supervised almost without exception — cross-graph representations trained from seed
alignments. Exact-OM has an unsupervised structural method (E02's anchor propagation) and no
supervised one.

This experiment adds the supervised structural channel and, more importantly, establishes
*where on the structural axis it pays*. That crossover is the scientific contribution; a matcher
that silently applies TBox-shaped evidence to a TBox-free graph is the failure it prevents.

## Research questions

- **RQ23.1**: On TBox-poor inputs, does a supervised graph-structure channel improve alignment
  over the current evidence model at fixed retrieval and fixed lexical evidence?
- **RQ23.2**: Where is the crossover? As an observable structural profile moves from deep-TBox to
  relation-dense, at what point does the graph channel begin to pay, and where does it stop?
- **RQ23.3**: How much does it beat E02's label-free anchor propagation, and how many seed
  alignments does that margin require?
- **RQ23.4**: Is the gain genuine relational structure, or feature volume and degree?
- **RQ23.5**: Does a transductive per-pair embedding earn its cost over an inductive structural
  head that transfers across graphs?
- **RQ23.6**: Can the channel deliver its gain through σ-mixing with the decomposition invariant
  intact, or only as a standalone matcher that replaces the evidence model?

## Hypotheses

On TBox-poor tracks the supervised graph channel improves macro F1 by at least 2 points over the
current model, with the gain concentrated on the non-exact-label slice. On Bio-ML and Anatomy it
is expected to be neutral to slightly negative: the hierarchy channel already carries that
structure and adding a correlated channel dilutes σ-mixing. The crossover should be predicted by
the structural profile rather than by domain — the falsifiable form of the claim, since domain
and structure are confounded across natural tracks.

Transductive embeddings should win within a pair and fail to transfer at all; the inductive head
should be weaker but portable, which likely makes it the better default. The degree-preserving
shuffle control is expected to absorb a substantial minority of the naive gain, and reporting the
gain net of that control is expected to change the ranking of arms.

## Change

A new channel through the standard σ-mixing, so the exact importance decomposition is preserved
by construction — `matching.channels.graph: off|inductive|transductive`:

- `inductive`: a supervised head over relational features computable without per-pair training —
  predicate-profile overlap, degree and type signatures, anchored neighborhood agreement reusing
  E02's machinery, and small motif counts. Portable across graphs; fits with E18's trainer.
- `transductive`: a cross-graph representation trained on training-split seed alignments over the
  normalized relational graph, producing `s_graph` from calibrated embedding similarity.
  Per-pair fitted artifact, bound to both graph hashes.
- Both supply `q_graph` from neighborhood coverage and degree reliability, so a sparse or hub
  entity contributes proportionately less authority rather than equally.
- `graph_only`: a standalone arm scoring on the graph channel alone. Diagnostic ceiling for
  RQ23.6; it bypasses the evidence model and cannot promote.

**Structural profile** (added to E00's `dataset_inventory.parquet`, and a deliverable regardless
of promotion): per task and per kind, hierarchy depth distribution, ancestor coverage, class-to-
instance ratio, axiom density, triples per entity, distinct predicates, and relational entropy,
reduced to a pre-registered `tbox_richness` and `relational_density` pair. The reduction is fixed
before any result is opened, since a profile fitted to the outcome would make RQ23.2 circular.

**Seed discipline**: seed alignments come from the training split only. The propagation or
message-passing graph must never contain validation or test pairs as edges — the standard leakage
failure in this method family, and the one the harness asserts explicitly here. Any contrastive
negative sampling follows the programme's reference-completeness rule: unlisted cross-graph pairs
from incomplete references remain unlabelled and require a filtered positive-unlabelled objective
or explicit semantic incompatibility.

Implementation boundary: one channel module plus its registration in the scorer's channel list,
following E02 and E06. Graph normalization is E13's, consumed and not re-implemented. Embedding
training is an experiment-side tool; runtime loads a fitted artifact and refuses one whose graph
hashes do not match. Adding `s_graph/q_graph` changes the fusion feature/explanation schema: an
E19 learned-fusion artifact fitted without that channel is incompatible and must be refit rather
than silently reused.

## Arms & validation

Eligible tracks must span the structural axis, not just its poor end — a crossover cannot be
located from one side. TBox-poor: OAEI-KG instance and class tasks, BioKG/KG-Align CSV pairs when
published. TBox-rich controls: Bio-ML and Anatomy.

**Controlled TBox ablation** (the causal arm for RQ23.2): take a TBox-rich pair and progressively
remove hierarchy depth and axioms while holding labels, candidates, entities, and seeds fixed,
synthesizing the poor regime without changing domain. Natural tracks confound structure with
domain, vocabulary, and reference quality; this arm does not. It is the primary evidence for the
crossover claim, with natural tracks as its external check.

The primary causal block is {graph off, inductive, transductive} × structural regime with
retrieval, fusion, and reranking pinned identically within every graph-off/on contrast. Run it
once under the promoted label-free stack and once under the current/promoted in-pair supervised
stack where labels exist; do not compare a graph-on arm with E18 enabled against a graph-off arm
without it. E02 anchor rescoring is the mandatory label-free structural comparator.

On development data, run graph × E18-reranker and graph × E19-fusion 2×2 interactions. Any
interaction that changes sign or exceeds the relevant MDE becomes a frozen confirmatory contrast,
and any promoted combination is rechecked in E17. Reuse E12's `relations_shuffled` negative
control unchanged for RQ23.4 — a degree- and predicate-count-preserving rewiring — and report
every graph gain both raw and net of it. `graph_only` remains a non-promotable diagnostic.

Three seeds. Primary: macro F1 on TBox-poor reporting tasks and the mandatory non-exact-label
slice, with the crossover location from the ablation arm as the other co-primary endpoint declared
before results open. Secondary: local MRR/Hits@1, candidate recall to prove retrieval was held fixed,
per-degree-quartile results, channel-importance and weight-entropy shifts, coverage/abstention,
embedding-training wall time and peak memory reported once per fitted artifact, and per-seed-count
curves feeding E22.

## Promotion

Standard criteria, applied within the structural regime rather than globally — a TBox-poor gain
must not be averaged against a TBox-rich regression, and the channel is expected to be promoted
per profile, as E06 anticipates for its non-biomedical case.

- The channel must earn its promotion **net of the shuffle control**. A gain that the
  degree-preserving rewiring reproduces is feature volume and does not promote.
- The decomposition invariant is a hard constraint: `graph_only` cannot promote regardless of its
  score, and a promoted channel must reconstruct the final score algebraically.
- Supervised graph structure ships as a `supervised` resolution of the structural component;
  **E02's anchor propagation is its `label_free` resolution** and remains the default wherever no
  seed alignments resolve. E22 fits the effective-seed × structural-profile policy under which
  `auto` should prefer the supervised channel — for the transductive arm this boundary is expected
  to be high.
- A transductive artifact is bound to both graph hashes and may not be reused across pairs. If
  its per-pair training cost exceeds the run it serves, it is reportable science but a poor
  default; the results note states that trade-off explicitly rather than burying it in a cost
  column.

**Pre-registered criterion-3 override**: the transductive arm may exceed the 1.2× wall-time bound
by its one-time per-pair training cost, reported separately from per-run inference cost, because
per-pair fitting is the capability under test. Inference-time cost keeps the default bound, and
the inductive arm receives no override.

**Paper contribution**: ontology matching and knowledge-graph entity alignment are largely
separate literatures with separate benchmarks, and the second is rarely evaluated inside a
TBox-oriented matcher or against a strong lexical/evidence baseline. Locating the crossover on a
measured structural axis — using a controlled TBox ablation rather than a comparison across
tracks that differ in every other way — is a result neither community currently reports, and it
converts "use a GNN for KGs" from folklore into a threshold a system can act on.

**Effort**: L. **Risks**: entity-alignment benchmarks are known to be partly solvable by string
similarity alone, so a headline gain here would be uninformative without the non-exact-label
slice — it is mandatory, not secondary; hub entities dominate neighborhood overlap, hence
degree-quartile reporting and degree-normalized evidence; seed edges leaking into the propagation
graph would invalidate the experiment silently, hence the explicit assertion; transductive
training cost may exceed any plausible deployment budget, which is a finding rather than a
failure; and a correlated structural channel can dilute σ-mixing on TBox-rich tracks, which is
why promotion is per profile and weight entropy is reported.
