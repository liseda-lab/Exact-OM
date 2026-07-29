# E12 — Instance-Equivalence Matching

**Motivation** (audit obs. 12; WP-F capability gap): instance matching is enabled, but
class-oriented lexical and hierarchy evidence is not sufficient for ABox identity. Names are
frequently ambiguous; types, relation neighborhoods, and literal values can be decisive.
Conversely, identifiers and near-unique literal values can leak trivial answers or cause false
matches if their semantics differ across KGs. Instance scale and hub structure also create a
different retrieval and statistical regime from TBox matching.

## Research questions

- **RQ12.1**: How much do type, ABox-neighborhood, and literal-value evidence improve instance
  equivalence over labels alone?
- **RQ12.2**: Which evidence is robust for ambiguous-name, sparse-neighborhood, hub, and
  cross-type instances rather than only for easy exact-label cases?
- **RQ12.3**: Does anchor-guided neighborhood consistency (E02) add value beyond independent
  per-instance scoring, and how much error propagation does it introduce?
- **RQ12.4**: Can Exact-OM retain candidate recall and runtime at realistic instance scale?

## Hypotheses

The full instance bundle improves F1 by at least 2 points on the ambiguous-name slice and
candidate Hits@1 by at least 1 point overall versus label-only. Type evidence should give the
largest precision gain; ABox neighborhoods should give the largest non-exact-label gain.
One-pass anchor rescoring is expected to help, while iterative rescoring may amplify hub errors.

## Experimental change

Expose independently switchable instance evidence groups behind experiment-only config:

- `labels`: normalized labels/aliases;
- `types`: asserted `rdf:type` plus one-level class closure, with matched-type confidence;
- `relations`: direction- and predicate-aware ABox neighborhoods, degree-normalized;
- `literals`: property-aware literal similarities with explicit identifier, numeric/unit,
  date, and free-text handling;
- `anchors`: the E02 one-pass structural agreement using only predicted high-confidence anchors.

Do not compare raw identifiers across sources unless the dataset descriptor declares their
namespace semantics compatible. Do not use a reference mapping to build neighborhoods,
features, or blocking keys.

Implementation boundary: bundle switches live under `matching.channels.instance.*` in
`exact/impl/datasets/pair_adaptive_context.py` and the existing channel scorer. Multi-view
blocking belongs to `exact/utils/candidate_generation.py`/the dataset assembly layer; it must
not alter class/property pools when disabled.

## Arms

At fixed label-only retrieval: label-only / +types / +relations / +literals / full, followed by
leave-one-group-out ablations. Separately test multi-view retrieval
{labels, labels+types, labels+relation-neighborhood signatures} at equal mean pool size. Cross
the best independent-pair arm with E02 {off, one-pass, iterate} on the primary instance task.

Add `relations_shuffled` as a negative control: independently rewire ABox relation targets
within each KG while preserving entity in/out-degree and predicate/direction counts, then build
the same number and shape of neighborhood features. Labels, types, literals, candidates, and
references remain untouched, and the shuffle never crosses KGs. Real-neighborhood gains count
as structural evidence only if they exceed this feature-volume control. E23 reuses this control
unchanged for its learned graph channel, where the same confound is stronger; keep it a shared
implementation rather than reimplementing it per experiment.

## Validation

OAEI-KG instance tasks are primary; add any pinned RDF/CSV KG task only when the inventory
confirms instance-kind references and a non-trivial test set. Mini fixtures validate behavior
but carry no performance claim. Three seeds, with source-entity paired bootstrap and a
connected-component cluster-bootstrap sensitivity analysis.

Primary metric: task-macro instance F1. Secondary: candidate recall@k, MRR/Hits@1, P/R,
coverage, peak memory, load/index time, and scoring time per 1,000 candidates. Pre-register
slices by label equality, source/target degree quartile, type availability, name ambiguity, and
literal availability. Report performance after removing exact-label pairs as the main
"non-trivial identity" diagnostic. Report real-versus-shuffled neighborhood deltas overall and
by degree quartile.

## Promotion

Standard criteria apply on instance-only results. The selected bundle must improve the
non-exact-label slice with CI excluding zero, keep each degree quartile within 1 F1 point of its
baseline, and stay within 1.5× peak memory and 1.2× per-candidate scoring time. Anchor rescoring
promotes separately and only if its wins survive the hub and sparse-neighborhood slices.

**Pre-registered criterion-2/3 overrides**: degree quartiles use a 1-point regression bound
because they subdivide already sparse task references; peak memory may reach 1.5× because the
ABox indexes are the capability under test. The default 0.5-point/1.2× gates are still reported,
and per-candidate scoring time remains capped at 1.2×.

Where instance training references exist, the supervised instance arm is the E18 ranking head and
E19 fusion weights fitted on the instance slice, reported against the E15 label-free selector —
not a separate instance-only learner. Instances are the kind most likely to have many labelled
candidate rows and the most ambiguous per-row evidence, so they are also E22's most informative
label-efficiency slice.

**Effort**: M–L. **Risks**: reference incompleteness is often worse for instances and makes
apparent false positives uncertain; literal values may contain personally sensitive data in
some KGs; high-degree nodes can dominate neighbor overlap. Record reference assumptions,
redact literal values from qualitative artifacts, and use degree-normalized evidence.
