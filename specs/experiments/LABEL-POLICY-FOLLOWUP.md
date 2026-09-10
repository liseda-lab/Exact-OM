# E22 count-policy follow-up

`exact.experiments.label_policy.materialize_followup(source, suite, manifests)`
prepares the bounded `E22-policy` comparison after E22 completes. It does not run
experiments. The campaign hook supplies the selected component recipe and declares:

```yaml
frozen_constants:
  label_policy_followup:
    producer: E22
    budget: 100
    fixed_minimum: 100
    components: [rerank, accept]
    practical_effect: 0.003
```

The initial implementation compares the joint rank/accept head on one independent
D1 development task at seed 17. The three passive E22 budgets must share a training
pool, label order, reporting population and entity kind. Each completed budget cell
supplies its actual `fitting/**/training_units.json`. The label-free cell supplies
the paired control. Missing, interrupted or ambiguous artifacts stop preparation.

The adapter uses the existing source-group bootstrap on those completed runs. It
freezes the smallest effective source count whose lower paired F1 interval exceeds
the declared practical effect. An inconclusive curve leaves the fitted policy
label-free. The fixed comparator uses 100 effective source groups by default.
Both policies receive the same passive D1 training budget; explicit confirmed
labels or a verified complete-reference policy are required. D1 training/reporting
source groups must be disjoint from the donor reporting/training populations at
the respective evaluation boundaries. No final-test labels are read.

Artifacts live in `<suite>/screen/policies/E22-policy/`: the paired evidence,
training counts, count policy, follow-up overlays and resolved experiment
configuration. Writes are immutable and identical replay is supported. Both arms
clear inherited rank artifacts so their intended training decision is applied.
The runtime checks the ontology identity and component-specific count definition.
Policy creation alone is not evidence of benefit: the separately executed paired
fixed-versus-fitted comparison must provide that evidence.

`tests/label_policy_test.py` covers supported and inconclusive curves, strict
configuration resolution, immutable replay, unknown labels, population overlap,
unequal joint counts, completed artifact discovery and real paired resampling with
synthetic evaluator counts. These are CPU implementation checks, not experiments.
