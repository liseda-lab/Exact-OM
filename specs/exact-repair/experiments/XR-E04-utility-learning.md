# XR-E04 — Neural benefit and proposal learning

Research question: RQ2.

## Data

Generated training parents; controlled real-structure adaptation; grouped held-out evaluation.

## Comparison

Compare uniform/confidence/symbolic controls, MLP features, HGT and R-GCN with matched attention readouts; unary versus pairwise benefit; regression plus ranking; exact versus declared partial supervision.

## Measurements

Measure full-repair benefit error, ranking accuracy, decoded teacher regret, useful proposal coverage, training/label compute and sensitivity to costs.

## Interpretation and acceptance

Keep loss targets and explicit costs separate. Proposal likelihood is not value. Complete-cache targets and partial comparisons must be reported separately. Do not use exact test regret for checkpoint selection.

Shared controls, split rules, resources and status reporting are defined in [02](../02-experimental-protocol.md).
