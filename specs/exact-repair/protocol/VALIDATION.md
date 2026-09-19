# XR-2 protocol validation — 19 September 2026

The resolved pilot and inherited smoke configuration pass the strict local schema and cross-field checks.

| Configuration | Train requested | Development requested | Test requested | Total |
|---|---:|---:|---:|---:|
| Pilot | 384 | 96 | 96 | 576 |
| Smoke | 32 | 32 | 32 | 96 |

Counts are requests before generation rejection, label limits and composition-holdout exclusions. They are not completed datasets.

Checks reject obsolete versions, unknown fields, missing actions/cohorts, duplicate seeds, invalid head dimensions, negative/zero required costs, invalid numeric values, inconsistent deadlines, inherited cycles, duplicate JSON fields and non-JSON NaN/Infinity constants.

Semantic regressions reject treating unknown as acceptance, excluding pending candidates from upper bounds, using pending assignments as logical cuts, mandatory all-off feasibility, test-label features, unconditioned mixture component weights, intersection-based desired-disjointness non-vacuity and exact teacher distributions from incomplete caches.

Both configurations retain all action families, HGT/matched controls, circuit generation, finite-pool optimisation, partial-label handling and generated/Conference/Bio-ML cohorts. No actual corpus generation, training, OWL/MaxSAT/circuit execution or real-ontology experiment was run by the static validator.

For the accompanying finite model see [reference/VALIDATION.md](../reference/VALIDATION.md).
