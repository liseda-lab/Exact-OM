# XR-2 common experimental protocol

Every XR-E study inherits these rules. Implementation conformance and research evidence are separate.

## Cohorts and progression

1. Generated clean theories, controlled corruption and typed intended consequences.
2. Controlled corruptions of training-side real ontology structure.
3. Actual held-out Conference matcher outputs.
4. Bio-ML scale/domain transfer, with 2024 selected-content and 2026 whole-ontology cohorts separated.

The main research programme includes Conference and Bio-ML; lack of licensed inputs or qualified reasoning is reported as unavailable/unknown, not silently replaced. The Complex-track Conference references are optional supplementary expression evidence, not preferred-repair labels. See [11](11-benchmark-evidence.md).

Captured alignments and evidence are identical across repair arms; do not rematch per arm. Fix ontology/import releases, alignment interpretation and entity-kind handling. Reference absence remains unspecified unless an independently justified negative is provided.

## Splits and leakage

- Generated train/development/test splits group by clean structural parent before renaming, score generation or corruption.
- Keep every corruption, orientation, explanation variant and matcher output of a Conference pair in one fold.
- Report pair holdout separately from whole-ontology holdout; familiar ontologies in new pairs are not unseen ontologies.
- Keep matcher transfer separate from ontology transfer.
- Training-side real structure and permitted reference supervision can be used in an adaptation arm. Test labels cannot select checkpoints, budgets, profiles or costs.
- Bio-ML public training/validation material may supervise declared training arms, with a separate development split if official validation is reported as evaluation. Hidden test labels are not accessed.
- Inference can read the presented test ontology/evidence. It cannot fit on hidden labels or post-repair explanations as if available before the decision.

Generated-only transfer remains a control, not the only allowed training regime. Once a test cohort informs redesign it is exploratory/regression data for that redesign.

## Required comparisons

No repair; score-based greedy deletion; exact fixed-cost deletion; directional weakening; richer mapping actions; selected ontology edits; deterministic symbolic scoring; HGT; R-GCN with the same readouts; no-graph control. Compare unary/pairwise benefit models and independent/correlated constrained proposal distributions.

Keep inventory, policy, resource limits and external evaluation objective fixed unless they are the intended contrast. Action ablations filter a common inventory without silent regeneration/refilling. Generator comparisons count retrieval, compilation, duplicate samples and verification as part of cost.

An exhaustive teacher is a controlled-case upper reference, not an expert oracle available on real inputs. Third-party repair systems are comparisons only when their output semantics and verification policy are compatible.

## Outcomes

Report every scheduled case and separate:
- logical status/scope, remaining witnessed violations, unknown and unsupported obligations;
- typed desired/unwanted consequence outcomes and non-vacuity;
- reference mapping measures with the evaluated entity kinds and partial reference interpretation;
- edits by family/provenance, changed consequences and profile sensitivity;
- vocabulary retrieval coverage, useful proposal coverage, value/selection errors and reasoner failures;
- objective and absolute integer gap, exact external regret only with a complete applicable teacher cache;
- total time, time to verified incumbent, search/verification/circuit/model costs, memory and bounded-stop behaviour.

One unsatisfiable class is not one explanation. Report discovered support counts with extraction budgets, not an invented average conflicts-per-mapping. Coherence and reference F1 do not establish the semantic acceptability of an ontology patch.

## Statistical and budget rules

Use clean structural groups and ontology pairs as grouped units; mappings, candidates, repeated corruptions and model seeds are nested observations. Conference pairs share ontologies; three Bio-ML pairs do not supply thousands of independent repair settings.

Publish paired effects, per-pair results, uncertainty intervals and all failures. Report verified-subset quality alongside coverage and an all-scheduled-case status table; never drop unknowns or impute them as coherent. Multiple comparisons and sample-size rationale are fixed before confirmatory evaluation. Pilot numerical thresholds are exploratory.

Freeze the external query basis independently of model predictions. A pilot may select parameters using development data; it cannot redefine success after observing held-out results. Record any later amendment.

Bound all stages, worker calls, memory where enforceable and cleanup. Persist partial results and unresolved jobs with their denominator. Finite retries are allowed within remaining budgets; retry-until-success is not.

## Reproducibility

Record source/dirty-patch hashes; ontology/import/capture hashes; schema/compiler/model/backend versions; grammar/menus/circuit/inventory/objective/policy/profile hashes; seeds/splits; hardware; cold/warm cache policy; verification/cut/replay records; and pending failures. Separate public file measurements, organiser-reported statistics, and newly measured experimental outcomes.

Any falsely authorised feasible result invalidates that arm's logical-validity claim. Finite tests do not prove universal reasoner soundness, and a negative experiment need not block reporting a correct implementation.
