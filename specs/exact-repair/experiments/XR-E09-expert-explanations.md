# XR-E09 — Expert review of state-level repair explanations

**Status:** planned; blocked until human-participant ethics/data-governance approval<br>
**Depends on:** XR-E00; XR-WP5 stable explanation/counterfactual schemas; XR-E01 eligibility gate<br>
**Question:** RQ10<br>
**Co-primary outcomes:** adjudicated review accuracy and task completion time

## Authorized claim

Whether certificate-derived state-level explanations with verified counterfactual alternatives help
the sampled ontology/domain experts review repair decisions more accurately and/or quickly than raw
mapping scores or axiom-justification views.

The study does not validate the logical oracle, infer general usability, or establish that expert
agreement equals ontology truth.

## Conditions

Use a randomized, counterbalanced within-participant or blocked design selected before approval:

1. provisional mapping, matcher score/evidence summary only;
2. axiom justification plus affected classes/queries;
3. structured state-level explanation: selected revision, responsible conflict, consequences,
   safety/optimality distinction, uncertainty, and verified counterfactual alternatives.

All conditions receive the same underlying repair case and no condition may reveal the adjudicated
answer. Generated prose, if used, is identical in factual content to the structured certificate and
is separately labelled/ablated.

## Case sampling

Freeze a balanced set across:

- deletion, direction weakening, guarded, qualification, and replacement decisions;
- simple versus multi-state conflicts;
- optimal safe, safe-with-gap, fallback, and unsafe-core diagnostics;
- multiple plausible/equal alternatives;
- at least the domains represented by recruited expertise.

Only cases with replayed certificates and an adjudicated reference are eligible. Pilot cases are
not reused in confirmatory measurement. Avoid protected ontology text or unpublished matcher data
unless participant consent/governance explicitly permits it.

## Participant and adjudication plan

Before recruitment, freeze:

- expertise inclusion criteria and target sample/power rationale;
- compensation, consent, withdrawal, storage, and anonymization policy;
- case-to-participant assignment and order randomization;
- adjudication panel/process and acceptable-answer sets;
- training/tutorial and washout procedures;
- exclusion rules based on preregistered attention/technical failures, not outcomes.

Record domain familiarity as a moderator. Expert corrections collected during this evaluation are
held out from model training until the study and confirmatory split are closed.

## Tasks and outcomes

For each case, participants choose/approve a repair, identify why a higher-value state was rejected,
and rate confidence/clarity. Measure:

- correctness against the adjudicated set;
- review time and completion rate;
- conflict/counterfactual comprehension;
- confidence calibration;
- inter-rater agreement;
- perceived clarity/usefulness/workload (secondary);
- requests for additional ontology context.

Instrumentation excludes idle time according to a frozen rule and logs condition/case/order without
capturing sensitive free text beyond consented fields.

## Analysis and decision rule

Use a preregistered mixed-effects model or paired analysis accounting for participant and case.
Correct the two co-primary tests or define a hierarchical decision rule at G4/ethics freeze.

RQ10 is supported only if the state-level condition improves the frozen accuracy/time criterion
without a material loss on the other co-primary outcome. Subjective preference alone is
insufficient. Report heterogeneity by expertise/domain as secondary unless powered/preregistered.

## Artifacts

- approved protocol/consent and data-management plan references;
- de-identified condition/case/response/time data where permitted;
- case certificates, UI build hash, and counterfactual replay reports;
- randomization and adjudication manifests;
- analysis code and disclosure-control report.

## Non-claims

- Expert approval does not override a reasoner violation.
- Faster review does not imply better repair quality unless accuracy meets the decision rule.
- Results generalize only to sampled experts, domains, tasks, and interfaces.
- Study labels cannot leak into XR-E04/XR-E06 confirmatory training/evaluation.
