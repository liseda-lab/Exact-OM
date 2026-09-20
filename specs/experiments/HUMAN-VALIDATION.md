# Human auditability: ranking and mixed-case validation

**Updated 2026-09-20. Accepted protocol direction; new participant study pending.** The previous
pilot remains historical evidence. The current primary task is partial reranking of five
candidates, with explicit none/insufficient-information responses and adjudicated cases in
which no displayed candidate is correct. This supersedes the earlier accept/reject-first study
outline; it does not alter the separate automatic-matching campaign or its 336–504 node-hour budget.

The normative protocol is [ranking study design](../explanation-framework/11-study-design-and-ranking.md).
Product/deployment boundaries are in [10](../explanation-framework/10-products-and-bundles.md),
transactional anonymous participation in [12](../explanation-framework/12-study-service-and-render.md),
and participant questions/setup in [13](../explanation-framework/13-study-questionnaires.md).
The [backend gate](../explanation-framework/07-validation-and-handoff.md) precedes full frontend
implementation; human recruitment and the final participant study are not backend prerequisites.

## Required design

Use a frozen two-condition within-participant design: explanations plus candidates/scores versus
the same candidates/scores with access to external ontology inspection. Participants install
Protégé and both ontologies before scored tasks; record actual external-resource/tool use after
every case in both conditions. Do not label optional Protégé use a randomized treatment.

Each participant sees disjoint cases under each condition; complementary assignments expose each
case under both conditions across participants. Counterbalance block order and original system
reference ranks, preserve real scores/order, and record seeds/allocations durably. Mixed case
counts are configurable; the recommended 24-case example and shorter 20-case alternative are
specified in 11. A duration pilot precedes final workload selection. Neither count is a power calculation.

Positive cases require independent adjudication of acceptable equivalents among all five. Negative
cases require none among those five, not merely missing reference entries, and do not imply
ontology-wide absence. Constructed negatives and natural retrieval failures retain separate
provenance/analysis. Keep unresolved cases and reference errors visible under predeclared rules.

Primary MRR and system-to-human improvement apply to answer-present cases; submitted omissions,
none and insufficient evidence score zero there. Negative-case correct-none, false endorsement
and uncertainty rates are separate outcomes. Do not assign invented reciprocal-rank credit to
none. Unsubmitted cases remain missing, with attrition/exclusion/sensitivity rules frozen in advance.
Report Top-1, ranking depth, omissions, helpful/harmful changes and task time alongside MRR.

## Preparation and claims

Use grounded OpenRouter outputs, meaning/hierarchy semantics and source/target coverage gates
before exposure. Formative usability/practice precedes confirmatory freeze. Original and generated
facts must not contradict each other through lost qualifiers or missing-as-negative interpretation.
Freeze real cases, answer keys, assets, forms, conditions, analysis and software versions before
collection. Final cases are distinct from prompt/UI development and the public demo.

Choose participant numbers and a smallest worthwhile improvement before results; account for
participant/case clustering and domain/ontology expertise. Small pilots remain descriptive and
cannot establish separate subgroup benefits. Measure component preference/usage descriptively;
it does not isolate causal component effectiveness. Fluency, trust or agreement with Exact does
not establish better human ranking or safe rejection of unsuitable candidates.

Case time includes legitimate external-tool work; browser invisibility must not automatically
stop timing. Save explicit breaks, unknown gaps, rankings, consultation and meaningful interaction
events independently. Follow consent/information and minimal-identification requirements in 12/13;
keep answer keys and invitation credentials out of participant payloads and analysis exports.

The study app must pass end-to-end recovery, isolation, Render redeploy/database restore and
launch-readiness checks after frontend implementation. Do not contact participants or launch a
public study through this specification-writing task. Recruitment, applicable consent/ethics and
participant availability remain separately scheduled from the unattended compute campaign.
