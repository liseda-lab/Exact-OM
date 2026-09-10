# Labels and final submissions

The user instruction of 2026-09-10 governs the execution handoff: private test references
must not enter optimization. Freeze the selected method using development data, then generate
mappings/rankings for the respective tracks. Organizer-side test scoring is separate; private
gold is not a prerequisite for generating a submission. The earlier internal E17 confirmation
path is not the reference-free submission path and must not be launched to work around missing
private references.

## Permitted label derivation

| Input | Permitted use | What absence does not establish |
| --- | --- | --- |
| Bio-ML official training candidate pools | Gold targets are positives; provided distractors are scoped benchmark negatives. Union known train alternatives per source before labeling. | Unlisted ontology pairs remain unknown; these pools provide no natural-NIL sources. |
| Bio-LLM 2024 historical subsets | Explicit unmatched sources are benchmark NIL; positive and negative labels apply only to provided candidates. Use the declared N0 source split and retain N1 for reporting. | No exhaustive ontology-wide absence guarantee; no label transfer to Bio-ML 2026 ontology versions. |
| DISO public candidate pools | Candidate inventories for inference and submission only. | NIL appears in every pool and public answers are withheld. Empty converted target columns are not NIL labels. |
| BioKG public train/valid references | Typed positive correspondences and release-defined ranking preferences. | A nonpreferred candidate/relation can have positive hierarchical relevance; preferred-label subtraction is not confirmed semantic-negative generation. |
| OAEI-KG reference | Listed correspondences are positive; explicitly designated public development partitions retain their recorded research scope. | The official reference is partial; absent pairs and unmatched sources are unknown. |

The existing Bio-ML training converter already implements the first rule. No natural-NIL
labels are fabricated from these public references. An explicitly rejected pair establishes
only a pair negative. Even DISO's annotated benchmark NIL definition uses an assumption that
no undiscovered correct target exists, rather than exhaustive ontology-wide verification.

A newer BioKG release describes candidate-by-candidate inconsistency certificates under its
specific curated OWL 2 RL theory. That guarantee must not be applied to the currently declared
active v0.2.0 release or replaced by ordinary graph cycles. A clean consistency check does not
prove a mapping or NIL; a conflict among multiple candidates does not identify which candidate
is wrong. This optional reasoning route remains outside the validation job.

Official sources: [Bio-ML local task](https://bio-ml.oaei-ml.org/tasks/local/),
[DISO ranking contract](https://github.com/city-artificial-intelligence/diso-oaei/blob/main/tasks/ranking/ranking_task_index.md),
[BioKG metrics](https://github.com/liseda-lab/BioKG-Align-kit/blob/main/documentation/metric_families.md),
[BioKG data card](https://github.com/liseda-lab/BioKG-Align-kit/blob/main/documentation/data_card.md),
[OAEI-KG partial-reference contract](https://oaei.ontologymatching.org/2026/knowledgegraph/index.html).

## Legacy Bio-LLM NIL examples

The Bio-LLM subsets introduced in 2023 and unchanged in 2024 each contain 50 matched and
50 unmatched source classes for NCIT–DOID and SNOMED–FMA. The documented pool size is 100;
the actual published NCIT–DOID file has one 70-candidate pool and 99 pools of 100 (9,970 pairs).
SNOMED–FMA has 10,000 pairs. Preserve these supplied pools without padding.
[Official Bio-LLM documentation](https://krr-oxford.github.io/DeepOnto/bio-ml/#oaei-bio-llm-2023),
[pinned NCIT–DOID candidates](https://github.com/KRR-Oxford/LLMap-Prelim/blob/1972457c9be2664c627f822b5f5086bcbd63e274/data/ncit2doid/test_cands.tsv),
[pinned SNOMED–FMA candidates](https://github.com/KRR-Oxford/LLMap-Prelim/blob/1972457c9be2664c627f822b5f5086bcbd63e274/data/snomed2fma/test_cands.tsv).

There is no separately released NIL training split. The user authorized selecting a public
benchmark on 2026-09-10; E04 now declares N0 NCIT–DOID as a stratified source-level 60/20/20
research split, seed 17. This gives 60 training, 20 validation and 20 held-out sources, each
balanced between mapped and benchmark-unmatched. N1 SNOMED–FMA remains a separate 100-source
reporting case. Keep every candidate for a source together. Only N0 training labels fit the
NIL head; validation selects, and reporting labels must remain inaccessible until freeze.

Retain the original [2024 ontology archives](https://zenodo.org/records/13119437). Their
main-track reference files are not used to supplement this subset. Source status, pair labels,
and candidate-only inference inputs are separate hashed files. Case references remain
`known_incomplete`, with `confirmed_only` pair negatives and `benchmark_pool` NIL semantics.
Unlisted ontology pairs remain unknown. The fitted artifact and evaluation report explicitly
exclude an ontology-wide NIL claim. Supervised D0/D1 artifacts can overlap these historical
source IRIs and must not be reused; a frozen fitting recipe is distinct from learned weights.

This public historical research split is not a blind official Bio-LLM result. It provides a
small biomedical rejection experiment, separate from the current-track submission populations.
G0's NCIT–DOID operational validation does not train this NIL head.

## Submission boundary

1. Select thresholds, model/head weights, retrieval policy and treatment using permitted
   train/development data only. Freeze that configuration and its artifact identities.
2. Build a separate inference configuration from public ontology/candidate inputs. Set no
   development source cap, remove all evaluation/test reference bindings, and run alignment
   with `run_eval=False`. Training inputs, if fitting is needed, remain the permitted train
   split only. Do not invoke experiment selection or evaluation on submission outputs.
3. Global Bio-ML uses the complete eligible ontology population, not the smaller local-test
   candidate source list. Local tracks preserve every original public query and its entire
   candidate inventory. Do not invent scores for missing candidates during export.
4. Export and structurally validate the official format without gold. Retain source coverage,
   configuration, input/artifact hashes and output checksum. Submit only after the development
   configuration is frozen; no hosted submission or external upload is automated here.

| Track/task | Output |
| --- | --- |
| Bio-ML global, OAEI-KG | OAEI Alignment RDF with full IRIs, relation and confidence. |
| Bio-ML local | `SrcEntity,TgtCandidate,Score` TSV, exactly the supplied 100 candidates per query. |
| DISO ranking | JSONL `{qid, ranking}`, every supplied candidate including NIL, original public query IDs. |
| BioKG active release | Release-defined typed TSV `SrcEntity,TgtEntity,Relation,Score`; preserve official query identities and release rules. |

The legacy Exact local TSV is not the official Bio-ML or DISO submission format. Use
`tools/export_experiment_submission.py` for the supported strict conversions once predictions
exist; the exporter takes public pools/source populations and scored outputs, never references.
DISO's original query JSONL is required; a flattened TSV cannot recover lost query IDs.

Official format specifications: [Bio-ML global](https://bio-ml.oaei-ml.org/tasks/global/submission-format/),
[Bio-ML local](https://bio-ml.oaei-ml.org/tasks/local/submission-format/),
[DISO](https://github.com/city-artificial-intelligence/diso-oaei/blob/main/tasks/ranking/submission-format.md).
