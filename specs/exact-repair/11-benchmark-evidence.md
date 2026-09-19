# XR-2 benchmark evidence and statistical scope

These are public dataset/result counts used for planning. They are not results produced by Exact-Repair and do not establish a training set of preferred repairs. Snapshot and hash every downloaded artefact in a run; do not mix annual releases.

## Conference 2025

The public Conference collection contains 16 ontology files. The ra1 reference scope covers 7 ontologies and 21 pairs. Its 305 reference correspondences comprise 259 class, 29 object-property and 17 data-property mappings: mean 14.52 per pair, median 15, range 4–25.

| Ontology | Named classes | Object properties | Data properties |
|---|---:|---:|---:|
| cmt | 29 | 49 | 10 |
| conference | 59 | 46 | 18 |
| confOf | 38 | 13 | 23 |
| edas | 103 | 30 | 20 |
| ekaw | 73 | 33 | 0 |
| iasted | 140 | 38 | 3 |
| sigkdd | 49 | 17 | 11 |
| Total | 491 | 226 | 85 |

Seven published system sets contain 147 outputs, 1,699 mapping occurrences, 532 distinct raw correspondence tuples and 135 distinct normalised alignment sets (ignoring confidence and normalising confOf spelling). These are different denominators: 147 outputs are not 147 independent ontology pairs.

The seven-ontology syntax inventory contains 566 expanded disjointness pairs, 178 existential restrictions, 13 universal restrictions, 36 cardinality restrictions, 301 domain and 301 range axioms, 61 functional-property declarations, 21 inverse-functional declarations and 4 transitive declarations. Syntax frequencies are not conflict frequencies.

The organiser's published logical evaluation gives:

| System | Mapping occurrences | Incoherent outputs | Unsatisfiable-class occurrences | Conservativity violations |
|---|---:|---:|---:|---:|
| ALIN | 325 | 7 | 107 | 111 |
| AgentOM | 275 | 8 | 136 | 101 |
| LSMatch | 147 | 0 | 0 | 2 |
| LogMap | 218 | 0 | 0 | 21 |
| LogMapLt | 208 | 3 | 18 | 97 |
| MDMapper | 216 | 3 | 39 | 81 |
| Matcha | 310 | 9 | 115 | 90 |
| Total | 1,699 | 30 | 415 | 503 |

Thus 30/147 = 20.4% of outputs are incoherent in that evaluation; 415/147 = 2.82 unsatisfiable-class occurrences per output, or 415/30 = 13.83 per incoherent output. None of these ratios is an average number of conflicts per mapping. A conflict explanation is a support set, often shared by several unsatisfiable classes, and one mapping can belong to many explanations. Our experiments must measure explanation incidence explicitly and report explanation completeness/limits.

One real cmt–ekaw AgentOM case contains RejectedPaper ≡ Rejection and Document ≡ Document. On the cmt side, RejectedPaper is below Paper and Document; on the ekaw side, Rejection is below Decision, disjoint from Document. Together the mappings make RejectedPaper unsatisfiable. The reverse first subsumption can instead make Rejection unsatisfiable. Direction selection needs joint verification; simply reversing an equivalence is not a universal repair.

The Conference portion of the Complex track provides auxiliary examples involving 3 ontologies, 78 simple and 79 complex 1:n correspondences. It is not a gold set of repair actions or user preferences.

Sources: [track and downloads](https://oaei.ontologymatching.org/2025/conference/index.html), [published evaluation](https://oaei.ontologymatching.org/2025/results/conference/index.html), [Complex track](https://oaei.ontologymatching.org/2025/complex/index.html).

## Bio-ML: separate historical and current releases

The 2024 selected-ontology benchmark has five task variants:

| 2024 task | Source classes | Target classes | Equivalence references | Subsumption references |
|---|---:|---:|---:|---:|
| OMIM–ORDO | 9,648 | 9,275 | 3,721 | 103 |
| NCIT–DOID | 15,762 | 8,465 | 4,686 | 3,338 |
| SNOMED–FMA | 34,418 | 88,955 | 7,256 | 5,453 |
| SNOMED–NCIT, pharmacology | 29,500 | 22,136 | 5,803 | 4,224 |
| SNOMED–NCIT, neoplasia | 22,971 | 20,247 | 3,804 | 213 |

The 2026 whole-ontology release uses three pairs. Public data snapshot: c454644a334ab43754bc1070c0fb7fdd56a90a1d.

| 2026 pair | Global matching train | Global matching validation | Local ranking train | Local ranking validation | Local ranking test |
|---|---:|---:|---:|---:|---:|
| NCIT–DOID | 3,262 | 530 | 3,845 | 634 | 1,819 |
| SNOMED–FMA | 3,239 | 541 | 6,817 | 1,178 | 3,414 |
| SNOMED–NCIT | 17,283 | 2,864 | 21,669 | 3,614 | 10,911 |
| Total | 23,784 | 3,935 | 32,331 | 5,426 | 16,144 |

Local ranking has 100 candidates per query. Those rows are correspondence/ranking supervision, not complete repair instances, explanations or expert preferences. No public global-test count is inferred from the local-ranking count.

The published baseline coherence snapshot ec436a97f49875227dedf634faa5280b1376bda9 reports merged class sizes 231,503, 490,694 and 597,931. A one-class NCIT–DOID denominator discrepancy across public sources is left explicit rather than used to recompute rates.

| Baseline | NCIT–DOID ELK / HermiT | SNOMED–FMA ELK | SNOMED–NCIT ELK |
|---|---:|---:|---:|
| LogMapLt | 37,944 / 37,946 | NA | 534,393 |
| AML | 64,366 / 64,371 | NA | 13,494 |
| LogMap | 8 / 8 | 97,408 | 448 |
| BERTMap | 14 / 14 | 76,300 | 9,801 |
| BERTMapLt | 10,444 / 10,535 | 127,556 | 77,844 |

These are published unsatisfiable-class counts, not our reruns. ELK is incomplete for unsupported expressive axioms; such detections are lower bounds. Published HermiT counts are available here only for NCIT–DOID. NA does not by itself identify a timeout.

The corresponding reference-coherence report gives 17,850 and 466,306 pre-repair ELK unsatisfiable classes for NCIT–DOID and SNOMED–NCIT; the SNOMED–FMA check failed with a 128 GB heap. Reported post-repair ELK zeroes do not certify full OWL coherence.

Sources: [current task description](https://bio-ml.oaei-ml.org/tasks/), [immutable data snapshot](https://huggingface.co/datasets/OAEI-ML/bio-ml/tree/c454644a334ab43754bc1070c0fb7fdd56a90a1d), [baseline coherence](https://github.com/liseda-lab/OAEI-Bio-ML/blob/ec436a97f49875227dedf634faa5280b1376bda9/baseline_coherence.json), [reference coherence](https://github.com/liseda-lab/OAEI-Bio-ML/blob/ec436a97f49875227dedf634faa5280b1376bda9/reference_coherence.json), [historical benchmark description](https://krr-oxford.github.io/DeepOnto/bio-ml/).

## Training and reporting consequences

- Keep every matcher output, corruption and extracted neighbourhood from the same parent pair in the same split. For stronger Conference transfer, holding out ekaw puts all six incident pairs in test and leaves fifteen pairs for training/development.
- Whole-ontology holdout is difficult on Bio-ML because two pairs share SNOMED and two share NCIT. Report overlap; random mapping splits do not establish ontology-level transfer.
- Controlled corruptions on real structure are training data with simulated intent. Untouched held-out matcher outputs are the real repair evaluation.
- Reference alignments offer correspondence supervision. Absence from a reference is not a negative, and reference correspondences can themselves cause incoherence.
- Pre-register explanation-count definitions, mapping incidence, baseline unsatisfiability, new unsatisfiability, verification coverage and failure status. Do not infer conflict frequency from syntax or unsatisfiable-class totals.
