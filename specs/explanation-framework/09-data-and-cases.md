# Dataset bindings and bounded case plan

Primary: Bio-ML 2026 **NCIT–DOID**, HF repository `OAEI-ML/bio-ml`, resolved revision `c454644a334ab43754bc1070c0fb7fdd56a90a1d`. Verify that revision and files before implementation; adopt a later revision only by an explicit data/design amendment, never a mutable silent update.

| Input | Hosted path | SHA-256 |
|---|---|---|
| NCIT | `ontologies/NCIT-bd7a9a8a75e5-Thesaurus.owl` | `1a7182a7327ebc4181f7d6b0f7e81ed04dd258f1a86bd8f560e4a0d61439d58a` |
| DOID | `ontologies/DOID-611355c44553-doid.owl` | `611355c445537fcf4bae2c519f1b3598af5a8fea793274316e35525b7d05e945` |
| FMA (targeted later check) | `ontologies/FMA-beb3dc47979a-fma.owl` | `beb3dc47979ad5434ef70fd02af4307f147f2023f7d8c2c57103b995191194c3` |

NCIT's filename pin is the source ZIP hash, not its extracted OWL hash. NCIT/DOID/FMA now ship in the dataset; only SNOMED is separately obtained. Update `bioml_hf.yaml`: actual `SHA256SUMS.txt`, hosted paths/hashes, public validation refs and no nonexistent repaired local pools. Verify reference/candidate input hashes and ontology IRI membership. Resolve filenames/paths from the user's available data; do not conclude data is unavailable from an old stub. Full-file hash validation must catch incomplete/empty reads regardless of apparent file metadata.

The verified root-file training profile is recorded in [evidence/context-profile-summary.json](evidence/context-profile-summary.json): NCIT definitions 3741/3741, DOID 2577/3625, FMA 548/6653 unique local-training reference endpoints. These are descriptive training-cohort counts, not held-out coverage or inferred completeness. Literal named subclass edges exclude parents inside equivalent expressions; imports were not resolved by the profile. DOID's external import and predicate labels require a separate lock for production.

## Case roles

- **Operational smoke:** 12 public-development pairs, deliberately including well-described, sparse, multiple-parent, restriction-heavy and close-alternative examples; supplemented by synthetic semantic/error cases. These prove functionality, not population benefit.
- **Prompt/UI development:** 48 pairs grouped by source, drawn deterministically from declared public training/development sources. Select approximately 12 each for well-described, missing/asymmetric information, structural complexity and close-alternative/scope difficulty. Strata may overlap: record tags and actual counts, never fabricate 'incorrect' or 'ambiguous' ground truth to fill a quota. Include several candidates from a frozen pool and a sparse legacy OMIM–ORDO case set.
- **Formative usability:** a coherent disease subset matched to participant expertise, with both ordinary and challenge cases, separate from final reporting cases. Model development can use these cases but cannot claim confirmatory results on them.
- **Wider validation:** small targeted SNOMED–FMA anatomy panel after both sides are profiled; SNOMED–NCIT scale/clinical verification if needed. These share ontologies and do not constitute independent ontology holdouts. Property matching/KGA require their actual data/reference semantics and adapters; class restriction rendering does not require a new property-matching experiment.
- **Final human study:** independently adjudicated answer-present and answer-absent five-candidate cases per 11. Use the configurable mixed-case design, original system-rank sampling and complementary condition allocation. Keep final cases unexposed during prompt/UI selection. Separate constructed-negative origin and unresolved cases; balanced sampling is not natural deployment prevalence.

A fresh Exact smoke run may score the selected source groups using their complete declared candidate pools, while preparation generates text for a bounded chosen candidate subset. Record candidate completeness and presentation selection. Browsing all eligible ontology context does not mean every entity needs an LLM summary in advance. Cache on demand through explicit preparation jobs; serving never triggers calls.

## References and fairness

Public global `refs_equiv` are repaired, and may include directional relations. The measured train counts are NCIT–DOID: 3085 `=`, 165 `<=`, 12 `>=`; SNOMED–FMA: 406 `=`, 2809 `<=`, 24 `>=`; SNOMED–NCIT: 16901 `=`, 354 `<=`, 28 `>=`. Preserve relation convention and reference basis. Membership is not proof of equivalence. Local ranking uses standard equivalence pools containing a reference target; it is not a natural ontology-wide NIL benchmark.

Independent clinical/semantic adjudication is required for human correctness claims, with unresolved judgments retained. Candidate omission does not establish NIL. Xrefs may overlap with reference construction: define equal-resource visibility controls for Exact and Protégé, applying them before generated descriptions. External enrichment is a separate declared source/condition, not silently added ontology knowledge. Do not reconstruct held-out benchmark answers or contact organizers/participants through this assignment.

Official references: [ontology inputs](https://bio-ml.oaei-ml.org/ontologies/), [global reference semantics](https://bio-ml.oaei-ml.org/tasks/global/), [local ranking](https://bio-ml.oaei-ml.org/tasks/local/), [release changes](https://bio-ml.oaei-ml.org/changelog/). Preserve source license/provenance and permissible packaging; the initial NCIT–DOID gate does not require public redistribution of SNOMED.

Mixed-case study keys must establish whether any of the displayed five is equivalent. No-match-within-five does not establish ontology-wide NIL. The old local pool's guaranteed reference property describes the benchmark, not a prohibition on separately adjudicated study negatives. Negative construction, if used, is a declared study transformation rather than a matching-method result. Preserve all original candidate/score provenance and obtain the final ontology resources and Protégé setup pack before study publication.
