# XR-2 machine-readable pilot protocol

[pilot.json](pilot.json) covers the current methodology: all action families, HGT with matched controls, conditioned probabilistic generation, unary/pairwise optimisation, typed symbolic supervision, generated-to-real adaptation, Conference and separate Bio-ML releases. Values are starting settings for an exploratory pilot, not empirically selected defaults.

[smoke.json](smoke.json) inherits the pilot and reduces counts/budgets. Resolve inheritance before validation or hashing. Dictionaries merge recursively; arrays replace. Unknown fields, cycles and obsolete XR-P1 versions are errors.

[schema.json](schema.json) is the strict schema for a resolved protocol. [validate_protocol.py](../reference/validate_protocol.py) checks the schema subset used here plus cross-field invariants without external dependencies. It prints the resolved hash and requested generated-case counts. It does not download data, reason over OWL, label cases or train a model.

The 16 generated families request 576 pilot cases: 384 training, 96 development and 96 test. Smoke requests 96: 32 per split. Generation/label deadlines can produce fewer; log requested, generated, verified and labelled counts separately.

## Semantic conventions behind the numbers

Each group is a clean structural parent, with independently recorded corruptions and clean controls. All siblings stay in one split. The mixed family is additionally evaluated in a composition-holdout regime: omit mixed training/development cases in that named arm rather than leaking the held-out mechanism.

The graph uses width 128, three HGT layers and four attention heads. The matched R-GCN control uses the same target/candidate attention readouts. Four circuit mixture components are compared with one component. These settings do not prescribe a specific graph or circuit library.

Profile costs count features of the actual selected edit once: mapping deletion; original inclusion directions removed; endpoints changed; subclass expressions specialised; new necessary conditions; new intersection/existential constructors; ontology objects changed; and changed human-authored ontology objects. Do not count a constructor merely for retaining an original axiom. Ontology keep has zero edit cost; human ontology edits incur both the general and human increments. A composite is costed by its emitted changes, not by adding duplicate labels for its component trace. Provenance-unknown edits use the general increment and are reported separately. These numbers are declared preferences, not inferred probabilities.

Desired and unwanted query families have frozen weights before comparing predictions. Normalise within each nonempty family as specified in [07](../07-corpus-and-training.md); the pilot uses equal desired-family weights and the stated unwanted-family penalty. Store the full resolved query list, labels and weights per case. Required/prohibited hard queries contain identifiers resolved by the case manifest, not arbitrary free text passed to a reasoner.

A protocol is supplemented by a run manifest containing exact input hashes, per-pair train/development/test membership, provenance/eligibility, hard query definitions, qualified backend and library versions, hardware and actual command arguments. The whole-ontology ekaw holdout constrains the Conference test split; all remaining pair assignments must be frozen before training. Bio-ML shared-ontology overlap must be disclosed.

## Required operational behaviour

- Compile and verification budgets are per call; the run/stage/campaign supervisors impose their own deadlines.
- No full initial classification is mandatory before diagnosis can make progress.
- Unknown assignments are pending and contribute to the global bound; finite retries are not logical exclusions.
- Exact teacher distributions require complete caches. Partial verified comparisons may train explicitly named losses.
- A missing complete development subset cannot silently switch checkpoint selection to test results. Record the unavailable criterion and stop that arm or use a predeclared development-only fallback.
- Ontology changes are returned as patches. Repair remains disabled by default in production.

The schema encodes the full-methodology pilot. Ablations are derived, separately named experiment manifests with explicit changed factors; do not relax validation silently to accommodate an old protocol.
