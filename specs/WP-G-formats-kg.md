# WP-G — I/O Formats & Pure-KG Sources (`exact/io/`)

**Depends on**: WP-B (implements `KnowledgeSource` for new sources; wraps `exact.ontology` for
OWL). Coordinates with WP-F (shared candidate-table schema) and WP-E (`typed-tsv` feeds typed
eval). **Size**: L. **Behavior**: additive — with OWL inputs and default
`output_formats: ["tsv-global","tsv-local"]`, outputs are byte-identical to baseline.

## Context

Inputs today are whatever mowl loaded from a single `.owl` path; outputs are two TSV shapes +
JSON explanations (`core/contracts/trainer.py:353-381`); no OAEI Alignment-RDF is emitted, and
pure KGs (no OWL at all) are unsupported. Target competitions/kits:
[BioKG-Align-kit](https://github.com/liseda-lab/BioKG-Align-kit) — CSV triples/graph files
(`subclass_of` hierarchies + entity relationships), TSV candidates (50/query), optional datalog,
submission format `SrcEntity\tTgtEntity\tRelation\tScore` with
`Relation ∈ {equivalent, source_subsumed_by_target, source_subsumes_target}`, task pairs
NCIT-DOID / SNOMED-FMA / SNOMED-NCIT. The same framework should also accept plain RDF.

## G1. Source layer (`exact/io/sources/`)

Registry `exact.io.sources.resolve(path, format="auto", options={}) -> KnowledgeSource`;
`auto` dispatches on extension (`.owl/.rdf/.owx/.ofn` → owl; `.ttl/.nt/.n3/.xml` → rdf;
`.csv/.tsv` or directory-with-descriptor → csv-kg). Config `io.input_format`,
`io.source_options` / `io.target_options` (contracts §9); CLI `--input-format`,
`--source-options/--target-options` as `key=value` pairs or a YAML file.

1. `owl.py` — thin wrapper returning WP-B's `OwlOntologySource` (options: label properties,
   include_abox).
2. `rdf.py` — `RdfSource` via **rdflib** (already a dependency; the only permitted importer
   besides `data/get_data.py`). For RDF that is not OWL-structured: options declare
   `label_predicates` (default rdfs:label + skos:prefLabel/altLabel), `hierarchy_predicates`
   (default rdfs:subClassOf + skos:broader), `type_predicates` (rdf:type), `entity_selector`
   (`subjects|configured type IRIs`). Maps onto `KnowledgeSource`: entities per kind (via
   types), labels, annotations (literal-valued predicates), hierarchy (declared predicates),
   `projection_edges` = all IRI-object triples (+ literal edges when asked). For files that ARE
   OWL in turtle (py-horned-owl may not parse turtle), `rdf.py` is the fallback path.
3. `csv_kg.py` — `CsvKgSource` for BioKG-style directories. Descriptor
   (`io.source_options` or `kg.yaml` in the data dir) declares: `triples_files:
   [(path, src_col, rel_col, dst_col)]` (CSV/TSV), `labels_file` (entity→label(s)) or
   `label_relation` id used inside triples, `hierarchy_relations: ["subclass_of"]`,
   `attribute_relations: [...]` (object is a literal). Everything maps as: entities = union of
   subjects/objects of IRI-ish ids; `direct_parents` = hierarchy relations (cycle-normalized,
   reusing WP-B's `hierarchy.py` index on generic edges — refactor that index to accept plain
   edge lists so both backends share it); `projection_edges` = the triples;
   `hierarchy_bundle` families = configured relation ids. Kinds: default all entities
   `INDIVIDUAL` unless the descriptor tags class-like ids (`class_relation: subclass_of`
   participation ⇒ CLASS) — BioKG entities are effectively ontology classes, so descriptor
   presets ship in `configs/profiles/biokg.yaml`.
4. `datalog.py` — parser for datalog **fact files** (`rel(a, b).` → `Edge(a, rel, b)`), merged
   into `CsvKgSource` triples via descriptor `datalog_files: [...]`. Rule evaluation and
   Soufflé/OWL2RL are **out of scope** (document: pre-materialize with the kit's own tooling).

All sources must pass WP-B's `KnowledgeSourceConformance` suite — add fixture data:
`tests/fixtures/kg_csv/` (mini BioKG-shaped dir: triples.csv, labels.csv, candidates.tsv,
answers.tsv modeled on the kit's "mini fixture") and `tests/fixtures/rdf/mini.ttl`.

## G2. Writer layer (`exact/io/writers/`)

Extract the output-writing currently inlined in `core/contracts/trainer.py:353-381`
(`save_alignment`) into writers registered by name (contracts §10); the trainer iterates
`configs.io.output_formats`. This is the one WP-D-adjacent file WP-G owns — coordinate merge
order with WP-D's PR-D3 (writers land after it).

1. `tsv.py` — `tsv-global`, `tsv-local`: byte-identical moves of the current code (golden-file
   tested).
2. `oaei_rdf.py` — OAEI Alignment Format (`align.rdf`: `Alignment/map/Cell/entity1/entity2/
   measure/relation`, xmlns `http://knowledgeweb.semanticweb.org/heterogeneity/alignment`).
   `data/get_data.py:24-43` already parses this format — round-trip test against that parser.
   Relation cell from the `Relation` column (`=`, `<`, `>`).
3. `typed_tsv.py` — `SrcEntity\tTgtEntity\tRelation\tScore` (header mandatory), BioKG
   submission-compatible: relation vocabulary `{equivalent, source_subsumed_by_target,
   source_subsumes_target}` (mapped from internal `=`,`<`,`>`), finite float scores, ranking by
   descending score. Multi-task submissions (one file covering all pairs) are assembled by a
   small `tools/build_biokg_submission.py` that concatenates per-pair runs and runs sanity
   checks mirroring the kit's `verify` rules (the kit itself remains the authority — document
   running its `verify` before submission).
4. `json.py` — mappings as JSON records (Src, Tgt, Score, Relation, Kind) for programmatic use.

## G3. Relation typing (`matching.relation_prediction`, v2-native key)

- `none` (default): all rows `Relation "="` — current behavior.
- `hierarchy_heuristic`: post-scoring pass (new `impl/models/relation_typer.py`, pure function
  over the candidate frame + both `KnowledgeSource`s): if tgt ∈ ancestors(src-image via anchor
  candidates) evidence dominates → `<`; descendant → `>`; else `=`. Uses only
  `ancestors/descendants` from the sources and existing channel scores as tie-breakers; no
  LLM call, no new model. Emits per-pair `relation_confidence` into explanations. This is a
  baseline capability, explicitly labeled heuristic in docs; learned typing is future work.

## G4. CLI/API & profile

- `--source/-s`, `--target/-t` accept files **or** KG directories; `--input-format`,
  `--output-formats`, `--relation-prediction` flags added (defaults preserve behavior).
- `configs/profiles/biokg.yaml`: candidate restriction from the kit's candidates TSV (existing
  `-c` path), `relation_prediction: hierarchy_heuristic`, `output_formats: ["typed-tsv"]`,
  csv-kg descriptor template, eval `backends: ["builtin","bioml"]` (typed metrics via WP-E once
  upstream lands).
- Data acquisition is WP-I's job: its `biokg` track provider (HF stub until the competition
  publishes) and `diso-oaei` provider materialize directly into this WP's source layouts —
  align the csv-kg descriptor fields with WP-I's `TaskLayout` so a `data:` config block feeds
  a KG run with no manual paths.
- End-to-end doc page (WP-H hosts it; write the draft here): "Running Exact-OM on
  BioKG-Align" — data layout, profile, submission build, kit `verify`.

## Tests

1. Golden-file: default OWL run byte-identical TSVs after writer extraction.
2. Conformance suite over `RdfSource` + `CsvKgSource` fixtures.
3. `oaei_rdf` round-trip via `data/get_data.py:_read_alignment`.
4. `typed_tsv` validated against the kit's documented rules (header, vocabulary, finite scores);
   if the kit is pip-installable at implementation time, add a `requires_data` test invoking its
   `verify` on our output.
5. Mini-BioKG end-to-end: csv-kg source + candidates file + hierarchy_heuristic → ranked typed
   submission; assert candidate coverage and relation vocabulary.
6. Relation heuristic unit tests on fixture hierarchies (equivalent/subsumes/subsumed cases).

## Out of scope

Datalog *rule* evaluation / Soufflé OWL2RL scoring; learned relation prediction; OBO parsing;
streaming very-large-KG support (document current memory model instead).

## Acceptance criteria

1. Baseline OWL runs byte-identical (writers + sources refactor invisible by default).
2. `exact -s tests/fixtures/kg_csv -t tests/fixtures/kg_csv --input-format csv-kg ...` completes
   end-to-end with LLM fakes, producing a valid `typed-tsv`.
3. All writer/source tests green; conformance suite green for all three sources.
4. BioKG profile + doc draft committed; submission builder produces a file passing our mirrored
   `verify` checks on the mini fixture.
