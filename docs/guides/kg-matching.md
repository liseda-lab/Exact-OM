# KG matching (BioKG-Align)

Exact-OM accepts plain RDF graphs and descriptor-driven CSV knowledge graphs without converting
them into OWL. Use `io.input_format: rdf` for Turtle/N-Triples/RDF/XML, or `csv-kg` for a
directory containing `kg.yaml`.

## CSV-KG layout

The descriptor identifies triple tables and their subject, relation, and object columns. It
may also declare label files, hierarchy relations, attribute relations, class tagging, and
pre-materialized datalog fact files.

```yaml
triples_files:
  - path: triples.csv
    src_col: source
    rel_col: relation
    dst_col: target
labels_file:
  path: labels.csv
  entity_col: entity
  label_col: label
hierarchy_relations: [subclass_of]
datalog_files: [facts.dl]
```

Datalog input is limited to facts such as `related(a, b).`; evaluate rules with the
competition toolkit before running Exact-OM.

## Outputs and relation typing

Select `typed-tsv` to emit the BioKG submission columns
`SrcEntity, TgtEntity, Relation, Score`. The optional `hierarchy_heuristic` relation pass uses
source/target ancestors and descendants to choose `equivalent`,
`source_subsumed_by_target`, or `source_subsumes_target`; it is a transparent baseline, not a
learned relation classifier.

The shipped profile keeps every supplied candidate and writes only the typed submission:

```console
exact align \
  -s DATA/source -t DATA/target -c DATA/candidates.tsv -o RUN \
  -y configs/profiles/biokg.yaml
```

For a nonstandard descriptor, override the adapter directly with
`--source-options key=value` / `--target-options key=value`, or pass one YAML mapping file to
each flag. The profile assumes the kit's `subclass_of` hierarchy relation; a directory's
`kg.yaml` supplies its triples and label-file declarations.

Assemble task outputs with:

```console
poetry run python tools/build_biokg_submission.py \
  --input ncit-doid=RUN/alignment/alignment.typed.tsv \
  --input snomed-fma=OTHER/alignment/alignment.typed.tsv \
  --output submission.tsv
```

The helper checks headers, finite scores, relation vocabulary, ordering, duplicate pairs, and
candidate coverage. Run the official kit's `verify` command before submitting; it remains the
competition authority.
