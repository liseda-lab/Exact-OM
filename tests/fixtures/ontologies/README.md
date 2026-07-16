# Mini ontology fixtures

`mini_src.owl` and `mini_tgt.owl` are paired RDF/XML ontologies used to test the
Java-free ontology backend. Each contains roughly thirty classes, including a
multi-parent hierarchy, named equivalence cycles, an equivalent intersection,
and a direct existential restriction.

The pair also contains multilingual labels, exact synonyms, definitions,
literal annotations, a label-free class, `use_in_alignment=false` and
`owl:deprecated=true` exclusions, object-property domains/ranges and a
subproperty, a data property, inverse and symmetric properties, and three named
individuals with class, object-property, and data-property assertions.

`mini_refs.tsv` and `mini_test.cands.tsv` use the Bio-ML reference and candidate
formats and cover five obvious cross-ontology matches.
