# Property and instance matching

`matching.entity_kinds` controls which kinds enter the candidate pipeline. The default is
`[class]`, preserving the class-only behavior of Exact-OM 1.x.

```yaml
matching:
  entity_kinds: [class, object_property, data_property, individual]
```

Candidate retrieval is strictly within kind: a class is never proposed for a property or
individual. Candidate frames carry `SrcKind` and `TgtKind` internally, while the traditional
TSV deliverables remain compatible. Use the JSON or typed-TSV writer when downstream code
needs explicit kind metadata.

Evidence is adapted rather than scored by a separate model:

| Kind | Hierarchy evidence | Graph and attribute evidence |
| --- | --- | --- |
| Class | Direct parents and configured existential families. | Projected ontology edges and annotations. |
| Object/data property | `subPropertyOf` parents. | Domain/range pseudo-edges, inverse/characteristics, annotations. |
| Individual | Asserted types plus one class-parent level. | ABox object assertions and data values. |

Set `include_abox: true` in source options when individuals are in scope. IRIs from a reference
or candidate file are looked up in the source signatures. An unknown IRI falls back to the
configured primary kind with a warning, so malformed inputs remain visible rather than being
silently scored across kinds.

Explanation records include the source and target kind. Evaluation filters references to the
selected kinds and reports ignored reference rows instead of turning them into false misses.
