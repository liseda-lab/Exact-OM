# Target-contract seed fixtures

`entity-rich.json` and `entity-missing-definition.json` contain actual original ontology annotations adapted from a checksum-verified raw-file profile. They are partial design fixtures, not outputs from the production backend. Missing expression/source-axiom/source-span fields are explicit. The profile did not preserve every literal datatype/language; unknown remains null.

The trace and generation-failure fixtures are explicitly synthetic; they do not claim a matcher run, clinical relation or provider call. Validate all four against `../protocol/contract.schema.json`. B0–B5 must add native/current-Exact fixtures and the semantic/error cases required by 01/07. A passing seed-schema check does not pass those runtime gates.

The [raw entity examples](../evidence/real-entity-context-examples.json) preserve richer original XML fragments for three actual training examples. They intentionally have an exploratory schema; they are not validated as target API responses.
