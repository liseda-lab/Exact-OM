# Declared observable graph schema

New larger-training protocols supply `model.graph_schema`, an embedded
`exact-repair/declared-graph-schema/v1` declaration. The strict protocol schema
edition is `specs/exact-repair/protocol/schema-v3.4.json`; earlier editions remain
unchanged. Historical protocols without this field retain their serialized
identity and observed-metadata behavior.

`generic_graph_schema()` derives syntax node kinds and relation roles from the
installed public pyowl-core structural AST, then adds the generic graph builder's
object occurrence, evidence, explanation, witness, reverse and self relations.
It takes no corpus inputs. Its language hash, sorted node/relation inventory and
schema version are frozen in each protocol before model construction. This
declares the observable input language, not native backend semantic support.

Declared models allocate the full metadata before training, including families
absent from the training examples. All three encoders reject undeclared input
types; no-graph still retains the observable adjacency for its shared readouts.
Checkpoints retain the declaration through model configuration and an explicit
schema hash. Training state also records metadata and verifies these fields on
resume. A declared warm start must have exactly compatible metadata. Existing
pilot checkpoints are never extended by this mechanism.

The frozen validation job tests missing/new relation admission, all three encoder
backward paths, portable checkpoints, protocol projection and resume rejection.
It audits every observable record in the 128-case train and 32-case development
releases, without opening evaluator payloads or held-out cases. It preserves all
unsupported rows in the denominator. Passing these checks does not qualify
labels, proposal-circuit backward, generated-pool checkpoint selection, GPU
capacity, or G0–G2. Those remain requirements of the original larger-training
preparation stage.
