# 01 — Version and compatibility contract

## 1. Published package set

Exact must declare compatible minor-line ranges while the lock and release evidence identify one
exact published set. The current lock resolves all four native distributions to `0.2.1`;
see the [published stack contract](../native-stack.md):

```toml
pyowl-core = ">=0.2.1,<0.3"
pyowl2vec-star-projector = ">=0.2.1,<0.3"
pyelk-reasoner = { version = ">=0.2.1,<0.3", optional = true }
pyhermit = { version = ">=0.2.1,<0.3", optional = true }
oaei-bioml-eval = { version = ">=0.2.1,<0.3", optional = true }
```

The final evidence must not use editable installs, source checkouts, development versions, release
candidates, or unpublished commits. The base installation requires only core and projector.
pyELK and pyHermiT remain in the `reasoning` extra; Bio-ML evaluation remains optional.

## 2. Core 0.2 public contract

Exact must negotiate the installed package through public attributes. The accepted minimum
contract is:

| Contract | Required value/behavior |
|---|---|
| package | released `pyowl-core==0.2.1` must work |
| API version | `(0, 2)` |
| model schema | `2` |
| wire | core-declared readable range includes version `2`; new payloads use the current writer |
| adapter protocol | public core-declared value; do not duplicate it in Exact |
| encoded structural schema | `2` |
| encoded descriptor | `pyowl_core.EncodedStructuralView.DESCRIPTOR_SHA256` |

Exact must not use the version-suffixed
`ENCODED_STRUCTURAL_DESCRIPTOR_SHA256_V1` constant with a dynamic schema number. It must not
hard-code the schema-2 digest when the public class supplies it.

Allowed behavior:

- inspect immutable public capability/version constants;
- pass an `OntologyView`, snapshot provider, or core wire/mmap owner to a public consumer API;
- compare consumer provenance with the expected public contract; and
- fail actionably when the consumer reports an incompatible contract.

Forbidden behavior:

- import `pyowl_core.backends` or consumer compiler/native implementation modules;
- construct, decode, flatten, or reinterpret `EncodedStructuralView` buffers in Exact;
- cache schema-local dense IDs;
- retry a failed consumer by handing it the original OWL path; or
- infer capability support only from a distribution version string.

## 3. Independent companion contracts

The core model migration does not renumber unrelated companion contracts. The implementation must
read projector/reasoner public constants and preserve their current values unless their own 0.2
release explicitly changes them.

In particular, do not mechanically alter:

- the `mowl-d993536-v1` semantic projection profile;
- projector compiler-cache schema version 1;
- pyELK compiler schema version 1; or
- pyHermiT compiler-cache, IR, or native-ABI schema version 1.

Projector Python/native output and reasoner Python/native output remain semantic equivalents for
the options Exact selects.

## 4. Persisted-data boundary

Core model schema 2 changes structural identities, including anonymous-individual/component
scoping. Every ontology-derived persisted value whose identity depends on the core model must be
invalidated.

Required actions:

- increment Exact's ontology backend/cache compatibility versions;
- include core API, model schema, current wire writer, encoded schema/descriptor, projector
  identity/options, and selected reasoner identity in applicable cache keys;
- reject schema-1 parsed-ontology, projection, compiler, and resumable workflow caches with an
  actionable rebuild message;
- rebuild them from the original source through core 0.2; and
- test that rejection occurs before unsafe unpickling or interpretation of dense IDs.

Immutable completed run artifacts may remain readable as historical results. A resumed run must
not reuse an ontology-derived schema-1 cache even when its path, mtime, or source digest matches.
No forward or backward cache converter is permitted.

## 5. Provenance and release manifest

For every source and target, record only public, reproducible values:

- exact distribution version;
- core distribution, API, model, wire, encoded schema, and descriptor digest;
- source/closure fingerprints;
- projector distribution, backend, profile, options, and compiler-cache schema;
- asserted/ELK/HermiT selection, distribution, backend, and public compiler/native schema;
- selected ingestion path and owner kind;
- whether the worker used verified core wire/mmap; and
- cache schema and cold/hit state.

Do not persist Python object IDs, pointers, private arena IDs, credentials, or machine-local
temporary paths.

`release/core-compatibility.json` must contain the exact final published package set and the core
schema-2 descriptor used by acceptance. Its `performance_claim` remains `false`.
