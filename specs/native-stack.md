# Published native dependency contract

As of 20 September 2026, Exact-OM uses the published `0.2.1` native ontology stack.
The dependency declarations in [pyproject.toml](../pyproject.toml) and the resolved
artifacts in [poetry.lock](../poetry.lock) are authoritative for the current installation.
This contract applies to matching, standalone or sequential repair, ontology context,
and native optimization work.

| Distribution | Published version in the lock | Supported range | Installation |
| --- | --- | --- | --- |
| `pyowl-core` | `0.2.1` | `>=0.2.1,<0.3` | Base |
| `pyowl2vec-star-projector` | `0.2.1` | `>=0.2.1,<0.3` | Base |
| `pyelk-reasoner` | `0.2.1` | `>=0.2.1,<0.3` | Optional `reasoning` extra |
| `pyhermit` | `0.2.1` | `>=0.2.1,<0.3` | Optional `reasoning` extra |

Use the locked published distributions for implementation and validation. Keep dependency
source revisions and source-path overrides out of runtime requirements. Record the actual
installed distribution versions and artifact/native-binary hashes in new run evidence.
Continue to qualify the public API, model/encoded schemas, ownership, backend and required
capabilities; a package version alone does not establish support for an operation.
The [version contract](pyowl-core-0.2/01-version-contract.md) defines these boundaries.

Historical source audits, implementation measurements, repair checks and the synthetic
ontology-context probe are retained in the [native-stack archive](../docs/archive/native-stack/README.md).
Their results remain bound to the artifacts actually tested. They do not establish new
validation or performance results for the published stack. The existing
[release compatibility manifest](../release/core-compatibility.json) also retains earlier
validation evidence; use the lockfile for current dependencies and regenerate release
acceptance evidence from the installed published packages when that validation is run.
