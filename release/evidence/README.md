# Exact 2.1 release evidence

Release evidence uses an implementation-candidate commit followed by an evidence-only commit.
This avoids a self-referential commit or artifact hash:

1. Finalize implementation, package metadata, `release/core-compatibility.json`, and version
   2.1.0 in implementation-candidate commit A.
2. Set `SOURCE_DATE_EPOCH` to commit A's Unix commit timestamp. Build one wheel and one sdist,
   run all release gates, and retain their hashes.
3. Run the frozen NCIT–DOID correctness acceptance against the final published dependency set.
4. In evidence-only commit B, set `legacy-compliance-closure.json` to `passed` and record commit
   A, its fixed epoch, wheel/sdist/lock/NCIT hashes, and stable job or test identifiers.
5. Tag commit B. The tag workflow verifies that A is its ancestor and that A..B changes only
   `release/evidence/**`, rebuilds once with the recorded epoch, verifies every hash, fans that
   one build through the Python 3.10–3.12 artifact smokes, and uploads the tested files without
   another rebuild.

`release/evidence/**` is explicitly excluded from both wheel and sdist. The packaged
`release/core-compatibility.json` must be final before commit A; it contains the exact published
package/schema contract but no commit or artifact hash.

The closure's `exact.commit` and `candidate.implementation_commit` both name commit A. They do
not name the later evidence-only commit B. `candidate.source_date_epoch` is exactly commit A's
Unix commit timestamp.

The NCIT–DOID command creates `pyowl-core-0.2-ncit-doid.json`. Do not hand-edit its acceptance
result, replace the frozen ontology digests, or add timing thresholds. Diagnostic timing fields
do not support a performance claim.
