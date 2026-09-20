# Core publication, selection and queries

Implementation progress and artifact identities: [IMPLEMENTATION.md](IMPLEMENTATION.md).

This file defines acceptance requirements; current completion is recorded separately.
This specifies pyOWLCore changes and the small Exact/consumer integrations they require.
The [shared contract](CONTRACT.md) and [validation plan](VALIDATION.md) govern every item.
Current dependency: published `pyowl-core==0.2.1` under the [stack contract](../native-stack.md).
The [historical audit](../experiments/NATIVE-STACK-AUDIT.md#2-pyowlcore-stop-reconstructing-native-data-in-python)
records the evidence that motivated these requirements.
The measured exclusion and publication intervals contain multiple operations; they do not
establish an isolated saving or justify changing ontology semantics.

## N02: Native validation and required execution

**Deliverable.** Extend existing core capabilities/validated-view negotiation to support
consumer `require_native_pipeline=False` from CONTRACT.md. Core retains
`LoadOptions.backend=NATIVE`; add no redundant loader flag. General defaults and scalar APIs
retain their behavior; Exact opts in after all consumers support the agreed requirement.
This requirement covers publication, structural validation and downstream compilation,
not merely the backend label. Unsupported capabilities/owners fail before scalar traversal,
canonical decoding, per-byte Python callbacks or Python compilation begins. Bounded public
argument checks and requested result wrappers may remain Python. N00 locks final API naming.

- Keep `EncodedStructuralViewV2`, its schema, canonical columns and snapshot-local cache.
  Move `_validate_columns` structural/range/order/reachability/digest checks into native code.
  Preserve valid outputs and reject invalid publications; never replace validation with trust.
- Publish an internal native-issued validation receipt bound to the exact immutable storage,
  owner lifetime, descriptor/model schema, selection and buffer identities. Reuse it across
  core/projector/reasoner boundaries only when the receiving contract is compatible.
  Consumer-specific semantic validation remains required; structural validation is insufficient.
- A caller-supplied Boolean, matching public fingerprint or backend label cannot mint a receipt.
  External, changed or incompatible buffers receive full native validation or an explicit
  unsupported error. Receipts are process-local and cannot authorize a relocated publication.
- Keep requested ROOT/DOCUMENT/CLOSURE metadata, document identity, overlay tombstones,
  composite scope maps and exact owner references. Bound every borrowed slice to live storage.
  Existing mutable/read-only-buffer distinctions, close behavior and mmap rules still apply.
- Stricter limits, cancellation and deadlines apply on cache hits. Validate supported limits
  from checked native summaries, or repeat the necessary native check; never reuse a weaker
  acceptance result. Charge retained/temporary allocations and aggregate accounting exactly.
- Preserve direct-fill buffers and genuine GIL release already present in the native writer.
  Record actual native validation, scalar decoding, Python byte access and compiler selection;
  fixed zero-valued report fields are not evidence. No new cache framework is required.

**Implementation surface.** `backends/native_views.py`, native column validation/publication,
existing view-cache metadata, and consumer capability checks. Keep transport changes internal
unless a minimal versioned capability is necessary; do not silently redefine the v2 schema.

**Differential/negative tests.** Compare columns, canonical root digest, structural fingerprint
and selections with the pinned reference implementation on direct, decoded, imported,
overlay and composite fixtures. Exercise malformed offsets/tags/cycles/order/duplicates,
foreign owners, forged receipts, mutated exporters, unsupported schemas, closed owners,
stricter limits, allocation failure and cancellation. Include mmap only where the complete
consumer chain safely supports it; otherwise assert rejection before traversal.
In strict bulk tests, forbid Python structural iterators, canonical decoders and scalar compilers.
Assert one native structural validation per newly accepted immutable publication and none
on a compatible receipt reuse; record native semantic checks separately. Repeated requests
must use the existing cache without relaxing scope or limits. Generic default-mode tests
must still cover their existing supported fallback behavior.

## N03: Proven ROOT/CLOSURE reuse

**Dependency.** N02 supplies the validated selection/ownership boundary; the projector uses
this proof before requesting another encoded view or constructing a disposable compiler.
Its annotation provenance and all projection semantics remain unchanged.

- Add a narrow native selection-equivalence query or equivalent checked internal metadata.
  For a direct immutable one-document snapshot with no imports or transformations, prove
  equality of selected ontology annotations, axioms, extensions and anonymous scopes.
  Document count alone is insufficient; an overlay can change the effective selection.
- Reuse the existing validated buffers through a view that retains the caller's requested
  scope metadata and owner. Do not alias table-local row IDs between independent views.
  Preserve root-annotation provenance when closure includes imported annotations.
- If equality is unproven, use the existing full native publication/selection path.
  Imports, overlays and composites remain supported according to their actual capabilities;
  lack of a shortcut is not permission for Python fallback in strict mode.
- Do not start with general cross-owner/content-addressed selection caching. Extend proof
  coverage beyond the direct no-import case only when differential evidence warrants it.

**Tests and operation counts.** A direct no-import fixture must produce identical full and
taxonomy edges/digests, with literals both enabled and disabled. Where the consumer previously
requested both selections, assert one column production and no duplicate compiler preparation.
Use imported root-only/closure annotations, hidden/tombstoned roots, overlay additions,
composite duplicate IRIs/blank nodes and distinct document scopes as negative reuse fixtures.
Assert correct native fallback and outputs, not merely a refusal to optimize. Closing either
permitted owner handle must preserve the established lifetime contract of retained views.

## N04: Native filtered annotations and typed indexes

**Deliverable.** Keep existing `AnnotationAssertionIndex`, `AxiomTypeIndex` and structural
view APIs. Add one bounded column-page query on the annotation index for bulk consumers;
proposed spelling `iter_columns(subjects=None, properties=None, max_rows=..., max_bytes=...)`.
Index options continue to select scope/document and origins. Final keyword spelling follows
the shared API conventions, but the following observable semantics are required.

- `None` means all values for that filter; an empty filter means no rows. Subject and property
  predicates combine by intersection, without language preference, lexical normalization,
  entailment or exclusion policy. Nested annotations remain a separately requested feature.
- Pages preserve canonical assertion order/identity and expose enough information to recover
  the existing result exactly: subject kind/scope, property IRI, value kind, exact lexical/IRI
  value, language, datatype, and assertion/annotation/origin identity where requested.
  Preserve canonical duplicate semantics and deterministic page boundaries/cursors.
- Build/query retained native subject/property or constructor indexes; decode only explicitly
  requested scalar results at the compatibility boundary. Bulk pages retain compact buffers
  and owners. No full Python object graph, per-row re-encode, deep sizing or scalar LRU admission
  is permitted for the bulk native path. Bound pages by both bytes and rows.
  An indivisible row exceeding `max_bytes` raises the existing resource-limit error before
  publishing that row; never omit, truncate or silently exceed the limit. Earlier complete
  pages remain valid; the failed stream must not report successful completion.
- Each published page retains immutable backing storage until release. Requests and cursor
  advancement follow the existing owner/index close contract; do not revoke independently
  retained data or reopen closed owners. Cursors bind owner, selection and filters; no reparse.
- Exact requests the two exclusion properties and applies its existing policy unchanged:
  `use_in_alignment` false/0 and `owl:deprecated` true/1 with the same lexical processing.
  Labels and other evidence continue to request their actual properties/subjects. Missing,
  unexpected or nonliteral values must not acquire new exclusion meanings.
- Route typed structural-view construction through native constructor partitions rather
  than repeated generic `iter_axioms(Type)` full scans. Build adjacency/domain/range tables
  on retained IDs and preserve asserted/equivalent/inverse/chain distinctions and origins.
  Do not materialize all roots when `include_nested=False` in annotation indexing.
- Keep postings, canonical sizes and identity ranges in native arrays/ranges. Validate and
  reserve aggregate checked metadata without tuples of one Python integer per axiom or
  per-row Python budget locks; preserve overflow checks, limits and cancellation frequency.
- Reuse the existing owner-local indexes. Native index construction may scan the required
  population once; repeated related queries must reuse it. Avoid a separate global cache.

**Dependencies.** N02 defines strict bulk execution and receipts; N03 is unnecessary for
ordinary annotation queries. The unused nested-root scan can be removed independently.
Coordinate projector/Exact integration without changing their selected evidence features.

**Differential/negative tests.** Compare complete annotation records, exclusions, selected
labels and asserted hierarchy/domain/range results against existing APIs. Cover exact IRI
spelling, class/property punning, literal datatype/language, anonymous subjects, duplicates,
nested annotations, imports, overlay removals/additions and composite scope isolation.
Test all/empty/intersected filters, absent subjects, exact limits, an oversized indivisible row,
continuation/cancellation, owner closure between pages, retained buffers and requested origins.
Native unsupported cases must fail explicitly; failed pages cannot silently drop data.
For selected-row fixtures, forbid scalar decode of unrelated rows and Python whole-ontology
iteration; assert no nested-root scan when disabled. Add unrelated axioms and verify that
warm typed/property queries do not scan them. Count native partition builds and page calls;
Python metadata/result allocations must scale with pages and requested scalar results,
not all ontology roots. Report retained index bytes separately from borrowed arena bytes.

## N10: Profile-gated canonical/compiler pass optimization

**Condition.** Instrument native discovery, canonical length calculation, sorting/comparison,
deduplication, validation and buffer filling first. Reassess after N02/N03 and localized
projector fixes; do not assign the remaining measured time to canonical sorting by subtraction.

- Use existing canonical-work/comparison-byte counters and add phase boundaries where missing.
  Compare long-common-prefix IRIs, nested/shared structures and ordinary ontology fixtures.
- If measurements justify it, reuse proven native canonical ranks/order or add bounded prefix
  acceleration with exact comparison on collisions. Combine passes only when their distinct
  validation/limit obligations remain satisfied. Retain exact canonical bytes, IDs and order.
- Hash order, approximate equality, materializing all expanded roots, changing the ontology
  model, skipping unsupported axioms and arbitrary thread-count increases are out of scope.
  Preserve shared arenas, borrowed roots, memory limits and deterministic parallel results.
- Differential tests must exercise equal prefixes, structural sharing, deep nesting, overflow,
  cancellation and multiple input orders. Assert unchanged column/edge digests and query results;
  require a measured reduction in the targeted operation count with declared memory accounting.
  Reject an optimization that merely moves work or creates unbounded retained state.

## Completion evidence and exclusions

Land these as separate reviewable changes with pinned package revisions, focused contract
tests, counters and before/after measurements under [VALIDATION.md](VALIDATION.md). Published
performance claims require matched inputs/settings; no fixed latency or speedup is promised.
Changes to matching thresholds, evidence features, exclusion meaning, gold labels, reasoner
semantics and production-scale orchestration are excluded. This document authorizes no run;
large measurements follow the shared validation stages and resource limits.
