# Projector optimization specification

Implementation progress and artifact identities: [IMPLEMENTATION.md](IMPLEMENTATION.md).

Correctness is a hard acceptance requirement; current completion is recorded separately.
Follow [CONTRACT](CONTRACT.md) and [VALIDATION](VALIDATION.md); resolve conflicts before coding.
Current dependency: published `pyowl2vec-star-projector==0.2.1` under the [stack contract](../native-stack.md).
The [source audit](../experiments/NATIVE-STACK-AUDIT.md#1-projector-replace-repeated-native-class-membership-scans-first)
records evidence and measurement limits. Neither item promises a speedup or changes projection semantics.
Use existing immutable encoded owners and native kernels; add no parser, shadow ontology or cache framework.
Any semantic defect requires a separate reproducer and separately documented correctness change.

## N01 — Exact native annotation membership index

### Trigger and bounded change

Before optimization, literal-inclusive projection checked class membership for each
whitelisted annotation. The [historical lookup assessment](../experiments/NATIVE-STACK-AUDIT.md#1-projector-replace-repeated-native-class-membership-scans-first)
found scans of all N encoded nodes, repeated during counting and emission: worst-case
O(A × N) work. The composite equivalent scanned C reachable class coordinates per annotation,
O(A × C). The following requirements define the replacement index.
A counts annotation occurrences that reach this predicate, including subjects absent from the class set.
Replace only this predicate's scan with a native index per prepared table and semantic class selection.
Preserve existing literal/taxonomy suppression; build at most once per successful, non-replayed pass.
Share that index across counting and every emission batch; never rebuild per annotation, batch or class.
Separate ROOT preparations may still build their own index before N03; count them independently.
Share indexes across preparations only when identical class universes and safe ownership are proven.

### Exact membership and annotation semantics

- **Direct:** include every existing encoded `Entity(kind=class)` node in that direct closure table.
  Do not restrict to declarations or replace this with a newly inferred/reachable class universe.
- **Composite:** use only the existing reachable `composite_class_nodes`, built after selections,
  exclusions and scope remapping. Indexing all nodes from all member tables is incorrect.
- Match exact IRI bytes without case folding, Unicode normalization, label matching or suffix matching.
  A punned class/property or class/individual IRI qualifies because the class-kind entity exists.
- Retain ROOT-only annotation selection and CLOSURE class membership, including imported classes.
  Equal ROOT/CLOSURE is a proven special case, never an assumption from matching filenames or counts.
- Preserve the exact annotation-property whitelist and relation spelling, including NCIT synonyms.
  Anonymous subjects remain nonprojecting; unknown/nonclass IRI subjects produce no annotation edge.
- Keep value handling unchanged: IRI and anonymous values, language validation, string/plain lexical
  forms, typed-literal rendering, escaping and blank-node scope retain the baseline contract.
- The index returns membership only. Its iteration/insertion order must not choose edge order,
  multiplicity, selected annotation order, semantic counts/reports or canonical artifact bytes.

### Native shape, ownership and failure behavior

Prefer one native exact-IRI hash set, charged by allocated capacity and retained string storage.
A sorted native vector is acceptable if measured memory favors it; state its O(log C) lookup cost.
Direct table-local IRI-ID bitsets require proof that every subject resolves to the same canonical
IRI identity; never compare IDs across separately encoded tables. Use exact strings otherwise.
Borrowed keys must retain their immutable exporter through the final batch; owned keys must be counted.
Bind the index to the prepared table's exact owner, descriptor, scope and selections; no cross-call cache.
Retain full admission validation; do not hide malformed nodes by indexing only apparently useful rows.
Check cancellation during index construction and between bounded lookup/emission units.
Charge workspace before allocation, use fallible allocation and retain typed resource errors.
On failure/cancellation, publish no completed index or successful report and release partial storage;
preserve existing cursor/role-state transaction and cleanup behavior for already active streams.
If added workspace exceeds a limit, fail explicitly; never silently change semantics or use Python fallback.
[N02](CORE.md#n02-native-validation-and-required-execution) owns reusable validation and strict execution.
[N03](CORE.md#n03-proven-rootclosure-reuse) owns ROOT/CLOSURE proofs; N01 must work without those optimizations.

### Required tests and deterministic work evidence

Compare outputs, semantic counts/reports and canonical digests; performance counters may change under CONTRACT:

1. Repeated whitelisted annotations on one class, many classes and absent/nonclass subjects;
   non-whitelisted properties, anonymous subjects and zero annotations/classes.
2. Declared and referenced classes, class/property and class/individual punning, exact-case variants,
   distinct Unicode spellings, same IRI in separate tables and excluded composite classes.
3. Root annotations on imported classes, import-only annotations, ROOT=CLOSURE and ROOT≠CLOSURE;
   direct, overlay and composite selections with reachable and unreachable class coordinates.
4. Every existing literal/value category, nested metadata and typed-rendering regression fixtures;
   both duplicate policies, both orders, literal/taxonomy options and batch boundaries 1/2/7.
5. Malformed columns/owners, expired or replaced exporters, allocation limits, cancellation during
   index construction/counting/emission, and retriable batch-publication failures.

Add bounded diagnostic/test counters: index builds, build-node/class visits, inserted keys,
membership queries, fallback full-scan visits, retained/peak bytes and optional key bytes hashed.
Measure each prepared table/semantic class selection over one successful, non-replayed count-and-emit pass.
For its direct table assert builds ≤1, build-node visits ≤N and fallback full-scan visits =0.
For a composite selection only the existing C class coordinates feed membership-index construction.
Count and emission together issue at most 2A queries; completed batches reuse the same index identity.
Track retries/replays and separate ROOT preparations independently; no facade-global bound is required.
Scale N and A independently on generated fixtures and assert these counters, never elapsed-time ratios.
These counts establish expected O(N + A) hash-table operations (plus IRI byte-processing cost),
not a worst-case collision guarantee. A sorted implementation must instead report O(N + C log C + A log C).
Force hash collisions in a bounded test where supported; exact equality must still determine membership.
Accept N01 only after shared validation gates and exact output parity pass; timing is supplementary.

## N09 — Native canonical output and profile-gated extensions

### Entry gate and scope

Minimal native sorting/deduplication for Exact's canonical profile is required for full strict-native
conformance. N01–N04 may ship as measured partial improvements without that claim.
After N01–N04, split generation, Python publication, sorting and spill timings on identical bounded inputs.
Record CPU/wall time, raw/final edges, publication objects, copy bytes, peak workspace and spill counters.
The old 920-second iterator boundary combines these activities; it cannot authorize a sorting claim.
Keep existing Edge iterator/list and artifact APIs; the required path must honor memory/spill limits.
Packed-edge APIs, dictionary encoding, generic native sinks and spill tuning are conditional extensions:
require a documented residual bottleneck, retain immutable owners and add no general framework.

### Required behavior and resource contract

- Canonical ordering is the exact lexicographic tuple of UTF-8 bytes `(source, relation, destination)`.
  Dictionary IDs, hashes, locale order and encounter order are not substitutes for that comparison.
- `preserve` keeps every identical triple occurrence; `unique` emits one per exact triple.
  Preserve raw/distinct/duplicate counts across batches and spill runs; encounter-mode APIs retain
  baseline sequence and first-occurrence behavior even if N09 only optimizes canonical consumers.
- Retain canonical payload/digest bytes and semantic report fields; package/timing/counter provenance
  may truthfully change under CONTRACT. Do not promise identical complete artifact hashes when
  their declared nonsemantic provenance differs; verify unchanged payload and metadata schema separately.
- A bounded in-memory path may avoid a needless single-run spill. Larger outputs retain bounded
  external sorting with current temporary/spill-byte limits, descriptor limits and fan-in behavior.
- Preserve checksum/corruption detection, truncation/invalid UTF-8 rejection, typed disk/allocation
  errors, cancellation, sink-failure cleanup and atomic final-artifact publication.
- Retain immutable-owner lifetime and transactional cursor guarantees. Avoid exposed mutable Python
  intermediates instead of deleting finalizer/mutation defenses or allowing partially committed batches.

### Acceptance and strict-native claim

Run a cross-product of duplicate/order modes, empty/small/multiple-run outputs, tiny batch/fan-in limits,
long and non-ASCII IRIs, literal escaping and duplicates straddling every boundary. Compare exact outputs.
Inject corrupted/truncated runs, disk-full/permission failures, cancellation and failing sinks;
verify typed errors, cleanup, no false success report and unchanged committed outputs on failure.
Compare Edge iterator/list, canonical digest and artifact payload, plus any added compact sink.
Counters must show bounded workspace and native sorting; verify any claimed removal of materialization;
retain standalone phase timings before claiming a benefit. Passing small tests is not corpus throughput evidence.
`require_native_pipeline=False` preserves package defaults; the opt-in contract is defined in CONTRACT.
Exact may claim strict native execution including canonical handling only after N09's required path
exists and passes N02 admission. Until then, strict requests must fail before expensive work.
N01 is independently useful and must not wait for N09; no output-semantic waiver follows from this gate.
