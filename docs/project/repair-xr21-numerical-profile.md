# Development numerical resource profile

The frozen schedule binds only the existing 16 development profiling parents
(32 clean/corrupted cases), their full 64-row native compilation report, and the
nonce-bound GPU qualification. The four original native timeouts remain present.
No larger-corpus or pilot model checkpoint is opened. Missing cases retain rows
in every stage and model arm.

The graph worker measures retrieval, graph construction, feature hashing and
sparse pair selection with the native v3 observable preparation. The teacher
worker measures a bounded fixed-inventory symbolic teacher independently. Its
labels are profiling artifacts and cannot silently become training labels.
Six separate GPU jobs cover HGT, R-GCN and no-graph, each unary and pairwise,
with identical graph/feature and model dimensions. They use the union of the
observed development graph metadata, fresh seed-13 weights for each case/arm,
one warmup and two measured optimizer steps, and a diagnostic squared readout
loss through unary, available pair, and plan-risk heads. All results include
time and CUDA memory, finite-gradient checks and profiling-only checkpoints.

This measures the numerical readouts on a fixed inventory. It does not measure
proposal-circuit gradients, active generated-pool collection, decoded checkpoint
selection, supervised learning quality or convergence. Graph omissions and cases
without selected interactions remain explicit. Existing native compilation
timings provide separate stage evidence, not fabricated end-to-end timings.

Every bounded call writes a hash-bound receipt. Completed rows resume without
rerunning; uncertain in-flight probes require prior-owner/cleanup reconciliation
before continuation and cannot silently restart their per-case budget. Each
logical worker has a finite operational slice with all costs retained in the
campaign ledger. A healthy continuation may extend that operational allowance
without changing the frozen probe budgets or counting as a software repair.

The final summary depends on every actual stage, reports 32 graph, 32 teacher
and 192 numerical outcomes, and gives provisional batch-size/checkpoint/slice
recommendations. Missing components limit capacity claims. Larger training
continues to require the separate native and historical requirement audits;
this profile does not pass G0-G2 or complete the training study.
