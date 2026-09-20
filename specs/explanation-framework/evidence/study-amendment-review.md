# Study/product amendment review — 2026-09-20

The accepted two-product and mixed-ranking study scope was reviewed separately for study design and backend persistence. This is specification evidence; no new runtime or participant results were produced.

Corrections incorporated before publication:

- Positive-case MRR stays separate from negative-case none/false-endorsement/uncertainty rates; unsubmitted cases remain missing.
- Counterbalanced 24-case and shorter 20-case schedules preserve actual positive system-rank strata; every case switches conditions across participants, never within one participant.
- Constructed negatives retain private production-rank provenance while public initial positions do not expose rank gaps.
- Actual external ontology inspection remains in task time; only the subsequent resource-use questionnaire is excluded.
- Top-1 improvements/deteriorations are relative to system advice, not an uncollected initial human judgment.
- Reusable invitation links restore server state; revocation/reissue invalidates old authenticated cookies as well as old links.
- Mutable answer revision checks, replay-safe invitation exchange and independently deduplicated events have distinct transaction rules.
- Consent is a durable gated transition; individualized/admin responses cannot enter shared HTTP caches.
- B5 tests real PostgreSQL with synthetic sessions; actual Render rehearsal and full frontend validation precede later participant launch.

See [study design](../11-study-design-and-ranking.md) and [study service](../12-study-service-and-render.md). The final case-count choice remains configurable and is frozen after a duration pilot; it is not an implementation blocker.
