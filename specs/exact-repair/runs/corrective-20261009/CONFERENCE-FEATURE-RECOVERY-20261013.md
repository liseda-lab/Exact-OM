# Conference provenance boundary correction

Early TEST input construction completed in step14451.218. The first pool shard
(14451.219) then failed before producing a scientific result: the v3 graph builder
rejected `matcher_member` as an unregistered feature. The same published input
also carries `matcher_row_index`. Both are provenance identifiers, not measurements.

The graph now excludes these two explicit identifiers, just as it already excludes
paths and archive hashes. Raw input provenance remains intact. Matcher confidence
remains observable; arbitrary unregistered channels and nested evaluator labels
remain forbidden. Neither the feature vocabulary nor the six model weights changes.
All available materialized Conference inputs were checked: their feature vectors
equal their confidence-only vectors. Fifty-two focused tests and targeted mypy
checks passed, including the strict feature boundary and bounded recovery.

Keep completed input construction, all original receipts and charges. Re-freeze
the 54 remaining jobs with the corrected source and unchanged scientific rows.
The one failed pool shard receives an explicitly linked first technical repair
attempt under the same logical job budget. Its failed row consumed4.900996seconds;
subtract this from the original300-second row budget and include it in cumulative
row costs. Validate the original nonce, completion, artifacts and retired owner
before retrying. No successful result, timeout or unqualified failure is replayed.

This is repair attempt1 for the stable provenance-boundary incident. Preserve
the supervisor's two-unsuccessful-repair limit across replacement jobs; do not
rename the incident or reset counters. Keep early TEST authorization, absolute
16October23UTC deadline/final8h reserve, all splits and the $35 ceiling unchanged.

The first replacement (14451.222) stopped during input handoff, before starting
any scientific row: the consumer required its source commit to equal the already
completed input producer's commit. The retained producer now has an explicit
frozen `public_input_producer.source_commit`. This admits exactly the recorded
producer version while all schedule, model-freeze, case, nonce, completion,
artifact and checksum guards remain active. Legacy manifests retain the original
same-source requirement. The next linked replacement preserves both failed job
charges and the original row's remaining budget. Validate the actual completed
producer receipts through this handoff before release.
