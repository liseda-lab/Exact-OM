# Post-implementation frontend review — 2026-10-03

**Reviewed checkout: `8755431` (frontend `6895bc6`/`10fea04` on backend `96b76fe`).** This
records four frontend defects that were reproduced after the S2/S3 handoff and that its
tests did not detect. They reopen R01 and R02 in
[17](../17-integration-corrective-programme.md); the required behavior is in
[19](../19-frontend-integration-corrections.md) F16–F19. Nothing here is evidence that the
backend emits malformed data or lacks a contract: the scoped routes already expose the
continuations, contexts and configurations needed.

| ID | Reopens | Priority | Reproduction | Observed defect |
|---|---|---|---|---|
| R06 | R01 (J03) | P2 | A generated ontology gave one entity 26 definitions. The initial entity context carried 20; the continuation returned the other 6. | `EntityCard` rendered the first 20 and said “all are shown” with no continuation. Definitions, alternate definitions, synonyms, defining facts and parents were all limited to their first page; only “Other recorded facts” continued. |
| R07 | R01 (J03) | P2 | In a study case, search the hierarchy for a non-focal entity. Its hierarchy appears and the scoped entity context succeeds. | `PairWorkspace` does not pass `onOpenContext` to the hierarchy browser, so the study lacks the main app's “Open full context”. Participants cannot inspect that entity's definitions and facts. |
| R08 | R02 (J04/J05) | P2 | Fault injection: one 200 entity-context response with the right entity but a missing required collection. Rendering fails. Restore valid responses and press Retry. | The shared read cache keeps the malformed result because only identity was validated. Retry reuses it without fetching, and ranking stays disabled until reload. |
| R09 | R02 (J04/J05) | P1 | Publish a valid study whose only component is `pair_comparison` (new form version). The backend accepts it and the comparison renders. | `RenderProbe` always waits for two entity cards. Cards are intentionally absent, so readiness times out and the case can never be answered. |

The S3 suite passed while these defects remained because:
- its fixture had no category beyond the first page except comments;
- its study cases never opened a non-focal entity's context;
- its fault tests injected transport failures, not malformed 200 responses;
- every publication used the full component set.

The corrections must add tests for each of these gaps.
