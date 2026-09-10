# Accepted review decisions and v2 changes

The 2026-09-07 review was accepted by the user on 2026-09-09. This file records scope decisions
so an implementation agent does not have to reconstruct the conversation.

| Accepted point | Normative destination |
| --- | --- |
| Global experiments must execute the selector/extractor even with a candidate file | E00, shared clarifications 1 |
| A real LLM-off control and a transferable quantile policy | E25, shared clarifications 4 |
| Fitted analytic fusion must replay shipped scores at neutral parameters | E19, shared clarifications 6 |
| Assignment must optimize eligible mappings with a meaningful unmatched option | E01, shared clarifications 8 |
| Quality, inactivity and confidence sharpening need separate tests | E26, before E19 |
| Missing relations are not automatically contradictions | E24/E08 and controlled missingness tests |
| Exact lexical matches need a soft-anchor comparison and noise audit | E02 |
| LLMs need direct facts, comparisons, useful routing and new evidence | E05/E07/E21/E25 |
| Retrieval misses, pool misses and genuine NIL must be separated | E00/E04/E05 |
| Training and final decision explanations need complete traces | E00/E18/E19 |
| FP+FN for a wrong mapping is ordinary F1 accounting | E03 |
| Test reuse across component selection biases final stack evidence | G4/G5 and E17 |
| Property/instance/typed/KG claims need actual feature-specific cases | RUN-PLAN case registry and E11–E14/E23 |
| Human utility needs controlled evidence/prose comparisons | HUMAN-VALIDATION.md |
| Available primitives and missing integrated paths must be distinguished | IMPLEMENTATION-STATUS and every E-file |
| Stop/restart and reuse after bugs must not discard unaffected work | CHECKPOINT-RECOVERY |

User constraints:

- One node, one RTX 5090, either 64 GB or 128 GB RAM.
- All generative LLM work uses OpenRouter, as in previous runs.
- OAEI and BioKG-Align data are available; binding/capability checks remain necessary.
- Broad method search using a specific informative case per feature, with broader validation
  at selected gates; NCIT–DOID is the general default, not a universal test dataset.
- Target 2–3 weeks of continuous execution, with calibrated budgets and reserved recovery time.
- OpenRouter spending has broad user discretion. Track/forecast cost and bound calls/tokens;
  do not block execution waiting for an additional arbitrary monetary cap.

The v1 matrices and the v1 implementation-only/no-real-runs handoff are superseded. No real
experiments or implementation changes were performed by this specification update.
