# Specialized frontend implementation — F1, after B5

**Do not start the specialized frontend agent until G5 in 07 passes.** This brief is written now to make backend obligations concrete; it does not require an early frontend design phase. Once admitted, the specialist owns full information architecture, visual design, responsive implementation and integration for both the exploration app and complete study application in 10–13. Use the actual backend handoff and fixtures; inspect existing React/Next/Cytoscape code before choosing a refactor or replacement.

## Product outcome

A domain expert must be able to understand each entity and find a distinguishing fact. An ontology expert must be able to navigate hierarchy and inspect original axioms. Both must be able to reject Exact's suggestion or defer for missing evidence. The default screen must support those tasks before showing dense metrics or a network. The frontend does not infer equivalence, reinterpret absent fields, calculate probabilities, parse OWL or call an LLM directly.

The specialist can choose the detailed design, but these requirements are fixed:

- Persistent, equally prominent source and target meaning cards: identity/kind, original definition or explicit absence, useful aliases, scope/defining facts, and discoverable context/provenance. Generated paraphrase is distinguished from original text.
- Independent hierarchy browsers: reveal entity, search, parent/child navigation, multiple inheritance/equivalence handling, explicit asserted/structural/inferred basis, counts/paging and preserved navigation. Changing candidate keeps source context. Context does not depend on expanding the match graph.
- Compact balanced comparison: shared information, differences/scope, actual incompatibilities and unknowns. Decision trace is a separate optional view. Scores are 'matching scores' unless the backend supplies a validated probability meaning.
- Stable evidence graph as an optional view. Ontology assertions, projected evidence and matcher bridges are visually and textually distinguishable. Selected edges expose direction, semantic reading, original expression and provenance. The graph has an equivalent accessible evidence list.
- Exploration-app review actions and study responses are distinct. Study tasks use a partial ranking or explicit none/insufficient information per 11; do not replace ranking with accept/reject or require every candidate to be ranked. Save state and display errors honestly. Study views respect server assignments, never client-side answer masking.

## User feedback acceptance

| Complaint | Required result |
|---|---|
| Overwhelming left panel / hover movement | No pointer movement changes geometry, scroll, selection or camera. Metrics collapsed initially; explicit persistent help. Honor reduced motion. |
| Values on property arrows | Hide numeric match weights by default; retain them in details. Meaningful ontology values/cardinalities remain accessible. |
| Selection order | Wide-screen reading/keyboard order: candidate choice, source–target comparison, optional details. |
| Similar blue/purple | Categories distinguishable without color through labels/icons/line patterns; verify contrast and both themes. |
| Ambiguous +/- | Magnifier plus/minus with accessible names; explicit fit/reset; distinct ontology expansion affordances. |
| Generic source label | 'Source class'/'Target class' for class tasks; actual kind for properties/individuals. |
| Small fonts | Primary text default 16 CSS px or larger; app-wide persistent size control; readable initial graph labels, not 6 px after fit. |
| Poor expansion | Separate usable hierarchy/context navigation; any optional graph expansion is explicit, cumulative/reversible and bounded with counts. |
| Sparse source | Clear available/missing/filtered/unloaded states; no padding with invented prose or duplicate-name nodes. |
| Complex relations | Faithful plain-language meaning plus original axiom/predicate/qualifiers; lossless fallback where no template exists. |
| Screen-size problems | Same essential information and actions at 320/360/390/768/1024/1440/2560 CSS px, portrait/landscape and 200% text enlargement. |

On phones, use readable sequential cards or labeled views and optional dedicated graph space; do not shrink the desktop graph and hide descriptions. Preserve selected entities/candidate, hierarchy navigation, explicit selections and reading position across breakpoints. No required hover; mouse, touch and keyboard have equivalent access. A click/tap inspects consistently rather than using a screen-size-dependent recenter-first rule.

Dialogs have accessible names/focus trapping/return; search/comboboxes and tree navigation use appropriate semantics and keyboard behavior. Narrative/control content reflows without document-wide horizontal scrolling; a truly 2D graph may pan within its own region. Loading, empty, stale, failed, truncated, unsupported and disconnected states are designed from the real API fixtures.

## Delivery and tests

Begin by running the provided backend package and examining actual good/sparse/complex cases. Build a complete end-to-end comparison flow before polishing every optional panel. Reuse shared components across breakpoints. Keep interpretation in the backend; request additive contract amendments explicitly if needed.

Deliver runnable application, design decisions, verified API integration, viewport/accessibility evidence and continuation commands. Test real data plus duplicate labels, multiple parents, missing definitions, long names/nested restrictions, server failures and resumed decisions. Verify no hover reflow, no accidental selection, no state loss on resize and no exposure of reference answers. Integrate fresh package versions through declared compatibility behavior rather than silently mixing runs.

Do not claim the new interface outperforms Protégé from screenshots or automated tests. Complete formative iterations and the later frozen evaluation in 07. No recruitment or outbound messages are implied by the frontend assignment.

## Full product workflows added 2026-09-20

Implement local bundle upload/import validation/progress/library and truthful legacy states; the public demo opens only its supplied bundle with no import controls. Both retain the main exploration flow. Implement the study as a complete first-party interface: private-link resumption, welcome/setup/background, practice, two assigned condition blocks, ranking/consultation, final ratings/comments and completion. Provide minimal authenticated researcher screens for study publication, link generation, progress and exports. Use the service contract, not frontend storage, as the authoritative participant state.

The baseline must use the same ranking component, source/candidate identities/scores and external resource access as the explanation condition, while integrated explanations remain unavailable server-side. Keep initial-list positions and participant ranks distinct; natural cases preserve system ranks, while constructed sets must not expose production-rank gaps that reveal case kind. Candidate labels/IDs do not change when dragged. Offer select-next, move, remove, undo and explicit keep-initial-order; keyboard/touch and partial lists are first-class. Do not auto-submit the displayed original order. None and insufficient information require explicit selection. Save/submit acknowledgements, conflicts, reconnects and an unfinished consultation step are visible and recoverable.

Prior-study feedback adds these acceptance cases: immediate explicit descriptions rather than delayed hover; parallel edges remain distinguishable; conventional drag, permanent fit/reset and recoverable graph position; practice progresses in complexity outside the scored set; source references are as discoverable as target context. Numeric agreement strength must not be mislabeled as semantic equivalence. Table, graph and generated text use the same qualified facts; 'only described on this side' does not mean contradiction. Preserve umbrella/conditional inheritance and onset qualifiers; show original axioms for verification. Add regression fixtures for the reported same-attribute conflict and mixed inheritance/onset cases without declaring the old cases medically wrong from feedback alone.

Verify the entire study across reload, closure/original-link return, slow/disconnected networks, narrow screens and keyboard use. A desktop with installed Protégé is the initial scored-study setup; mobile resumption must preserve state and explain any unmet setup requirement rather than silently changing condition. Prototype and integration tests use synthetic participant records, never fabricated study findings. Final live launch requires the separate readiness in 07/12.
