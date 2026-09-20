# Two products and three deployments — B0/B3/F1

**Accepted scope amendment, 2026-09-20; implementation pending.** This document and 11–13 extend the backend and frontend deliverables. They supersede the earlier idea that a restricted viewer embedded in an external survey is sufficient. Ranking is the study's primary task; ordinary application review actions remain a separate capability.

## Product boundaries

| Product / deployment | Required behavior | Excluded behavior |
|---|---|---|
| Exploration app / local | User starts local service, imports an Exact inspection bundle through the app, selects it, explores both ontologies and saved matches/explanations | No implicit matching or paid generation on upload/view |
| Exploration app / online demo | Same exploration components with the operator's fixed preloaded bundle | No public bundle import, arbitrary server path/URL input, study records or researcher administration |
| Study application / Render | Complete participant flow and researcher management, using frozen case packages and shared explanation components | No external survey dependency, participant upload or unrestricted research/demo routes |

Use one shared ontology/explanation engine and reusable UI components, with explicit server-owned deployment profiles `local_app`, `public_demo`, `study`. Client mode switches cannot enable capabilities. The two hosted profiles should use separate services/origins and storage scopes. The public demo must not expose final study cases or answer-bearing artifacts; use development cases instead. A participant cannot access a case's hidden explanation through its ordinary app API during the baseline condition.

The local app remains the main development product. Shared exploration gets the primary engineering investment; the survey shell adds workflow rather than a second ontology engine. The specialized frontend agent implements both products only after B5 includes the new services. Backend development may use fixture clients without an early frontend specialist.

## Portable inspection bundle

Define versioned `InspectionBundle` with root manifest, package/schema IDs, content hashes, ontology/context versions and import scope, Exact run/config identity, candidates/scores/decision traces, evidence links, prepared entity profiles/comparisons, capabilities/completeness and license/export treatment. Relative artifact locators must survive relocation. The bundle must provide the context indexes/original axiom material needed for its declared exploration coverage, not just the nodes selected into support graphs.

An inspection bundle is an exported, prepared product of an Exact run. A raw run directory and the old study JSON are not automatically full bundles. Preserve explicit legacy adapters: show available matching evidence with truthful missing-context messages, and offer a documented preparation path to a full bundle. Do not silently invent full-ontology support or fetch imports when opening old data.

Original ontology bytes are included when permitted/needed for offline exploration and external tools, or carried in an independently verified, explicitly bound local package. A self-contained portable claim requires successful reopening without the original run/input directories. A merely linked bundle must identify missing dependencies before it is usable. The study additionally needs downloadable, frozen ontology files and setup instructions compatible with its information policy.

Reuse `exact-inspect prepare` and extend it with an explicit portable export stage. No matching or LLM generation is implied by packaging. Define the archive format and compatibility/migration rules in B0; a versioned ZIP of inert data is a suitable default. No deserialization of executable objects, plugins or scripts from uploaded archives.

## Local import workflow

1. Select/upload a bundle; show manifest identity, versions, size and declared coverage.
2. Stream into bounded staging storage; verify checksums, schema and artifact references before atomic publication to the local bundle library.
3. Reject unsafe paths, archive traversal, symlinks, duplicate entries and excessive decompression; set configurable byte/file limits and cancellation. Do not load training/model weights or execute bundle contents.
4. Present usable, partial, incompatible and corrupt states with actionable messages. A failed import must preserve the current usable bundle.
5. Persist selected bundle/navigation where appropriate; reopening or copying the bundle must preserve stable identities. Deleting a library copy must not delete the user's original run.

The local service binds to loopback by default. Import/progress operations are a local-only mutation surface; read-only exploration remains lazy and bounded per 04. State clearly that upload goes to the locally running service. Hosted demo/study reject import requests on the server, including direct requests that bypass the UI.

## Study package and admission

A `StudyDefinition` references immutable participant-safe case packages, ontology resources, questionnaires, tutorials, candidate presentation, scoring/adjudication version, assignment design and software compatibility. Keep the researcher answer key separately controlled. Publication validates every candidate ID, score provenance, resource hash, condition policy and adjudicated case kind.

Both conditions receive identical source/candidate identities, saved scores and frozen initial order for the same case, with constructed-set provenance handled per 11. They have the same permitted external ontology resources. Only the explanation condition receives the integrated meaning/hierarchy/evidence/comparison components. Basic ranking controls and instructions remain identical. Never expose the answer-present/absent label, quotas, acceptable target set or correctness feedback while the study is active.

If mapping xrefs/annotations are prohibited, the downloaded ontology files, prepared indexes and generated inputs must implement the same declared policy. Removing a field from API JSON while leaving it in the downloadable file is not blinding. Preserve the modified-resource identity and filtering manifest; reject unsuitable cases rather than silently deleting semantic axioms to force an outcome.

## Acceptance

Demonstrate local import, cancellation/corruption handling, compatible legacy degradation, switching two bundles, and offline reopening on a clean machine. Prove that direct import requests fail in hosted profiles. Verify study and demo isolation, equal permitted ontology resources, answer-key separation and no runtime provider/model work. Demonstrate the same real prepared case in the local app and explanation-study fixture, plus the server-enforced restricted baseline representation.
