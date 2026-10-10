# Backend S1 contracts for frontend S2

This is the integration contract for specs 20–22. The accepted frontend base is
`ffd390f35c108abe47f5603b91d28bd1a54e590d`. The runtime schema files under
`specs/explanation-framework/protocol/` are generated from the implemented models.
The final S1 review and receipt record the tested implementation and data identities.

## Compatibility and publication purpose

Existing `exact-study/1.0` and `exact-study/2.0` publications, forms/1–2, saved answers,
mutation receipts and analysis/2–3 retain their meanings. An exact publication retry reads
its frozen materialization before applying new admission rules. Changed revisions still
undergo normal admission and cannot replace published content.

The study envelope remains `exact-study/2.0`. A new review publication explicitly uses:

```json
{
  "purpose": "development_review",
  "synthetic": false,
  "correctness_metrics": "disabled",
  "form_version": "exact-study-forms/3",
  "protocol_versions": {
    "forms": "exact-study-forms/3",
    "setup": "setup/3",
    "consultation": "consultation/3",
    "methods": "study-methods/3",
    "export": "exact-study-analysis/4"
  }
}
```

This excerpt is not a complete publication. Use the generated publication schema for all
required fields. Real biomedical cases have unresolved private correctness keys; an empty
acceptable set is never presented as an adjudicated negative. Review publications cannot
issue research invitations, require a loopback origin, and do not carry launch approvals.
Tutorial content remains invented and disjoint from the full real ontology indexes.

For new forms/3 sessions, POST `/api/v1/study/session` must include
`{"secret":"<private invitation capability>","client_contract":"study-review/1"}`.
An old client omitting that declaration receives 409 before answer collection. Legacy
publications continue accepting the historical secret-only request. S2 must explicitly
recognize all new form/method versions before making this declaration. Do not merely add
the string to a client that still renders old form semantics.

Concrete delivered examples are in
[the S2 response excerpts](explanation-product-review-backend-s1/s2-response-examples.json).
They retain the actual public HTTP fields, full-response hash, status and extraction pointers;
they are bounded subsets, not complete payloads for schema validation. The actual public forms
and protocol dictionaries are captured from a successful negotiated session response with all
participant state and capabilities omitted. Use the generated schemas for complete envelopes.

## Tutorial actions and safe rejection

Keep the existing mutation envelope, revision check, idempotency key and acknowledged-state
recovery. Entity-bearing actions require the exact `ontology_version_id`, `iri` and explicit
`kind`. No class default is inserted at the tutorial action request boundary. Shared legacy
entity models keep their historical defaults.

| Action | Required evidence |
|---|---|
| `open_citation` | An admitted fact cited by a prepared claim in this tutorial |
| `open_original_axiom` | The admitted original fact actually inspected |
| `locate_in_evidence_list`, `inspect_graph_or_list` | An available selected feature linked to the inspected fact |
| `search_entity`, `copy_iri` | The admitted typed entity actually used |
| `navigate_parent`, `navigate_child` | A recorded typed hierarchy transition from a reachable current entity |
| `return_to_compared` | A compared typed entity reached after real navigation away |
| Bridge/context selection | Selection/inspection telemetry; it is not original-feature completion |

An additive strict selected-feature action is:

```json
{
  "requirement_id": "<frozen requirement ID>",
  "action": "locate_in_evidence_list",
  "candidate_id": "<current tutorial candidate ID>",
  "evidence_id": "<recorded selected evidence ID>",
  "fact_id": "<linked inspected original fact ID>"
}
```

`evidence_id` remains optional for valid historical fact-only payloads. If supplied it must
resolve to that fact and candidate; ambiguous evidence identities require candidate scope.
A bridge ID cannot be substituted for a fact. Null new evidence fields are excluded from the
legacy request hash so old receipts replay. New publications are checked for reachable
search, parent/child, return, citation and selected-feature tasks for every permitted candidate.
Five assessment answers and completion stay server-owned; optional graph work stays optional.

New errors retain a short safe `detail` and add:

```json
{
  "detail": "Original fact is outside the synthetic tutorial",
  "validation": {
    "code": "tutorial_fact_out_of_scope",
    "requirement_ids": ["<authored requirement>"],
    "action_ids": ["open_original_axiom"]
  },
  "correlation_id": "<request UUID>"
}
```

The exact detail is not a machine key. Malformed request structure uses
`tutorial_action_payload_invalid` and safe numeric action positions. The response correlation
matches `X-Request-ID`. Rejection writes neither progress nor revision/history/receipt; the
next valid mutation may reuse the rejected idempotency key. S2 should preserve S0 rollback
and two-tab recovery and use additive validation only for diagnostics.

## Hierarchy and full context

Exploration routes:

- GET `/api/v1/ontologies/{ontology_id}/entry-points`
- GET `/api/v1/ontologies/{ontology_id}/equivalents?iri=...&kind=class`
- Existing GET `/api/v1/hierarchy?ontology_version_id=...&iri=...&kind=class&basis=...`

Study routes use `/api/v1/study/workspace/{scope_id}/entry-points` and `/equivalents`, with
`ontology_version_id` as a query parameter, and the existing scoped `/hierarchy` route.
Each request/page rechecks the session, generation, stage, current assignment, condition,
resource scope and policy. The ontology baseline cannot query explanation workspace routes.
Use the current state's authorized workspace descriptor, never another session's locator.

Entry points accept `kind`, `basis`, `language`, `limit` (1–100) and opaque `cursor`.
The advertised contract is `exact-hierarchy/1`; bases are `literal_asserted`,
`structural_navigation`, and `reasoner_inferred`. Responses retain the normal Page envelope
and add `navigation_contract`, `basis`, `default_entity`, `completeness`, and `inference`.
An item contains a typed `entity`, `preferred_label`, `entry_kind` and `component_size`.
`entry_kind` is `named_root`, `component_entry` or `entity_listing`. A cycle entry represents
navigation choice, not a newly asserted relationship to owl:Thing. Individuals use a listing.

Prepared inference is a named direct class taxonomy, not all ancestor pairs. Equivalent
classes share a recorded group; top and bottom/unsatisfiable membership are explicit in
`entity_semantics`. `/equivalents` provides bounded group membership. Properties and
individuals do not acquire invented class inference. No minimal justification is claimed.
Failed/unsupported/not-prepared inference is distinct from an available empty page; S2 must
show its status/reason and retain asserted browsing. Context, policy, basis, kind and query
enter cursor identities. Do not reuse an old cursor after changing any of them.

Classification occurs only during explicit preparation through the installed native reasoner,
using admitted logical input. No endpoint launches a parser, model or reasoner. Context indexes,
navigation sidecars and their manifests travel with the portable bundle. Root-only and derived
DOID provenance remains visible; it is not the unresolved original/import-closure gate passing.

## Description and evidence meaning

New prepared descriptions use `exact-description/2`. The served metadata is explicit:

| `preparation_kind` | `support_status` | Meaning |
|---|---|---|
| `grounded_description` | `typed_template_support` | Tested rendering of admitted definitions/typed necessary conditions/equivalence |
| `grounded_description` | `independently_reviewed` | Saved OpenRouter prose admitted by a separate hash-bound semantic review |
| `fact_extract` | `exact_support` | Original literal extracts; do not title them synthesized descriptions |
| `factual_fallback` | `fallback` | Generation/review unavailable or failed; retain the reason |

Missing legacy metadata maps at read time to fact extract or fallback without rewriting its
frozen bytes. A label alone is not a biomedical definition. Necessary superclass conditions
are not equivalence; universal restrictions do not assert existence; cardinality/quantifiers
remain exact. The original ontology definition stays a separate original fact.

Independent prose is an explicit offline process: generate and save response; build a review
request; have a separate reviewer approve/reject each exact claim against cited originals;
then derive a new artifact. The `exact-description-review/1` receipt binds the packet, policy,
source manifest, original response, claim inventory and decisions. Provider output cannot supply
its own approval. The original failed/fallback artifact remains immutable. Serving verifies
both receipt bindings and the actual authoritative original facts/ASTs. Changing approved text,
limitations, reasons, generation status or provenance invalidates the artifact. New study
`independent_reviews` envelopes also carry the complete `generation_manifest`; its computed
digest, source reconstruction and review binding are checked rather than trusting matching
caller-supplied digest strings.

Comparison wording has its own new dependency identity. S2 must render returned claims intact,
not delete old caveat substrings. New short excerpts remain explicitly excerpts; asymmetric
information remains different from opposition. Original pre-review profile dependencies are
retained in the portable package, separately from the final displayed artifacts.

Pair traces add `exact_basis` only as recorded metadata. A protected flag or score one does
not establish normalized-label equality. Historical absent triggers stay `not_recorded`;
internal `prefilter` stage IDs remain compatible while UI copy says Exact matching.

`exact-evidence-metadata/1` conserves selected-feature counts: resolved + unresolved +
not_exported + withheld equals selected. Graph supported + unsupported + not_recorded also
equals selected. Missing mappings do not disappear as zero selected features. Explicit bridges
identify either `channel_comparison` or `feature_correspondence`, with typed endpoints and
recorded IDs/provenance. A frontend must not fabricate a bridge from matching channel names,
including source-only channels. Original records remain inspectable when graph projection is
unsupported. Preserve list/graph identity and distinguish added context from matcher evidence.

### Recorded comparison metadata in both workspaces

The main `/api/v1/runs/{run_id}/pair` response and authorised study
`/api/v1/study/workspace/{scope_id}/evidence-metadata?candidate_id=...` expose the same
`exact-evidence-metadata/1` value. The study response contains `candidate_id`, typed `source`,
typed `target`, and `evidence_metadata`. It repeats the current stage/condition/session/
presentation checks; a baseline or alien-candidate request receives 403.

Check the study scope capability `evidence_metadata_contract` before requesting this route.
A legacy index reports null and the route returns 404, meaning not exported. New metadata
indexes use `exact-workspace-index/2`; the optional index version is omitted when serializing
old scopes, preserving historical publication hashes. Neither absence nor failure creates a
comparison bridge in the client.

The importer now reads the producer's retained `cross_side_provenance`. Matrix feature links
resolve by side, typed entity, channel, hierarchy family and exact saved alias. Attribute anchors
may cross channels; an omitted hierarchy family is resolved only when one selected feature
matches. Lexical entity-to-entity and difference feature-to-entity links remain
`channel_comparison`, not paired original facts. `source_channel`, `target_channel` and their
family fields describe each side. `comparison_basis` distinguishes `matrix_support`,
`attribute_anchor`, `difference_endpoint`, and `lexical_score`. `recorded_score` is a matcher
metric, not equivalence or a calibrated correctness probability.

Every retained legacy comparison gets a `comparison_records` entry with a hashed identity and
`resolved`, `unresolved`, `ambiguous`, or `unsupported` status; `comparison_resolution` conserves
that inventory. Missing mappings cannot become `not_recorded`. Unavailable originals cause
withheld bridge support while retaining the recorded comparison identity. S2 must inspect both
identity-resolution status and bridge visibility, and must not draw an unsupported feature
correspondence from shared channel names.

## Forms/3 and restricted analysis

Use server-ordered definitions, conditional question IDs, new dictionaries and validation.
Experience codes are `none`, `less_than_1`, `1_3`, `4_5`, `6_9`, `10_plus`. Count completed
years without recoding old bins. Newly published option sets omit `prefer_not_to_say`;
optional fields remain optional. Background Other questions require nonblank conditional
text on final submission but permit unfinished drafts.

Method codes are `protege`, `plain_files`, `other_method`, displayed as Protégé, Text editor,
Other method. `plain_files` has this narrowed meaning only under `study-methods/3`.
Familiarity additionally permits exclusive `none`; an unanswered empty list is different.
`familiar_other_method` and per-case `other_method` accept 1–160 trimmed characters on final
submission when Other is selected. Deselecting Other or selecting No clears current detail;
historical drafts remain stored. External inspection remains optional, combinable and mutable
between cases. Readiness never requires installing a named tool. Setup is not actual use.

Setup/3 example (IDs/revision are supplied by the live client):

```json
{
  "idempotency_key": "<new stable mutation key>", "expected_revision": 1,
  "setup_version": "setup/3", "instructions_acknowledged": true,
  "external_inspection_optional_understood": true, "resource_access": "available",
  "familiar_methods": ["plain_files", "other_method"],
  "familiar_other_method": "An ontology viewer", "submitted": true
}
```

A consultation/3 final request retains the live presentation ID, form version and revision:

```json
{
  "idempotency_key": "<new stable mutation key>", "expected_revision": 20,
  "presentation_id": "<current presentation>", "form_version": "exact-study-forms/3",
  "consultation_version": "consultation/3", "consulted_external_ontologies": true,
  "methods": ["protege", "other_method"], "other_method": "A local RDF tool",
  "resource_scope": null
}
```

No uses an empty method list and null detail. Do not prefill a case report from prior cases,
setup, downloads or telemetry. Tutorial method-report actions use the same new dictionary.

Analysis/4 preserves analysis/3 page-clock timing semantics. Default researcher exports redact
named method details. Only authenticated explicit `include_named_methods=true` adds current
setup/consultation method names and their case/session/version associations. It does not include
general role, specialty, comments or arbitrary tutorial free text. JSON preserves Unicode;
CSV escapes formula prefixes. Hidden, unanswered, redacted and included are different states.
Review exports identify purpose/test status and disable correctness summaries. Existing saved
analysis/2–3 files and codebooks are unchanged.

## Review discovery and exact S2 work

GET `/api/v1/review-bundle` provides strict public-safe `exact-review-about/1` metadata;
review study state includes an aggregate projection of `review_bundle`. It identifies counts, ontology scope,
policy/provenance hashes, coverage and limitations without private paths, credentials or keys.
The main review endpoint's `cases` array gives each review `case_id`, typed `source` and ordered typed `targets`;
`source_run_id` selects the corresponding exploration run. Use these actual identities to
compare the same 24 sources and 120 pairs during owner review. Participant state always has
`cases: []`, including before assignment, both conditions and historical mutation replay.
Obtain only the current authorized study case through its existing case route; do not fetch
the main catalogue from the participant app. Frozen publication and receipt bytes are unchanged.
The underlying exploration
run retains the larger recorded candidate universe. Private original production ranks remain
empty when the producer did not record ranks; recorded array positions are catalogue provenance,
not matcher ranks. Non-review publications cannot use unresolved correctness keys.
Expose one About this review bundle entry. A legacy package returns 404 for this optional
metadata; capability discovery distinguishes it from a failed real request.

S2 must implement the negotiated forms and methods, new preparation/support presentation,
root/equivalence/inferred navigation with honest failed states, and the public About entry.
Consume actual frozen tutorial/evidence identities, preserve the accepted assessment recovery,
and finish F02/F08/F09 presentation work against the real bundle. Then perform the complete
browser journey and side-by-side rendered Q01–Q12 checks. Backend HTTP/regression results are
not frontend accessibility, usability or joint acceptance. Protégé remains a recommendation.

Decision adapter `decision-adapter/6` retains the complete verified `dependencies` map in the
run manifest and declares its canonical `dependency_inventory_hash` under
`exact-decision-dependency-inventory/1`. Legacy/default bridge and comparison-resolution
`artifact_refs` contain that inventory hash; resolve it against the same verified run manifest
to obtain the original dependency hashes. It denotes an inventory reference, not a newly
recorded producer file. Explicit producer artifact references remain unchanged. Admission checks
the declared inventory schema and recomputed hash, so missing or altered inventories fail closed.

Adapter `decision-adapter/7` retains the same inventory contract and unions typed candidates
recorded in `source_decisions`, including candidates without explanation exports. On
`GET /api/v1/runs/{run_id}/candidates`, optional `source_decision_record` binds the zero-based
`source_record_index` and `candidate_index`, `ordering="source_decisions_candidates_array"`,
and separately retained `scores` to `artifact_ref`, the exact source-decision file hash.
Array location is not a retrieval or production rank. Only fixed scalar score/control fields
enter the adapter. Missing stage/evidence exports remain explicit. A uniquely declared named
global mapping is admitted only after contained-path, hash and size verification; historical
canonical mappings retain their existing seam and acquire consumed-byte hashes.

Policy export `exact-policy-decisions/5` also requires each selected-evidence citation to
reference the feature's exact focal IRI and kind. Generic context resolution can legitimately
find an original axiom of a neighbouring concept; that alone does not support a focal citation.
Such selected records remain counted with `status: "not_exported"`,
`origin_resolution: "unresolved"`, empty `fact_ids` and a specific typed-entity reason. Mixed
valid/neighbour citations are conservatively unresolved as a whole record. Originals excluded
by policy remain `withheld`, a separate state. Serving historical indexes applies the same
projection without rewriting their bytes. Keep the unresolved record visible in the list;
do not treat it as zero selected evidence or create an original-fact graph edge from it.


### Delivered-data boundaries for S2

The interim bundle contains 140 unique typed entity profiles and 120 comparisons, all
`grounded_description` with `typed_template_support`. It contains no approved provider prose
and no `factual_fallback` result; those are supported runtime branches with separate tests.
Both real inference preparations report failure. All 3,443 selected evidence records have
graph projection `not_recorded`, including the 301 records with admitted focal original links.
All 120 selected pair traces report the `prefilter` event and `exact_basis` as `not_recorded`.
Do not derive route-event copy from saved protected flags or score 1, or graph support from
original-fact resolution. The owner catalogue and final handoff record these coverage gaps.
