# Independent entity profiles and comparisons — B4

All generative calls use existing OpenRouter integration and configured credentials. Do not add local LLM requirements. Reuse provider routing, durable request/cost accounting and error handling, while separating these explanation products from matcher decisions. Read-only APIs serve completed artifacts; generation is explicit preparation work.

## Entity profile

Input: one EntityRef, allowed ontology facts/context, language, interpretation/missingness status and input-selection manifest. Exclude the counterpart, candidate pool/rank, Exact score/verdict and all reference/adjudication answers. Cache once per entity/context/policy/prompt/model, reusable across candidates.

Output: concise meaning, scope, identifying facts/qualifiers, aliases when useful, and unknowns. Each factual claim references one or more input fact IDs. Retain original ontology definitions separately from generated paraphrase. A missing definition may be described using supported labels/hierarchy/restrictions with that limitation; it must not be reconstructed from the target or uncited biomedical memory. If meaningful prose is unsupported, show original available facts and a missing-information statement.

## Pair comparison

Input: both independently generated/validated profiles plus selected allowed original facts and explicit missingness. The primary comparative prompt excludes matcher score, rank and verdict. It distinguishes agreements, scope differences, explicit incompatibilities, one-sided information and unresolved questions. It can disagree with Exact or abstain. Do not call a target-only assertion conflicting unless an actual opposing statement/qualified logical basis exists. A common broad parent is weak contextual similarity, not equivalence proof.

Output structured arrays of claims/citations plus optional tentative semantic relation and limitations. Do not require a forced binary match decision or a confidence number. Semantic relation is distinct from reviewer action; generated suggestions are distinct from Exact's saved decision trace. Reviewers can inspect facts and the system trace without reading generated text.

## Grounding and visibility

Apply study/research visibility policy **before** packet construction, retrieval and generation; policy hash enters every derived artifact/cache key. Exclude reference answers and prohibited mapping-xref/advice fields. A profile produced from unrestricted facts cannot be reused in a restricted condition merely by removing its citations. See 06.

Treat ontology descriptions as data, not instructions. Prompts state this boundary. Require structured output and schema validation; bound output length and evidence count. Verify citation existence, entity/side/snapshot/policy scope, and claim support. ID existence alone is not factuality verification. Deterministic checks catch contradictions such as claiming a missing source definition exists; supported semantic templates and blinded expert review assess paraphrase meaning. An optional LLM checker is advisory, never the sole truth oracle.

Record `validated`, `unverified` or `rejected` grounding state with reasons. Invalid/unsupported output is not displayed as approved explanation. At most two controlled repair attempts per request family by default; then retain error/output provenance and use factual fallback. Do not turn provider refusal, timeout or missing input into a negative mapping decision.

## Models, jobs and replay

Resolve one existing OpenRouter profile as baseline and one stronger configured challenger for the bounded development comparison. Record requested/returned model, provider, routing, parameters, prompt/schema hashes, input manifest and response IDs/content hashes. Do not assume mutable names are immutable snapshots. If provider revision cannot be pinned, record the limitation and freeze the actual generated artifacts used in the study. Keep provider secrets outside manifests.

Request ledger states: pending, dispatched, response_saved, validated, failed, ambiguous. Persist the request identity before dispatch and save response atomically before validation. A crash after remote completion may make exact-once billing impossible; retry with provider idempotency only where supported, otherwise record ambiguity and possible duplicate cost. Never claim a guarantee the provider does not offer. Checkpoint every completed entity/pair; process a bounded queue with retry/backoff, rate/concurrency limits and STOP handling.

Cache identity includes ontology/context and visibility hashes, ordered fact packet and truncation policy, profile dependencies for comparisons, prompt text, output schema, model/provider configuration, parameters and language. An entity profile must not depend on candidate order or counterpart. Changing the prompt/model explicitly regenerates affected text even when old strings are populated. Keep previous generations immutable and linked as superseded; do not rerun matching for a text-only change.

## Bounded calibration and acceptance

Use 48 development pairs from the declared development universe (09), plus semantic fixtures. Compare current rationale as historical baseline, revised profiles/comparison on the current model, and the same revised task on one stronger model. A fourth old-prompt/stronger-model arm is optional only for a declared prompt×model analysis. Hold evidence/visibility fixed. Selection criteria are supported facts, discriminating information, missingness handling and reviewer decision quality; fluency alone does not select the model.

B5 requires real OpenRouter outputs on at least 12 varied development pairs, including missing information and close alternatives, with schema/citation checks and a documented claim audit. This is an operational gate, not proof of human benefit or a requirement to recruit participants before F1. The implementing agent can use the configured baseline provisionally and record model selection as pending; never fabricate external expert approval. All 48-case comparison artifacts can follow within bounded development, before final human-study freeze.

Verify candidate-independence, order independence of entity profiles, no forced advocacy, explicit unsupported states, policy cache isolation, prompt-only selective regeneration, resumed completion and actual request/cost records. The deterministic factual fallback must remain usable when LLM services are unavailable.


The mixed study in 11 also requires development examples where none of five candidates is equivalent. Check that profiles/comparisons do not assume a correct option exists, force a match or convert missing information into incompatibility. Keep answer keys/case-kind labels out of generation inputs. Prepared text for negative cases follows the same prompts, evidence selection, visibility and quality checks as positive cases; no wording may reveal a private negative-case flag. This adds coverage to bounded development, not live generation during the study.
