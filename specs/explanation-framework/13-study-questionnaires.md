# Participant flow and questionnaires — B0/B3/F1

All questions, wording, answer codes, skip rules and required/optional status are versioned before publication. Participant-facing text is plain language, not backend identifiers. The following is the initial wording to implement and pilot; revisions create a new form version. No name, email, institution or exact age is requested.

## Welcome, setup and practice

Explain the task, study duration estimated from the pilot, voluntary participation, response/click timing collection and link-based resumption. Ask users to retain the original private invitation link and not share it or include identifying details in comments. Consent/information text is supplied by the study owner before launch; do not fabricate approval or consent wording beyond the supplied protocol.

Before scored cases, participants must confirm that Protégé is installed and both supplied ontology files can be opened. Provide pinned downloads/hashes and platform-neutral setup instructions ahead of participation. Record setup success/failure and self-reported Protégé version where known. Ask them to locate a practice source and inspect its definition/parents. Do not require Proof of installation or inspect their computer; mark checks as self-reported/task-confirmed. Failure routes to setup help or save-and-return, not silent entry into a different study condition.

Provide an unscored tutorial explaining source/candidate identity, scores versus correctness, entity descriptions, hierarchy, evidence relations and ranking controls. Include simple, complex, partial-ranking and none-of-these practice. Make clear that some scored cases have no equivalent among the displayed candidates, and that insufficient evidence is a valid distinct response. No scored-case answers appear in tutorials or the public demo.

## Background form

Unless marked optional, an answer is required with `prefer_not_to_say` available for background items. Do not infer missing expertise from that response. Years mean approximate combined study/work experience, counting overlapping years once; they are not a claim of proficiency.

| ID | Participant question | Options / behavior |
|---|---|---|
| `cs_experience_years` | How much study or professional experience do you have in computer science or computing? | None; less than 1 year; 1–2; 3–4; 5–9; 10 or more; prefer not to say |
| `health_bio_experience_years` | How much study or professional experience do you have in health, medicine or biosciences? | Same scale |
| `domain_specialty` | What is your main health or bioscience field, if any? | Broad short text, optional; prompt against identifying institution/employer |
| `semantic_web_experience` | Have you used semantic-web technologies or worked with ontologies in study or professional work? | Yes; no; prefer not to say |
| `clinical_experience` | Do you have practical experience in clinical medicine? | Yes; no; prefer not to say |
| `bioinformatics_experience` | Do you have practical experience in bioinformatics? | Yes; no; prefer not to say |
| `doid_familiarity` | Before this study, how familiar were you with the Disease Ontology (DOID)? | Had not heard of it; had heard of it but not used it; had used it; prefer not to say |
| `doid_contexts` | In which contexts have you used DOID? | Show if used; multi-select research, clinical work, teaching, curation, other, prefer not to say |
| `ncit_familiarity` | Before this study, how familiar were you with the NCI Thesaurus (NCIT)? | Same familiarity scale |
| `ncit_contexts` | In which contexts have you used NCIT? | Same conditional contexts |
| `roles` | Which roles describe your current work or study? | Multi-select undergraduate student, postgraduate student, researcher, senior researcher, medical doctor/clinician, other, prefer not to say |
| `correspondence_confidence` | How confident are you in identifying when two medical or disease terms refer to the same concept? | Not at all; slightly; moderately; highly; prefer not to say |
| `ontology_activities` | Which of the following have you done with ontologies? | Multi-select described/annotated data; used in automated analysis or software; used for manual curation; developed/maintained ontologies; other; never used; prefer not to say |
| `protege_experience` | Before preparing for this study, how often had you used Protégé? | Never; tried occasionally; used regularly; prefer not to say |

`never used` and `prefer not to say` are exclusive with substantive multi-select answers. Other may expose optional short text, but is not a requirement to identify an employer or project. Validate hidden conditional answers consistently when a parent answer changes; preserve prior draft history without exporting stale answers as current. Role, years and confidence remain distinct; do not automatically label someone an expert solely from one checkbox.

## Per-case task and consultation

Instruction: 'Rank the candidates you consider plausible equivalents of the source concept, with the best first. You may use fewer than five. If none appears equivalent, choose None of these; if you cannot judge, choose Insufficient information. The initial order and system scores are suggestions, not answers.' Use a visible source label/IRI, stable candidate identities and clear participant rank numbers. Response codes are in 11.

Immediately after the ranking is committed, ask in **both** conditions:

1. `consulted_external_ontologies`: 'For this case, did you consult the ontology resources outside the study's explanation panels?' Yes / No. This is required; internal panel use is captured separately and must not be confused with external-file consultation.
2. If Yes, `consultation_methods`: 'How did you consult them? Select all that apply.' Protégé / Another ontology editor / The ontology files directly (for example, in a text editor) / Another ontology resource or viewer.
3. Optional `other_editor` / `other_resource` short text when appropriate. No URL, institution or personal name is needed.

A participant may use several methods. A clicked link is not automatically recorded as Yes; this question records self-report, separate from telemetry. No means an empty methods list. Save this step separately and return to it after a refresh without changing the already submitted ranking. Time spent completing this post-case consultation questionnaire is excluded from ranking-task duration. Time spent inspecting ontology resources before ranking submission remains included.

Optional per-case confidence or difficulty questions must be declared before publication, identical across conditions and included in burden estimates. They are not required for the initial study; avoid extra questions that interrupt every comparison unnecessarily.

## Final form

| ID | Question | Options / behavior |
|---|---|---|
| `component_usefulness` | How helpful was each component for deciding your ranking? | One rating for original entity definitions/context; generated entity descriptions; hierarchy browser; comparison/evidence table; evidence graph; generated pair comparison. Not helpful; slightly; moderately; very helpful; did not use/cannot judge |
| `most_helpful_components` | Which explanation component or components were most helpful? | Multi-select the same components; all equally helpful; none helpful; cannot judge. Special choices exclusive |
| `workflow_preference` | Which approach would you prefer for similar tasks? | Explanation interface; candidates/scores with separate ontology inspection; no preference; cannot judge |
| `mental_effort` | How much mental effort did each approach require? | Separate ratings for the two conditions: very low; low; moderate; high; very high; cannot judge |
| `preference_reason` | Why did you prefer those components or that approach? | Open text, optional |
| `comments` | What was confusing, missing or difficult, and what would you change? | Open text, optional |

Keep preference separate from measured performance. Display only components actually provided in the frozen interface; adding/removing a component changes the form version. Distinguish original definitions from generated prose. Completion confirms receipt and permits return to the completion page; it does not reveal the answer key while recruitment is active.

## Acceptance

Test branching, exclusive options, multi-role answers, question order, validation errors, keyboard/touch use, saving indicators and restored drafts. Required steps cannot be skipped by manipulating a URL, while optional free text remains optional. Questionnaire edits during an active study do not rewrite historical responses. Export codes, labels, versions and skipped/not-answered states distinctly.
