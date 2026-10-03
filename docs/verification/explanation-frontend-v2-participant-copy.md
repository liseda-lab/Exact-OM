# Proposed exact-study/2.0 participant copy (for review)

**Status:** proposal for the study owner and backend owner to review and freeze (specs 14–16). It was extracted on 2026-10-03 from the code that renders it (`SetupV2.tsx`, `Resources.tsx`, `Tutorial.tsx`, `ConsultationV2.tsx`, `src/study/v2/tutorialContent.ts`) and matched the preview then. If the two ever differ, the code is authoritative. It is task wording, not consent or ethics text. Answer keys below are **synthetic tutorial material**, never research case keys; they stay on the server and are never sent to participants.

## Setup (tool-neutral)

- Title: “Before you start: the task and the ontology files”
- Lead: “You do not need to install or use any particular program. Please read the task, check that you can get the two files, and confirm that you understand how outside inspection works in this study.”
- 1 · The task: study instructions and owner setup notes, then the checkbox “I have read the task instructions.”
- 2 · The two ontology files: “These are the exact ontology versions and the information this study supplies. Another release, an online viewer or another source may show different or additional information, including mappings this study withholds. You may use any inspection method; if you consult a different source, you can say so after the case.”
  - Under the files: “How to open them: ontology editors (Protégé is one) and many ontology viewers open these files directly, and any text editor shows them as plain text. Opening them is optional; nothing on your computer is checked.”
  - “Can you get the two files?” Options: “Yes, I can download or open them if I want to” / “I have a problem getting them”. A problem shows retry, Pause and save-and-return, and contact help; it never forces a false confirmation or changes the condition.
- 3 · “Inspecting the ontologies outside this site is optional”: “You may inspect the supplied ontologies using any method or combination of methods. Protégé is one option; installing or using it is not required. You may also complete a case without external tools. Use the supplied ontology versions and follow the information guidance above. You can change methods between cases. After each ranking, we will ask which methods, if any, you used for that source and its candidates.” Checkbox: “I understand that outside inspection is optional, that I can combine and change methods between cases, and that I will be asked about my methods after each case.”
- 4 · Methods you are familiar with (optional): “This only describes your experience. It does not commit you to using anything, and leaving it empty is fine.”
- Continue requires the two acknowledgements and file access; draft saves never advance.

## Tutorial

Intro: “This tutorial uses an invented practice case in the same tools as the study. Nothing here is scored, and every name, fact, score and description in it is made up. Your progress is saved, so you can pause and come back on any device with your private link.”

### Lesson 1: The source and its candidates

1. This practice case uses two small invented ontologies. The source concept comes from one; the five candidates come from the other.
2. Candidates are listed in the system’s initial order, with a matching score. The initial position is a suggestion, not an answer: your rank is whatever you decide.
3. Inspect a different candidate, then go back to one you inspected before. Inspecting a candidate never ranks it.

Required actions (shown as a checklist): “Inspect a candidate other than the first” (`inspect_other_candidate`); “Return to a candidate you inspected before” (`return_to_candidate`).

### Lesson 2: Context: parents, children and missing information

1. Open Hierarchy. The source and the candidate each have their own browser.
2. Search the source ontology for “crate” and choose a result. Then open one of its parents and one of its children.
3. The source concept has two parents (multiple inheritance). Some candidates have no definition; the page says so instead of leaving a gap, and missing information is not a contradiction.
4. Use “Return to …” to come back to the compared entity.

Required actions (shown as a checklist): “Find an entity with search” (`search_entity`); “Open a parent” (`navigate_parent`); “Open a child” (`navigate_child`); “Return to the compared entity” (`return_to_compared`).

### Lesson 3: Where each statement comes from

1. Statements marked “Original” come from the ontology. Generated text sits in a dashed “Generated” frame and cites the records it rests on.
2. Open a citation under a generated description to see the exact original record.
3. In a card, use “Show original axiom” to see a restriction in its original form.
4. Open Evidence and select a feature (“Show in graph”), or use “Show where it appears” from a citation, to find the same record in the list.

Required actions (shown as a checklist): “Open a citation” (`open_citation`); “Show an original axiom” (`open_original_axiom`); “Find a record in the evidence list” (`locate_in_evidence_list`).

### Lesson 4: The optional evidence graph (optional)

1. The evidence graph shows the same items as the evidence list. The list is a complete alternative if you prefer it.
2. Select a line in the graph to read what it means, or select an item in the list. Try zoom or “Fit all”.
3. With the graph or list open, add or move a candidate in your practice answer: the view stays where it was.

Required actions (shown as a checklist): “Inspect a line in the graph or an item in the list” (`inspect_graph_or_list`); “Zoom or fit the graph” (`change_graph_view`, or `inspect_graph_or_list`); “Change your practice answer with the graph or list open” (`rank_with_details_open`).

### Lesson 5: Your answer

1. Add the candidates you consider equivalent, best first. Move one, remove one and use Undo.
2. Try “Keep initial order”, then make a partial ranking of one to four candidates and check it.
3. Try “None of these” and “Insufficient information”: they are different answers. An empty answer is never submitted.
4. This source keeps its qualifiers: a closed lid, only round pieces, at least two pieces. A broader or narrower candidate is not the same meaning.

Required actions (shown as a checklist): “Add a candidate” (`add_rank`); “Move a candidate” (`move_rank`); “Remove a candidate” (`remove_rank`); “Undo a change” (`undo_rank`); “Use “Keep initial order”” (`keep_initial_order`); “Check a partial ranking” (`check_partial_ranking`); “Choose “None of these”” (`choose_none`); “Choose “Insufficient information”” (`choose_insufficient`).

### Lesson 6: The block without explanations, and your methods

1. In one block of cases the study shows only names, IRIs and scores. You can inspect the ontologies in any way you like, or not at all.
2. Copy an IRI, and open “Ontology files” to find the two downloads. Downloading is optional.
3. After each case you will say which outside methods, if any, you used for it. Practise one report with two methods and one report with No.

Required actions (shown as a checklist): “Copy an IRI” (`copy_iri`); “Find the two downloads” (`locate_downloads`); “Practise a report with two or more methods” (`report_multiple_methods`); “Practise a report with No” (`report_no_methods`).

Practice reports in lesson 6: “Suppose that for a case you looked the source up in Protégé and also searched the ontology files in a text editor.” and “Suppose that for another case you used only the study pages and nothing outside them.”

## Five short questions

Introduction: “These five questions check that the task is clear. They are not a test of medical or ontology knowledge, and you can try each one again as often as you like.”

### Q1 `score_meaning`: Matching scores

Prompt: “A candidate is first in the initial order and has the highest matching score. What does this tell you?”

Options (in order): `advice` “The matcher suggests it; I should inspect the evidence.”; `certain` “It must be equivalent.”; `probability` “The score is necessarily a probability of being correct.”.

Pass predicate (server-side): `{"choice":"advice"}`. Lesson linked for revisiting: `identity`.

- Correct feedback: “Right. The initial position and the matching score are recorded advice from the matcher. Neither proves equivalence, and the score is not a calibrated probability.”
- Corrective feedback: “Not quite. A first-place, high-scoring candidate can still be wrong, and the score is not a probability of being correct. Treat it as a suggestion and inspect the evidence.”

### Q2 `equivalence_scope`: Same meaning?

Prompt: “In a practice ontology, the source means red circles. The candidate means circles of any colour and explicitly includes blue circles. Are these the same concept?”

Options (in order): `same` “Yes.”; `broader` “No, the candidate is broader.”; `unrelated` “No, they are unrelated.”.

Pass predicate (server-side): `{"choice":"broader"}`. Lesson linked for revisiting: `answers`.

- Correct feedback: “Right. Circles of any colour include red circles but also blue ones, so the candidate is broader. Overlap or a broader category is not the same meaning.”
- Corrective feedback: “Not quite. Compare the two definitions: the candidate also covers blue circles, so it is broader than the source. A broader or overlapping concept is related, but it is not equivalent.”

### Q3 `response_states`: Three kinds of answer

Prompt: “Match each situation to the answer it describes.”

Rows: `judged_none` “You judged that none of the displayed candidates is equivalent.”; `cannot_judge` “You cannot make a justified judgment.”; `not_answered` “You have not chosen or submitted an answer.”. Choices per row: `none_of_these` “None of these”; `insufficient_evidence` “Insufficient information”; `unanswered` “An empty, unsubmitted draft”.

Pass predicate (server-side): `{"matches":{"judged_none":"none_of_these","cannot_judge":"insufficient_evidence","not_answered":"unanswered"}}`. Lesson linked for revisiting: `answers`.

- Correct feedback: “Right. None of these is an explicit judgment, Insufficient information says you cannot judge, and an empty draft is not a submitted answer.”
- Corrective feedback: “Not quite. None of these is a judgment that no candidate is equivalent. Insufficient information means you cannot make a justified judgment. An empty draft has not been answered or submitted at all.”

### Q4 `evidence_origin`: Where information comes from

Prompt: “Match each kind of information to its origin.”

Rows: `statement` “A statement in the ontology”; `feature` “A feature selected or projected by Exact”; `summary` “A prepared generated summary”. Choices per row: `original` “Original ontology fact”; `matcher` “Matcher evidence”; `generated` “Generated text”.
Part B: “The candidate has no definition in the loaded scope. Does that alone establish a contradiction with the source?” (yes / no).

Pass predicate (server-side): `{"matches":{"statement":"original","feature":"matcher","summary":"generated"},"part_b":"no"}`. Lesson linked for revisiting: `evidence`.

- Correct feedback: “Right. Each kind of information keeps its origin, and information missing from the loaded scope does not prove incompatibility.”
- Corrective feedback: “Not quite. Statements in the ontology are original facts, features Exact selected are matcher evidence, and prepared summaries are generated text that cites original facts. A missing, filtered or unprepared definition is not a contradiction.”

### Q5 `external_methods`: Outside inspection

Prompt: “Which statements are true? Select all that apply.”

Options (in order): `optional` “Inspecting the ontologies outside the study pages is optional.”; `combine_change` “I may combine methods and change them between cases.”; `per_case` “After each source-and-candidates case, I report the methods I actually used.”; `protege_required` “I must use Protégé.”; `locked` “I must keep one method for the whole study.”.

Pass predicate (server-side): `{"choices":["optional","combine_change","per_case"]}`. Lesson linked for revisiting: `baseline`.

- Correct feedback: “Right. Outside inspection is optional, you can combine and change methods, and you report what you actually used after each case.”
- Corrective feedback: “Not quite. No tool is required, and you can change methods from case to case. After each case you report the methods you actually used for that source and its candidates, not for each candidate separately.”

## After each case

- Title: “About this case: inspection outside the study pages”. Lead: “Your ranking for “‹source›” is saved and can no longer change. This question is not timed as part of the case.” Note: “This case is the source and its five candidates together. Using the study's own panels does not count here; it is recorded separately.”
- Question: “For the case you just completed, did you inspect ontology information outside this study interface?” Yes / No.
- If Yes: “Select all methods you actually used. If you used several methods for different candidates, include all of them.” Protégé; Another ontology editor; The ontology files directly (for example, in a text editor); Another ontology resource or viewer; Queries or scripts; A reasoner; Another method. Optional short names for the three “other” methods (no web address, person or institution).
- Optional: “Which ontology information did you use outside the study pages?” Only the files supplied by this study / A different version or additional sources as well / Not sure. An unanswered question stays unknown.
- Reuse, offered only while unanswered: “If you used the same methods as in your previous case, you can copy that answer and then check it.” After copying: “Copied from your previous case: … Check that it describes this case before you save; change anything that differs.”
- Footer: “You can use different methods, or none, in every case. Your answer here does not change your ranking.”

## During cases

- Baseline guidance: “This block shows no explanations. Inspect the source and candidates in any way you choose: an ontology editor such as Protégé, a viewer, the files directly, queries, or none of these. Copy an IRI to find an entity in the files or in your tool.” and “Time you spend inspecting before you submit counts as part of this case. Switching windows does not pause anything.”
- Tutorial help (v2): “These are the practice lessons you completed, with their invented examples. Your answer to this case is unchanged, and the case continues while this is open.”
- Gap question after a closed page: “Working on this case outside the study pages (for example, in an ontology tool or the files)” / “Taking a break” / “Not sure”.
