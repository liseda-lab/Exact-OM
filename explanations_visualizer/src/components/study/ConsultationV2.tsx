"use client";

// exact-study/2.0 per-case report (14 decisions 3–6, 16 B4). A case is one source and its
// five candidates. After the ranking is committed the participant reports actual outside
// inspection for that case: Yes with any combination of methods, or No. Drafts autosave
// through the outbox and survive reload without advancing or touching the ranking; the final
// save commits and advances once. A previous case's report is offered only on request and
// is never preselected or submitted for the participant.

import { useEffect, useRef, useState } from "react";

import { METHOD_LABELS } from "@/components/study/SetupV2";
import { consultationProblem, EMPTY_CONSULTATION, normalizeConsultation, type ConsultationAnswer } from "@/study/consultation";
import { orderedEntries, orderedKeys } from "@/study/formOrder";
import type { StudySession } from "@/study/session";
import type { ConsultationMethod, ResourceScope, StudyState } from "@/study/types";

const SCOPES: [ResourceScope, string][] = [
  ["supplied_only", "Only the files supplied by this study"],
  ["different_or_additional", "A different version or additional sources as well"],
  ["unsure", "Not sure"],
];

const NAME_FIELDS: { method: ConsultationMethod; field: "other_editor" | "other_resource" | "other_method"; label: string }[] = [
  { method: "other_editor", field: "other_editor", label: "Which other ontology editor?" },
  { method: "other_resource", field: "other_resource", label: "Which other resource or viewer?" },
  { method: "other_method", field: "other_method", label: "Which other method?" },
];

type Answer = ConsultationAnswer;
const EMPTY = EMPTY_CONSULTATION;

function fromDraft(state: StudyState): Answer {
  const draft = state.consultation_draft;
  if (!draft || draft.presentation_id !== state.current_presentation_id) return EMPTY;
  return {
    consulted_external_ontologies: draft.consulted_external_ontologies,
    methods: draft.methods,
    other_editor: draft.other_editor,
    other_resource: draft.other_resource,
    other_method: draft.other_method,
    resource_scope: draft.resource_scope,
  };
}

function summary(answer: Answer, labels: Map<string, string>): string {
  if (answer.consulted_external_ontologies === false) return "No outside inspection";
  if (answer.consulted_external_ontologies === null) return "Unanswered";
  return `Yes · ${answer.methods.map((method) => labels.get(method) ?? method).join(", ") || "no method selected"}`;
}

export function ConsultationStageV2({ state, session, sourceLabel }: { state: StudyState; session: StudySession; sourceLabel: string | null }) {
  const form = state.forms.consultation;
  const question = (id: string) => form.find((item) => item.id === id);
  const methodsQuestion = question("methods");
  const methodOptions: [string, string][] = methodsQuestion?.options ? orderedEntries(methodsQuestion.options, orderedKeys(methodsQuestion, "options", state.forms.version)) : METHOD_LABELS;
  const labels = new Map(methodOptions);
  const [answer, setAnswer] = useState<Answer>(() => fromDraft(state));
  const [copied, setCopied] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const edited = useRef(false);
  useEffect(() => {
    if (!edited.current) setAnswer(fromDraft(state));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.consultation_draft]);
  const handledConflict = useRef(0);
  useEffect(() => {
    if (session.save.kind !== "conflict" || session.save.at === handledConflict.current) return;
    handledConflict.current = session.save.at;
    edited.current = false;
    setAnswer(fromDraft(state));
  }, [session.save, state]);

  const caseId = state.current_case_id;
  const payload = (value: Answer) => ({ presentation_id: state.current_presentation_id, form_version: state.forms.version, ...normalizeConsultation(value) });
  const update = (changes: Partial<Answer>) => {
    if (!caseId) return;
    edited.current = true;
    const next = { ...answer, ...changes };
    setAnswer(next);
    setError(null);
    void session
      .mutate({ method: "PUT", path: `/api/v1/study/cases/${encodeURIComponent(caseId)}/consultation/draft`, body: payload(next), coalesce: `consultation:${caseId}`, debounceMs: 600 })
      .catch(() => undefined);
  };
  const save = async () => {
    const problem = consultationProblem(answer);
    if (problem || !caseId) {
      setError(problem);
      return;
    }
    setBusy(true);
    try {
      await session.mutate({ method: "PUT", path: `/api/v1/study/cases/${encodeURIComponent(caseId)}/consultation`, body: payload(answer), transition: true });
    } catch {
      /* the header explains; the answer stays for another try */
    } finally {
      setBusy(false);
    }
  };
  const previous = state.previous_consultation ?? null;
  const next = state.completed_cases + 2 <= state.assigned_case_count ? `Save and go to case ${state.completed_cases + 2}` : "Save and continue";
  const restored = state.consultation_draft?.presentation_id === state.current_presentation_id && state.consultation_draft.consulted_external_ontologies !== null;

  return (
    <div className="study-page">
      <div className="study-intro">
        <h1>About this case: inspection outside the study pages</h1>
        <p className="lead">Your ranking{sourceLabel ? ` for “${sourceLabel}”` : ""} is saved and can no longer change. This question is not timed as part of the case.</p>
        <p className="muted">This case is the source and its five candidates together. Using the study&apos;s own panels does not count here; it is recorded separately.</p>
        {restored && <p className="meta">Your unfinished answer was restored from the study server.</p>}
      </div>

      {previous && answer.consulted_external_ontologies === null && (
        <div className="note note-info">
          <span>If you used the same methods as in your previous case, you can copy that answer and then check it.</span>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => {
              const value: Answer = {
                consulted_external_ontologies: previous.consulted_external_ontologies,
                methods: previous.methods,
                other_editor: previous.other_editor,
                other_resource: previous.other_resource,
                other_method: previous.other_method,
                resource_scope: previous.resource_scope,
              };
              update(value);
              setCopied(summary(value, labels));
            }}
          >
            Copy my answer from the previous case
          </button>
        </div>
      )}
      {copied && (
        <p className="note" role="status">
          Copied from your previous case: {copied}. Check that it describes this case before you save; change anything that differs.
        </p>
      )}

      <fieldset className="question">
        <legend className="question-label">{question("consulted_external_ontologies")?.label ?? "For the case you just completed, did you inspect ontology information outside this study interface?"}</legend>
        <div className="option-grid">
          {(
            [
              [true, "Yes"],
              [false, "No"],
            ] as [boolean, string][]
          ).map(([value, label]) => (
            <label key={String(value)} className={answer.consulted_external_ontologies === value ? "option on" : "option"}>
              <input type="radio" name="consulted" checked={answer.consulted_external_ontologies === value} onChange={() => update(value ? { consulted_external_ontologies: true } : { ...EMPTY, consulted_external_ontologies: false })} />
              <span>{label}</span>
            </label>
          ))}
        </div>
      </fieldset>

      {answer.consulted_external_ontologies === true && (
        <>
          <fieldset className="question">
            <legend className="question-label">
              {methodsQuestion?.label ?? "Select all methods you actually used. If you used several methods for different candidates, include all of them."} <span className="meta">Select all that apply</span>
            </legend>
            <div className="option-grid option-grid-column">
              {methodOptions.map(([code, label]) => (
                <label key={code} className={answer.methods.includes(code as ConsultationMethod) ? "option on" : "option"}>
                  <input
                    type="checkbox"
                    checked={answer.methods.includes(code as ConsultationMethod)}
                    onChange={() =>
                      update({ methods: answer.methods.includes(code as ConsultationMethod) ? answer.methods.filter((item) => item !== code) : [...answer.methods, code as ConsultationMethod] })
                    }
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
            {NAME_FIELDS.filter((item) => answer.methods.includes(item.method)).map((item) => (
              <div key={item.field} className="field">
                <label htmlFor={`name-${item.field}`}>
                  {question(item.field)?.label ?? item.label} <span className="meta">Optional · a short name only, no web address, person or institution</span>
                </label>
                <input id={`name-${item.field}`} className="input" maxLength={160} value={answer[item.field] ?? ""} onChange={(event) => update({ [item.field]: event.target.value })} />
              </div>
            ))}
          </fieldset>
          <fieldset className="question">
            <legend className="question-label">
              {question("resource_scope")?.label ?? "Which ontology information did you use outside the study pages?"} <span className="meta">Optional</span>
            </legend>
            <div className="option-grid option-grid-column">
              {SCOPES.map(([value, label]) => (
                <label key={value} className={answer.resource_scope === value ? "option on" : "option"}>
                  <input type="radio" name="resource-scope" checked={answer.resource_scope === value} onChange={() => update({ resource_scope: value })} />
                  <span>{label}</span>
                </label>
              ))}
            </div>
            {answer.resource_scope && (
              <button type="button" className="btn btn-quiet btn-sm" onClick={() => update({ resource_scope: null })}>
                Clear this optional answer
              </button>
            )}
          </fieldset>
        </>
      )}

      <div className="study-actions">
        <button type="button" className="btn btn-primary btn-large" disabled={busy} onClick={save}>
          {busy ? "Saving…" : next}
        </button>
        {error && (
          <span className="field-error" role="alert">
            {error}
          </span>
        )}
      </div>
      <p className="meta">You can use different methods, or none, in every case. Your answer here does not change your ranking.</p>
    </div>
  );
}
