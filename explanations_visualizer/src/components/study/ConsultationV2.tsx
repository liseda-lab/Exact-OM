"use client";

// exact-study/2.0 per-case report (14 decisions 3–6, 16 B4; consultation/3 in 22 B06). A case
// is one source and its five candidates. After the ranking is committed the participant reports
// actual outside inspection for that case: Yes with any combination of methods, or No. Drafts
// autosave through the outbox and survive reload without advancing or touching the ranking;
// the final save commits and advances once. Nothing is prefilled: setup familiarity never
// populates actual use, and a previous case's report is copied only on an explicit request.

import { useEffect, useRef, useState } from "react";

import { METHOD_LABELS } from "@/components/study/SetupV2";
import { consultationProblem, EMPTY_CONSULTATION, normalizeConsultation, type ConsultationAnswer } from "@/study/consultation";
import { orderedEntries, orderedKeys } from "@/study/formOrder";
import type { StudySession } from "@/study/session";
import { isReviewProtocol, type ConsultationMethod, type ConsultationReceiptV2, type ResourceScope, type StudyState } from "@/study/types";

/** consultation/2 wording, kept for historical publications. */
const SCOPES_V2: [ResourceScope, string][] = [
  ["supplied_only", "Only the files supplied by this study"],
  ["different_or_additional", "A different version or additional sources as well"],
  ["unsure", "Not sure"],
];

const NAME_FIELDS_V2: { method: ConsultationMethod; field: "other_editor" | "other_resource" | "other_method"; label: string }[] = [
  { method: "other_editor", field: "other_editor", label: "Which other ontology editor?" },
  { method: "other_resource", field: "other_resource", label: "Which other resource or viewer?" },
  { method: "other_method", field: "other_method", label: "Which other method?" },
];

type Answer = ConsultationAnswer;
const EMPTY = EMPTY_CONSULTATION;

function answerOf(value: Partial<ConsultationReceiptV2> | null | undefined): Answer {
  if (!value) return EMPTY;
  return {
    consulted_external_ontologies: value.consulted_external_ontologies ?? null,
    methods: value.methods ?? [],
    other_editor: value.other_editor ?? null,
    other_resource: value.other_resource ?? null,
    other_method: value.other_method ?? null,
    resource_scope: value.resource_scope ?? null,
  };
}

function fromDraft(state: StudyState): Answer {
  const draft = state.consultation_draft;
  return draft && draft.presentation_id === state.current_presentation_id ? answerOf(draft) : EMPTY;
}

function summary(answer: Answer, labels: Map<string, string>): string {
  if (answer.consulted_external_ontologies === false) return "No outside inspection";
  if (answer.consulted_external_ontologies === null) return "Unanswered";
  const named = answer.methods.map((method) => (method === "other_method" && answer.other_method ? `${labels.get(method) ?? method} (${answer.other_method})` : labels.get(method) ?? method));
  return `Yes · ${named.join(", ") || "no method selected"}`;
}

export function ConsultationStageV2({ state, session, sourceLabel }: { state: StudyState; session: StudySession; sourceLabel: string | null }) {
  const v3 = isReviewProtocol(state);
  const form = state.forms.consultation;
  const question = (id: string) => form.find((item) => item.id === id);
  const methodsQuestion = question("methods");
  const methodOptions: [string, string][] = methodsQuestion?.options ? orderedEntries(methodsQuestion.options, orderedKeys(methodsQuestion, "options", state.forms.version)) : METHOD_LABELS;
  const scopeQuestion = question("resource_scope");
  const scopes: [ResourceScope, string][] = v3 && scopeQuestion?.options ? (orderedEntries(scopeQuestion.options, orderedKeys(scopeQuestion, "options", state.forms.version)) as [ResourceScope, string][]) : SCOPES_V2;
  const nameFields = v3 ? [{ method: "other_method" as ConsultationMethod, field: "other_method" as const, label: question("other_method")?.label ?? "Which other methods did you use?" }] : NAME_FIELDS_V2;
  const labels = new Map(methodOptions);
  const [answer, setAnswer] = useState<Answer>(() => fromDraft(state));
  const [copied, setCopied] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<ReturnType<typeof consultationProblem>>(null);
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
  const payload = (value: Answer) => ({
    presentation_id: state.current_presentation_id,
    form_version: state.forms.version,
    ...(v3 ? { consultation_version: state.protocol_versions?.consultation ?? "consultation/3" } : {}),
    ...normalizeConsultation(value, v3),
  });
  const update = (changes: Partial<Answer>) => {
    if (!caseId) return;
    edited.current = true;
    const next = { ...answer, ...changes };
    setAnswer(next);
    setProblem(null);
    void session
      .mutate({ method: "PUT", path: `/api/v1/study/cases/${encodeURIComponent(caseId)}/consultation/draft`, body: payload(next), coalesce: `consultation:${caseId}`, debounceMs: 600 })
      .catch(() => undefined);
  };
  const save = async () => {
    const found = consultationProblem(answer, v3);
    if (found || !caseId) {
      setProblem(found);
      if (found) document.getElementById(`consultation-${found.field}`)?.focus();
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
  const errorFor = (field: NonNullable<typeof problem>["field"]) =>
    problem?.field === field ? (
      <p className="field-error" id={`consultation-${field}-error`} role="alert">
        {problem.message}
      </p>
    ) : null;

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
              const value = answerOf(previous);
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

      <fieldset className={problem?.field === "consulted" ? "question invalid" : "question"} id="consultation-consulted" tabIndex={-1} aria-describedby={problem?.field === "consulted" ? "consultation-consulted-error" : undefined}>
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
        {errorFor("consulted")}
      </fieldset>

      {answer.consulted_external_ontologies === true && (
        <>
          <fieldset className={problem?.field === "methods" ? "question invalid" : "question"} id="consultation-methods" tabIndex={-1} aria-describedby={problem?.field === "methods" ? "consultation-methods-error" : undefined}>
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
            {errorFor("methods")}
            {nameFields
              .filter((item) => answer.methods.includes(item.method))
              .map((item) => (
                <div key={item.field} className={v3 && problem?.field === "other_method" ? "field invalid" : "field"}>
                  <label htmlFor={v3 ? "consultation-other_method" : `name-${item.field}`}>
                    {v3 ? item.label : question(item.field)?.label ?? item.label}{" "}
                    <span className="meta">{v3 ? "Short names, separated by commas" : "Optional · a short name only, no web address, person or institution"}</span>
                  </label>
                  <input
                    id={v3 ? "consultation-other_method" : `name-${item.field}`}
                    className="input"
                    maxLength={160}
                    value={answer[item.field] ?? ""}
                    aria-invalid={v3 && problem?.field === "other_method" ? true : undefined}
                    aria-describedby={v3 && problem?.field === "other_method" ? "consultation-other_method-error" : undefined}
                    onChange={(event) => update({ [item.field]: event.target.value })}
                  />
                  {v3 && errorFor("other_method")}
                </div>
              ))}
          </fieldset>
          <fieldset className="question">
            <legend className="question-label">
              {scopeQuestion?.label ?? "Which ontology information did you use outside the study pages?"} {!/optional/i.test(scopeQuestion?.label ?? "") && <span className="meta">Optional</span>}
            </legend>
            <div className="option-grid option-grid-column">
              {scopes.map(([value, label]) => (
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
      </div>
      <p className="meta">You can use different methods, or none, in every case. Your answer here does not change your ranking.</p>
      {v3 && state.forms.collection_notice && <p className="meta">{state.forms.collection_notice}</p>}
    </div>
  );
}
