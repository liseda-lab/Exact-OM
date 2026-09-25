"use client";

// Renders a frozen, versioned questionnaire exactly as published: wording, codes, branching
// and required status come from the server. Special choices (prefer not to say, never used,
// cannot judge…) are exclusive; hidden branches are cleared when their parent changes.

import { useId } from "react";

import type { Question } from "@/study/types";

export const SPECIAL = new Set(["prefer_not_to_say", "never_used", "all_equally", "none_helpful", "cannot_judge"]);

export type Answers = Record<string, unknown>;

export function isVisible(question: Question, answers: Answers): boolean {
  if (!question.show_if) return true;
  return Object.entries(question.show_if).every(([key, expected]) => {
    const actual = answers[key];
    if (expected && typeof expected === "object" && "contains" in (expected as Record<string, unknown>)) {
      return Array.isArray(actual) && actual.includes((expected as { contains: unknown }).contains);
    }
    return actual === expected;
  });
}

/** Drop answers to questions that are no longer shown, repeatedly for nested branches. */
export function pruneHidden(questions: Question[], answers: Answers): Answers {
  let current = { ...answers };
  for (let pass = 0; pass < 4; pass += 1) {
    const next = { ...current };
    for (const question of questions) if (!isVisible(question, next)) delete next[question.id];
    if (Object.keys(next).length === Object.keys(current).length) return next;
    current = next;
  }
  return current;
}

export function isAnswered(question: Question, value: unknown): boolean {
  if (value === undefined || value === null || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (question.matrix && typeof value === "object") return Object.keys(question.matrix).every((row) => (value as Record<string, unknown>)[row]);
  return true;
}

export function missingRequired(questions: Question[], answers: Answers): Question[] {
  return questions.filter((question) => question.required && isVisible(question, answers) && !isAnswered(question, answers[question.id]));
}

function Choice({
  question,
  value,
  onChange,
  disabled,
  name,
}: {
  question: Question;
  value: unknown;
  onChange: (next: unknown) => void;
  disabled: boolean;
  name: string;
}) {
  const options = Object.entries(question.options ?? {});
  if (question.multiple) {
    const selected = Array.isArray(value) ? (value as string[]) : [];
    const toggle = (code: string) => {
      if (selected.includes(code)) return onChange(selected.filter((item) => item !== code));
      if (SPECIAL.has(code)) return onChange([code]);
      return onChange([...selected.filter((item) => !SPECIAL.has(item)), code]);
    };
    return (
      <div className="option-grid">
        {options.map(([code, label]) => (
          <label key={code} className={selected.includes(code) ? "option on" : "option"}>
            <input type="checkbox" name={name} value={code} checked={selected.includes(code)} disabled={disabled} onChange={() => toggle(code)} />
            <span>{label}</span>
          </label>
        ))}
      </div>
    );
  }
  return (
    <div className="option-grid">
      {options.map(([code, label]) => (
        <label key={code} className={value === code ? "option on" : "option"}>
          <input type="radio" name={name} value={code} checked={value === code} disabled={disabled} onChange={() => onChange(code)} />
          <span>{label}</span>
        </label>
      ))}
    </div>
  );
}

function Matrix({ question, value, onChange, disabled, name }: { question: Question; value: unknown; onChange: (next: unknown) => void; disabled: boolean; name: string }) {
  const rows = Object.entries(question.matrix ?? {});
  const columns = Object.entries(question.options ?? {});
  const current = (value && typeof value === "object" ? value : {}) as Record<string, string>;
  return (
    <div className="matrix-wrap">
      <table className="matrix">
        <thead>
          <tr>
            <th scope="col">
              <span className="sr-only">Item</span>
            </th>
            {columns.map(([code, label]) => (
              <th key={code} scope="col">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([row, rowLabel]) => (
            <tr key={row}>
              <th scope="row">{rowLabel}</th>
              {columns.map(([code, label]) => (
                <td key={code}>
                  <input
                    type="radio"
                    name={`${name}-${row}`}
                    aria-label={`${rowLabel}: ${label}`}
                    checked={current[row] === code}
                    disabled={disabled}
                    onChange={() => onChange({ ...current, [row]: code })}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="matrix-cards" aria-hidden="false">
        {rows.map(([row, rowLabel]) => (
          <fieldset key={row} className="matrix-card">
            <legend>{rowLabel}</legend>
            <div className="option-grid">
              {columns.map(([code, label]) => (
                <label key={code} className={current[row] === code ? "option on" : "option"}>
                  <input type="radio" name={`${name}-card-${row}`} checked={current[row] === code} disabled={disabled} onChange={() => onChange({ ...current, [row]: code })} />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </fieldset>
        ))}
      </div>
    </div>
  );
}

export function QuestionnaireForm({
  questions,
  answers,
  onChange,
  errors,
  disabled = false,
  revealNote,
}: {
  questions: Question[];
  answers: Answers;
  onChange: (next: Answers, changed: string) => void;
  errors: Record<string, string>;
  disabled?: boolean;
  revealNote?: (question: Question) => string | null;
}) {
  const base = useId();
  return (
    <div className="questionnaire">
      {questions.map((question) => {
        if (!isVisible(question, answers)) return null;
        const name = `${base}-${question.id}`;
        const value = answers[question.id];
        const error = errors[question.id];
        const update = (next: unknown) => onChange({ ...answers, [question.id]: next }, question.id);
        const hint = question.required ? null : "Optional";
        const note = revealNote?.(question);
        if (!question.options) {
          return (
            <div key={question.id} className={error ? "question invalid" : "question"} id={`q-${question.id}`}>
              <label htmlFor={name} className="question-label">
                {question.label} {hint && <span className="meta">{hint}</span>}
              </label>
              <textarea id={name} className="textarea" rows={3} maxLength={2000} disabled={disabled} value={typeof value === "string" ? value : ""} onChange={(event) => update(event.target.value)} />
              {error && <p className="field-error">{error}</p>}
            </div>
          );
        }
        return (
          <fieldset key={question.id} className={error ? "question invalid" : "question"} id={`q-${question.id}`} aria-describedby={error ? `${name}-error` : undefined}>
            <legend className="question-label">
              {question.label} {question.multiple && <span className="meta">Select all that apply</span>} {hint && <span className="meta">{hint}</span>}
            </legend>
            {note && <p className="reveal-note">{note}</p>}
            {question.matrix ? <Matrix question={question} value={value} onChange={update} disabled={disabled} name={name} /> : <Choice question={question} value={value} onChange={update} disabled={disabled} name={name} />}
            {error && (
              <p className="field-error" id={`${name}-error`}>
                {error}
              </p>
            )}
          </fieldset>
        );
      })}
    </div>
  );
}
