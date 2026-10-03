// Presentation order of questionnaire options and matrix rows. Never derived from object
// key order (canonical storage sorts keys) or from sorting labels. A v2 question declares
// `option_order`/`row_order`; a legacy v1 form uses its versioned declared-order table. A
// form with neither cannot be shown faithfully and fails visibly before collecting answers.
// Pure module so it runs under Node's test runner.

import type { Question } from "./types";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { LEGACY_FORM_ORDER } from "./legacyFormOrder.ts";

export class FormOrderError extends Error {}

function check(order: string[], keys: string[], what: string, question: string): string[] {
  const unique = new Set(order);
  if (unique.size !== order.length) throw new FormOrderError(`Duplicate ${what} in the declared order of ${question}`);
  const missing = keys.filter((key) => !unique.has(key));
  const unknown = order.filter((key) => !keys.includes(key));
  if (missing.length || unknown.length) throw new FormOrderError(`The declared ${what} order of ${question} does not match its ${what}s`);
  return order;
}

/** Ordered codes for a question's options or matrix rows. */
export function orderedKeys(question: Question, kind: "options" | "rows", formVersion: string): string[] {
  const source = kind === "options" ? question.options : question.matrix;
  if (!source) return [];
  const keys = Object.keys(source);
  const explicit = kind === "options" ? question.option_order : question.row_order;
  if (explicit) return check(explicit, keys, kind === "options" ? "option" : "row", question.id);
  const legacy = (LEGACY_FORM_ORDER as Record<string, Record<string, { options?: string[]; rows?: string[] }>>)[formVersion]?.[question.id]?.[kind];
  if (legacy) {
    const present = legacy.filter((key: string) => keys.includes(key));
    if (present.length !== keys.length) throw new FormOrderError(`${question.id} has ${kind} that its versioned presentation does not declare`);
    return present;
  }
  throw new FormOrderError(`${question.id} has no declared ${kind === "options" ? "option" : "row"} order`);
}

/** Every ordering problem in a form; an empty list means it can be shown as published. */
export function formOrderProblems(questions: Question[], formVersion: string): string[] {
  const problems: string[] = [];
  for (const question of questions) {
    for (const kind of ["options", "rows"] as const) {
      try {
        orderedKeys(question, kind, formVersion);
      } catch (error) {
        problems.push(error instanceof Error ? error.message : String(error));
      }
    }
  }
  return problems;
}

export function orderedEntries(record: Record<string, string> | null, order: string[]): [string, string][] {
  if (!record) return [];
  return order.map((key) => [key, record[key]]);
}
