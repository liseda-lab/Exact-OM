// Helpers for browser tests on a real review publication (the Luna prose revision): resolve
// frozen cases to main-app pairs and to the served artifacts, and reach a case in the
// explanation condition through test sessions. Cases are chosen from current presentation
// metadata, not from assumed positions. Seeding is used only to reach a case for a regression
// check; the acceptance journey (study-review-journey.spec.ts) seeds nothing.

import { readFileSync } from "node:fs";

import { expect, type Page } from "@playwright/test";

import { harness, ParticipantApi, seedAdvance, seedToCase } from "./v2Harness";

export const reviewMain = process.env.EXACT_E2E_REVIEW_MAIN_URL ?? null;

export interface EntityRef {
  ontology_version_id: string;
  iri: string;
  kind: string;
}

export interface CaseDef {
  case_id: string;
  source: EntityRef;
  candidates: { candidate_id: string; display_position: number; entity: EntityRef }[];
}

export interface Artifact {
  explanation_id: string;
  entities: EntityRef[];
  claims: { claim_id?: string; text: string; fact_ids: string[]; category: string }[];
  support_status?: string;
  preparation_kind?: string;
  presentation?: { kind: string; covered_sides: string[]; claim_sides: Record<string, string[]> };
  support_packet?: { facts?: { fact_id: string; subject: EntityRef }[]; selection?: { referenced_labels?: { fact_id: string; subject: EntityRef }[] } };
}

export function reviewCases(): CaseDef[] {
  return (JSON.parse(readFileSync(harness!.publication, "utf8")).definition.cases as CaseDef[]).map((item) => ({ ...item }));
}

async function json<T>(page: Page, url: string): Promise<T> {
  const response = await page.request.get(url);
  expect(response.ok(), url).toBe(true);
  return (await response.json()) as T;
}

const params = (entity: EntityRef, prefix = "") => ({ [`${prefix}ontology_version_id`]: entity.ontology_version_id, [`${prefix}iri`]: entity.iri, [`${prefix}kind`]: entity.kind });

/** The served artifact for a profile or comparison (validated first), as the frontend selects it. */
export async function artifactFor(page: Page, task: "entity_profile" | "pair_comparison", subject: EntityRef, counterpart?: EntityRef): Promise<Artifact | null> {
  const query = new URLSearchParams({ task, limit: "100", ...params(subject), ...(counterpart ? params(counterpart, "counterpart_") : {}) });
  const page_ = await json<{ items: { explanation_id: string; grounding_status: string }[] }>(page, `${reviewMain}/api/v1/explanations?${query}`);
  if (!page_.items.length) return null;
  const chosen = page_.items.find((item) => item.grounding_status === "validated") ?? page_.items[0];
  return json<Artifact>(page, `${reviewMain}/api/v1/explanations/${encodeURIComponent(chosen.explanation_id)}`);
}

let runId: string | null = null;
/** The main app's saved pair for a frozen case candidate. */
export async function mainPair(page: Page, item: CaseDef, position: number) {
  runId ??= (await json<{ items: { run_id: string }[] }>(page, `${reviewMain}/api/v1/runs`)).items[0].run_id;
  const target = item.candidates.find((candidate) => candidate.display_position === position)!.entity;
  let cursor: string | null = null;
  for (let guard = 0; guard < 20; guard += 1) {
    const query = new URLSearchParams({ source: item.source.iri, source_kind: item.source.kind, limit: "100", ...(cursor ? { cursor } : {}) });
    const result: { items: { pair_id: string; target: EntityRef }[]; next_cursor: string | null } = await json(page, `${reviewMain}/api/v1/runs/${encodeURIComponent(runId)}/candidates?${query}`);
    const found = result.items.find((candidate) => candidate.target.iri === target.iri && candidate.target.kind === target.kind);
    if (found) return { run: runId, source: item.source.iri, pair: found.pair_id, target };
    cursor = result.next_cursor;
    if (!cursor) break;
  }
  throw new Error(`No saved pair for ${item.case_id} candidate ${position}`);
}

/** Open a case candidate's pair in the main app and wait for both descriptions to settle. */
export async function openMainPair(page: Page, item: CaseDef, position: number) {
  const pair = await mainPair(page, item, position);
  await page.goto(`${reviewMain}/?${new URLSearchParams({ run: pair.run, source: pair.source, pair: pair.pair })}`);
  await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible({ timeout: 60_000 });
  await settledDescriptions(page);
  return pair;
}

/** Descriptions are ready only when no loading placeholder remains (R5): never capture a skeleton. */
export async function settledDescriptions(page: Page) {
  await expect(page.locator(".comparison")).toBeVisible({ timeout: 60_000 });
  await expect(page.locator(".comparison [aria-busy='true'], .comparison .skeleton-row")).toHaveCount(0, { timeout: 60_000 });
  await expect(page.locator(".generated-loading")).toHaveCount(0, { timeout: 60_000 });
}

/**
 * A test session whose current case is `caseId` in the explanation condition: sessions rotate
 * through the counterbalanced schedules, so a session that shows it as baseline is set aside.
 */
export async function studyCaseInExplanation(page: Page, caseId: string): Promise<{ api: ParticipantApi; current: Record<string, any> }> {
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const api = await seedToCase(page);
    for (let index = 0; index < 30; index += 1) {
      const current = await api.currentCase();
      if (current.case_id === caseId) {
        if (current.condition === "explanation") {
          await page.reload();
          await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible({ timeout: 60_000 });
          return { api, current };
        }
        break;
      }
      await seedAdvance(api, current);
    }
  }
  throw new Error(`No session reached ${caseId} in the explanation condition`);
}

/** Inspect a candidate by its initial position in a study case. */
export async function inspectCandidate(page: Page, position: number) {
  const button = page.locator(`[data-tour="inspect"][data-position="${position}"]`);
  if ((await button.getAttribute("aria-pressed")) !== "true") await button.click();
  await expect(button).toHaveAttribute("aria-pressed", "true");
  await settledDescriptions(page);
}
