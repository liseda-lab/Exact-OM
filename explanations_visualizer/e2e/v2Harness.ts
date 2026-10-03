// Shared helpers for browser tests against the real exact-study/2.0 service started by
// `python -m tools.serve_explanation_v2_e2e` (HTTPS, PostgreSQL, compiled frontend, normal
// CSP). The private config written by that launcher is read from EXACT_E2E_STUDY_V2_CONFIG.
// Seeding through the API is used only for fault scenarios; the acceptance journey in
// study-v2-backend.spec.ts performs every tutorial interaction in the browser.

import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";

import { expect, type APIResponse, type Page, type Request } from "@playwright/test";

export interface HarnessConfig {
  origin: string;
  researcher_token: string;
  publication: string;
  study_revision: string;
}

export const harness: HarnessConfig | null = process.env.EXACT_E2E_STUDY_V2_CONFIG ? JSON.parse(readFileSync(process.env.EXACT_E2E_STUDY_V2_CONFIG, "utf8")) : null;

/** The frozen synthetic publication; used only to seed fault scenarios, never sent to the page. */
function publication() {
  return JSON.parse(readFileSync(harness!.publication, "utf8")) as {
    definition: {
      information_version: string;
      tutorial: {
        version: string;
        lessons: { lesson_id: string; requirements: { requirement_id: string; action: string }[] }[];
        case: { source: unknown; candidates: { candidate_id: string }[]; ontology_resources: { asset_id: string }[] };
        grading: Record<string, { response: Record<string, unknown> }>;
      };
    };
  };
}

export async function newInvitation(page: Page): Promise<string> {
  const response = await page.request.post(`${harness!.origin}/api/v1/admin/studies/${encodeURIComponent(harness!.study_revision)}/invitations`, {
    headers: { Authorization: `Bearer ${harness!.researcher_token}` },
    data: { count: 1, test: true },
  });
  expect(response.ok()).toBe(true);
  return (await response.json()).invitations[0].invitation as string;
}

/** Same-origin participant API through the page's own cookie jar. */
export class ParticipantApi {
  constructor(private page: Page) {}

  async state(): Promise<Record<string, any>> {
    const response = await this.page.request.get(`${harness!.origin}/api/v1/study/state`);
    expect(response.ok()).toBe(true);
    return response.json();
  }

  async write(method: "PUT" | "POST", path: string, body: Record<string, unknown>): Promise<Record<string, any>> {
    const state = await this.state();
    const response: APIResponse = await this.page.request.fetch(`${harness!.origin}/api/v1/study${path}`, {
      method,
      headers: { Origin: harness!.origin, "Content-Type": "application/json", "X-Study-Session": state.session_id },
      data: { ...body, idempotency_key: randomUUID().replace(/-/g, ""), expected_revision: state.revision },
    });
    if (!response.ok()) throw new Error(`${method} ${path} → ${response.status()} ${await response.text()}`);
    return response.json();
  }

  async currentCase(): Promise<Record<string, any>> {
    const response = await this.page.request.get(`${harness!.origin}/api/v1/study/cases/current`);
    if (!response.ok()) throw new Error(`GET /cases/current → ${response.status()} ${await response.text()} (stage ${(await this.state()).stage})`);
    return response.json();
  }
}

/** Typed requirement evidence for every frozen lesson, built from the synthetic fixture. */
function tutorialActions() {
  const { tutorial } = publication().definition;
  const resource = JSON.parse(readFileSync(join(dirname(harness!.publication), "practice-explanations.json"), "utf8"));
  const candidate = tutorial.case.candidates[1].candidate_id;
  const initial = tutorial.case.candidates.map((item) => item.candidate_id);
  const actions: Record<string, unknown>[] = [];
  for (const lesson of tutorial.lessons)
    for (const requirement of lesson.requirements) {
      const action: Record<string, unknown> = { requirement_id: requirement.requirement_id, action: requirement.action };
      const kind = requirement.action;
      if (["inspect_other_candidate", "return_to_candidate"].includes(kind)) action.candidate_id = candidate;
      else if (["navigate_parent", "navigate_child"].includes(kind)) action.entity = resource.hierarchy[0][kind === "navigate_parent" ? "parent" : "child"];
      else if (["search_entity", "return_to_compared", "copy_iri"].includes(kind)) action.entity = tutorial.case.source;
      else if (["locate_in_evidence_list", "inspect_graph_or_list"].includes(kind)) action.fact_id = resource.evidence[0].fact_ids[0];
      else if (["open_citation", "open_original_axiom"].includes(kind)) action.fact_id = resource.facts[0].fact_id;
      else if (["choose_none", "choose_insufficient"].includes(kind)) Object.assign(action, { response_type: kind === "choose_none" ? "none_of_these" : "insufficient_evidence", ranked_candidate_ids: [] });
      else if (["add_rank", "move_rank", "remove_rank", "undo_rank", "keep_initial_order", "check_partial_ranking", "rank_with_details_open"].includes(kind))
        Object.assign(action, { response_type: "ranked_candidates", ranked_candidate_ids: kind === "keep_initial_order" ? initial : [candidate] });
      else if (["report_multiple_methods", "report_no_methods"].includes(kind)) action.methods = kind === "report_multiple_methods" ? ["queries_scripts", "reasoner"] : [];
      else if (kind === "locate_downloads") action.asset_ids = tutorial.case.ontology_resources.map((asset) => asset.asset_id);
      actions.push(action);
    }
  return actions;
}

/** Fault scenarios only: consent, setup and background through the API. */
export async function seedToTutorial(page: Page) {
  const invitation = await newInvitation(page);
  await page.goto(harness!.origin + invitation);
  await expect(page.getByRole("button", { name: "I agree to take part", exact: true })).toBeVisible();
  const api = new ParticipantApi(page);
  const { definition } = publication();
  await api.write("PUT", "/consent", { information_version: definition.information_version, accepted: true });
  await api.write("PUT", "/setup", { setup_version: "setup/2", instructions_acknowledged: true, external_inspection_optional_understood: true, resource_access: "available", familiar_methods: [], submitted: true });
  const state = await api.state();
  const answers: Record<string, unknown> = {};
  for (const question of state.forms.background) {
    if (!question.required || question.show_if) continue;
    const choices: string[] = question.option_order;
    const answer = choices.includes("prefer_not_to_say") ? "prefer_not_to_say" : choices[0];
    answers[question.id] = question.multiple ? [answer] : answer;
  }
  await api.write("PUT", "/questionnaires/background", { form_version: state.forms.version, answers, submitted: true });
  return api;
}

/** Fault scenarios only: API-completed training, so the scored case can be loaded directly. */
export async function seedToCase(page: Page) {
  const api = await seedToTutorial(page);
  const { tutorial } = publication().definition;
  const actions = tutorialActions();
  await api.write("PUT", "/tutorial/progress", { tutorial_version: tutorial.version, lesson_id: tutorial.lessons.at(-1)!.lesson_id, actions, completed_requirements: actions.map((action) => action.requirement_id) });
  for (const [question, rule] of Object.entries(tutorial.grading))
    await api.write("POST", "/tutorial/assessment", { tutorial_version: tutorial.version, question_id: question, attempt_id: randomUUID().replace(/-/g, ""), response: rule.response });
  await api.write("POST", "/tutorial/complete", { tutorial_version: tutorial.version });
  return api;
}

/** Submit the current seeded case and its report through the API. */
export async function seedAdvance(api: ParticipantApi, current: Record<string, any>) {
  await api.write("POST", `/cases/${encodeURIComponent(current.case_id)}/submit`, { presentation_id: current.presentation_id, response_type: "insufficient_evidence", ranked_candidate_ids: [] });
  const state = await api.state();
  await api.write("PUT", `/cases/${encodeURIComponent(current.case_id)}/consultation`, { presentation_id: current.presentation_id, form_version: state.forms.version, consulted_external_ontologies: false, methods: [], other_editor: null, other_resource: null, other_method: null, resource_scope: null });
}

/** Advance seeded cases through the API until the current case has the wanted condition. */
export async function seedCondition(page: Page, api: ParticipantApi, condition: "explanation" | "ontology_baseline") {
  for (let index = 0; index < 8; index += 1) {
    const current = await api.currentCase();
    if (current.condition === condition) return current;
    await seedAdvance(api, current);
  }
  throw new Error(`No ${condition} case was assigned`);
}

/** Records participant traffic so tests can assert on actual requests, not appearances. */
export function recordTraffic(page: Page) {
  const requests: { method: string; path: string; status: number | null; body: unknown }[] = [];
  const failures: string[] = [];
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error" && /Content Security Policy|Refused to/.test(message.text())) errors.push(message.text());
    if (message.text().startsWith("csp-violation ")) errors.push(message.text());
  });
  // Name the source of any CSP violation so it can be explained, not only counted.
  void page.addInitScript(() => {
    document.addEventListener("securitypolicyviolation", (event) => {
      console.warn(`csp-violation ${event.violatedDirective} ${event.sourceFile}:${event.lineNumber}:${event.columnNumber} ${event.sample} @ ${location.pathname} ${document.querySelector("h1")?.textContent ?? ""}`);
    });
  });
  page.on("requestfailed", (request: Request) => failures.push(`${request.method()} ${new URL(request.url()).pathname} ${request.failure()?.errorText ?? ""}`));
  page.on("requestfinished", async (request: Request) => {
    const url = new URL(request.url());
    if (!url.pathname.startsWith("/api/")) return;
    const response = await request.response();
    let body: unknown = null;
    try {
      body = request.postDataJSON();
    } catch {
      body = null;
    }
    requests.push({ method: request.method(), path: url.pathname + url.search, status: response?.status() ?? null, body });
  });
  return { requests, failures, errors };
}

export const scopedPath = (scope: string) => `/api/v1/study/workspace/${encodeURIComponent(scope)}`;
