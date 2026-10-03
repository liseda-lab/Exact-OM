import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

type Config = { origin: string; researcher_token: string; study_revision: string };
const config: Config | null = process.env.EXACT_E2E_STUDY_CONFIG
  ? JSON.parse(readFileSync(process.env.EXACT_E2E_STUDY_CONFIG, "utf8")) : null;
test.use({ screenshot: "off", trace: "off" });
test.beforeEach(() => test.skip(!config, "Requires the private synthetic PostgreSQL/HTTPS harness"));

async function current(page: Page) {
  const response = await page.request.get(`${config!.origin}/api/v1/study/state`);
  expect(response.ok()).toBe(true);
  return response.json();
}
async function mutate(page: Page, path: string, body: Record<string, unknown>, method = "PUT") {
  const before = await current(page);
  const response = await page.request.fetch(`${config!.origin}/api/v1/study/${path}`, {
    method, headers: { Origin: config!.origin, "X-Study-Session": before.session_id },
    data: { ...body, expected_revision: before.revision, idempotency_key: randomUUID() },
  });
  expect(response.ok()).toBe(true);
  return response.json();
}

/** API setup isolates recovery checks from the separately exercised complete UI journey. */
async function prepareCases(page: Page, request: APIRequestContext) {
  const issued = await request.post(`${config!.origin}/api/v1/admin/studies/${encodeURIComponent(config!.study_revision)}/invitations`, {
    headers: { Authorization: `Bearer ${config!.researcher_token}` }, data: { count: 1, test: true },
  });
  expect(issued.ok()).toBe(true);
  const invitation = (await issued.json()).invitations[0];
  const secret = new URLSearchParams(invitation.invitation.split("#")[1]).get("invite");
  const exchanged = await page.request.post(`${config!.origin}/api/v1/study/session`, { headers: { Origin: config!.origin }, data: { secret } });
  expect(exchanged.ok()).toBe(true);
  let state = await exchanged.json();
  await mutate(page, "consent", { information_version: state.information_version, accepted: true });
  const setup = { protege_installed: true, source_opened: true, target_opened: true, practice_source_located: true, practice_definition_parents_inspected: true, protege_version: null, completed_tutorial_steps: [] };
  state = await mutate(page, "setup", setup);
  const answers: Record<string, unknown> = {};
  for (const question of state.forms.background) {
    if (question.show_if || !question.required) continue;
    const options = Object.keys(question.options ?? {});
    const value = options.includes("prefer_not_to_say") ? "prefer_not_to_say" : options[0];
    answers[question.id] = question.multiple ? [value] : value;
  }
  state = await mutate(page, "questionnaires/background", { form_version: state.forms.version, answers, submitted: true });
  state = await mutate(page, "setup", { ...setup, completed_tutorial_steps: state.tutorial_steps.map((_: string, index: number) => index) });
  expect(state.stage).toBe("case");
  return state;
}

async function useExplanationCase(page: Page) {
  for (let count = 0; count < 3; count += 1) {
    const response = await page.request.get(`${config!.origin}/api/v1/study/cases/current`);
    expect(response.ok()).toBe(true);
    const item = await response.json();
    if (item.condition === "explanation") return item;
    await mutate(page, `cases/${encodeURIComponent(item.case_id)}/submit`, { presentation_id: item.presentation_id, response_type: "none_of_these", ranked_candidate_ids: [] }, "POST");
    await mutate(page, `cases/${encodeURIComponent(item.case_id)}/consultation`, { consulted_external_ontologies: false, methods: [], other_editor: null, other_resource: null });
  }
  throw new Error("Synthetic schedule has no explanation case");
}
async function dismissGap(page: Page) {
  const skip = page.getByRole("button", { name: "Skip this question", exact: true });
  const shown = await skip.isVisible();
  if (shown) await skip.click();
  return shown;
}

test("a stale tab cannot save into another equal-revision session sharing the cookie", async ({ page, context, request }) => {
  const first = await prepareCases(page, request);
  await page.goto(`${config!.origin}/participate/`);
  await expect(page.getByRole("button", { name: /^Add .* as rank 1$/ }).first()).toBeEnabled();
  const gapAnswered = await dismissGap(page);
  if (gapAnswered) await expect.poll(async () => (await current(page)).revision).toBe(first.revision + 1);
  const firstBeforeSwitch = await current(page);
  // Wait for the optional gap acknowledgement before capturing the final owner revision.
  await expect(page.getByRole("dialog")).toHaveCount(0);
  const other = await context.newPage();
  const second = await prepareCases(other, request);
  // The default resume question can advance the first revision once. Bring B to that same
  // revision explicitly, proving revision alone cannot identify the owner.
  const latestFirstRevision = firstBeforeSwitch.revision;
  let before = second;
  while (before.revision < latestFirstRevision) before = await mutate(other, "resume", { gap_activity: "unknown" }, "POST");
  const rejected = page.waitForResponse((response) => response.url().endsWith("/draft") && response.request().method() === "PUT");
  await page.getByRole("button", { name: /^Add .* as rank 1$/ }).first().click();
  expect((await rejected).status()).toBe(409);
  await expect(page.getByRole("button", { name: "Download unsaved changes", exact: true })).toBeVisible();
  const after = await current(other);
  expect(after.session_id).toBe(second.session_id);
  expect(after.revision).toBe(before.revision);
  expect(after.ranking).toBeNull();
});

test("failed explanation content blocks ranking and pending submission does not claim success", async ({ page, request }) => {
  await prepareCases(page, request);
  await useExplanationCase(page);
  await page.route("**/api/v1/study/resources/*", (route) => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Synthetic resource outage" }) }));
  await page.goto(`${config!.origin}/participate/`);
  await expect(page.getByRole("button", { name: "Retry loading this case", exact: true })).toBeVisible();
  await dismissGap(page);
  await expect(page.getByRole("button", { name: /^Add .* as rank 1$/ }).first()).toBeDisabled();
  await expect(page.getByRole("button", { name: "Submit answer", exact: true })).toBeDisabled();
  await page.unroute("**/api/v1/study/resources/*");
  await page.getByRole("button", { name: "Retry loading this case", exact: true }).click();
  await expect(page.getByRole("button", { name: /^Add .* as rank 1$/ }).first()).toBeEnabled();
  await page.getByRole("button", { name: /^Add .* as rank 1$/ }).first().click();
  let release!: () => void;
  const allowed = new Promise<void>((resolve) => { release = resolve; });
  let reached!: () => void;
  const intercepted = new Promise<void>((resolve) => { reached = resolve; });
  await page.route("**/api/v1/study/cases/*/submit", async (route) => { reached(); await allowed; await route.continue(); });
  await page.getByRole("button", { name: "Submit answer", exact: true }).click();
  await intercepted;
  await expect(page.getByText("Answer received and saved by the study server.", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Submitting…", exact: true })).toBeDisabled();
  release();
  await expect(page.getByRole("heading", { name: "One question about this case", exact: true })).toBeVisible();
});
