import { readFileSync } from "node:fs";
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

type Config = { origin: string; researcher_token: string; study_revision: string; publication: string };
const config: Config | null = process.env.EXACT_E2E_STUDY_CONFIG
  ? JSON.parse(readFileSync(process.env.EXACT_E2E_STUDY_CONFIG, "utf8")) : null;

test.use({ screenshot: "off" });

test.beforeEach(() => test.skip(!config, "Set the private configuration from the PostgreSQL/HTTPS harness"));

async function issue(request: APIRequestContext) {
  const response = await request.post(`${config!.origin}/api/v1/admin/studies/${encodeURIComponent(config!.study_revision)}/invitations`, {
    headers: { Authorization: `Bearer ${config!.researcher_token}` }, data: { count: 1, test: true },
  });
  expect(response.ok()).toBe(true);
  return (await response.json()).invitations[0] as { session_id: string; invitation: string };
}

async function state(page: Page) {
  const response = await page.request.get(`${config!.origin}/api/v1/study/state`);
  expect(response.ok()).toBe(true);
  return response.json();
}

async function answerForm(page: Page) {
  // Use visible native controls, including matrix rows and revealed branches.
  for (let pass = 0; pass < 3; pass += 1) {
    const radios = page.locator('input[type="radio"]:visible');
    const seen = new Set<string>();
    for (let index = 0; index < await radios.count(); index += 1) {
      const radio = radios.nth(index);
      const name = await radio.getAttribute("name") ?? `radio-${index}`;
      if (!seen.has(name)) { await radio.check(); seen.add(name); }
    }
    for (const question of await page.locator("fieldset.question:visible").all()) {
      const checks = question.locator('input[type="checkbox"]:visible');
      if (await checks.count() && !(await checks.evaluateAll((elements) => elements.some((element) => (element as HTMLInputElement).checked)))) {
        await checks.first().check();
      }
    }
  }
}

async function startCases(page: Page, request: APIRequestContext) {
  const invitation = await issue(request);
  await page.goto(config!.origin + invitation.invitation);
  await page.getByRole("button", { name: "I agree to take part", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Set up Protégé and the two ontologies" })).toBeVisible();
  // These are synthetic self-reports; this test does not claim to run Protégé.
  for (const checkbox of await page.getByRole("checkbox").all()) await checkbox.check();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page.getByRole("button", { name: "Continue to practice" })).toBeVisible();
  await answerForm(page);
  await page.getByRole("button", { name: "Continue to practice" }).click();
  await expect(page.getByRole("button", { name: "Start the scored cases" })).toBeVisible();
  for (const checkbox of await page.getByRole("checkbox").all()) await checkbox.check();
  for (let lesson = 0; lesson < 4; lesson += 1) {
    if (lesson < 3) await page.getByRole("button", { name: /^Add .* as rank 1$/ }).first().click();
    else await page.getByRole("radio", { name: /^None of these/ }).click();
    await page.getByRole("button", { name: "Check practice action", exact: true }).click();
    if (lesson < 3) await page.getByRole("button", { name: "Next practice", exact: true }).click();
  }
  await page.getByRole("button", { name: "Start the scored cases", exact: true }).click();
  await expect(page.getByRole("button", { name: "Submit answer", exact: true })).toBeVisible();
  return invitation;
}

test("complete HTTPS/PostgreSQL participant journey, pause, reload, timing and condition isolation", async ({ page, request, context }) => {
  test.setTimeout(180_000);
  const errors: string[] = [];
  const rejectedTiming: number[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => {
    if (response.url().endsWith("/study/timing") && !response.ok()) rejectedTiming.push(response.status());
  });
  const invitation = await startCases(page, request);
  const cookies = await context.cookies(`${config!.origin}/api/v1/study/state`);
  expect(cookies.some((cookie) => cookie.secure && cookie.httpOnly && cookie.sameSite === "Strict")).toBe(true);
  const conditions = new Set<string>();
  for (let index = 0; index < 3; index += 1) {
    await expect(page.getByRole("radio", { name: /^None of these/ })).toBeEnabled();
    const current = await (await page.request.get(`${config!.origin}/api/v1/study/cases/current`)).json();
    conditions.add(current.condition);
    if (current.condition === "ontology_baseline") {
      await expect(page.getByRole("button", { name: "Inspect", exact: true })).toHaveCount(0);
      const resource = await page.request.get(`${config!.origin}/api/v1/study/resources/explanation-0`);
      expect(resource.status()).toBe(403);
    }
    if (index === 0) {
      await page.getByRole("button", { name: /^Add .* as rank 1$/ }).first().click();
      await page.getByRole("button", { name: /Pause/ }).click();
      await expect(page.getByRole("heading", { name: "Paused", exact: true })).toBeVisible();
      await page.getByRole("button", { name: "Resume", exact: true }).click();
      await expect(page.getByRole("list", { name: "Your ranking" }).getByRole("listitem")).toHaveCount(1);
      await page.reload();
      await expect(page.getByRole("list", { name: "Your ranking" }).getByRole("listitem")).toHaveCount(1);
    } else {
      await page.getByRole("radio", { name: index === 1 ? /^None of these/ : /^Insufficient information/ }).click();
    }
    await page.getByRole("button", { name: "Submit answer", exact: true }).click();
    await expect(page.getByRole("heading", { name: "One question about this case" })).toBeVisible();
    await page.getByRole("radio", { name: "No", exact: true }).check();
    await page.getByRole("button", { name: /^Save and (go to case|continue)/ }).click();
  }
  expect([...conditions].sort()).toEqual(["explanation", "ontology_baseline"]);
  await expect(page.getByRole("heading", { name: "What helped, and what was hard?" })).toBeVisible();
  await answerForm(page);
  await page.getByRole("button", { name: "Finish the study", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Thank you. Your responses are recorded." })).toBeVisible();
  await page.goto(config!.origin + invitation.invitation);
  await expect(page.getByRole("heading", { name: "Thank you. Your responses are recorded." })).toBeVisible();
  expect((await state(page)).stage).toBe("completed");
  expect(errors).toEqual([]);
  expect(rejectedTiming).toEqual([]);
  const exported = await request.post(`${config!.origin}/api/v1/admin/studies/${encodeURIComponent(config!.study_revision)}/exports?include_test=true`, {
    headers: { Authorization: `Bearer ${config!.researcher_token}` },
  });
  expect(exported.ok()).toBe(true);
  const record = (await exported.json()).data.sessions.find((item: { session_id: string }) => item.session_id === invitation.session_id);
  expect(record.cases).toHaveLength(3);
  for (const item of record.cases) expect(item.timing.observed_segment_seconds).toBeGreaterThan(0);
  expect(record.timing_segments.filter((segment: { stage: string }) => segment.stage === "case").length).toBeGreaterThanOrEqual(4);
});

test("offline answers survive reload and reconnect before submission", async ({ page, request, context }) => {
  test.setTimeout(180_000);
  await startCases(page, request);
  await expect(page.getByRole("radio", { name: /^None of these/ })).toBeEnabled();
  // Block API writes while leaving the static page available for a real reload.
  await page.route("**/api/v1/study/**", (route) => route.request().method() === "GET" ? route.continue() : route.abort("internetdisconnected"));
  await page.getByRole("button", { name: /^Add .* as rank 1$/ }).first().click();
  await page.getByRole("button", { name: /^Add .* as rank 2$/ }).first().click();
  await expect(page.getByText(/Offline|offline/).first()).toBeVisible();
  await page.reload();
  await page.unroute("**/api/v1/study/**");
  await context.setOffline(true);
  await context.setOffline(false);
  await expect.poll(async () => (await state(page)).ranking?.ranked_candidate_ids ?? []).toHaveLength(2);
  await expect(page.getByRole("list", { name: "Your ranking" }).getByRole("listitem")).toHaveCount(2);
});

test("researcher browser authentication, publication, links, progress, exports and closure", async ({ page, request }) => {
  test.setTimeout(120_000);
  const publication = JSON.parse(readFileSync(config!.publication, "utf8"));
  const revision = `browser-admin-${Date.now()}`;
  publication.definition.study_revision = revision;
  await page.goto(`${config!.origin}/admin/`);
  await page.getByLabel("Researcher token", { exact: true }).fill(config!.researcher_token);
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await page.getByLabel("Publish a study revision…").setInputFiles({ name: "synthetic-publication.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(publication)) });
  await expect(page.getByText(`Published and frozen: ${revision}. Every case, resource and key was validated.`)).toBeVisible();
  await page.getByRole("button", { name: "Load progress", exact: true }).click();
  await expect(page.getByRole("table")).toBeVisible();
  await page.getByLabel("How many", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Create links", exact: true }).click();
  await expect(page.locator(".link-list li")).toHaveCount(2);
  const sessionId = await page.locator(".link-list li").first().locator("span").first().innerText();
  await page.getByLabel("Session ID", { exact: true }).fill(sessionId);
  await page.getByRole("button", { name: "Issue a replacement link", exact: true }).click();
  await expect(page.getByText(/Replacement link \(shown once\)/)).toBeVisible();
  await page.getByRole("button", { name: "Revoke", exact: true }).click();
  await expect(page.getByText(/Link revoked/)).toBeVisible();
  await page.getByLabel("Include test sessions", { exact: true }).check();
  for (const format of ["JSON", "CSV archive"]) {
    await page.getByRole("radio", { name: format, exact: true }).click();
    const download = page.waitForEvent("download");
    await page.getByRole("button", { name: "Create export", exact: true }).click();
    expect(await (await download).failure()).toBeNull();
  }
  await page.getByRole("button", { name: "Close study", exact: true }).click();
  await page.getByRole("button", { name: "Close the study", exact: true }).click();
  await expect(page.getByText(`${revision} is closed.`, { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page.getByLabel("Researcher token", { exact: true })).toHaveValue("");
  await expect(page.locator(".link-list")).toHaveCount(0);
  const landing = await request.get(`${config!.origin}/`, { maxRedirects: 0 });
  expect(landing.status()).toBe(307);
  expect(landing.headers().location).toBe("/participate/");
  for (const path of ["/browse/", "/library/", "/api/v1/runs"]) {
    expect((await request.get(`${config!.origin}${path}`)).status()).toBe(404);
  }
});
