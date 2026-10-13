import { readFileSync } from "node:fs";

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type Response } from "@playwright/test";

import { StyleDiagnostics } from "./styleDiagnostics";
import { harness, harnessForms, recordTraffic, REVIEW_FORMS, scopedPath } from "./v2Harness";

// The unseeded review journey on a forms/3 publication (23 C04/C05, 21 F01–F07): a new test
// invitation on the real service (HTTPS, PostgreSQL, compiled frontend, normal CSP) through
// consent, setup/3, the server-ordered background form, every tutorial lesson with one
// deliberate rejection and recovery, five assessment items with wrong retries, all assigned
// cases in both conditions with per-case method changes, the final form, completion and the
// analysis/4 exports. Nothing is completed through the API, and the answers come from the
// rendered page. Run against the corrected review publication:
//   EXACT_E2E_STUDY_V2_CONFIG=<private harness.json> npx playwright test e2e/study-review-journey.spec.ts
// The harness file names the study origin, researcher token, publication file and assets
// directory; it is private and never committed. EXACT_E2E_SHOTS (optional) keeps screenshots.
const shots = process.env.EXACT_E2E_SHOTS ?? null;
test.use({ screenshot: "off", trace: "off", actionTimeout: 20_000, viewport: { width: 1280, height: 720 } });
test.beforeEach(() => test.skip(!harness || harnessForms !== REVIEW_FORMS, "Set EXACT_E2E_STUDY_V2_CONFIG to a forms/3 review publication"));

// HR1: the application stylesheet must still apply after each reload and stage change; the
// diagnostics keep sanitized delivery evidence (see styleDiagnostics.ts).
let styles: StyleDiagnostics | null = null;
test.afterEach(async ({}, info) => {
  await styles?.finish(info);
  styles = null;
});
// Opt-in investigation only (spec 24 §3.3: at most ten): extra reloads at the assessment boundary.
const boundaryReloads = Math.min(10, Number(process.env.EXACT_E2E_BOUNDARY_RELOADS ?? 0) || 0);

async function noViolations(page: Page) {
  const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  expect(result.violations.map((item) => `${item.id}: ${item.nodes.map((node) => node.target.join(" ")).join(", ")}`)).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false);
}

const saved = (page: Page, count: number) => expect(page.locator(".lesson-checklist li .meta").filter({ hasText: "Done · saved" })).toHaveCount(count);
const step = (page: Page, label: string) => page.locator(".lesson-checklist li").filter({ hasText: label }).locator(".meta");
const snap = async (page: Page, name: string) => {
  if (shots) await page.screenshot({ path: `${shots}/${name}.png`, fullPage: false });
};

test("review journey: forms/3, tutorial recovery, both conditions, changing methods, completion and analysis/4 exports", async ({ page, context }) => {
  test.setTimeout(1_200_000);
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  styles = await StyleDiagnostics.attach(page, "study-review-journey", harness!.identity ?? {});
  const traffic = recordTraffic(page);
  const progress: { status: number; body: { actions?: { action: string; fact_id?: string }[] } }[] = [];
  page.on("response", async (response: Response) => {
    if (!response.url().endsWith("/api/v1/study/tutorial/progress") || response.request().method() !== "PUT") return;
    progress.push({ status: response.status(), body: response.request().postDataJSON() });
  });
  const requestErrors: string[] = [];
  page.on("response", (response) => {
    if (response.status() < 400) return;
    requestErrors.push(`${response.status()} ${response.request().method()} ${new URL(response.url()).pathname}`);
    if (process.env.EXACT_E2E_DEBUG) void response.text().then((text) => console.log("request error", response.status(), new URL(response.url()).pathname, text.slice(0, 300)));
  });
  const api = async () => (await page.request.get(`${harness!.origin}/api/v1/study/state`)).json();
  const methodName = `Journey method ${Date.now() % 100000}`;

  // F01: the session exchange declares the review client contract.
  const created = await page.request.post(`${harness!.origin}/api/v1/admin/studies/${encodeURIComponent(harness!.study_revision)}/invitations`, { headers: { Authorization: `Bearer ${harness!.researcher_token}` }, data: { count: 1, test: true } });
  expect(created.ok()).toBe(true);
  const [exchange] = await Promise.all([
    page.waitForRequest((request) => new URL(request.url()).pathname === "/api/v1/study/session" && request.method() === "POST"),
    page.goto(harness!.origin + (await created.json()).invitations[0].invitation),
  ]);
  expect(exchange.postDataJSON()).toMatchObject({ client_contract: "study-review/1" });
  await expect(page.getByRole("button", { name: "I agree to take part", exact: true })).toBeVisible();
  await styles.expectStyled("study-entry");
  expect((await api()).forms.version).toBe(REVIEW_FORMS);
  await page.getByRole("button", { name: "I agree to take part", exact: true }).click();

  // F01 setup/3: familiarity is a recommendation question; Other needs a name; None is exclusive.
  await expect(page.getByRole("heading", { name: "Before you start: the task and the ontology files" })).toBeVisible();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page.locator("#setup-status")).toContainText("Please confirm");
  await page.getByRole("checkbox", { name: "I have read the task instructions." }).check();
  await page.getByRole("radio", { name: /Yes, I can download or open them/ }).check();
  await page.getByRole("checkbox", { name: /outside inspection is optional/ }).check();
  await page.getByRole("checkbox", { name: "None of these", exact: true }).check();
  await page.getByRole("checkbox", { name: "Protégé", exact: true }).check();
  await expect(page.getByRole("checkbox", { name: "None of these", exact: true })).not.toBeChecked();
  await page.getByRole("checkbox", { name: "Other method", exact: true }).check();
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page.locator("#familiar-other-method")).toBeFocused();
  await page.locator("#familiar-other-method").fill(methodName);
  await snap(page, "journey-setup");
  await page.getByRole("button", { name: "Continue", exact: true }).click();

  // F01 background: server order, agreed bands, no "Prefer not to say", conditional Other text.
  await expect(page.getByRole("heading", { name: "About your background" })).toBeVisible();
  expect((await api()).setup).toMatchObject({ setup_version: "setup/3", familiar_methods: ["protege", "other_method"], familiar_other_method: methodName, submitted: true });
  const forms = (await api()).forms;
  await expect(page.locator("#q-cs_experience_years label").first()).toHaveText("None");
  expect(await page.getByText("Prefer not to say", { exact: true }).count()).toBe(0);
  const order = await page.locator(".questionnaire .question").evaluateAll((nodes) => nodes.map((node) => node.id.replace(/^q-/, "")));
  expect(order).toEqual(forms.background.filter((question: { show_if?: unknown }) => !question.show_if).map((question: { id: string }) => question.id));
  for (const question of forms.background as { id: string; required: boolean; show_if?: unknown; multiple?: boolean; option_order?: string[] }[]) {
    if (question.show_if || !question.required) continue;
    const box = page.locator(`#q-${question.id}`);
    if (question.id === "ncit_familiarity") await box.getByRole("radio").nth(2).check(); // "used": reveals its contexts
    else if (question.id === "roles") await box.getByRole("checkbox").last().check(); // "Other": reveals roles_other
    else await box.locator("input").first().check();
  }
  await page.locator("#q-ncit_contexts").getByRole("checkbox").first().check();
  await page.getByRole("button", { name: "Continue to the tutorial", exact: true }).click();
  await expect(page.locator("#q-roles_other")).toBeVisible();
  await expect(page.locator("#q-roles_other input, #q-roles_other textarea").first()).toBeFocused();
  await page.locator("#q-roles_other input, #q-roles_other textarea").first().fill("Ontology engineer");
  await snap(page, "journey-background");
  await page.getByRole("button", { name: "Continue to the tutorial", exact: true }).click();

  // F02 lesson 1: one adjacent task with Show me; progress shows only once acknowledged.
  await expect(page.getByRole("heading", { level: 1, name: /^Lesson 1 of 6/ })).toBeVisible();
  const coach = page.locator(".coach");
  await expect(coach.getByRole("button", { name: "Show me" })).toBeVisible();
  await coach.getByRole("button", { name: "Show me" }).click();
  await expect(page.locator('[data-tour="inspect"]:focus')).toHaveCount(1);
  await page.getByRole("button", { name: /^Inspect / }).nth(2).click();
  await page.getByRole("button", { name: /^Inspect / }).nth(0).click();
  await saved(page, 2);
  await snap(page, "journey-tutorial-lesson1");
  await noViolations(page);

  // Lesson 2: search, parent, child, and the instructed return to the compared concept.
  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of 6/ })).toBeFocused();
  await styles.beforeNavigation("lesson-2 reload");
  await page.reload();
  await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of 6/ })).toBeVisible();
  await styles.expectStyled("lesson-2-reload");
  await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
  const practice = page.getByRole("region", { name: "Source ontology browser" });
  await practice.getByRole("combobox").fill("crate");
  await expect(practice.getByRole("listbox").getByRole("option", { name: /^crate/ })).toBeVisible();
  await practice.getByRole("combobox").press("ArrowDown");
  await practice.getByRole("combobox").press("Enter");
  await expect(practice.locator(".focus-label")).toHaveText("crate");
  await practice.locator(".tree .tree-label").first().click();
  await practice.locator(".child-list .tree-label").first().click();
  if (await practice.getByRole("button", { name: "Back to compared concept" }).count()) await practice.getByRole("button", { name: "Back to compared concept" }).click();
  await expect(practice.getByText("Viewing compared concept")).toBeVisible();
  await saved(page, 4);

  // Lesson 3: citation, original axiom, evidence located in the graph.
  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
  await page.locator("button.citation").first().click();
  await expect(page.getByRole("dialog")).toContainText("Origin");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Show original axiom" }).first().click();
  await page.getByRole("tab", { name: "Evidence", exact: true }).click();
  await page.getByRole("button", { name: "Show in graph" }).first().click();
  await saved(page, 3);

  // Lesson 4 (F02/Q01): ontology context added to the graph is inspection only and submits
  // nothing; one rejected record (injected diagnostic: the REAL server rejects a rewritten fact)
  // is dropped and the next valid action saves without reload.
  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: /^Lesson 4 of 6/ })).toBeVisible();
  await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
  const items = page.locator(".graph-items");
  await items.getByRole("button", { name: /^Compared / }).first().click();
  await page.locator(".graph-inspector").getByRole("button", { name: "Add its parents" }).click();
  await items.locator("section").filter({ hasText: "Added from the ontology" }).getByRole("button").first().click();
  await expect(page.locator(".graph-inspector")).toContainText("Not one of Exact's selected features");
  await saved(page, 0);
  const sentBefore = progress.length;
  const bad = `sha256:${"0".repeat(64)}`;
  let injected = false;
  await page.route("**/api/v1/study/tutorial/progress", async (route) => {
    const body = route.request().postDataJSON();
    if (!injected && route.request().method() === "PUT" && body.actions?.some((action: { action: string }) => action.action === "inspect_graph_or_list")) {
      injected = true;
      body.actions = body.actions.map((action: { action: string; fact_id?: string }) => (action.action === "inspect_graph_or_list" ? { ...action, fact_id: bad } : action));
      return route.continue({ postData: JSON.stringify(body) });
    }
    return route.continue();
  });
  await items.locator('[data-tour="graph-feature"]').first().click();
  await expect(page.locator(".coach-notice")).toContainText("Not saved");
  await expect(step(page, "Inspect a line")).toHaveText("To do");
  await page.unroute("**/api/v1/study/tutorial/progress");
  await snap(page, "journey-tutorial-rejection");
  await items.locator('[data-tour="graph-feature"]').first().click();
  await expect(step(page, "Inspect a line")).toHaveText("Done · saved");
  await page.getByRole("button", { name: "Fit all" }).click();
  await page.getByRole("button", { name: /^Add .* as rank 1$/ }).first().click();
  await saved(page, 3);
  // Context clicks never sent an original-fact identity; only the injected record was refused.
  const lesson4 = progress.slice(sentBefore);
  expect(lesson4.filter((save) => save.status >= 400)).toHaveLength(1);
  expect(JSON.stringify(lesson4.filter((save) => save.status < 400).map((save) => save.body))).not.toContain(bad);

  // Lesson 5: the ranking tools.
  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
  await expect(page.getByRole("heading", { name: /^Lesson 5 of 6/ })).toBeVisible();
  const add = () => page.getByRole("button", { name: /^Add .* as rank \d$/ }).first().click();
  await page.getByRole("button", { name: /^Remove / }).first().click();
  await add();
  await add();
  await page.getByRole("button", { name: /^Move .* up$/ }).last().click();
  await page.getByRole("button", { name: /^Remove / }).first().click();
  await page.getByRole("button", { name: "Undo", exact: true }).click();
  await page.getByRole("button", { name: "Keep initial order", exact: true }).click();
  await page.getByRole("button", { name: /^Remove / }).first().click();
  await page.getByRole("button", { name: /^Remove / }).first().click();
  await page.getByRole("button", { name: "Check practice answer", exact: true }).click();
  await page.getByRole("radio", { name: /^None of these/ }).click();
  await page.getByRole("radio", { name: /^Insufficient information/ }).click();
  await saved(page, 8);

  // Lesson 6: the baseline block and methods/3 practice reports (Other needs a name).
  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
  await page.getByRole("button", { name: /^Copy IRI of / }).first().click();
  await page.getByRole("button", { name: /Ontology files/ }).click();
  await expect(page.getByRole("dialog")).toContainText("Source ontology");
  await page.keyboard.press("Escape");
  const reports = page.locator(".practice-reports fieldset.question");
  await reports.nth(0).getByRole("radio", { name: "Yes", exact: true }).check();
  expect(await reports.nth(0).getByRole("group", { name: "Methods used" }).getByRole("checkbox").count()).toBe(3);
  await reports.nth(0).getByRole("checkbox", { name: "Protégé", exact: true }).check();
  await reports.nth(0).getByRole("checkbox", { name: "Other method", exact: true }).check();
  await reports.nth(0).getByRole("button", { name: "Check practice report" }).click();
  await expect(reports.nth(0)).toContainText("name it, or clear Other method");
  await reports.nth(0).getByRole("checkbox", { name: "Other method", exact: true }).uncheck();
  await reports.nth(0).getByRole("checkbox", { name: "Text editor", exact: true }).check();
  await reports.nth(0).getByRole("button", { name: "Check practice report" }).click();
  await reports.nth(1).getByRole("radio", { name: "No", exact: true }).check();
  await reports.nth(1).getByRole("button", { name: "Check practice report" }).click();
  await saved(page, 4);
  await snap(page, "journey-tutorial-lesson6");
  await styles.expectStyled("lesson-6");

  // F02 assessment: server-graded, wrong answers retried, one question at a time.
  await page.getByRole("button", { name: "Go to the five questions", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Check your understanding" })).toBeFocused();
  const item = page.locator(".assessment-item");
  const next = () => page.getByRole("button", { name: "Next question", exact: true }).click();
  await item.getByRole("radio", { name: "It must be equivalent." }).check();
  await item.getByRole("button", { name: "Check answer" }).click();
  await expect(item).toContainText("Not quite");
  await expect(page.getByRole("button", { name: "Finish the tutorial and start the scored cases" })).toBeDisabled();
  await item.getByRole("radio", { name: /The matcher suggests it/ }).check();
  await item.getByRole("button", { name: "Check again" }).click();
  await expect(item.getByText("Passed")).toBeVisible();
  await next();
  await item.getByRole("radio", { name: "No, the candidate is broader." }).check();
  await item.getByRole("button", { name: "Check answer" }).click();
  await expect(item.getByText("Passed")).toBeVisible();
  await next();
  const q3 = item.locator("fieldset.match-row");
  await q3.nth(0).getByRole("radio", { name: "None of these" }).check();
  await q3.nth(1).getByRole("radio", { name: "Insufficient information" }).check();
  await q3.nth(2).getByRole("radio", { name: "An empty, unsubmitted draft" }).check();
  await item.getByRole("button", { name: "Check answer" }).click();
  await expect(item.getByText("Passed")).toBeVisible();
  await next();
  const q4 = item.locator("fieldset.match-row");
  await q4.nth(0).getByRole("radio", { name: "Original ontology fact" }).check();
  await q4.nth(1).getByRole("radio", { name: "Matcher evidence" }).check();
  await q4.nth(2).getByRole("radio", { name: "Generated text" }).check();
  await q4.nth(3).getByRole("radio", { name: "No", exact: true }).check();
  await item.getByRole("button", { name: "Check answer" }).click();
  await expect(item.getByText("Passed")).toBeVisible();
  await next();
  // Wrong first: "I must use Protégé" is false under methods/3.
  for (const label of [/is optional/, /combine methods/, /must use Protégé/]) await item.getByRole("checkbox", { name: label }).check();
  await item.getByRole("button", { name: "Check answer" }).click();
  await expect(item).toContainText("Not quite");
  await styles.beforeNavigation("assessment reload");
  await page.reload();
  await expect(page.getByRole("heading", { level: 1, name: "Check your understanding" })).toBeVisible();
  await styles.expectStyled("assessment-q5-reload");
  for (let extra = 1; extra <= boundaryReloads; extra += 1) {
    await styles.beforeNavigation(`assessment extra reload ${extra}`);
    await page.reload();
    await expect(page.getByRole("heading", { level: 1, name: "Check your understanding" })).toBeVisible();
    await styles.expectStyled(`assessment-q5-extra-reload-${extra}`);
  }
  await expect(item.getByRole("checkbox", { name: /must use Protégé/ })).toBeChecked();
  await item.getByRole("checkbox", { name: /must use Protégé/ }).uncheck();
  await item.getByRole("checkbox", { name: /After each source-and-candidates case/ }).check();
  await item.getByRole("button", { name: "Check again" }).click();
  await expect(item.getByText("Passed")).toBeVisible();
  await expect(page.locator(".question-nav button").filter({ hasText: "(passed)" })).toHaveCount(5);
  await page.getByRole("button", { name: "Finish the tutorial and start the scored cases" }).click();
  await expect(page.getByRole("complementary", { name: "Candidates and your answer" })).toBeVisible({ timeout: 60_000 });
  await styles.expectStyled("first-case");

  // Scored cases (F03–F07): both conditions; methods change between cases.
  const conditions = new Set<string>();
  const explanationScopes = new Set<string>();
  let explored = false;
  let baselineSeen = false;
  const plans: ({ consulted: false } | { consulted: true; methods: string[]; copy?: boolean })[] = [
    { consulted: true, methods: ["Protégé", "Other method"] },
    { consulted: true, methods: ["Text editor"] },
    { consulted: false },
    { consulted: true, methods: ["Protégé", "Text editor"] },
    { consulted: true, methods: ["Protégé"], copy: true },
  ];
  const expected: Record<string, string[] | null> = {};
  await expect(page.getByRole("complementary", { name: "Candidates and your answer" })).toBeVisible({ timeout: 60_000 });
  const assigned = (await api()).assigned_case_count as number;
  expect(assigned).toBeGreaterThan(1);
  for (let index = 0; index < assigned; index += 1) {
    await expect(page.getByRole("complementary", { name: "Candidates and your answer" })).toBeVisible();
    const current = await (await page.request.get(`${harness!.origin}/api/v1/study/cases/current`)).json();
    conditions.add(current.condition);
    if (index === 0) styles.record({ type: "allocation", first_condition: current.condition });
    await expect(page.getByRole("button", { name: /^Add .* as rank 1$/ }).first()).toBeEnabled();
    if (current.condition === "ontology_baseline") {
      await expect(page.getByRole("tab")).toHaveCount(0);
      expect(current.workspace).toBeNull();
      if (!baselineSeen) {
        baselineSeen = true;
        await styles.expectStyled(`first-baseline-case-${index + 1}`);
        await snap(page, "journey-case-baseline");
        await noViolations(page);
      }
    } else {
      const scope = current.workspace.scope_id as string;
      explanationScopes.add(scope);
      await expect.poll(() => traffic.requests.some((request) => request.path.startsWith(`${scopedPath(scope)}/capabilities`) && request.status === 200)).toBe(true);
      if (!explored) {
        explored = true;
        await expect(page.locator(".entity-card-source .entity-title")).not.toBeEmpty();
        await styles.expectStyled(`first-explanation-case-${index + 1}`);
        await snap(page, "journey-case-explanation");
        await noViolations(page);
        // F06: partial ELK is offered from the scope's navigation and explains itself.
        await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
        const browser = page.getByRole("region", { name: "Source ontology browser" });
        const modes = browser.getByRole("radiogroup", { name: "Relationships shown" });
        await expect(modes.getByRole("radio", { name: "Asserted" })).toBeVisible();
        // Keyboard: one tab stop, arrows choose the next mode.
        await modes.getByRole("radio", { name: "Asserted" }).focus();
        await page.keyboard.press("ArrowRight");
        await expect(modes.getByRole("radio", { name: "Structural" })).toHaveAttribute("aria-checked", "true");
        await expect(modes.getByRole("radio", { name: "Structural" })).toBeFocused();
        const capabilities = await (await page.request.get(`${harness!.origin}${scopedPath(scope)}/capabilities`, { headers: { "X-Study-Session": (await api()).session_id } })).json();
        const partial = Object.values(capabilities.navigation ?? {}).some((entry) => (entry as { partial_inference?: { status?: string } }).partial_inference?.status === "partial");
        if (partial) {
          await modes.getByRole("radio", { name: "ELK inferred · partial" }).click();
          await expect(browser).toContainText("A missing link is unknown, not absent.");
          await expect(browser.getByRole("button", { name: "About ELK partial results" })).toBeVisible();
          await expect.poll(() => traffic.requests.some((request) => request.path.startsWith(`${scopedPath(scope)}/hierarchy`) && request.path.includes("basis=elk_inferred_partial") && request.status === 200)).toBe(true);
          await snap(page, "journey-case-elk");
          await modes.getByRole("radio", { name: "Asserted" }).click();
        }
        await page.getByRole("tab", { name: "Evidence", exact: true }).click();
        await expect.poll(() => traffic.requests.some((request) => request.path.startsWith(`${scopedPath(scope)}/evidence?`) && request.status === 200)).toBe(true);
        await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
        await expect(page.locator(".graph-canvas, .graph-empty").first()).toBeVisible();
        if (await page.locator(".graph-canvas").count())
          await expect.poll(() => traffic.requests.some((request) => request.path.startsWith(`${scopedPath(scope)}/evidence-metadata?`) && request.status === 200)).toBe(true);
        await snap(page, "journey-case-graph");
        await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
      }
    }
    // A different answer kind per case, all through the page.
    if (index % 3 === 0) await page.getByRole("button", { name: /^Add .* as rank 1$/ }).first().click();
    else if (index % 3 === 1) await page.getByRole("radio", { name: /^Insufficient information/ }).click();
    else await page.getByRole("radio", { name: /^None of these/ }).click();
    await page.getByRole("button", { name: "Submit answer", exact: true }).click();
    await expect(page.getByRole("heading", { name: "About this case: inspection outside the study pages" })).toBeVisible();
    const plan = plans[index % plans.length];
    if (plan.consulted && plan.copy && (await page.getByRole("button", { name: "Copy my answer from the previous case" }).count())) {
      await page.getByRole("button", { name: "Copy my answer from the previous case" }).click();
      await expect(page.getByRole("status").filter({ hasText: "Copied from your previous case" })).toBeVisible();
      await page.getByRole("checkbox", { name: "Text editor", exact: true }).uncheck();
    } else if (plan.consulted) {
      await page.getByRole("radio", { name: "Yes", exact: true }).check();
      for (const name of plan.methods) await page.getByRole("checkbox", { name, exact: true }).check();
      if (index > 0 && plan.methods.includes("Other method")) await page.locator("#consultation-other_method").fill(methodName);
    } else await page.getByRole("radio", { name: "No", exact: true }).check();
    if (index === 0) {
      // Other method without a name is refused on the page; the draft is restored after reload.
      await page.getByRole("button", { name: /^Save and / }).click();
      await expect(page.locator("#consultation-other_method")).toBeFocused();
      await page.locator("#consultation-other_method").fill(methodName);
      await expect(page.locator(".save-indicator")).toContainText("Saved");
      await styles.beforeNavigation("consultation reload");
      await page.reload();
      await expect(page.getByText("Your unfinished answer was restored from the study server.")).toBeVisible();
      await styles.expectStyled("consultation-reload");
      await expect(page.getByRole("checkbox", { name: "Other method", exact: true })).toBeChecked();
      await snap(page, "journey-consultation");
      await noViolations(page);
    }
    expected[current.case_id] = plan.consulted ? (plan.copy ? ["Protégé"] : plan.methods) : null;
    // P3: no conflict happened, so no warning; the rejected tutorial record is only a quiet retained copy.
    await expect(page.locator(".study-banner")).toHaveCount(0);
    const [receipt] = await Promise.all([
      page.waitForResponse((response) => response.request().method() === "PUT" && new URL(response.url()).pathname.endsWith("/consultation")),
      page.getByRole("button", { name: /^Save and / }).click(),
    ]);
    expect(receipt.ok()).toBe(true);
    const body = receipt.request().postDataJSON();
    // consultation/3 is strict: no legacy editor/resource fields.
    expect(Object.keys(body).filter((key) => ["other_editor", "other_resource"].includes(key))).toEqual([]);
    expect(body.consultation_version).toBe("consultation/3");
    expect(body.methods.includes("other_method") ? body.other_method : null).toBe(body.methods.includes("other_method") ? methodName : null);
  }
  expect(conditions).toEqual(new Set(["explanation", "ontology_baseline"]));
  expect(explored && baselineSeen).toBe(true);

  // Final form, rendered in server order; then completion.
  await expect(page.getByRole("heading", { name: "What helped, and what was hard?" })).toBeVisible();
  for (const row of await page.locator("table.matrix tbody tr").all()) await row.getByRole("radio").first().check();
  const finalForm = (await api()).forms.final as { id: string; required: boolean; show_if?: unknown; type?: string; rows?: unknown[] }[];
  for (const question of finalForm) {
    if (!question.required || question.show_if || question.rows) continue;
    await page.locator(`#q-${question.id} input`).first().check();
  }
  await page.getByRole("button", { name: "Finish the study", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Thank you. Your responses are recorded." })).toBeVisible();
  const final = await api();
  expect(final.stage).toBe("completed");
  expect(final.completed_cases).toBe(final.assigned_case_count);
  await styles.expectStyled("completion");
  await snap(page, "journey-complete");

  // Only study routes; baseline cases read no workspace; no key material reached the page.
  expect(traffic.requests.filter((request) => !request.path.startsWith("/api/v1/study/") && !request.path.startsWith("/api/health"))).toEqual([]);
  expect(traffic.requests.filter((request) => request.path.startsWith("/api/v1/study/workspace/") && !explanationScopes.has(decodeURIComponent(request.path.split("/")[5])))).toEqual([]);
  expect(requestErrors.filter((error) => !error.startsWith("422 PUT /api/v1/study/tutorial/progress"))).toEqual([]);
  expect(traffic.errors).toEqual([]);
  // HR1: no application stylesheet transfer failed (a cancellation by a reload is not a failure).
  expect(styles.summary().css_failed_not_canceled).toBe(0);

  // Exports (researcher page): analysis/4, method names only on request; familiarity never fills use.
  const exportPage = await context.newPage();
  await exportPage.goto(`${harness!.origin}/admin/`);
  await exportPage.getByLabel("Researcher token").fill(harness!.researcher_token);
  await exportPage.getByRole("button", { name: "Continue", exact: true }).click();
  await exportPage.getByRole("textbox", { name: "Study revision" }).fill(harness!.study_revision);
  await exportPage.getByRole("button", { name: /Load published revisions|Load more revisions/ }).click();
  await exportPage.getByRole("checkbox", { name: "Include test sessions" }).check();
  const named = exportPage.getByRole("checkbox", { name: /named Other methods/ });
  await expect(named).toBeVisible();
  const download = async () => {
    const [file] = await Promise.all([exportPage.waitForEvent("download"), exportPage.getByRole("button", { name: "Create export" }).click()]);
    return JSON.parse(readFileSync((await file.path())!, "utf8"));
  };
  const redacted = await download();
  expect(redacted.manifest.schema).toBe("exact-study-analysis/4");
  await named.check();
  const withNames = await download();
  expect(withNames.manifest.schema).toBe("exact-study-analysis/4");
  const rows = (data: { data: { method_responses: Record<string, unknown>[] } }) => data.data.method_responses;
  const mine = rows(withNames).find((row) => row.stage === "setup_familiarity" && row.other_method === methodName)!;
  expect(mine).toBeTruthy();
  const session = mine.session_id;
  expect(mine.method_codes).toEqual(["protege", "other_method"]);
  const mineRedacted = rows(redacted).filter((row) => row.session_id === session);
  expect(mineRedacted.length).toBeGreaterThan(1);
  expect(mineRedacted.every((row) => !("other_method" in row))).toBe(true);
  expect(mineRedacted.find((row) => row.stage === "setup_familiarity")?.named_detail_state).toBe("redacted");
  const used = rows(withNames).filter((row) => row.session_id === session && row.stage === "consultation");
  expect(used).toHaveLength(assigned);
  const labels: Record<string, string> = { "Protégé": "protege", "Text editor": "plain_files", "Other method": "other_method" };
  for (const row of used) {
    const plan = expected[row.case_id as string];
    expect([...(row.method_codes as string[])].sort()).toEqual(plan ? plan.map((label) => labels[label]).sort() : []);
    expect(row.method_answer_state).toBe(plan ? "yes" : "no");
  }
  await exportPage.getByRole("radio", { name: "CSV archive" }).click();
  const [csv] = await Promise.all([exportPage.waitForEvent("download"), exportPage.getByRole("button", { name: "Create export" }).click()]);
  expect((await csv.suggestedFilename()).endsWith(".zip")).toBe(true);
  await expect(exportPage.getByRole("status").filter({ hasText: /exact-study-csv\/4/ })).toBeVisible();
  await exportPage.close();
});
