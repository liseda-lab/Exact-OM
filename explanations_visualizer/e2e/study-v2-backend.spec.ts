import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

import { harness, recordTraffic, scopedPath } from "./v2Harness";

// The unseeded acceptance journey (specs 17 S3, 19 F15): a new synthetic invitation on the
// real service (HTTPS, PostgreSQL, compiled frontend, normal CSP) through consent, setup,
// background, every tutorial interaction, all five items, both conditions, per-case reports,
// the final form and completion. Nothing is completed through the API. It asserts the actual
// requests, not only what appears. EXACT_E2E_MAIN_APP_URL (optional) serves the same frozen
// contexts in the exploration app for the J03 comparison (see prepare_main_app_package.py).
const mainApp = process.env.EXACT_E2E_MAIN_APP_URL ?? null;
test.use({ screenshot: "off", trace: "off", actionTimeout: 20_000, viewport: { width: 1280, height: 720 } });
test.beforeEach(() => test.skip(!harness, "Set EXACT_E2E_STUDY_V2_CONFIG for the PostgreSQL/HTTPS harness"));

async function noViolations(page: Page) {
  const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  expect(result.violations.map((item) => `${item.id}: ${item.nodes.map((node) => node.target.join(" ")).join(", ")}`)).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false);
}

async function saved(page: Page, count: number) {
  await expect(page.locator(".lesson-checklist li .meta").filter({ hasText: "Done · saved" })).toHaveCount(count);
}

async function position(page: Page) {
  return (await (await page.request.get(`${harness!.origin}/api/v1/study/state`)).json()).tutorial_progress.position;
}

/**
 * The study reads a physically policy-filtered copy of the same frozen context. Only its
 * declared metadata may differ: the derived copy's revision, its category inventory and
 * extraction status (`complete_for_policy`), and query-bound cursors. Facts, identities,
 * counts and order must be identical.
 */
const DECLARED_DIFFERENCES = ["context_revision", "capabilities_metadata", "extraction", "next_cursor"];
function normalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([key]) => !DECLARED_DIFFERENCES.includes(key))
        .map(([key, item]) => [key, normalize(item)]),
    );
  return value;
}

test("fresh browser journey: training, both conditions, reports and completion over the scoped workspace", async ({ page, context, browser }) => {
  test.setTimeout(420_000);
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const traffic = recordTraffic(page);
  const bodies: { path: string; text: string }[] = [];
  page.on("response", async (response) => {
    const path = new URL(response.url()).pathname;
    if (!path.startsWith("/api/v1/study/") || !(response.headers()["content-type"] ?? "").includes("json")) return;
    try {
      bodies.push({ path, text: await response.text() });
    } catch {
      /* navigation discarded the body */
    }
  });
  const requestErrors: string[] = [];
  page.on("response", (response) => {
    if (response.status() >= 400) requestErrors.push(`${response.status()} ${response.request().method()} ${new URL(response.url()).pathname}`);
  });
  const commitReport = async (name: RegExp) => {
    const [receipt] = await Promise.all([
      page.waitForResponse((response) => response.request().method() === "PUT" && new URL(response.url()).pathname.endsWith("/consultation")),
      page.getByRole("button", { name }).click(),
    ]);
    expect(receipt.ok()).toBe(true);
  };
  const response = await page.request.post(`${harness!.origin}/api/v1/admin/studies/${encodeURIComponent(harness!.study_revision)}/invitations`, { headers: { Authorization: `Bearer ${harness!.researcher_token}` }, data: { count: 1, test: true } });
  expect(response.ok()).toBe(true);
  const invitation = (await response.json()).invitations[0];
  await page.goto(harness!.origin + invitation.invitation);
  await expect(page.getByRole("button", { name: "I agree to take part", exact: true })).toBeVisible();
  const state0 = await (await page.request.get(`${harness!.origin}/api/v1/study/state`)).json();
  expect(state0.integration_contract).toBe("study-integration/1");
  await page.getByRole("button", { name: "I agree to take part", exact: true }).click();

  // C12: no installation, opening or tool-use claim; drafts never advance.
  await expect(page.getByRole("heading", { name: "Before you start: the task and the ontology files" })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: /installed|opens in Protégé/ })).toHaveCount(0);
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(page.locator("#setup-status")).toContainText("Please confirm");
  await page.getByRole("checkbox", { name: "I have read the task instructions." }).check();
  await page.getByRole("radio", { name: /Yes, I can download or open them/ }).check();
  await page.getByRole("checkbox", { name: /outside inspection is optional/ }).check();
  await expect(page.getByRole("heading", { name: "Before you start: the task and the ontology files" })).toBeVisible();
  await page.getByRole("button", { name: "Continue", exact: true }).click();

  // C05: options follow the declared order.
  await expect(page.getByRole("heading", { name: "About your background" })).toBeVisible();
  await expect(page.locator("#q-cs_experience_years label").first()).toHaveText("None");
  for (const name of ["cs_experience_years", "health_bio_experience_years"]) await page.locator(`#q-${name} input[type=radio]`).first().check();
  for (const question of await page.locator("fieldset.question").all()) {
    const choice = question.getByRole("radio", { name: "Prefer not to say", exact: true });
    const multi = question.getByRole("checkbox", { name: "Prefer not to say", exact: true });
    if (await choice.count()) await choice.check();
    else if (await multi.count()) await multi.check();
  }
  await page.getByRole("button", { name: "Continue to the tutorial", exact: true }).click();

  // C10/C14: lessons in the shared workspace, saved only once acknowledged.
  await expect(page.getByRole("heading", { name: /^Lesson 1 of 6/ })).toBeVisible();
  await page.getByRole("button", { name: /^Inspect / }).nth(2).click();
  await page.getByRole("button", { name: /^Inspect / }).nth(0).click();
  await saved(page, 2);
  await noViolations(page);

  // R03/J06: navigate, wait for the acknowledgement, reload WITHOUT any lesson-2 action.
  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of 6/ })).toBeFocused();
  await expect(page.locator(".save-indicator")).toContainText("Saved");
  expect(await position(page)).toMatchObject({ view: "lesson", question_id: null });
  await page.reload();
  await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of 6/ })).toBeVisible();
  await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of 6/ })).not.toBeFocused();

  await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
  const source = page.getByRole("region", { name: "Source ontology browser" });
  await source.getByRole("combobox").fill("crate");
  await expect(source.getByRole("listbox").getByRole("option", { name: /^crate/ })).toBeVisible();
  await source.getByRole("combobox").press("ArrowDown");
  await source.getByRole("combobox").press("Enter");
  await expect(source.locator(".focus-label")).toHaveText("crate");
  await source.locator(".tree .tree-label").first().click();
  await source.locator(".child-list .tree-label").first().click();
  await source.getByRole("button", { name: /^Return to / }).click();
  await expect(source.locator(".browser-section .section-head .meta").first()).toContainText("multiple inheritance");
  await saved(page, 4);

  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
  await page.locator("button.citation").first().click();
  await expect(page.getByRole("dialog")).toContainText("Origin");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Show original axiom" }).first().click();
  await page.getByRole("tab", { name: "Evidence", exact: true }).click();
  await expect(page.getByRole("button", { name: "Show in graph" }).first()).toBeVisible();
  await page.getByRole("button", { name: "Show in graph" }).first().click();
  await saved(page, 3);

  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
  await expect(page.getByRole("heading", { name: /^Lesson 5 of 6/ })).toBeVisible();
  const add = () => page.getByRole("button", { name: /^Add .* as rank \d$/ }).first().click();
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

  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
  await page.getByRole("button", { name: /^Copy IRI of / }).first().click();
  await page.getByRole("button", { name: /Ontology files/ }).click();
  await expect(page.getByRole("dialog")).toContainText("Source ontology");
  await page.keyboard.press("Escape");
  const reports = page.locator(".practice-reports fieldset.question");
  await reports.nth(0).getByRole("radio", { name: "Yes", exact: true }).check();
  await reports.nth(0).getByRole("checkbox", { name: "Protégé", exact: true }).check();
  await reports.nth(0).getByRole("checkbox", { name: /The ontology files directly/ }).check();
  await reports.nth(0).getByRole("button", { name: "Check practice report" }).click();
  await reports.nth(1).getByRole("radio", { name: "No", exact: true }).check();
  await reports.nth(1).getByRole("button", { name: "Check practice report" }).click();
  await saved(page, 4);

  // R03/J06: the assessment landing persists without an answer.
  await page.getByRole("button", { name: "Go to the five questions", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Check your understanding" })).toBeFocused();
  await expect(page.locator(".save-indicator")).toContainText("Saved");
  expect(await position(page)).toEqual({ view: "assessment", lesson_id: null, question_id: null });
  await page.reload();
  await expect(page.getByRole("heading", { level: 1, name: "Check your understanding" })).toBeVisible();

  // C11: server-graded items with specific feedback and unlimited retries.
  const items = page.locator(".assessment-item");
  await items.nth(0).getByRole("radio", { name: "It must be equivalent." }).check();
  await items.nth(0).getByRole("button", { name: "Check answer" }).click();
  await expect(items.nth(0)).toContainText("Not quite");
  await expect(items.nth(0).getByRole("button", { name: /^Revisit the lesson/ })).toBeVisible();
  await expect(page.getByRole("button", { name: "Finish the tutorial and start the scored cases" })).toBeDisabled();
  await noViolations(page);
  await items.nth(0).getByRole("radio", { name: /The matcher suggests it/ }).check();
  await items.nth(0).getByRole("button", { name: "Check again" }).click();
  await items.nth(1).getByRole("radio", { name: "No, the candidate is broader." }).check();
  await items.nth(1).getByRole("button", { name: "Check answer" }).click();
  const q3 = items.nth(2).locator("fieldset.match-row");
  await q3.nth(0).getByRole("radio", { name: "None of these" }).check();
  await q3.nth(1).getByRole("radio", { name: "Insufficient information" }).check();
  await q3.nth(2).getByRole("radio", { name: "An empty, unsubmitted draft" }).check();
  await items.nth(2).getByRole("button", { name: "Check answer" }).click();
  const q4 = items.nth(3).locator("fieldset.match-row");
  await q4.nth(0).getByRole("radio", { name: "Original ontology fact" }).check();
  await q4.nth(1).getByRole("radio", { name: "Matcher evidence" }).check();
  await q4.nth(2).getByRole("radio", { name: "Generated text" }).check();
  await q4.nth(3).getByRole("radio", { name: "No", exact: true }).check();
  await items.nth(3).getByRole("button", { name: "Check answer" }).click();
  for (const label of [/is optional/, /combine methods/, /After each source-and-candidates case/]) await items.nth(4).getByRole("checkbox", { name: label }).check();
  await items.nth(4).getByRole("button", { name: "Check answer" }).click();
  await expect(page.locator(".assessment-item.passed")).toHaveCount(5);
  await page.getByRole("button", { name: "Finish the tutorial and start the scored cases" }).click();

  // Scored cases in both conditions, each followed by its own report.
  const conditions = new Set<string>();
  const explanationScopes = new Set<string>();
  let explored = false;
  for (let index = 0; index < 8; index += 1) {
    const state = await (await page.request.get(`${harness!.origin}/api/v1/study/state`)).json();
    if (state.stage === "final") break;
    await expect(page.getByRole("complementary", { name: "Candidates and your answer" })).toBeVisible();
    const current = await (await page.request.get(`${harness!.origin}/api/v1/study/cases/current`)).json();
    conditions.add(current.condition);
    await expect(page.getByRole("button", { name: /^Add .* as rank 1$/ }).first()).toBeEnabled();
    if (current.condition === "ontology_baseline") {
      await expect(page.getByRole("tab")).toHaveCount(0);
      expect(current.workspace).toBeNull();
    } else {
      // J01: the descriptor survives HTTP; the page reads only that scope.
      const scope = current.workspace.scope_id as string;
      explanationScopes.add(scope);
      const reads = traffic.requests.filter((item) => item.path.startsWith(scopedPath(scope)));
      expect(reads.some((item) => item.path.startsWith(`${scopedPath(scope)}/capabilities`) && item.status === 200)).toBe(true);
      const contexts = reads.filter((item) => item.path.startsWith(`${scopedPath(scope)}/entity-context`) && item.status === 200);
      for (const entity of [current.source, ...current.candidates.map((candidate: { entity: unknown }) => candidate.entity)] as { iri: string }[])
        expect(contexts.some((item) => new URL(item.path, harness!.origin).searchParams.get("iri") === entity.iri)).toBe(true);
      if (!explored) {
        explored = true;
        await noViolations(page);
        // J03: search outside the focal excerpt, follow multiple parents, exhaust children, open a citation.
        await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
        const studyBrowser = page.getByRole("region", { name: "Source ontology browser" });
        await expect(studyBrowser.locator(".browser-section .section-head .meta").nth(1)).toHaveText("50 of 65");
        await studyBrowser.getByRole("button", { name: "Load next 50 children" }).click();
        await expect(studyBrowser.locator(".browser-section .section-head .meta").nth(1)).toHaveText("65 of 65");
        const children = await studyBrowser.locator(".child-list li .iri").allTextContents();
        expect(new Set(children).size).toBe(65);
        await studyBrowser.getByRole("combobox").fill("Navigation source 012");
        await expect(studyBrowser.getByRole("listbox").getByRole("option", { name: /^Navigation source 012/ })).toBeVisible();
        await studyBrowser.getByRole("combobox").press("ArrowDown");
        await studyBrowser.getByRole("combobox").press("Enter");
        await expect(studyBrowser.locator(".focus-label")).toHaveText("Navigation source 012");
        await expect(studyBrowser.locator(".browser-section .section-head .meta").first()).toHaveText("5 · multiple inheritance");
        await studyBrowser.getByRole("button", { name: /^Return to / }).click();
        // Multi-page fact category: continue the context's own cursor to the end, no duplicates.
        const sourceCard = page.locator(".entity-card-source");
        await sourceCard.getByRole("button", { name: /^Other recorded facts/ }).click();
        const comments = sourceCard.locator(".more-category").filter({ hasText: "Comments" });
        await expect(comments.locator("h4 .meta")).toHaveText("20 of 65");
        while (await comments.getByRole("button", { name: "Load more comments" }).count()) await comments.getByRole("button", { name: "Load more comments" }).click();
        await expect(comments.locator("h4 .meta")).toHaveText("65 of 65");
        const factIds = await comments.locator("[data-fact-id]").evaluateAll((nodes) => nodes.map((node) => node.getAttribute("data-fact-id")));
        expect(new Set(factIds).size).toBe(65);
        await page.locator("button.citation").first().click();
        const dialog = page.getByRole("dialog");
        await expect(dialog.locator(".iri").first()).toHaveText(/^urn:(source|target):\d$/);
        await expect(dialog).toContainText(/Original|Asserted|Origin/);
        await page.keyboard.press("Escape");
        if (mainApp) {
          // The same frozen information in the main app; only declared scope metadata differs.
          const entity = current.source as { ontology_version_id: string; iri: string; kind: string };
          const study = await (await page.request.get(`${harness!.origin}${scopedPath(scope)}/entity-context`, { params: entity, headers: { "X-Study-Session": state.session_id } })).json();
          const main = await (await page.request.get(`${mainApp}/api/v1/entity-context`, { params: entity })).json();
          expect(normalize(study)).toEqual(normalize(main));
          expect(study.completeness.extraction).toBe("complete_for_policy");
          for (const [route, params] of [
            ["/hierarchy", { ...entity, direction: "children", limit: 100 }],
            ["/hierarchy", { ...entity, direction: "parents" }],
            ["/entities", { ontology_version_id: entity.ontology_version_id, term: "Navigation source 012" }],
          ] as const) {
            const scoped = await (await page.request.get(`${harness!.origin}${scopedPath(scope)}${route}`, { params: params as never, headers: { "X-Study-Session": state.session_id } })).json();
            const unscoped = await (await page.request.get(`${mainApp}/api/v1${route}`, { params: params as never })).json();
            expect(normalize(scoped)).toEqual(normalize(unscoped));
          }
          // Every page of the comments category, through each product's own continuation.
          const allFacts = async (base: string, headers: Record<string, string>) => {
            const ids: string[] = [];
            let cursor: string | null = null;
            do {
              const page_: { items: { fact_id: string }[]; next_cursor: string | null } = await (await page.request.get(`${base}/entity-facts`, { params: { ...entity, category: "comments", limit: 50, ...(cursor ? { cursor } : {}) }, headers })).json();
              ids.push(...page_.items.map((item) => item.fact_id));
              cursor = page_.next_cursor;
            } while (cursor);
            return ids;
          };
          const studyFacts = await allFacts(`${harness!.origin}${scopedPath(scope)}`, { "X-Study-Session": state.session_id });
          expect(studyFacts).toHaveLength(65);
          expect(studyFacts).toEqual(await allFacts(`${mainApp}/api/v1`, {}));
          const other = await browser.newPage();
          await other.goto(`${mainApp}/browse/?so=${encodeURIComponent(entity.ontology_version_id)}&s=${encodeURIComponent(entity.iri)}&sk=class`);
          const mainBrowser = other.getByRole("region", { name: "Source ontology browser" });
          await expect(mainBrowser.locator(".browser-section .section-head .meta").nth(1)).toHaveText("50 of 65");
          await mainBrowser.getByRole("combobox").fill("Navigation source 012");
          await expect(mainBrowser.getByRole("listbox").getByRole("option", { name: /^Navigation source 012/ })).toBeVisible();
          await mainBrowser.getByRole("combobox").press("ArrowDown");
          await mainBrowser.getByRole("combobox").press("Enter");
          await expect(mainBrowser.locator(".focus-label")).toHaveText("Navigation source 012");
          await expect(mainBrowser.locator(".browser-section .section-head .meta").first()).toHaveText("5 · multiple inheritance");
          await other.close();
        }
      }
    }
    if (index === 0) {
      await add();
      await page.getByRole("button", { name: "Submit answer", exact: true }).click();
      await expect(page.getByRole("heading", { name: "About this case: inspection outside the study pages" })).toBeVisible();
      await page.getByRole("radio", { name: "Yes", exact: true }).check();
      await page.getByRole("checkbox", { name: "Queries or scripts" }).check();
      await page.getByRole("checkbox", { name: "A reasoner" }).check();
      await expect(page.locator(".save-indicator")).toContainText("Saved");
      await page.reload();
      await expect(page.getByRole("checkbox", { name: "A reasoner" })).toBeChecked();
      await expect(page.getByText("Your unfinished answer was restored from the study server.")).toBeVisible();
      await noViolations(page);
      await commitReport(/^Save and go to case 2/);
      continue;
    }
    await page.getByRole("radio", { name: /^Insufficient information/ }).click();
    await page.getByRole("button", { name: "Submit answer", exact: true }).click();
    await expect(page.getByRole("radio", { name: "Yes", exact: true })).not.toBeChecked();
    await expect(page.getByRole("radio", { name: "No", exact: true })).not.toBeChecked();
    await page.getByRole("radio", { name: "No", exact: true }).check();
    await commitReport(/^Save and (go to case|continue)/);
  }
  expect(conditions).toEqual(new Set(["explanation", "ontology_baseline"]));
  expect(explored).toBe(true);
  await expect(page.getByRole("heading", { name: "What helped, and what was hard?" })).toBeVisible();
  await expect(page.locator("#q-component_usefulness thead th").nth(1)).toHaveText("Not helpful");
  await expect(page.locator("#q-component_usefulness tbody th").first()).toHaveText("Original entity definitions and context");
  for (const row of await page.locator("table.matrix tbody tr").all()) await row.getByRole("radio").first().check();
  await page.locator("#q-most_helpful_components").getByRole("checkbox", { name: "Cannot judge", exact: true }).check();
  await page.locator("#q-workflow_preference").getByRole("radio", { name: "Cannot judge", exact: true }).check();
  await page.getByRole("button", { name: "Finish the study", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Thank you. Your responses are recorded." })).toBeVisible();
  const final = await (await page.request.get(`${harness!.origin}/api/v1/study/state`)).json();
  expect(final.stage).toBe("completed");
  expect(final.completed_cases).toBe(final.assigned_case_count);

  // The participant used only study routes, never the prepared-excerpt resources for v2
  // scored cases or unrestricted exploration routes, and no response carried private keys.
  expect(traffic.requests.filter((item) => !item.path.startsWith("/api/v1/study/") && !item.path.startsWith("/api/health"))).toEqual([]);
  expect(traffic.requests.filter((item) => /\/api\/v1\/study\/resources\/case-/.test(item.path))).toEqual([]);
  // Baseline cases made no workspace request: every scoped read belongs to an explanation case.
  expect(traffic.requests.filter((item) => item.path.startsWith("/api/v1/study/workspace/") && !explanationScopes.has(decodeURIComponent(item.path.split("/")[5])))).toEqual([]);
  expect(bodies.filter((item) => /"grading"|researcher_case_keys|"answer_key|"adjudicat/.test(item.text)).map((item) => item.path)).toEqual([]);
  expect(requestErrors).toEqual([]);
  expect(traffic.failures).toEqual([]);
  expect(traffic.errors).toEqual([]);
});
