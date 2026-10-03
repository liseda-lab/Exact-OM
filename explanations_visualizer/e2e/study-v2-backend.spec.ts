import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

// Full corrected workflow against a synthetic admitted publication on real HTTPS/PostgreSQL.
const config = process.env.EXACT_E2E_STUDY_V2_CONFIG ? JSON.parse(readFileSync(process.env.EXACT_E2E_STUDY_V2_CONFIG, "utf8")) : null;
test.use({ screenshot: "off", trace: "off", actionTimeout: 20_000, viewport: { width: 1280, height: 720 } });
test.beforeEach(() => test.skip(!config, "Set EXACT_E2E_STUDY_V2_CONFIG for the PostgreSQL/HTTPS fixture"));

async function noViolations(page: Page) {
  const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  expect(result.violations.map((item) => `${item.id}: ${item.nodes.map((node) => node.target.join(" ")).join(", ")}`)).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false);
}

async function saved(page: Page, count: number) {
  await expect(page.locator(".lesson-checklist li .meta").filter({ hasText: "Done · saved" })).toHaveCount(count);
}

test("tool-neutral setup, interactive tutorial, server-graded assessment and durable per-case reports", async ({ page, context, request }) => {
  test.setTimeout(240_000);
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const errors: string[] = [];
  const requestErrors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400 && ["POST", "PUT"].includes(response.request().method())) requestErrors.push(`${response.status()} ${new URL(response.url()).pathname}`);
  });
  const commitReport = async (name: RegExp) => {
    // A click can finish before the queued mutation commits. Observe that receipt
    // before querying the next assignment, especially across a condition boundary.
    const [receipt] = await Promise.all([
      page.waitForResponse((response) => response.request().method() === "PUT" && new URL(response.url()).pathname.endsWith("/consultation")),
      page.getByRole("button", { name }).click(),
    ]);
    expect(receipt.ok()).toBe(true);
  };
  const response = await request.post(`${config.origin}/api/v1/admin/studies/${encodeURIComponent(config.study_revision)}/invitations`, { headers: { Authorization: `Bearer ${config.researcher_token}` }, data: { count: 1, test: true } });
  expect(response.ok()).toBe(true);
  const invitation = (await response.json()).invitations[0];
  await page.goto(config.origin + invitation.invitation);
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

  // C05: options follow the declared order although the service sends reversed keys.
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

  await page.getByRole("button", { name: "Next lesson", exact: true }).click();
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

  // Reload mid-tutorial: position and acknowledged evidence are restored from the server.
  await page.reload();
  await expect(page.getByRole("heading", { name: /^Lesson 2 of 6/ })).toBeVisible();
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

  // C11: server-graded items with specific feedback and unlimited retries.
  await page.getByRole("button", { name: "Go to the five questions", exact: true }).click();
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

  // C06/C15: the case layout, then a durable draft report restored after reload.
  await expect(page.getByRole("complementary", { name: "Candidates and your answer" })).toBeVisible();
  await expect(page.locator(".answer-peek")).toContainText("Your answer");
  await noViolations(page);
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

  // Remaining cases keep their assigned conditions and independent reports.
  const conditions = new Set<string>();
  for (let index = 0; index < 8; index += 1) {
    const state = await (await page.request.get(`${config.origin}/api/v1/study/state`)).json();
    if (state.stage === "final") break;
    const current = await (await page.request.get(`${config.origin}/api/v1/study/cases/current`)).json();
    conditions.add(current.condition);
    if (current.condition === "ontology_baseline") await expect(page.getByRole("tab")).toHaveCount(0);
    else {
      await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
      await expect(page.getByRole("region", { name: "Source ontology browser" })).toBeVisible();
      await page.getByRole("tab", { name: "Evidence", exact: true }).click();
    }
    await page.getByRole("radio", { name: /^Insufficient information/ }).click();
    await page.getByRole("button", { name: "Submit answer", exact: true }).click();
    await expect(page.getByRole("radio", { name: "Yes", exact: true })).not.toBeChecked();
    await expect(page.getByRole("radio", { name: "No", exact: true })).not.toBeChecked();
    await page.getByRole("radio", { name: "No", exact: true }).check();
    await commitReport(/^Save and (go to case|continue)/);
  }
  expect(conditions.has("ontology_baseline")).toBe(true);
  await expect(page.getByRole("heading", { name: "What helped, and what was hard?" })).toBeVisible();
  await expect(page.locator("#q-component_usefulness thead th").nth(1)).toHaveText("Not helpful");
  await expect(page.locator("#q-component_usefulness tbody th").first()).toHaveText("Original entity definitions and context");
  for (const row of await page.locator("table.matrix tbody tr").all()) await row.getByRole("radio").first().check();
  await page.locator("#q-most_helpful_components").getByRole("checkbox", { name: "Cannot judge", exact: true }).check();
  await page.locator("#q-workflow_preference").getByRole("radio", { name: "Cannot judge", exact: true }).check();
  await page.getByRole("button", { name: "Finish the study", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Thank you. Your responses are recorded." })).toBeVisible();
  const final = await (await page.request.get(`${config.origin}/api/v1/study/state`)).json();
  expect(final.stage).toBe("completed");
  expect(final.completed_cases).toBe(final.assigned_case_count);
  expect(requestErrors).toEqual([]);
  expect(errors).toEqual([]);
});
