import { expect, test, type Page, type Request } from "@playwright/test";

import { harness, seedToTutorial, type ParticipantApi } from "./v2Harness";

// S0 regressions for the product review (20/21: U01–U05, Q01–Q04) against the real v2 study
// service (HTTPS, PostgreSQL, compiled frontend, normal CSP) and the served main app. Sessions
// are API-seeded to the tutorial only; every lesson action is performed in the browser. The one
// injected fault (Q02) rewrites a request body so the REAL server rejects it; it is labelled.
test.use({ screenshot: "off", trace: "off", viewport: { width: 1280, height: 900 } });

type Progress = { status: number; body: { actions?: { requirement_id: string; action: string; fact_id?: string; entity?: { iri: string } }[]; completed_requirements?: string[]; help_opened?: boolean } };

function recordProgress(page: Page) {
  const saves: Progress[] = [];
  page.on("requestfinished", async (request: Request) => {
    if (!request.url().endsWith("/api/v1/study/tutorial/progress") || request.method() !== "PUT") return;
    const response = await request.response();
    saves.push({ status: response?.status() ?? 0, body: request.postDataJSON() });
  });
  return saves;
}

const step = (page: Page, label: string) => page.locator(".lesson-checklist li").filter({ hasText: label }).locator(".meta");

async function lesson(page: Page, api: ParticipantApi, lessonId: string) {
  const { tutorial } = await api.state();
  await api.write("PUT", "/tutorial/progress", { tutorial_version: tutorial.version, position: { view: "lesson", lesson_id: lessonId, question_id: null } });
  await page.reload();
  await expect(page.locator(".coach")).toBeVisible();
  return tutorial;
}

async function practiceEvidence(page: Page, api: ParticipantApi) {
  const { tutorial } = await api.state();
  const resources = await Promise.all(tutorial.case.explanation_refs.map(async (ref: string) => (await page.request.get(`${harness!.origin}/api/v1/study/resources/${encodeURIComponent(ref)}`)).json()));
  return resources.flatMap((resource: { evidence: { evidence_id: string; candidate_id: string; fact_ids: string[] }[] }) => resource.evidence);
}

test.describe("U01/Q01 graph lesson: interaction is not evidence", () => {
  test.beforeEach(() => test.skip(!harness, "Set EXACT_E2E_STUDY_V2_CONFIG"));

  test("one-sided channels draw no comparison line; added ontology context saves nothing; a feature saves its exact record for the inspected candidate", async ({ page }) => {
    const saves = recordProgress(page);
    const api = await seedToTutorial(page);
    await lesson(page, api, "graph");
    const evidence = await practiceEvidence(page, api);
    // Inspect a second candidate first: its own evidence set, not the first candidate's.
    await page.getByRole("button", { name: /^Inspect / }).nth(1).click();
    const candidate = (await api.state()).tutorial.case.candidates[1].candidate_id;
    await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
    const items = page.locator(".graph-items");
    // S2: comparison lines come only from recorded comparisons; one-sided channels are named, not bridged.
    await expect(items).toContainText(/Features on one side only: .+\(target\)/);
    await expect(page.locator(".graph-items button").filter({ hasText: /correspondence|compared$/ })).toHaveCount(0);
    // Ontology context: add the compared entity's parents and select the added line.
    await items.getByRole("button", { name: /^Compared / }).first().click();
    await page.locator(".graph-inspector").getByRole("button", { name: "Add its parents" }).click();
    await items.locator("section").filter({ hasText: "Added from the ontology" }).getByRole("button").first().click();
    await expect(page.locator(".graph-inspector")).toContainText("Not one of Exact's selected features");
    await page.waitForTimeout(1200);
    expect(saves.filter((save) => save.body.actions?.some((action) => action.action === "inspect_graph_or_list"))).toEqual([]);
    await expect(step(page, "Inspect a line")).toHaveText("To do");
    // A selected feature: its record must belong to this candidate's evidence.
    await items.locator('[data-tour="graph-feature"]').first().click();
    await expect(step(page, "Inspect a line")).toHaveText("Done · saved");
    const sent = saves.find((save) => save.body.actions?.some((action) => action.action === "inspect_graph_or_list"))!;
    expect(sent.status).toBe(200);
    const fact = sent.body.actions!.find((action) => action.action === "inspect_graph_or_list")!.fact_id;
    expect(evidence.filter((item) => item.candidate_id === candidate).flatMap((item) => item.fact_ids)).toContain(fact);
    expect(saves.filter((save) => save.status >= 400)).toEqual([]);
  });
});

test.describe("U02/Q02 rejected evidence recovers without reload", () => {
  test.beforeEach(() => test.skip(!harness, "Set EXACT_E2E_STUDY_V2_CONFIG"));

  test("a rejected action is dropped, never resent, and the next valid action, help, navigation and reload all work", async ({ page }) => {
    const saves = recordProgress(page);
    const api = await seedToTutorial(page);
    await lesson(page, api, "graph");
    const bad = `sha256:${"0".repeat(64)}`;
    // Injected diagnostic: rewrite the first feature evidence so the real server rejects it (422).
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
    await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
    await page.locator('.graph-items [data-tour="graph-feature"]').first().click();
    await expect(page.locator(".coach-notice")).toContainText("Not saved");
    await expect(step(page, "Inspect a line")).toHaveText("To do");
    await expect(page.locator(".study-banner.banner-bad")).toHaveCount(0);
    expect(saves.find((save) => save.status === 422)).toBeTruthy();
    // The next valid action saves; nothing later carries the rejected record.
    await page.locator('.graph-items [data-tour="graph-feature"]').first().click();
    await expect(step(page, "Inspect a line")).toHaveText("Done · saved");
    await page.getByRole("button", { name: "Help", exact: true }).click();
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: /^Add .* as rank 1$/ }).first().click();
    await expect(step(page, "Change your practice answer")).toHaveText("Done · saved");
    const after = saves.slice(saves.findIndex((save) => save.status === 422) + 1);
    expect(after.length).toBeGreaterThan(1);
    expect(after.every((save) => save.status === 200)).toBe(true);
    expect(JSON.stringify(after.map((save) => save.body))).not.toContain(bad);
    await page.getByRole("button", { name: "Next lesson", exact: true }).click();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 5 of / })).toBeFocused();
    await page.reload();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 5 of / })).toBeVisible();
    const progress = (await api.state()).tutorial_progress;
    expect(progress.completed_requirements).toEqual(expect.arrayContaining(["graph.inspect", "graph.view", "graph.rank"]));
  });

  test("offline evidence shows Saving… without a tick and saves after reconnecting", async ({ page, context }) => {
    const api = await seedToTutorial(page);
    await lesson(page, api, "evidence");
    await context.setOffline(true);
    await page.locator('[data-tour="axiom"]').first().click();
    await expect(step(page, "Show an original axiom")).toHaveText("Saving…");
    await expect(page.locator(".lesson-checklist li.saving .check-mark svg")).toHaveCount(0);
    await context.setOffline(false);
    await expect(step(page, "Show an original axiom")).toHaveText("Done · saved");
  });
});

test.describe("U03/Q03 lesson 2: the instructed return is possible and acknowledged", () => {
  test.beforeEach(() => test.skip(!harness, "Set EXACT_E2E_STUDY_V2_CONFIG"));

  test("search → a child that is the compared concept counts as the return; Back appears whenever focus is away", async ({ page }) => {
    const saves = recordProgress(page);
    const api = await seedToTutorial(page);
    const tutorial = await lesson(page, api, "context");
    await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
    const source = page.getByRole("region", { name: "Source ontology browser" });
    await expect(source.getByText("Viewing compared concept")).toBeVisible();
    await source.getByRole("combobox").fill("crate");
    await source.getByRole("listbox").getByRole("option", { name: /^crate/ }).first().click();
    await expect(source.locator(".focus-label")).toHaveText("crate");
    await expect(source.getByRole("button", { name: "Back to compared concept" })).toBeVisible();
    // The compared concept is one of crate's children: reaching it is the return.
    await source.locator(".child-list li").filter({ hasText: tutorial.case.source_label }).getByRole("button").first().click();
    await expect(source.getByText("Viewing compared concept")).toBeVisible();
    await expect(step(page, "Return to the compared entity")).toHaveText("Done · saved");
    await expect(step(page, "Open a child")).toHaveText("Done · saved");
    const ret = saves.flatMap((save) => save.body.actions ?? []).find((action) => action.action === "return_to_compared")!;
    expect(ret.entity?.iri).toBe(tutorial.case.source.iri);
    await source.locator(".tree .tree-label").first().click();
    await expect(step(page, "Open a parent")).toHaveText("Done · saved");
    // The target side offers the same way back.
    const target = page.getByRole("region", { name: "Target ontology browser" });
    await target.locator(".tree .tree-label").first().click();
    await target.getByRole("button", { name: "Back to compared concept" }).click();
    await expect(target.getByText("Viewing compared concept")).toBeVisible();
    expect(saves.filter((save) => save.status >= 400)).toEqual([]);
  });
});

test.describe("F02 guided task", () => {
  test.beforeEach(() => test.skip(!harness, "Set EXACT_E2E_STUDY_V2_CONFIG"));

  test("Show me reveals and focuses the control without doing the step", async ({ page }) => {
    const saves = recordProgress(page);
    const api = await seedToTutorial(page);
    await lesson(page, api, "identity");
    await expect(page.locator("#coach-task")).toHaveText("Inspect a candidate other than the first");
    await page.locator(".coach").getByRole("button", { name: "Show me" }).click();
    const focused = page.locator(":focus");
    await expect(focused).toHaveAttribute("data-tour", "inspect");
    await expect(focused).not.toHaveAttribute("data-position", "1");
    await expect(focused).toHaveClass(/tour-target/);
    await expect(focused).toBeInViewport();
    await page.waitForTimeout(800);
    expect(saves.filter((save) => save.body.actions?.length)).toEqual([]);
    await expect(step(page, "Inspect a candidate other than the first")).toHaveText("To do");
    // In lesson 2 the step opens Hierarchy and focuses a parent link; the coach stays visible.
    await lesson(page, api, "context");
    await page.locator(".lesson-checklist").getByRole("button", { name: "Choose step: Open a parent" }).click();
    await page.locator(".coach").getByRole("button", { name: "Show me" }).click();
    await expect(page.locator(":focus")).toHaveAttribute("data-tour", /^parent-/);
    await expect(page.locator(":focus")).toBeInViewport();
    await expect(page.locator(".coach")).toBeInViewport();
    await expect(step(page, "Open a parent")).toHaveText("To do");
    // A step whose control does not exist here says so specifically (the compared concept is a leaf).
    await page.locator(".lesson-checklist").getByRole("button", { name: "Choose step: Open a child" }).click();
    await page.locator(".coach").getByRole("button", { name: "Show me" }).click();
    await expect(page.locator(".coach-missing")).toContainText("has no children");
  });
});

test.describe("U05/Q04 explicit navigation reveals and focuses its destination", () => {
  const sizes = [
    { name: "1280×720", viewport: { width: 1280, height: 720 } },
    { name: "1440×900", viewport: { width: 1440, height: 900 } },
    { name: "375×812", viewport: { width: 375, height: 812 } },
  ];
  for (const size of sizes)
    test(`Open decision trace, card parent and Show in graph at ${size.name}`, async ({ page }) => {
      await page.setViewportSize(size.viewport);
      await page.emulateMedia({ reducedMotion: "reduce" });
      await page.goto("/");
      await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible();
      await page.getByRole("button", { name: "Open decision trace", exact: true }).click();
      const heading = page.locator(".trace-heading");
      await expect(heading).toBeFocused();
      await expect(heading).toBeInViewport();
      // Again, from the top: it still reveals.
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.getByRole("button", { name: "Open decision trace", exact: true }).click();
      await expect(heading).toBeInViewport();
      await expect(heading).toBeFocused();
      await expect(page.getByRole("heading", { name: "How this pair was decided" })).toBeVisible();
      // A card parent opens Hierarchy at that entity.
      await page.evaluate(() => window.scrollTo(0, 0));
      const parent = page.locator(".entity-card-target .parent-link").first();
      await parent.scrollIntoViewIfNeeded();
      await parent.click();
      const focus = page.locator(".browser-target .focus-card");
      await expect(focus).toBeInViewport();
      await expect(focus).toBeFocused();
      // Show in graph from the evidence list reveals the graph's selection.
      await page.getByRole("tab", { name: "Evidence", exact: true }).click();
      await page.getByRole("button", { name: "Show in graph" }).first().click();
      await expect(page.locator(".graph-inspector")).toBeInViewport();
      await expect(page.locator(".graph-inspector")).toContainText("feature");
    });

  test("at 200% text the trace still reveals below the sticky header", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible();
    for (let index = 0; index < 5; index += 1) await page.getByRole("button", { name: "Increase text size" }).click();
    await expect(page.locator("html")).toHaveAttribute("style", /--text-scale:\s*2/);
    await page.getByRole("button", { name: "Open decision trace", exact: true }).click();
    const heading = page.locator(".trace-heading");
    await expect(heading).toBeFocused();
    await expect(heading).toBeInViewport();
    const header = await page.locator(".app-header").boundingBox();
    const box = await heading.boundingBox();
    expect(box!.y).toBeGreaterThanOrEqual(header!.y + header!.height - 1);
  });
});
