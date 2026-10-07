import { readFileSync } from "node:fs";
import { join } from "node:path";

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type Route } from "@playwright/test";

import { chooseContinuedParent, exhaustBrowserParents, exhaustPagedCard, TYPED } from "./factPagination";
import { harness, newInvitation, ParticipantApi, publishVariant, recordTraffic, scopedPath, seedAdvance, seedCondition, seedToCase, seedToTutorial } from "./v2Harness";

// Integration regressions for specs 17–19 against the real v2 service (HTTPS, PostgreSQL,
// compiled frontend, normal CSP). Sessions here are seeded through the API to reach fault
// scenarios quickly; the unseeded journey is study-v2-backend.spec.ts. Every injected
// failure says so in its body; a "diagnostic" interception modifies a real response and is
// fault evidence only, never acceptance of the happy path.
test.use({ screenshot: "off", trace: "off", actionTimeout: 20_000, viewport: { width: 1280, height: 900 } });
test.beforeEach(() => test.skip(!harness, "Set EXACT_E2E_STUDY_V2_CONFIG for the PostgreSQL/HTTPS harness"));

const INJECTED = { code: "injected_test_failure", message: "Injected by the browser test; not a service response", retryable: true };
const fail = (status = 503) => (route: Route) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(INJECTED) });
const delay = (ms: number) => async (route: Route) => {
  await new Promise((resolve) => setTimeout(resolve, ms));
  await route.continue().catch(() => undefined);
};
const add1 = (page: Page) => page.getByRole("button", { name: /^Add .* as rank 1$/ }).first();

async function savedIndicator(page: Page) {
  await expect(page.locator(".save-indicator")).toContainText("Saved");
}

function caseReady(traffic: ReturnType<typeof recordTraffic>) {
  return traffic.requests.filter((item) => item.path === "/api/v1/study/events" && JSON.stringify(item.body).includes('"case_ready"'));
}

/** The service's answer to the next event batch carrying case_ready; register it before the action. */
function caseReadyAcknowledged(page: Page) {
  return page
    .waitForResponse((response) => new URL(response.url()).pathname === "/api/v1/study/events" && response.request().method() === "POST" && (response.request().postData() ?? "").includes('"case_ready"'), { timeout: 20_000 })
    .then(async (response) => {
      const events = (response.request().postDataJSON() as { events: { event_id: string; type: string }[] }).events;
      const acknowledged = response.ok() ? ((await response.json()) as { acknowledged_event_ids: string[] }).acknowledged_event_ids : [];
      return { status: response.status(), sent: events.filter((event) => event.type === "case_ready").map((event) => event.event_id), acknowledged };
    });
}

function caseSegments(traffic: ReturnType<typeof recordTraffic>) {
  return traffic.requests.filter((item) => item.path === "/api/v1/study/timing" && (item.body as { stage?: string })?.stage === "case");
}

const routeFor = (scope: string, suffix: string, match: (params: URLSearchParams) => boolean = () => true) => (url: URL) =>
  url.pathname === `${scopedPath(scope)}${suffix}` && match(url.searchParams);

async function explanationCase(page: Page, revision?: string) {
  const api = await seedToCase(page, revision);
  const current = await seedCondition(page, api, "explanation");
  const byPosition = (n: number) => current.candidates.find((item: { display_position: number }) => item.display_position === n);
  return { api, current, scope: current.workspace.scope_id as string, byPosition };
}

test.describe("R03 tutorial position (J06)", () => {
  test("lesson 1 → 2 then reload without any lesson-2 action restores lesson 2, with focus on the new heading", async ({ page }) => {
    const api = await seedToTutorial(page);
    await page.reload();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 1 of / })).toBeVisible();
    await page.getByRole("button", { name: "Next lesson", exact: true }).click();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of / })).toBeFocused();
    await savedIndicator(page);
    const state = await api.state();
    expect(state.tutorial_progress.position).toEqual({ view: "lesson", lesson_id: state.tutorial.lessons[1].lesson_id, question_id: null });
    await page.reload();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of / })).toBeVisible();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of / })).not.toBeFocused();
  });

  test("the assessment landing, a partial draft and a focused question all survive reload", async ({ page }) => {
    const api = await seedToTutorial(page);
    await page.reload();
    await page.getByRole("button", { name: /Five short questions/ }).click();
    await expect(page.getByRole("heading", { level: 1, name: "Check your understanding" })).toBeFocused();
    await savedIndicator(page);
    expect((await api.state()).tutorial_progress.position).toEqual({ view: "assessment", lesson_id: null, question_id: null });
    await page.reload();
    await expect(page.getByRole("heading", { level: 1, name: "Check your understanding" })).toBeVisible();
    // A partial (draft) answer is restored; a draft never changes the position.
    const items = page.locator(".assessment-item");
    await items.nth(2).locator("fieldset.match-row").nth(0).getByRole("radio", { name: "None of these" }).check();
    await savedIndicator(page);
    expect((await api.state()).tutorial_progress.position).toEqual({ view: "assessment", lesson_id: null, question_id: null });
    // Deliberate question navigation moves focus and is saved as the destination.
    const third = await page.locator(".question-nav button").nth(2).textContent();
    await page.locator(".question-nav button").nth(2).click();
    await expect(items.nth(2).getByRole("heading", { level: 2 })).toBeFocused();
    await savedIndicator(page);
    const state = await api.state();
    expect(state.tutorial_progress.position).toEqual({ view: "assessment", lesson_id: null, question_id: state.tutorial.assessment[2].question_id });
    await page.reload();
    await expect(items.nth(2).locator("fieldset.match-row").nth(0).getByRole("radio", { name: "None of these" })).toBeChecked();
    await expect(page.locator(".question-nav button").nth(2)).toHaveAttribute("aria-current", "location");
    expect(third).toContain("3 ·");
    await expect(items.nth(2)).toBeInViewport();
  });

  test("an incorrect attempt gives feedback, a retry passes on the server, and both receipts survive reload", async ({ page }) => {
    const api = await seedToTutorial(page);
    await page.reload();
    await page.getByRole("button", { name: /Five short questions/ }).click();
    const first = page.locator(".assessment-item").first();
    await first.getByRole("button", { name: "Check answer" }).click();
    await expect(first.getByRole("alert")).toBeFocused();
    await first.getByRole("radio", { name: "It must be equivalent." }).check();
    await first.getByRole("button", { name: "Check answer" }).click();
    await expect(first).toContainText("Not quite");
    await first.getByRole("radio", { name: /The matcher suggests it/ }).check();
    await first.getByRole("button", { name: "Check again" }).click();
    await expect(first.getByText("Passed")).toBeVisible();
    await page.reload();
    await expect(page.locator(".assessment-item").first().getByText("Passed")).toBeVisible();
    const progress = (await api.state()).tutorial_progress;
    expect(progress.attempts.map((attempt: { correct: boolean }) => attempt.correct)).toEqual([false, true]);
    expect(progress.position.view).toBe("assessment");
  });

  test("rapid Next/Back while saves are delayed keeps the last destination, on screen and on the server", async ({ page }) => {
    const api = await seedToTutorial(page);
    await page.reload();
    await page.route("**/api/v1/study/tutorial/progress", delay(1200));
    await page.getByRole("button", { name: "Next lesson", exact: true }).click();
    await page.getByRole("button", { name: "Next lesson", exact: true }).click();
    await page.getByRole("button", { name: "Back", exact: true }).click();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of / })).toBeVisible();
    await expect.poll(async () => (await api.state()).tutorial_progress.position.lesson_id, { timeout: 20_000 }).toBe((await api.state()).tutorial.lessons[1].lesson_id);
    await savedIndicator(page);
    await page.unroute("**/api/v1/study/tutorial/progress");
    await page.reload();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of / })).toBeVisible();
  });

  test("navigation while offline is kept in this tab and saved after reconnecting", async ({ page, context }) => {
    const api = await seedToTutorial(page);
    await page.reload();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 1 of / })).toBeVisible();
    await context.setOffline(true);
    await page.getByRole("button", { name: "Next lesson", exact: true }).click();
    await expect(page.locator(".save-indicator")).toContainText("Offline");
    expect((await (await fetchState(page)).json()).tutorial_progress.position.lesson_id).not.toBe("context");
    await context.setOffline(false);
    await savedIndicator(page);
    const state = await api.state();
    expect(state.tutorial_progress.position).toEqual({ view: "lesson", lesson_id: state.tutorial.lessons[1].lesson_id, question_id: null });
  });

  test("a stale second tab adopts the server's position after its conflicting navigation", async ({ page, context }) => {
    const api = await seedToTutorial(page);
    await page.reload();
    const other = await context.newPage();
    await other.goto(`${harness!.origin}/participate/`);
    await expect(other.getByRole("heading", { level: 1, name: /^Lesson 1 of / })).toBeVisible();
    await page.getByRole("button", { name: "Next lesson", exact: true }).click();
    await page.getByRole("button", { name: "Next lesson", exact: true }).click();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 3 of / })).toBeVisible();
    await savedIndicator(page);
    await other.getByRole("button", { name: "Next lesson", exact: true }).click();
    await expect(other.getByRole("alert").filter({ hasText: /changed in another tab/ })).toBeVisible();
    await expect(other.getByRole("heading", { level: 1, name: /^Lesson 3 of / })).toBeVisible();
    const state = await api.state();
    expect(state.tutorial_progress.position.lesson_id).toBe(state.tutorial.lessons[2].lesson_id);
    await other.close();
  });

  test("opening tutorial help during a scored case keeps completion, assignment and the case", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { api } = await explanationCase(page);
    const before = await api.state();
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    // Background label lookups for the cards may still be finishing; mark after they settle.
    await page.waitForLoadState("networkidle");
    const mark = traffic.requests.length;
    await page.getByRole("button", { name: "Tutorial help", exact: true }).click();
    await expect(page.getByRole("dialog", { name: "Tutorial help" })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("button", { name: "Tutorial help", exact: true })).toBeFocused();
    const after = await api.state();
    expect(after.stage).toBe("case");
    expect(after.current_presentation_id).toBe(before.current_presentation_id);
    expect(after.assignment_id).toBe(before.assignment_id);
    expect(after.tutorial_progress.completed_at).toBe(before.tutorial_progress.completed_at);
    expect(traffic.requests.slice(mark).filter((item) => item.path.startsWith("/api/v1/study/tutorial") || item.path.includes("/workspace/"))).toEqual([]);
    await expect(add1(page)).toBeEnabled();
  });
});

async function fetchState(page: Page) {
  return page.request.get(`${harness!.origin}/api/v1/study/state`);
}

test.describe("R01/R02 scored explanation case (J01, J04, J05)", () => {
  test("the scored case reads its scoped workspace: six contexts, admitted descriptions and comparisons, then one case_ready", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { current, scope } = await explanationCase(page);
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    const reads = traffic.requests.filter((item) => item.path.startsWith(scopedPath(scope)));
    expect(reads.some((item) => item.path.startsWith(`${scopedPath(scope)}/capabilities`) && item.status === 200)).toBe(true);
    const contextIris = reads.filter((item) => item.path.startsWith(`${scopedPath(scope)}/entity-context`)).map((item) => new URL(item.path, harness!.origin).searchParams.get("iri"));
    for (const entity of [current.source, ...current.candidates.map((item: { entity: unknown }) => item.entity)] as { iri: string }[]) expect(contextIris).toContain(entity.iri);
    expect(new Set(contextIris).size).toBe(contextIris.length);
    expect(traffic.requests.filter((item) => /\/api\/v1\/study\/resources\/case-/.test(item.path))).toEqual([]);
    await expect.poll(() => caseReady(traffic).length).toBe(1);
    // Rapid candidate switches neither restart the case nor refetch the validated contexts.
    const mark = traffic.requests.length;
    for (const button of await page.getByRole("button", { name: /^Inspect / }).all()) await button.click();
    await page.getByRole("button", { name: /^Inspect / }).first().click();
    await page.waitForTimeout(800);
    expect(traffic.requests.slice(mark).filter((item) => item.path.includes("/entity-context"))).toEqual([]);
    expect(caseReady(traffic).length).toBe(1);
    expect(traffic.errors).toEqual([]);
  });

  test("a v2 explanation case without its workspace descriptor is blocked, never served from prepared excerpts", async ({ page }) => {
    const traffic = recordTraffic(page);
    await explanationCase(page);
    // Diagnostic interception: the real response with its descriptor removed.
    await page.route("**/api/v1/study/cases/current", async (route) => {
      const response = await route.fetch();
      const body = await response.json();
      delete body.workspace;
      await route.fulfill({ response, json: body });
    });
    await page.reload();
    await expect(page.getByRole("alert").filter({ hasText: /study service/ })).toBeVisible();
    await expect(add1(page)).toBeDisabled();
    await page.waitForTimeout(1000);
    expect(traffic.requests.filter((item) => /\/api\/v1\/study\/resources\/case-/.test(item.path) || item.path.includes("/workspace/"))).toEqual([]);
    expect(caseReady(traffic)).toEqual([]);
  });

  test("capabilities for another scope are incompatible and block the case", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { scope } = await explanationCase(page);
    await page.route(routeFor(scope, "/capabilities"), async (route) => {
      const response = await route.fetch();
      await route.fulfill({ response, json: { ...(await response.json()), scope_id: "diagnostic-other-scope" } });
    });
    await page.reload();
    await expect(page.getByRole("alert").filter({ hasText: /different scope/ })).toBeVisible();
    await expect(add1(page)).toBeDisabled();
    expect(traffic.requests.filter((item) => item.path.includes("/entity-context"))).toEqual([]);
    expect(caseReady(traffic)).toEqual([]);
  });

  test("a target-context failure blocks ranking and case_ready until a retry succeeds, reusing valid reads", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { scope, byPosition } = await explanationCase(page);
    const pattern = routeFor(scope, "/entity-context", (params) => params.get("iri") === byPosition(3).entity.iri);
    await page.route(pattern, fail());
    await page.reload();
    await expect(page.getByRole("alert").filter({ hasText: /Ontology information for candidate 3/ })).toBeVisible();
    await expect(add1(page)).toBeDisabled();
    await expect(page.getByRole("radio", { name: /^Insufficient information/ })).toBeDisabled();
    const audit = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
    expect(audit.violations.map((item) => item.id), "blocked case state").toEqual([]);
    // The retry follows the case toolbar in keyboard order.
    await page.getByRole("button", { name: "Tutorial help", exact: true }).focus();
    let reached = false;
    for (let step = 0; step < 6 && !reached; step += 1) {
      await page.keyboard.press("Tab");
      reached = await page.getByRole("button", { name: "Retry loading this case" }).evaluate((node) => node === document.activeElement);
    }
    expect(reached).toBe(true);
    await page.waitForTimeout(1200);
    expect(caseReady(traffic)).toEqual([]);
    expect(caseSegments(traffic)).toEqual([]);
    await page.unroute(pattern);
    const mark = traffic.requests.length;
    await page.getByRole("button", { name: "Retry loading this case" }).click();
    await expect(add1(page)).toBeEnabled();
    await expect.poll(() => caseReady(traffic).length).toBe(1);
    const retried = traffic.requests.slice(mark).filter((item) => item.path.includes("/entity-context"));
    expect(retried.map((item) => new URL(item.path, harness!.origin).searchParams.get("iri"))).toEqual([byPosition(3).entity.iri]);
  });

  test("a slow fifth candidate keeps the case loading; case_ready and case timing start only after it", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { scope, byPosition } = await explanationCase(page);
    let released = 0;
    await page.route(routeFor(scope, "/entity-context", (params) => params.get("iri") === byPosition(5).entity.iri), async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 3000));
      released = Date.now();
      await route.continue();
    });
    const started = Date.now();
    await page.reload();
    await expect(page.getByRole("status").filter({ hasText: /Loading this case's information/ })).toBeVisible();
    await expect(add1(page)).toBeDisabled();
    await expect(add1(page)).toBeEnabled({ timeout: 20_000 });
    expect(released).toBeGreaterThan(started);
    await expect.poll(() => caseReady(traffic).length).toBe(1);
    expect(caseSegments(traffic)).toEqual([]);
    expect(traffic.errors).toEqual([]);
  });

  test("a comparison failure blocks with its own label; a profile failure likewise", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { scope, byPosition, current } = await explanationCase(page);
    const comparison = routeFor(scope, "/explanations", (params) => params.get("task") === "pair_comparison" && params.get("counterpart_iri") === byPosition(2).entity.iri);
    await page.route(comparison, fail());
    await page.reload();
    await expect(page.getByRole("alert").filter({ hasText: /Comparison with candidate 2/ })).toBeVisible();
    await expect(add1(page)).toBeDisabled();
    await page.unroute(comparison);
    const profile = routeFor(scope, "/explanations", (params) => params.get("task") === "entity_profile" && params.get("iri") === current.source.iri);
    await page.route(profile, fail(500));
    await page.reload();
    await expect(page.getByRole("alert").filter({ hasText: /Generated description of the source concept/ })).toBeVisible();
    await page.unroute(profile);
    const acknowledged = caseReadyAcknowledged(page);
    await page.getByRole("button", { name: "Retry loading this case" }).click();
    await expect(add1(page)).toBeEnabled();
    // The UI can recover before the service acknowledges case_ready: wait for that response.
    const ack = await acknowledged;
    expect(ack.status).toBe(200);
    expect(ack.sent).toHaveLength(1);
    expect(ack.acknowledged).toEqual(expect.arrayContaining(ack.sent));
    await expect.poll(() => caseReady(traffic).map((item) => item.status)).toEqual([200]);
  });

  test("an authorized, recorded absence is a usable terminal status, not a failure (diagnostic)", async ({ page }) => {
    const { scope, byPosition } = await explanationCase(page);
    // Diagnostic: the service's own terminal-absence shape for one comparison.
    const absence = JSON.parse(readFileSync(join(__dirname, "../../docs/verification/explanation-integration-backend-examples.json"), "utf8")).examples.terminal_absence.body;
    await page.route(routeFor(scope, "/explanations", (params) => params.get("task") === "pair_comparison" && params.get("counterpart_iri") === byPosition(1).entity.iri), (route) => route.fulfill({ status: 200, json: absence }));
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    await expect(page.getByRole("alert").filter({ hasText: /could not be loaded/ })).toHaveCount(0);
  });

  test("a 200 response describing another entity is an invalid binding and blocks readiness", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { scope, byPosition } = await explanationCase(page);
    await page.route(routeFor(scope, "/entity-context", (params) => params.get("iri") === byPosition(2).entity.iri), async (route) => {
      const response = await route.fetch();
      await route.fulfill({ response, json: { ...(await response.json()), entity: byPosition(1).entity } });
    });
    await page.reload();
    await expect(page.getByRole("alert").filter({ hasText: /different entity/ })).toBeVisible();
    await expect(add1(page)).toBeDisabled();
    expect(caseReady(traffic)).toEqual([]);
  });

  test("an optional hierarchy failure stays local and does not block submission", async ({ page }) => {
    const { scope } = await explanationCase(page);
    await page.route(routeFor(scope, "/hierarchy"), fail());
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
    await expect(page.getByRole("region", { name: "Source ontology browser" }).getByRole("alert").first()).toBeVisible();
    await page.getByRole("radio", { name: /^Insufficient information/ }).click();
    await expect(page.getByRole("button", { name: "Submit answer", exact: true })).toBeEnabled();
  });

  test("a blocking failure keeps a saved partial ranking and its order through retry", async ({ page }) => {
    const { api, current, scope, byPosition } = await explanationCase(page);
    await api.write("PUT", `/cases/${encodeURIComponent(current.case_id)}/draft`, { presentation_id: current.presentation_id, response_type: "ranked_candidates", ranked_candidate_ids: [byPosition(3).candidate_id, byPosition(1).candidate_id] });
    const pattern = routeFor(scope, "/entity-context", (params) => params.get("iri") === byPosition(2).entity.iri);
    await page.route(pattern, fail());
    await page.reload();
    await expect(page.getByRole("alert").filter({ hasText: /could not be loaded/ })).toBeVisible();
    const ranking = page.getByRole("list", { name: "Your ranking" });
    await expect(ranking.locator("li")).toHaveCount(2);
    await expect(ranking.locator("li").first()).toContainText(byPosition(3).label);
    await page.unroute(pattern);
    await page.getByRole("button", { name: "Retry loading this case" }).click();
    await expect(page.getByRole("button", { name: "Submit answer", exact: true })).toBeEnabled();
    await expect(ranking.locator("li").first()).toContainText(byPosition(3).label);
    await expect(ranking.locator("li").nth(1)).toContainText(byPosition(1).label);
  });

  test("replacing the session aborts the old case: nothing from it becomes ready", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { scope, current } = await explanationCase(page);
    await page.route(routeFor(scope, "/entity-context"), delay(4000));
    await page.reload();
    await expect(page.getByRole("status").filter({ hasText: /Loading this case's information/ })).toBeVisible();
    const invitation = await newInvitation(page);
    await page.evaluate((hash) => { window.location.hash = hash; }, invitation.split("#")[1]);
    await expect(page.getByRole("button", { name: "I agree to take part", exact: true })).toBeVisible();
    await page.waitForTimeout(5000);
    expect(caseReady(traffic).filter((item) => JSON.stringify(item.body).includes(current.presentation_id))).toEqual([]);
    expect(traffic.errors).toEqual([]);
  });

  test("a second tab loads its own readiness and reports its own ready observation (J05)", async ({ page, context }) => {
    const traffic = recordTraffic(page);
    await explanationCase(page);
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    const other = await context.newPage();
    const otherTraffic = recordTraffic(other);
    await other.goto(`${harness!.origin}/participate/`);
    await other.getByRole("button", { name: /Continue|Resume|Back to the case/ }).first().click({ timeout: 3000 }).catch(() => undefined);
    await expect(add1(other)).toBeEnabled();
    await expect.poll(() => caseReady(otherTraffic).length).toBe(1);
    const pages = new Set([...caseReady(traffic), ...caseReady(otherTraffic)].flatMap((item) => (item.body as { events: { page_instance_id: string; type: string }[] }).events.filter((event) => event.type === "case_ready").map((event) => event.page_instance_id)));
    expect(pages.size).toBe(2);
    await other.close();
  });

  test("pause and resume never reopen case timing before the content is usable again", async ({ page }) => {
    const traffic = recordTraffic(page);
    await explanationCase(page);
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    await page.getByRole("button", { name: "Pause", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Paused" })).toBeVisible();
    const mark = traffic.requests.length;
    await page.getByRole("button", { name: "Resume", exact: true }).click();
    await expect(add1(page)).toBeEnabled();
    await expect.poll(() => caseReady({ ...traffic, requests: traffic.requests.slice(mark) }).length).toBe(1);
  });
});

test.describe("J02 participant boundaries (browser subset)", () => {
  test("a baseline case reads no workspace, and a previous explanation scope is denied after advancing", async ({ page }) => {
    const traffic = recordTraffic(page);
    const api = await seedToCase(page);
    const checkBaseline = async (baseline: Record<string, any>) => {
      expect(baseline.workspace).toBeNull();
      expect(baseline.explanation_refs).toEqual([]);
      const mark = traffic.requests.length;
      await page.reload();
      await expect(add1(page)).toBeEnabled();
      await expect(page.getByRole("tab")).toHaveCount(0);
      await page.getByRole("button", { name: "Tutorial help", exact: true }).click();
      await page.keyboard.press("Escape");
      expect(traffic.requests.slice(mark).filter((item) => item.path.includes("/workspace/") || /\/study\/resources\/case-/.test(item.path))).toEqual([]);
    };
    // Schedules differ between sessions: handle a baseline before or after the explanation case.
    let current = await api.currentCase();
    let baselineChecked = false;
    if (current.condition === "ontology_baseline") {
      await checkBaseline(current);
      baselineChecked = true;
      current = await seedCondition(page, api, "explanation");
    }
    const scope = current.workspace.scope_id as string;
    await seedAdvance(api, current);
    const state = await api.state();
    const stale = await page.request.get(`${harness!.origin}${scopedPath(scope)}/capabilities`, { headers: { "X-Study-Session": state.session_id } });
    expect(stale.status()).toBe(403);
    if (!baselineChecked) await checkBaseline(await seedCondition(page, api, "ontology_baseline"));
  });
});

test.describe("J08 researcher export (browser)", () => {
  async function signIn(page: Page) {
    await page.goto(`${harness!.origin}/admin/`);
    await page.getByLabel("Researcher token").fill(harness!.researcher_token);
    await page.getByRole("button", { name: "Continue", exact: true }).click();
    await page.getByRole("textbox", { name: "Study revision" }).fill(harness!.study_revision);
  }

  test("v2 exports request and verify analysis 3 in JSON and CSV", async ({ page }) => {
    const exports: string[] = [];
    page.on("request", (request) => {
      if (request.url().includes("/exports")) exports.push(new URL(request.url()).search);
    });
    await signIn(page);
    await page.getByRole("checkbox", { name: "Include test sessions" }).check();
    const [json] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Create export" }).click()]);
    await expect(page.getByRole("status").filter({ hasText: /derived schema exact-study-analysis\/3/ })).toBeVisible();
    const data = JSON.parse(readFileSync((await json.path())!, "utf8"));
    expect(data.manifest.schema).toBe("exact-study-analysis/3");
    expect(data.manifest.source_protocol_versions).toBeTruthy();
    const timing = data.data.sessions.flatMap((session: { cases: { timing?: Record<string, unknown> }[] }) => session.cases.map((row) => row.timing)).filter(Boolean);
    for (const summary of timing) expect(summary).toMatchObject({ coverage_status: "not_established", unique_elapsed_coverage_seconds: null, unobserved_elapsed_seconds: null });
    await page.getByRole("radio", { name: "CSV archive" }).click();
    const [csv] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Create export" }).click()]);
    await expect(page.getByRole("status").filter({ hasText: /exact-study-csv\/3/ })).toBeVisible();
    expect((await csv.suggestedFilename()).endsWith(".zip")).toBe(true);
    expect(exports.every((search) => search.includes("analysis_schema=exact-study-analysis%2F3"))).toBe(true);
  });

  test("a returned schema other than analysis 3 is reported and not saved (diagnostic)", async ({ page }) => {
    await signIn(page);
    await page.route("**/exports?*", async (route) => {
      const response = await route.fetch();
      const body = await response.json();
      await route.fulfill({ response, json: { ...body, manifest: { ...body.manifest, schema: "exact-study-analysis/2" } } });
    });
    let downloaded = false;
    page.on("download", () => (downloaded = true));
    await page.getByRole("button", { name: "Create export" }).click();
    await expect(page.getByRole("alert").filter({ hasText: /returned exact-study-analysis\/2 instead of exact-study-analysis\/3/ })).toBeVisible();
    expect(downloaded).toBe(false);
  });
});

test.describe("J10 layout and accessibility on the real service", () => {
  for (const scheme of ["light", "dark"] as const)
    test(`setup, a tutorial lesson and an explanation case reflow from 320 to 2560 px at 100% and 200% text (${scheme})`, async ({ page }) => {
      test.setTimeout(300_000);
      await page.emulateMedia({ colorScheme: scheme });
      const metrics: string[] = [];
      const check = async (stage: string, control: RegExp | string) => {
        for (const scale of [1, 2]) {
          await page.evaluate((value) => localStorage.setItem("exact.textScale", String(value)), scale);
          await page.reload();
          await expect(page.getByRole("button", { name: control }).first()).toBeVisible();
          for (const width of [320, 390, 768, 1280, 2560]) {
            await page.setViewportSize({ width, height: 800 });
            // The layout reacts to the resize asynchronously; measure once it has settled.
            await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), { message: `${stage} overflow at ${width}px ${scale}x` }).toBe(true);
            const actual = await page.evaluate(() => ({ inner: innerWidth, scroll: document.documentElement.scrollWidth, font: getComputedStyle(document.documentElement).fontSize }));
            metrics.push(`${stage} ${scheme} ${scale}x requested ${width} → innerWidth ${actual.inner}, scrollWidth ${actual.scroll}, root font ${actual.font}`);
            expect(actual.inner).toBe(width);
            // Crossing the narrow breakpoint re-mounts the rail; retry until the layout settles.
            await expect(async () => {
              const button = page.getByRole("button", { name: control }).first();
              await button.scrollIntoViewIfNeeded({ timeout: 2000 });
              await expect(button).toBeInViewport({ timeout: 1000 });
            }).toPass({ timeout: 15_000 });
          }
          await page.setViewportSize({ width: 1280, height: 900 });
          const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
          expect(result.violations.map((item) => item.id), `${stage} ${scale}x`).toEqual([]);
        }
      };
      const invitation = await newInvitation(page);
      await page.goto(harness!.origin + invitation);
      const api = new ParticipantApi(page);
      await expect(page.getByRole("button", { name: "I agree to take part", exact: true })).toBeVisible();
      const definition = JSON.parse(readFileSync(harness!.publication, "utf8")).definition;
      await api.write("PUT", "/consent", { information_version: definition.information_version, accepted: true });
      await check("setup", "Continue");
      await page.context().clearCookies();
      await seedToTutorial(page);
      await check("tutorial", /^Add .* as rank 1$/);
      await page.context().clearCookies();
      await explanationCase(page);
      await check("case", /^Add .* as rank 1$/);
      test.info().annotations.push({ type: "viewport-metrics", description: metrics.join("\n") });
    });
});

const submitButton = (page: Page) => page.getByRole("button", { name: "Submit answer", exact: true });
const definitionOpened = (traffic: ReturnType<typeof recordTraffic>, side: string) =>
  traffic.requests.some((item) => item.path === "/api/v1/study/events" && JSON.stringify(item.body).includes('"definition_open"') && JSON.stringify(item.body).includes(`context_${side}`));

async function focusSearched(page: Page, term: string) {
  await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
  const browser = page.getByRole("region", { name: "Source ontology browser" });
  await browser.getByRole("combobox").fill(term);
  await expect(browser.getByRole("listbox").getByRole("option", { name: new RegExp(`^${term}`) })).toBeVisible();
  await browser.getByRole("combobox").press("ArrowDown");
  await browser.getByRole("combobox").press("Enter");
  await expect(browser.locator(".focus-label")).toHaveText(term);
  return browser;
}

test.describe("R06/R07 study full context and fact pagination (19 F16, F17)", () => {
  test("a searched non-focal entity's full context pages every category; closing keeps the case", async ({ page }) => {
    test.setTimeout(120_000);
    const traffic = recordTraffic(page);
    const { api, current, scope, byPosition } = await explanationCase(page);
    await api.write("PUT", `/cases/${encodeURIComponent(current.case_id)}/draft`, { presentation_id: current.presentation_id, response_type: "ranked_candidates", ranked_candidate_ids: [byPosition(3).candidate_id, byPosition(1).candidate_id] });
    await page.reload();
    await expect(submitButton(page)).toBeEnabled();
    await page.getByRole("button", { name: /^Inspect / }).nth(2).click();
    const browser = await focusSearched(page, "Paged facts source");
    await exhaustBrowserParents(browser);
    const mark = traffic.requests.length;
    await browser.getByRole("button", { name: "Open full context", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Full context" });
    await exhaustPagedCard(dialog.locator(".entity-card"), { alternate: false });
    // Facts open as original axioms; no description is read for a non-focal entity.
    await dialog.locator('[data-category="restrictions"]').getByRole("button", { name: "Show original axiom" }).first().click();
    await expect(dialog.locator('[data-category="restrictions"] .original-axiom').first()).toBeVisible();
    await expect(dialog.getByRole("region", { name: "Generated description" })).toHaveCount(0);
    const reads = traffic.requests.slice(mark);
    expect(reads.filter((item) => item.path.includes("/explanations"))).toEqual([]);
    expect(reads.filter((item) => (item.status ?? 0) >= 400)).toEqual([]);
    expect(reads.filter((item) => /\/(entity-facts|hierarchy|entity-context|axioms)/.test(item.path) && !item.path.startsWith(scopedPath(scope)))).toEqual([]);
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    // The ranked pair, inspected candidate, draft and navigation are as they were.
    await expect(browser.locator(".focus-label")).toHaveText("Paged facts source");
    await expect(page.getByText("Inspecting initial position 3 of 5")).toBeVisible();
    const ranking = page.getByRole("list", { name: "Your ranking" });
    await expect(ranking.locator("li").first()).toContainText(byPosition(3).label);
    await expect(ranking.locator("li").nth(1)).toContainText(byPosition(1).label);
    await expect(submitButton(page)).toBeEnabled();
    await expect.poll(() => caseReady(traffic).map((item) => item.status)).toEqual([200]);
    await expect.poll(() => definitionOpened(traffic, "source"), { timeout: 20_000 }).toBe(true);
    expect(traffic.errors).toEqual([]);
  });

  test("a focal entity's full context shows its description; a citation opens and returns; a parent navigates", async ({ page }) => {
    const traffic = recordTraffic(page);
    await explanationCase(page);
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
    const browser = page.getByRole("region", { name: "Source ontology browser" });
    await browser.getByRole("button", { name: "Open full context", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Full context" });
    const citation = dialog.getByRole("region", { name: "Generated description" }).locator("button.citation").first();
    await citation.click();
    const cited = page.getByRole("dialog", { name: /^Cited fact/ });
    await expect(cited.locator(".iri").first()).toHaveText(/^urn:source:\d$/);
    await page.keyboard.press("Escape");
    await expect(cited).toHaveCount(0);
    await expect(citation).toBeFocused();
    await dialog.locator('[data-category="parents"] li').first().getByRole("button").click();
    await expect(dialog).toHaveCount(0);
    await expect(browser.locator(".focus-label")).not.toHaveText(/^Synthetic source \d$/);
    await expect(add1(page)).toBeEnabled();
    expect(traffic.requests.filter((item) => (item.status ?? 0) >= 400)).toEqual([]);
  });
});

test.describe("R08 retry refetches unusable responses (19 F18)", () => {
  test("a malformed 200 context injected once blocks the case; Retry refetches it and restores usability in place", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { api, current, scope, byPosition } = await explanationCase(page);
    await api.write("PUT", `/cases/${encodeURIComponent(current.case_id)}/draft`, { presentation_id: current.presentation_id, response_type: "ranked_candidates", ranked_candidate_ids: [byPosition(4).candidate_id] });
    const target = byPosition(2).entity.iri;
    let injected = 0;
    // Diagnostic interception, once: the real response with a required collection removed.
    await page.route(routeFor(scope, "/entity-context", (params) => params.get("iri") === target), async (route) => {
      if (injected) return route.continue();
      injected += 1;
      const response = await route.fetch();
      const body = await response.json();
      delete body.synonyms;
      await route.fulfill({ response, json: body });
    });
    const contextReads = () => traffic.requests.filter((item) => item.path.startsWith(`${scopedPath(scope)}/entity-context`) && new URL(item.path, harness!.origin).searchParams.get("iri") === target);
    await page.reload();
    await expect(page.getByRole("alert").filter({ hasText: /Ontology information for candidate 2: The response has no synonyms page/ })).toBeVisible();
    await expect(submitButton(page)).toBeDisabled();
    await page.waitForTimeout(800);
    expect(contextReads()).toHaveLength(1);
    expect(caseReady(traffic)).toEqual([]);
    await page.evaluate(() => ((window as unknown as { __sameDocument: boolean }).__sameDocument = true));
    await page.getByRole("button", { name: "Retry loading this case" }).click();
    await expect(submitButton(page)).toBeEnabled();
    expect(contextReads()).toHaveLength(2);
    expect(await page.evaluate(() => (window as unknown as { __sameDocument?: boolean }).__sameDocument)).toBe(true);
    await expect(page.getByRole("list", { name: "Your ranking" }).locator("li").first()).toContainText(byPosition(4).label);
    await expect.poll(() => caseReady(traffic).length).toBe(1);
    expect(traffic.errors).toEqual([]);
  });

  test("a slow required read finishing after another failed cannot unblock the case", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { scope, byPosition } = await explanationCase(page);
    await page.route(routeFor(scope, "/entity-context", (params) => params.get("iri") === byPosition(2).entity.iri), fail());
    await page.route(routeFor(scope, "/entity-context", (params) => params.get("iri") === byPosition(5).entity.iri), delay(2500));
    await page.reload();
    await expect(page.getByRole("alert").filter({ hasText: /Ontology information for candidate 2/ })).toBeVisible();
    await page.waitForTimeout(3500);
    await expect(page.getByRole("alert").filter({ hasText: /Ontology information for candidate 2/ })).toBeVisible();
    await expect(add1(page)).toBeDisabled();
    expect(caseReady(traffic)).toEqual([]);
  });
});

test.describe("Optional reads refused with 403 (19 F18)", () => {
  test("a refused optional read stays local; only a refused scope blocks the case", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { scope } = await explanationCase(page);
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    await page.route(routeFor(scope, "/hierarchy"), fail(403));
    await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
    await expect(page.getByRole("region", { name: "Source ontology browser" }).getByRole("alert").first()).toBeVisible();
    await page.waitForTimeout(1000);
    await expect(add1(page)).toBeEnabled();
    expect(traffic.requests.some((item) => item.path === `${scopedPath(scope)}/capabilities` && item.status === 200)).toBe(true);
    // Now the scope itself is refused: the same kind of read means access was lost.
    await page.route(routeFor(scope, "/capabilities"), fail(403));
    await page.getByRole("region", { name: "Source ontology browser" }).getByRole("button", { name: "Try again" }).first().click();
    await expect(page.getByRole("alert").filter({ hasText: /no longer has access/ })).toBeVisible();
    await expect(add1(page)).toBeDisabled();
  });
});

test.describe("R09 readiness follows the admitted components (19 F19)", () => {
  const variants = [
    { suffix: "cmp", components: ["pair_comparison"], cards: 0, comparison: true, profiles: false, tabs: 0 },
    { suffix: "ctx", components: ["original_context"], cards: 2, comparison: false, profiles: false, tabs: 0 },
    { suffix: "desc", components: ["entity_description"], cards: 2, comparison: false, profiles: true, tabs: 0 },
    { suffix: "nav", components: ["hierarchy", "evidence_table", "evidence_graph"], cards: 0, comparison: false, profiles: false, tabs: 3 },
  ] as const;
  for (const variant of variants)
    test(`${variant.components.join(" + ")} only: usable once exactly its content renders; a failure blocks and Retry recovers`, async ({ page }) => {
      const traffic = recordTraffic(page);
      const revision = await publishVariant(page, variant.suffix, [...variant.components]);
      const { scope, byPosition } = await explanationCase(page, revision);
      // First load: one required read fails, nothing is ready, Retry recovers.
      const failing = variant.comparison
        ? routeFor(scope, "/explanations", (params) => params.get("task") === "pair_comparison" && params.get("counterpart_iri") === byPosition(1).entity.iri)
        : variant.profiles
          ? routeFor(scope, "/explanations", (params) => params.get("task") === "entity_profile" && params.get("iri") === byPosition(1).entity.iri)
          : routeFor(scope, "/entity-context", (params) => params.get("iri") === byPosition(1).entity.iri);
      await page.route(failing, fail());
      await page.reload();
      await expect(page.getByRole("alert").filter({ hasText: /could not be loaded/ })).toBeVisible();
      await expect(add1(page)).toBeDisabled();
      expect(caseReady(traffic)).toEqual([]);
      await page.unroute(failing);
      await page.getByRole("button", { name: "Retry loading this case" }).click();
      await expect(add1(page)).toBeEnabled({ timeout: 15_000 });
      await expect.poll(() => caseReady(traffic).length).toBe(1);
      // Exactly the admitted components are shown and awaited; nothing else is read.
      await expect(page.locator(".pair-question")).toBeVisible();
      await expect(page.locator(".card-pair .entity-card")).toHaveCount(variant.cards);
      await expect(page.locator(".comparison")).toHaveCount(variant.comparison ? 1 : 0);
      await expect(page.getByRole("tab")).toHaveCount(variant.tabs);
      await expect(page.locator(".card-pair").getByRole("region", { name: "Generated description" })).toHaveCount(variant.profiles ? 2 : 0);
      const explanations = traffic.requests.filter((item) => item.path.startsWith(`${scopedPath(scope)}/explanations?`));
      expect(explanations.some((item) => item.path.includes("task=entity_profile"))).toBe(variant.profiles);
      expect(explanations.some((item) => item.path.includes("task=pair_comparison"))).toBe(variant.comparison);
      if (variant.cards) await expect(page.locator(".card-pair .entity-card").first().getByRole("heading", { name: "Definition" })).toHaveCount((variant.components as readonly string[]).includes("original_context") ? 1 : 0);
      if (variant.tabs) {
        await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
        // Original context is not admitted, so the browser offers no full-context view.
        await expect(page.getByRole("button", { name: "Open full context" })).toHaveCount(0);
      }
      expect(traffic.errors).toEqual([]);
    });

  test("comparison only: a recorded absence is usable (diagnostic)", async ({ page }) => {
    const revision = await publishVariant(page, "cmp", ["pair_comparison"]);
    const { scope, byPosition } = await explanationCase(page, revision);
    const absence = JSON.parse(readFileSync(join(__dirname, "../../docs/verification/explanation-integration-backend-examples.json"), "utf8")).examples.terminal_absence.body;
    await page.route(routeFor(scope, "/explanations", (params) => params.get("task") === "pair_comparison" && params.get("counterpart_iri") === byPosition(1).entity.iri), (route) => route.fulfill({ status: 200, json: absence }));
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    await expect(page.locator(".comparison")).toBeVisible();
    await expect(page.getByRole("alert").filter({ hasText: /could not be loaded/ })).toHaveCount(0);
  });
});

/** Holds scope capability checks until released, then refuses them (labelled fault injection). */
async function holdChecks(page: Page, scope: string) {
  const releases: (() => void)[] = [];
  const done: Promise<void>[] = [];
  let passthrough = false;
  await page.route(routeFor(scope, "/capabilities"), async (route) => {
    if (passthrough) return route.continue();
    let release!: () => void;
    const gate = new Promise<void>((resolve) => (release = resolve));
    releases.push(release);
    let finished!: () => void;
    done.push(new Promise<void>((resolve) => (finished = resolve)));
    await gate;
    await route.fulfill({ status: 403, json: { detail: "Injected access-check refusal" } }).catch(() => undefined);
    finished();
  });
  return {
    count: () => releases.length,
    release: async (index: number) => {
      releases[index]();
      await done[index];
    },
    passThrough: () => (passthrough = true),
  };
}

test.describe("R10 access checks belong to the attempt that issued them (19 F20)", () => {
  test("a delayed check failure cannot undo a successful retry; the draft and cached contexts survive", async ({ page }) => {
    const traffic = recordTraffic(page);
    const { api, current, scope, byPosition } = await explanationCase(page);
    await api.write("PUT", `/cases/${encodeURIComponent(current.case_id)}/draft`, { presentation_id: current.presentation_id, response_type: "ranked_candidates", ranked_candidate_ids: [byPosition(2).candidate_id, byPosition(5).candidate_id] });
    await page.reload();
    await expect(submitButton(page)).toBeEnabled();
    const checks = await holdChecks(page, scope);
    const optional = routeFor(scope, "/hierarchy");
    await page.route(optional, fail(403));
    await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
    await expect.poll(() => checks.count()).toBeGreaterThanOrEqual(2);
    // The first check is refused within this attempt: a genuine loss, so the case blocks.
    await checks.release(0);
    await expect(page.getByRole("alert").filter({ hasText: /no longer has access/ })).toBeVisible();
    await expect(submitButton(page)).toBeDisabled();
    // Access is restored; Retry recovers without refetching the validated contexts.
    checks.passThrough();
    await page.unroute(optional);
    const contextsBefore = traffic.requests.filter((item) => item.path.includes("/entity-context")).length;
    await page.getByRole("button", { name: "Retry loading this case" }).click();
    await expect(submitButton(page)).toBeEnabled();
    expect(traffic.requests.filter((item) => item.path.includes("/entity-context")).length).toBe(contextsBefore);
    // The older, still-held check now fails: it belongs to the superseded attempt.
    await checks.release(1);
    await page.waitForTimeout(500);
    await expect(page.getByRole("alert").filter({ hasText: /no longer has access/ })).toHaveCount(0);
    await expect(submitButton(page)).toBeEnabled();
    const ranking = page.getByRole("list", { name: "Your ranking" });
    await expect(ranking.locator("li").first()).toContainText(byPosition(2).label);
    await expect(ranking.locator("li").nth(1)).toContainText(byPosition(5).label);
    expect((await api.state()).ranking.ranked_candidate_ids).toEqual([byPosition(2).candidate_id, byPosition(5).candidate_id]);
  });

  test("a check still in flight from a submitted presentation cannot affect the next case", async ({ page }) => {
    const { scope } = await explanationCase(page);
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    const checks = await holdChecks(page, scope);
    await page.route(routeFor(scope, "/hierarchy"), fail(403));
    await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
    await expect.poll(() => checks.count()).toBeGreaterThanOrEqual(1);
    await page.getByRole("radio", { name: /^Insufficient information/ }).click();
    await submitButton(page).click();
    await page.getByRole("radio", { name: "No", exact: true }).check();
    await page.getByRole("button", { name: /^Save and (go to case|continue)/ }).click();
    await expect(page.getByRole("complementary", { name: "Candidates and your answer" })).toBeVisible();
    await expect(add1(page)).toBeEnabled();
    await checks.release(0);
    await page.waitForTimeout(500);
    await expect(page.getByRole("alert").filter({ hasText: /no longer has access/ })).toHaveCount(0);
    await expect(add1(page)).toBeEnabled();
  });

  test("a check still in flight from a replaced session cannot affect the new session", async ({ page }) => {
    const { scope } = await explanationCase(page);
    await page.reload();
    await expect(add1(page)).toBeEnabled();
    const checks = await holdChecks(page, scope);
    await page.route(routeFor(scope, "/hierarchy"), fail(403));
    await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
    await expect.poll(() => checks.count()).toBeGreaterThanOrEqual(1);
    const invitation = await newInvitation(page);
    await page.evaluate((hash) => { window.location.hash = hash; }, invitation.split("#")[1]);
    await expect(page.getByRole("button", { name: "I agree to take part", exact: true })).toBeVisible();
    await checks.release(0);
    await page.waitForTimeout(500);
    await expect(page.getByRole("alert").filter({ hasText: /no longer has access/ })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "I agree to take part", exact: true })).toBeVisible();
  });
});

test.describe("R11 continued parents keep their entity type (19 F21)", () => {
  for (const typed of TYPED)
    test(`study: a ${typed.name}'s later-page superproperty opens as a ${typed.name}; the case is kept`, async ({ page }) => {
      const traffic = recordTraffic(page);
      const { api, current, scope, byPosition } = await explanationCase(page);
      await api.write("PUT", `/cases/${encodeURIComponent(current.case_id)}/draft`, { presentation_id: current.presentation_id, response_type: "ranked_candidates", ranked_candidate_ids: [byPosition(4).candidate_id] });
      await page.reload();
      await expect(submitButton(page)).toBeEnabled();
      const browser = await focusSearched(page, typed.label);
      await expect(browser.getByText(`Focused ${typed.name}`)).toBeVisible();
      await browser.getByRole("button", { name: "Open full context", exact: true }).click();
      const dialog = page.getByRole("dialog", { name: "Full context" });
      const iri = await chooseContinuedParent(dialog, typed.kind);
      await expect(dialog).toHaveCount(0);
      await expect(browser.locator(".focus-label")).toHaveText(typed.parent);
      await expect(browser.getByText(`Focused ${typed.name}`)).toBeVisible();
      await browser.getByRole("button", { name: "Open full context", exact: true }).click();
      await expect(page.getByRole("dialog", { name: "Full context" }).locator(".entity-title")).toHaveText(typed.parent);
      const reads = traffic.requests.filter((item) => item.path.startsWith(scopedPath(scope)) && new URL(item.path, harness!.origin).searchParams.get("iri") === iri);
      expect(reads.some((item) => item.path.includes("/entity-context") && item.status === 200 && new URL(item.path, harness!.origin).searchParams.get("kind") === typed.kind)).toBe(true);
      expect(reads.some((item) => item.path.includes("/hierarchy") && item.status === 200)).toBe(true);
      expect(reads.filter((item) => new URL(item.path, harness!.origin).searchParams.has("kind") && new URL(item.path, harness!.origin).searchParams.get("kind") !== typed.kind)).toEqual([]);
      await page.keyboard.press("Escape");
      await expect(page.getByRole("list", { name: "Your ranking" }).locator("li").first()).toContainText(byPosition(4).label);
      await expect(submitButton(page)).toBeEnabled();
      expect(traffic.errors).toEqual([]);
    });
});


test.describe("R12 tutorial card parents keep their recorded type (19 F21)", () => {
  const PARENTS = [
    { name: "crate", iri: "https://example.org/practice/a#Crate", parent: "container" },
    { name: "lidded object", iri: "https://example.org/practice/a#LiddedObject", parent: "object" },
  ];

  test("lesson 1's source card opens both parents as typed classes; lesson 2 saves a typed parent action", async ({ page }) => {
    const traffic = recordTraffic(page);
    const api = await seedToTutorial(page);
    await page.reload();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 1 of / })).toBeVisible();
    const { tutorial } = await api.state();
    const ontology = tutorial.case.source.ontology_version_id as string;
    // Workspace and lesson state that the card navigation must keep.
    await page.getByRole("button", { name: /^Inspect / }).nth(2).click();
    await expect(page.locator(".lesson-checklist li .meta").filter({ hasText: "Done · saved" })).toHaveCount(1);
    const before = (await api.state()).tutorial_progress;
    const card = page.locator(".entity-card-source");
    await expect(card.locator('[data-category="parents"] li')).toHaveCount(PARENTS.length);
    const browser = page.getByRole("region", { name: "Source ontology browser" });
    for (const parent of PARENTS) {
      const row = card.locator(`[data-category="parents"] li[data-iri="${parent.iri}"]`);
      await expect(row).toHaveAttribute("data-kind", "class");
      await expect(row).toHaveAttribute("data-fact-id", /^sha256:/);
      const button = row.getByRole("button");
      await expect(button).toBeEnabled();
      await expect(button).toContainText(parent.name);
      await button.click();
      await expect(page.getByRole("tab", { name: "Hierarchy", exact: true })).toHaveAttribute("aria-selected", "true");
      await expect(browser.locator(".focus-label")).toHaveText(parent.name);
      await expect(browser.locator(".focus-card .iri")).toHaveText(parent.iri);
      await expect(browser.locator(".focus-card .eyebrow")).toHaveText("Focused class");
      // The typed lookup finds the parent's own recorded parent; a wrong kind or ontology finds none.
      await expect(browser.locator("ul.tree > li.tree-item > .tree-row .tree-label")).toHaveText([parent.parent]);
    }
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 1 of / })).toBeVisible();
    await expect(page.getByText(/inspecting initial position 3 of 5/)).toBeVisible();
    const after = (await api.state()).tutorial_progress;
    expect(after.position).toEqual(before.position);
    expect(after.completed_requirements).toEqual(expect.arrayContaining(before.completed_requirements));

    // In the context lesson the same card link is accepted by the service as a typed parent action.
    await page.getByRole("button", { name: "Next lesson", exact: true }).click();
    await expect(page.getByRole("heading", { level: 1, name: /^Lesson 2 of / })).toBeFocused();
    await card.locator(`[data-category="parents"] li[data-iri="${PARENTS[0].iri}"]`).getByRole("button").click();
    const requirement = tutorial.lessons[1].requirements.find((item: { action: string }) => item.action === "navigate_parent");
    await expect(page.locator(".lesson-checklist li").filter({ hasText: requirement.label }).locator(".meta")).toHaveText(/Done · saved/);
    expect((await api.state()).tutorial_progress.completed_requirements).toContain(requirement.requirement_id);
    const saves = () => traffic.requests.filter((item) => item.method === "PUT" && item.path === "/api/v1/study/tutorial/progress");
    const typedSaves = () => saves().filter((item) => ((item.body as { actions?: { action: string; entity?: unknown }[] }).actions ?? []).some((action) => action.action === "navigate_parent"));
    // Traffic is recorded once a request finishes, which can follow the UI's acknowledgement.
    await expect.poll(() => typedSaves().length).toBeGreaterThan(0);
    for (const save of typedSaves()) {
      expect(save.status).toBe(200);
      const action = (save.body as { actions: { action: string; entity?: unknown }[] }).actions.find((item) => item.action === "navigate_parent");
      expect(action?.entity).toEqual({ ontology_version_id: ontology, iri: PARENTS[0].iri, kind: "class" });
    }
    expect(saves().filter((item) => (item.status ?? 0) >= 400)).toEqual([]);
    expect(traffic.errors).toEqual([]);
  });
});
