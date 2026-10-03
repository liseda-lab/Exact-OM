import { readFileSync } from "node:fs";
import { join } from "node:path";

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page, type Route } from "@playwright/test";

import { harness, newInvitation, ParticipantApi, recordTraffic, scopedPath, seedAdvance, seedCondition, seedToCase, seedToTutorial } from "./v2Harness";

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

function caseSegments(traffic: ReturnType<typeof recordTraffic>) {
  return traffic.requests.filter((item) => item.path === "/api/v1/study/timing" && (item.body as { stage?: string })?.stage === "case");
}

const routeFor = (scope: string, suffix: string, match: (params: URLSearchParams) => boolean = () => true) => (url: URL) =>
  url.pathname === `${scopedPath(scope)}${suffix}` && match(url.searchParams);

async function explanationCase(page: Page) {
  const api = await seedToCase(page);
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
    await page.getByRole("button", { name: "Retry loading this case" }).click();
    await expect(add1(page)).toBeEnabled();
    expect(caseReady(traffic).length).toBe(1);
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
