import { test, expect } from "@playwright/test";
import { harness, newInvitation, ParticipantApi, seedToTutorial } from "./v2Harness";

// Explicit opt-in: the harness must use a private validation database. Invitations
// created by these helpers are test sessions, excluded from research exports.
test.skip(!harness || process.env.EXACT_PRODUCTION_ACCEPTANCE !== "1", "Requires the final production publication in an isolated validation database");

test("final participant information is complete and readable", async ({ page }) => {
  await page.goto(harness!.origin + await newInvitation(page));
  await expect(page.getByRole("button", { name: "I agree to take part", exact: true })).toBeVisible();
  const information = page.locator(".consent-text");
  await expect(information).toContainText("National Cancer Institute Thesaurus (NCIT)");
  await expect(information).toContainText("Disease Ontology (DOID)");
  await expect(information).toContainText("retained for one year");
  await expect(information).toContainText("pgcotovio@ciencias.ulisboa.pt");
  await expect(information).not.toContainText(/Study period:|Time to take part:|Ethics review:|to be confirmed|clinical decisions|professional abilities/);
  expect(await information.evaluate(el => getComputedStyle(el).maxHeight)).toBe("none");
  await page.screenshot({ path: "test-results/production-welcome-desktop.png", fullPage: true });
  await page.setViewportSize({ width: 375, height: 900 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({ path: "test-results/production-welcome-mobile.png", fullPage: true });
  const state = await new ParticipantApi(page).state();
  await page.request.post(`${harness!.origin}/api/v1/admin/invitations/${state.session_id}/revoke`, { headers: { Authorization: `Bearer ${harness!.researcher_token}` } });
});

test("real case renders evidence, hierarchy and both colour themes", async ({ page }) => {
  test.setTimeout(180000);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  const api = await seedToTutorial(page);
  const state = await api.state();
  await api.write("POST", "/tutorial/complete", { tutorial_version: state.tutorial.version, skip_confirmed: true });
  let current = await api.currentCase();
  while (current.condition !== "explanation") {
    expect(current.workspace).toBeNull();
    expect(current.explanation_refs).toEqual([]);
    await api.write("POST", `/cases/${current.case_id}/submit`, { presentation_id: current.presentation_id, response_type: "ranked_candidates", ranked_candidate_ids: current.candidates.map((c: { candidate_id: string }) => c.candidate_id) });
    const after = await api.state();
    await api.write("PUT", `/cases/${current.case_id}/consultation`, { presentation_id: current.presentation_id, form_version: after.forms.version, consultation_version: "consultation/3", consulted_external_ontologies: false, methods: [] });
    current = await api.currentCase();
  }
  expect(JSON.stringify(current)).not.toMatch(/reference_rank|acceptable_candidate_ids|reference_target/);
  await page.route("**/api/v1/study/cases/current", async route => {
    await new Promise(resolve => setTimeout(resolve, 1500));
    await route.continue();
  });
  await page.reload();
  await expect(page.getByRole("status").filter({ hasText: "Loading this case's information" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Keep initial order", exact: true })).toBeEnabled({ timeout: 30000 });
  await page.screenshot({ path: "test-results/production-case-light.png", fullPage: true });
  await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
  await expect(page.locator(".graph-canvas canvas").first()).toBeVisible();
  await page.locator(".graph-wrap").scrollIntoViewIfNeeded();
  await page.locator(".details").screenshot({ path: "test-results/production-graph-light.png" });
  await page.getByLabel("Colour theme").selectOption("dark");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.locator(".details").screenshot({ path: "test-results/production-graph-dark.png" });
  await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
  await expect(page.getByRole("region", { name: "Source ontology browser" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("button", { name: "Keep initial order", exact: true })).toBeEnabled();
  expect(errors).toEqual([]);
  const final = await api.state();
  await page.request.post(`${harness!.origin}/api/v1/admin/invitations/${final.session_id}/revoke`, { headers: { Authorization: `Bearer ${harness!.researcher_token}` } });
});

test("a stalled initial case offers a retry and then recovers", async ({ page }) => {
  test.setTimeout(90000);
  const api = await seedToTutorial(page);
  const state = await api.state();
  await api.write("POST", "/tutorial/complete", { tutorial_version: state.tutorial.version, skip_confirmed: true });
  await page.route("**/api/v1/study/cases/current", () => new Promise(() => {}));
  await page.reload();
  await expect(page.getByRole("status").filter({ hasText: "Loading this case's information" })).toBeVisible();
  await expect(page.getByText("Loading this case took too long. Please retry.", { exact: false })).toBeVisible({ timeout: 50000 });
  await page.unrouteAll({ behavior: "ignoreErrors" });
  await page.getByRole("button", { name: "Retry loading this case", exact: true }).click();
  await expect(page.getByRole("button", { name: "Keep initial order", exact: true })).toBeEnabled({ timeout: 30000 });
  const final = await api.state();
  await page.request.post(`${harness!.origin}/api/v1/admin/invitations/${final.session_id}/revoke`, { headers: { Authorization: `Bearer ${harness!.researcher_token}` } });
});
