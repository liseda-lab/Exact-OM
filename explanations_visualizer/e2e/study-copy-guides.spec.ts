import { test, expect } from "@playwright/test";
import { harness, newInvitation, ParticipantApi, seedToCase, seedCondition } from "./v2Harness";

test.skip(!harness, "Requires an explicitly configured test-only study");

async function revoke(page: import("@playwright/test").Page) {
  const state = await new ParticipantApi(page).state();
  await page.request.post(`${harness!.origin}/api/v1/admin/invitations/${state.session_id}/revoke`, {
    headers: { Authorization: `Bearer ${harness!.researcher_token}` },
  });
}

test("participant copy and both optional inspection guides", async ({ page }) => {
  await page.goto(harness!.origin + await newInvitation(page));
  await expect(page.getByRole("heading", { name: "Keep your private link secure" })).toBeVisible();
  await expect(page.getByText("Your answers, the actions you take in the study pages and how long each step takes will be recorded.", { exact: true })).toBeVisible();
  await expect(page.getByText(/Version synthetic-information|Provisional real development/)).toHaveCount(0);
  await page.getByRole("button", { name: "I agree to take part", exact: true }).click();
  await expect(page.getByRole("button", { name: "Protégé guide", exact: true })).toBeVisible();
  const api = new ParticipantApi(page);
  const before = await api.state();
  for (const name of ["Protégé guide", "Text editor guide"]) {
    await page.getByRole("button", { name, exact: true }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    const download = page.waitForEvent("download");
    await dialog.getByRole("button", { name: "Download the tiny practice ontology" }).click();
    expect((await download).suggestedFilename()).toBe("inspection-practice.ofn");
    await dialog.getByRole("button", { name: "Next", exact: true }).click();
    await expect(dialog.locator("figure")).toBeVisible();
    await dialog.getByRole("button", { name: "Next", exact: true }).click();
    await dialog.getByRole("button", { name: "Show answer" }).click();
    await expect(dialog.locator(".note")).toContainText(/container/i);
    await dialog.getByRole("button", { name: "Done", exact: true }).click();
    await page.getByRole("button", { name, exact: true }).click();
    await dialog.getByRole("button", { name: "Skip guide", exact: true }).last().click();
    await expect(dialog).toHaveCount(0);
  }
  const after = await api.state();
  expect(after.stage).toBe(before.stage);
  expect(after.tutorial_progress).toEqual(before.tutorial_progress);
  await revoke(page);
});

for (const condition of ["ontology_baseline", "explanation"] as const) {
  test(`ranking survives tutorial completion and reload in ${condition}`, async ({ page }) => {
    test.setTimeout(180000);
    const errors: string[] = [];
    page.on("pageerror", e => errors.push(e.message));
    const api = await seedToCase(page);
    await seedCondition(page, api, condition);
    await page.reload();
    // Dismiss the legitimate return question if the API-seeded transition did not set lastSeen.
    const gap = page.getByRole("dialog");
    if (await gap.isVisible()) {
      const no = gap.getByRole("button", { name: /not working|No,/i });
      if (await no.count()) await no.first().click();
    }
    await expect(page.getByRole("button", { name: "Submit answer", exact: true })).toBeVisible({ timeout: 60000 });
    await page.getByRole("button", { name: "Keep initial order", exact: true }).click();
    await expect(page.getByRole("button", { name: "Submit answer", exact: true })).toBeEnabled();
    await page.waitForTimeout(1500);
    await page.reload();
    await expect(page.getByRole("button", { name: "Submit answer", exact: true })).toBeEnabled({ timeout: 60000 });
    expect(await page.locator("html").evaluate(e => getComputedStyle(e).getPropertyValue("--paper").trim())).not.toBe("");
    expect(errors).toEqual([]);
    await revoke(page);
  });
}

test("admin selects, replaces and revokes an active link", async ({ page }) => {
  let generation = 1; let revoked = false;
  const paths: string[] = [];
  await page.route("**/api/v1/admin/**", async route => {
    const req = route.request(); const path = new URL(req.url()).pathname; paths.push(`${req.method()} ${path}`);
    let body: unknown;
    if (path.endsWith("/invitations") && req.method() === "GET") body = { items: revoked ? [] : [{ session_id: "selectable-session", generation, test: true, stage: "tutorial" }], next_cursor: null, study_closed: false };
    else if (path.endsWith("/reissue")) { generation++; body = { invitation: "/participate#invite=illustration-only" }; }
    else if (path.endsWith("/revoke")) { revoked = true; body = { revoked: true }; }
    else body = { items: [], next_cursor: null };
    await route.fulfill({ json: body });
  });
  await page.goto(harness!.origin + "/admin/");
  await page.getByLabel("Researcher token").fill("test-only-browser-placeholder-token");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await page.getByLabel("Study revision", { exact: true }).fill("test-revision");
  const select = page.getByLabel("Active links", { exact: true });
  await expect(select.locator("option")).toHaveCount(2);
  await select.selectOption("selectable-session");
  await page.getByRole("button", { name: "Issue a replacement link" }).click();
  await expect(page.getByText(/Replacement link \(shown once\)/)).toBeVisible();
  await expect(select.locator("option").last()).toContainText("link 2");
  await page.getByRole("button", { name: "Revoke", exact: true }).click();
  await expect(page.getByText(/No active links in this study/)).toBeVisible();
  expect(paths).toContain("POST /api/v1/admin/invitations/selectable-session/reissue");
  expect(paths).toContain("POST /api/v1/admin/invitations/selectable-session/revoke");
});

test("a rendering failure has a recovery screen and reload preserves the session", async ({ page }) => {
  await page.goto(harness!.origin + await newInvitation(page));
  await expect(page.getByRole("button", { name: "I agree to take part", exact: true })).toBeVisible();
  const api = new ParticipantApi(page);
  const saved = await api.state();
  await page.route("**/api/v1/study/state", async route => {
    // Deliberately malformed transport data tests the route error boundary, not a claimed
    // reproduction of the user's intermittent black screen.
    await route.fulfill({ json: { ...saved, stage: "background", forms: { ...saved.forms, background: null } } });
  });
  await page.reload();
  await expect(page.getByRole("heading", { name: "The study page could not be displayed" })).toBeVisible();
  await page.unroute("**/api/v1/study/state");
  await page.getByRole("button", { name: "Reload study", exact: true }).click();
  await expect(page.getByRole("button", { name: "I agree to take part", exact: true })).toBeVisible();
  expect((await api.state()).session_id).toBe(saved.session_id);
  await revoke(page);
});
