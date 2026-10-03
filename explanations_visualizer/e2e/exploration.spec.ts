import { expect, test, type Page } from "@playwright/test";

async function ready(page: Page) {
  await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible();
  await expect(page.locator(".source-list button").first()).not.toContainText("Loading label");
  await expect(page.getByRole("heading", { level: 1 })).not.toContainText(/NCIT:|DOID:/);
}

// These assertions use the served package, not a mocked exploration backend.
test("prepared comparison, evidence, graph and saved pair navigation", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await ready(page);
  await expect(page.getByText("Not a probability of being correct.", { exact: true })).toBeVisible();
  await expect(page.locator(".entity-card")).toHaveCount(2);
  for (const name of ["Hierarchy", "Evidence", "Decision trace", "Evidence graph", "Scores"]) {
    await page.getByRole("tab", { name, exact: true }).click();
    await expect(page.getByRole("tabpanel")).toBeVisible();
    if (name === "Evidence graph") {
      await expect(page.getByRole("img", { name: /^Evidence graph with/ })).toBeVisible();
      await page.getByRole("button", { name: "Zoom in", exact: true }).click();
      await page.getByRole("button", { name: "Fit all", exact: true }).click();
      await page.getByRole("button", { name: "Reset view", exact: true }).click();
    }
  }
  const first = page.url();
  await page.locator(".source-list button").nth(1).click();
  await ready(page);
  await expect.poll(() => page.url()).not.toBe(first);
  const selected = page.url();
  const heading = await page.getByRole("heading", { level: 1 }).innerText();
  await page.reload();
  await ready(page);
  expect(page.url()).toBe(selected);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(heading);
  expect(errors).toEqual([]);
});

test("every generated-text citation opens its exact original record and returns focus (C03)", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  const citations = page.locator("button.citation");
  await expect(citations.first()).toBeVisible();
  const count = await citations.count();
  expect(count).toBeGreaterThan(0);
  for (let index = 0; index < count; index += 1) {
    const citation = citations.nth(index);
    await citation.focus();
    await page.keyboard.press("Enter");
    const dialog = page.getByRole("dialog");
    await expect(dialog).toContainText("Origin");
    await expect(dialog).not.toContainText("not available in the open bundle");
    await expect(dialog.locator(".inspector-subjects")).not.toContainText("Subject not recorded");
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(citation).toBeFocused();
  }
});

test("the graph keeps its view across tabs and draws only supported, legend-matched lines (C07, C08)", async ({ page }) => {
  const warnings: string[] = [];
  page.on("console", (message) => { if (/style property|invalid/i.test(message.text())) warnings.push(message.text()); });
  await page.goto("/?details=graph");
  await ready(page);
  await expect(page.getByRole("img", { name: /^Evidence graph with/ })).toBeVisible();
  const view = () => page.locator(".graph-canvas").evaluate((element) => {
    const cy = (element as unknown as { _cyreg: { cy: { zoom: () => number; pan: () => { x: number; y: number } } } })._cyreg.cy;
    return { zoom: cy.zoom(), pan: cy.pan() };
  });
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  const zoomed = await view();
  await page.getByRole("tab", { name: "Evidence", exact: true }).click();
  await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
  await expect(page.getByRole("img", { name: /^Evidence graph with/ })).toBeVisible();
  await expect.poll(view).toEqual(zoomed);
  const styles = await page.locator(".graph-canvas").evaluate((element) => {
    const cy = (element as unknown as { _cyreg: { cy: { edges: () => { data: (k: string) => string; style: (k: string) => string }[] } } })._cyreg.cy;
    return Array.from(cy.edges(), (edge) => [edge.data("kind"), edge.style("line-style")]);
  });
  expect(styles.every(([, style]) => ["solid", "dashed", "dotted"].includes(style))).toBe(true);
  expect(warnings).toEqual([]);
});

test("run-discovery failures remain errors and can be retried", async ({ page }) => {
  await page.route("**/api/v1/runs?*", (route) => route.fulfill({
    status: 409, contentType: "application/json",
    body: JSON.stringify({ code: "invalid_prepared_resource", message: "Invalid fixture metadata" }),
  }));
  await page.goto("/");
  await expect(page.locator("#main[role=alert]")).toContainText("The bundle could not be read");
  await expect(page.getByText("This bundle has no saved matching run")).toHaveCount(0);
  await page.unroute("**/api/v1/runs?*");
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await ready(page);
});

test("a capped candidate list declares incomplete ranking coverage", async ({ page }) => {
  let number = 0;
  await page.route("**/api/v1/runs/*/candidates?*", async (route) => {
    const response = await route.fetch({ url: route.request().url().replace(/&cursor=[^&]*/, "") });
    const body = await response.json();
    const candidate = body.items[0];
    number += 1;
    await route.fulfill({ response, json: {
      ...body, items: [{ ...candidate, pair_id: number === 1 ? candidate.pair_id : `fixture-${number}`,
        ordinal_ranks: { candidate_joint_rank: number } }],
      next_cursor: `page-${number}`, total_count: 600,
    } });
  });
  await page.goto("/");
  await expect(page.getByText(/Showing a loaded subset: 5 of 600 candidates/)).toBeVisible();
  await expect(page.getByText(/additional candidates may rank higher/)).toBeVisible();
});

test("ontology search, independent focus, hierarchy and keyboard dialog", async ({ page }) => {
  await page.goto("/browse/");
  const source = page.getByRole("region", { name: "Source ontology browser" });
  const target = page.getByRole("region", { name: "Target ontology browser" });
  const search = page.getByRole("combobox", { name: "Search the source ontology" });
  await search.fill(process.env.EXACT_E2E_SEARCH_TERM ?? "myopathy");
  await expect(page.getByRole("listbox").getByRole("option").first()).toBeVisible();
  await search.press("ArrowDown");
  await search.press("Enter");
  await expect(source.locator(".focus-card")).toBeVisible();
  await expect(target.locator(".focus-card")).toHaveCount(0);
  await expect(source.getByRole("radio", { name: /not prepared/ })).toHaveAttribute("aria-disabled", "true");
  await source.getByRole("button", { name: "Open full context" }).click();
  await expect(page.getByRole("dialog", { name: "Full context" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(source.getByRole("button", { name: "Open full context" })).toBeFocused();
});

test("responsive comparison and persistent theme/text preferences", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  for (const width of [320, 375, 768, 1440, 2560]) {
    await page.setViewportSize({ width, height: 1000 });
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), `overflow at ${width}px`).toBe(true);
  }
  await page.getByRole("combobox", { name: "Colour theme" }).selectOption("dark");
  for (let i = 0; i < 5; i += 1) await page.getByRole("button", { name: "Increase text size" }).click();
  await expect(page.getByRole("button", { name: "Increase text size" })).toBeDisabled();
  await page.reload();
  await ready(page);
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  expect(await page.locator("html").evaluate((element) => element.style.getPropertyValue("--text-scale"))).toBe("2");
  await page.setViewportSize({ width: 375, height: 1000 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
});

test("local library rejects corrupt input and opens a verified ZIP", async ({ page }) => {
  const fixtureURL = process.env.EXACT_E2E_FIXTURE_URL;
  const zip = process.env.EXACT_E2E_BUNDLE_ZIP;
  test.skip(!fixtureURL || !zip, "Set the isolated fixture URL and exported ZIP path");
  await page.goto(`${fixtureURL}/library/`);
  await page.getByLabel("Choose file").setInputFiles({ name: "corrupt.zip", mimeType: "application/zip", buffer: Buffer.from("not a zip") });
  await page.getByRole("button", { name: "Import this bundle" }).click();
  await expect(page.getByRole("heading", { name: "corrupt.zip was not imported" })).toBeVisible();
  await page.getByLabel("Choose file").setInputFiles(zip!);
  await expect(page.getByText(/2 ontologies/)).toBeVisible();
  await page.getByRole("button", { name: "Import this bundle" }).click();
  await page.getByRole("button", { name: "Open this bundle" }).click();
  await ready(page);
  await expect(page.locator(".candidate-list button")).toHaveCount(5);
  const original = page.url();
  await page.getByRole("button", { name: "Next candidate", exact: true }).click();
  await expect.poll(() => page.url()).not.toBe(original);
  await page.reload();
  await ready(page);
});

test("public demo keeps exploration and excludes private/import routes", async ({ page, request }) => {
  const demo = process.env.EXACT_E2E_DEMO_URL;
  test.skip(!demo, "Set the public-demo URL");
  await page.goto(demo!);
  await ready(page);
  await expect(page.getByRole("link", { name: "Library", exact: true })).toHaveCount(0);
  for (const path of ["/library/", "/participate/", "/admin/", "/api/v1/study/state", "/api/v1/bundles"]) {
    const response = await request.get(`${demo}${path}`);
    expect([403, 404]).toContain(response.status());
  }
});
