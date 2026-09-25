import { expect, test, type Page } from "@playwright/test";
import type { Core, NodeSingular } from "cytoscape";

// Explicit opt-in: use a saved run server, never the participant/prepared service.
const legacyURL = process.env.EXACT_E2E_LEGACY_URL;
test.use({ baseURL: legacyURL, viewport: { width: 1600, height: 1000 } });
test.skip(!legacyURL, "Requires an existing exact-inspect serve --run-dir server");

async function graphState(page: Page) {
  return page.evaluate(() => {
    type GraphElement = HTMLElement & { _cyreg?: { cy: Core } };
    const container = Array.from(document.querySelectorAll<GraphElement>("div"))
      .find((element) => element._cyreg?.cy);
    const cy = container?._cyreg?.cy;
    if (!container || !cy) return null;
    const source = cy.nodes().filter((node) => node.data("type") === "Source").first() as NodeSingular;
    if (!source.length) return null;
    const point = source.renderedPosition();
    const bounds = container.getBoundingClientRect();
    return {
      nodes: cy.nodes().length,
      edges: cy.edges().length,
      sourceContext: cy.nodes().filter((node) => node.data("type") === "source-context").length,
      zoom: cy.zoom(),
      source: source.data("label") as string,
      sourcePoint: { x: bounds.left + point.x, y: bounds.top + point.y },
      fontSize: Number.parseFloat(source.style("font-size")),
    };
  });
}

async function ready(page: Page) {
  await expect(page.getByRole("button", { name: "Zoom in", exact: true })).toBeVisible();
  await expect.poll(async () => (await graphState(page))?.nodes ?? 0).toBeGreaterThan(1);
}

test("saved-run API and root redirect retain the same-origin run", async ({ page, request }) => {
  const errors: string[] = [];
  const foreignRequests: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    if (new URL(request.url()).origin !== new URL(legacyURL!).origin) {
      foreignRequests.push(request.url());
    }
  });
  const health = await request.get("/api/health");
  expect(health.ok()).toBe(true);
  expect(await health.json()).toMatchObject({ mode: "open", frontend_mode: "static" });
  expect((await request.get("/api/v1/health")).status()).toBe(404);
  const sources = await (await request.get("/api/study/sources")).json();
  expect(sources).toHaveLength(12);
  const response = await request.get("/api/study/source", { params: { source: sources[0].source_id } });
  expect(response.ok()).toBe(true);
  const bundle = await response.json();
  expect(bundle.source_id).toBe(sources[0].source_id);
  expect(bundle.targets.length).toBeGreaterThan(0);
  expect(bundle.targets[0].score).toBeGreaterThan(0);
  await page.goto("/");
  await expect(page).toHaveURL(/\/legacy\//);
  await ready(page);
  expect(new URL(page.url()).searchParams.get("source")).toBe(sources[0].source_id);
  expect((await graphState(page))?.source).toBe(bundle.source_label);
  expect(foreignRequests).toEqual([]);
  expect(errors).toEqual([]);
});

test("saved-run canvas, filters and source navigation remain interactive", async ({ page, request }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/legacy/");
  await ready(page);
  const initial = (await graphState(page))!;
  expect(initial.edges).toBeGreaterThan(0);
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  await expect.poll(async () => (await graphState(page))!.zoom).toBeGreaterThan(initial.zoom);
  await page.getByRole("button", { name: "Increase graph label size", exact: true }).click();
  await expect.poll(async () => (await graphState(page))!.fontSize).toBeGreaterThan(initial.fontSize);
  const selected = (await graphState(page))!;
  await page.mouse.click(selected.sourcePoint.x, selected.sourcePoint.y);
  await expect(page.getByText("Selected node", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Hide inspector", exact: true }).click();
  const sourceContext = page.getByRole("checkbox", { name: "Source context", exact: true });
  await sourceContext.uncheck();
  await expect.poll(async () => (await graphState(page))!.sourceContext).toBe(0);
  await sourceContext.check();
  await expect.poll(async () => (await graphState(page))!.sourceContext).toBeGreaterThan(0);
  const sources = await (await request.get("/api/study/sources")).json();
  const alternate = sources.find((source: { source_id: string }) => source.source_id !== new URL(page.url()).searchParams.get("source"));
  const picker = page.getByPlaceholder("Search labels or IRIs");
  await picker.fill(alternate.source_id);
  await picker.press("Enter");
  await expect.poll(() => new URL(page.url()).searchParams.get("source")).toBe(alternate.source_id);
  const bundle = await (await request.get("/api/study/source", { params: { source: alternate.source_id } })).json();
  await expect.poll(async () => (await graphState(page))?.source).toBe(bundle.source_label);
  await page.reload();
  await ready(page);
  expect(new URL(page.url()).searchParams.get("source")).toBe(alternate.source_id);
  expect((await graphState(page))?.source).toBe(bundle.source_label);
  expect(errors).toEqual([]);
});
