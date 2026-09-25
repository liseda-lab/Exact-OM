import { expect, test } from "@playwright/test";

const PROPERTY = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#R176";
const CLASS = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C114940";

// Use the full prepared NCIT package. No response invents a property, kind or ontology fact.
test("NCIT property search preserves kind in hierarchy, URL, reload and full context", async ({ page, request }) => {
  const ontologies = await (await request.get("/api/v1/ontologies")).json();
  const ncit = ontologies.items.find((item: { name?: string }) => item.name === "NCIT");
  expect(ncit).toBeTruthy();
  const query = new URLSearchParams({ ontology_version_id: ncit.ontology_version_id, term: PROPERTY, limit: "20" });
  const found = await (await request.get(`/api/v1/entities?${query}`)).json();
  const property = found.items.find((item: { entity: { iri: string; kind: string } }) => item.entity.iri === PROPERTY && item.entity.kind === "object_property");
  expect(property).toBeTruthy();
  const queries: URL[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (["/api/v1/hierarchy", "/api/v1/entity-context"].includes(url.pathname) && url.searchParams.get("iri") === PROPERTY) queries.push(url);
  });
  await page.goto(`/browse/?${new URLSearchParams({ so: ncit.ontology_version_id })}`);
  const source = page.getByRole("region", { name: "Source ontology browser" });
  await source.getByRole("combobox").fill(PROPERTY);
  await source.getByRole("option").filter({ hasText: "object property" }).click();
  await expect(source.locator(".focus-card")).toContainText("Focused object property");
  await expect.poll(() => new URL(page.url()).searchParams.get("sk")).toBe("object_property");
  await expect.poll(() => queries.filter((url) => url.pathname === "/api/v1/hierarchy").length).toBeGreaterThanOrEqual(2);
  await page.reload();
  await expect(source.locator(".focus-card")).toContainText("Focused object property");
  await expect.poll(() => queries.filter((url) => url.pathname === "/api/v1/hierarchy").length).toBeGreaterThanOrEqual(4);
  const contextResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.pathname === "/api/v1/entity-context" && url.searchParams.get("iri") === PROPERTY;
  });
  await source.getByRole("button", { name: "Open full context", exact: true }).click();
  const context = await contextResponse;
  expect(context.ok()).toBe(true);
  expect((await context.json()).entity.kind).toBe("object_property");
  await expect(page.getByRole("dialog", { name: "Full context", exact: true })).toContainText("Source object property");
  expect(queries.every((url) => url.searchParams.get("kind") === "object_property")).toBe(true);
});

test("an old in-flight search cannot replace the new term's real results", async ({ page, request }) => {
  const ontologies = await (await request.get("/api/v1/ontologies")).json();
  const ncit = ontologies.items.find((item: { name?: string }) => item.name === "NCIT");
  expect(ncit).toBeTruthy();
  let release!: () => void;
  const held = new Promise<void>((resolve) => { release = resolve; });
  let started!: () => void;
  const loading = new Promise<void>((resolve) => { started = resolve; });
  await page.route("**/api/v1/entities?*", async (route) => {
    if (new URL(route.request().url()).searchParams.get("term") !== PROPERTY) return route.continue();
    const response = await route.fetch();
    started();
    await held;
    try { await route.fulfill({ response }); } catch { /* The changed query canceled this request. */ }
  });
  await page.goto(`/browse/?${new URLSearchParams({ so: ncit.ontology_version_id })}`);
  const source = page.getByRole("region", { name: "Source ontology browser" });
  const search = source.getByRole("combobox");
  await search.fill(PROPERTY);
  await loading;
  await search.fill(CLASS);
  await expect(source.getByRole("option").first()).toContainText("NCIT:C114940");
  release();
  await expect(source.getByRole("option")).toHaveCount(1);
  await expect(source.getByRole("option").first()).not.toContainText("object property");
  await search.press("Enter");
  await expect(source.locator(".focus-card")).toContainText(CLASS);
  expect(new URL(page.url()).searchParams.get("sk")).toBe("class");
});
