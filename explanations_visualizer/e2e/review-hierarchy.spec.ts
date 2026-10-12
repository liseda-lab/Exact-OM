import { expect, test, type Locator, type Page } from "@playwright/test";

// Real-data hierarchy continuation (23 C05, 21 F06) in the main app on the corrected review
// package: multi-page asserted children, multi-page partial-ELK children, a basis change that
// never reuses a cursor, discovered entry points and independent side focus. Set
// EXACT_E2E_REVIEW_MAIN_URL to the exploration service of the combined review publication.
// The classes are real: DOID_0080015 (103 asserted children) and NCIT C9305 (72 ELK children).
const main = process.env.EXACT_E2E_REVIEW_MAIN_URL ?? null;
const DOID_PHYSICAL = "http://purl.obolibrary.org/obo/DOID_0080015";
const NCIT_MALIGNANT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C9305";
test.use({ viewport: { width: 1440, height: 900 } });
test.beforeEach(() => test.skip(!main, "Set EXACT_E2E_REVIEW_MAIN_URL to the review package's exploration service"));

async function versions(page: Page): Promise<Record<string, string>> {
  const body = await (await page.request.get(`${main}/api/v1/ontologies`)).json();
  return Object.fromEntries(body.items.map((item: { name: string; ontology_version_id: string }) => [item.name, item.ontology_version_id]));
}

/** Load every page of children and return the distinct identifiers shown. */
async function allChildren(browser: Locator, total: number): Promise<string[]> {
  const meta = browser.locator(".browser-section").filter({ hasText: "Children" }).locator(".section-head .meta");
  await expect(meta).toContainText(`of ${total}`);
  // Each request disables the control ("Loading…"); wait for its page before asking again.
  for (let page = 0; page < 10 && (await meta.textContent()) !== `${total} children`; page += 1) {
    const before = await meta.textContent();
    await browser.getByRole("button", { name: "Load more children" }).click();
    await expect(meta).not.toHaveText(before ?? "");
  }
  await expect(meta).toHaveText(`${total} children`);
  return browser.locator(".child-list li .iri").allTextContents();
}

test("asserted and ELK children page to their recorded totals; a basis change restarts paging", async ({ page }) => {
  test.setTimeout(180_000);
  const ids = await versions(page);
  const requests: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname === "/api/v1/hierarchy") requests.push(url.search);
  });
  await page.goto(`${main}/browse/?${new URLSearchParams({ so: ids.NCIT, s: NCIT_MALIGNANT, sk: "class", to: ids.DOID, t: DOID_PHYSICAL, tk: "class" })}`);
  const source = page.getByRole("region", { name: "Source ontology browser" });
  const target = page.getByRole("region", { name: "Target ontology browser" });
  await expect(source.locator(".focus-label")).toHaveText("Malignant Neoplasm");
  await expect(target.locator(".focus-label")).toHaveText("physical disorder");

  // Asserted, three pages of 50: 103 distinct children, in order, without duplicates.
  const asserted = await allChildren(target, 103);
  expect(new Set(asserted).size).toBe(103);

  // ELK on the other side only: two pages, 72 distinct children; the target keeps its mode.
  await source.getByRole("radiogroup", { name: "Relationships shown" }).getByRole("radio", { name: "ELK inferred · partial" }).click();
  await expect(source).toContainText("A missing link is unknown, not absent.");
  const elk = await allChildren(source, 72);
  expect(new Set(elk).size).toBe(72);
  await expect(target.getByRole("radio", { name: "Asserted" })).toHaveAttribute("aria-checked", "true");
  const continued = requests.filter((search) => search.includes("cursor="));
  expect(continued.some((search) => search.includes("basis=elk_inferred_partial"))).toBe(true);

  // Back to Asserted: the first page is requested again; no ELK cursor is sent with it.
  const before = requests.length;
  await source.getByRole("radio", { name: "Asserted" }).click();
  await expect(source.locator(".browser-section").filter({ hasText: "Children" }).locator(".section-head .meta")).toHaveText("12 children");
  const after = requests.slice(before).filter((search) => search.includes("direction=children"));
  expect(after.length).toBeGreaterThan(0);
  expect(after.every((search) => !search.includes("cursor="))).toBe(true);

  // Independent focus: moving the target never moves the source, and the URL keeps both.
  await target.locator(".child-list .tree-label").first().click();
  await expect(source.locator(".focus-label")).toHaveText("Malignant Neoplasm");
  expect(new URL(page.url()).searchParams.get("s")).toBe(NCIT_MALIGNANT);
  await page.reload();
  await expect(source.locator(".focus-label")).toHaveText("Malignant Neoplasm");
  await expect(target.locator(".focus-label")).not.toHaveText("physical disorder");
});

test("discovered entry points page and focus one side", async ({ page }) => {
  const ids = await versions(page);
  await page.goto(`${main}/browse/?${new URLSearchParams({ so: ids.NCIT, s: NCIT_MALIGNANT, sk: "class", to: ids.DOID, tk: "class" })}`);
  const source = page.getByRole("region", { name: "Source ontology browser" });
  const target = page.getByRole("region", { name: "Target ontology browser" });
  // No target focus: its top-level classes are the starting point, with continuation.
  const entries = target.locator(".entry-points");
  await expect(entries.locator(".child-list li").first()).toBeVisible();
  const first = await entries.locator(".child-list li").count();
  await target.getByRole("button", { name: /^Load more / }).first().click();
  await expect.poll(() => entries.locator(".child-list li").count()).toBeGreaterThan(first);
  const shown = await entries.locator(".child-list li .iri").allTextContents();
  expect(new Set(shown).size).toBe(shown.length);
  await entries.locator(".tree-label").first().click();
  await expect(target.locator(".focus-label")).toBeVisible();
  await expect(source.locator(".focus-label")).toHaveText("Malignant Neoplasm");
  // The source's own top-level classes open on request without changing its focus.
  await source.getByRole("button", { name: "Top-level classes" }).click();
  await expect(source.locator(".entry-points .child-list li").first()).toBeVisible();
  await expect(source.locator(".focus-label")).toHaveText("Malignant Neoplasm");
});
