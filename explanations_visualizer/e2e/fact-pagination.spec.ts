import { expect, test } from "@playwright/test";

import { chooseContinuedParent, exhaustBrowserParents, exhaustPagedCard, TYPED } from "./factPagination";

// R06 / 19 F16 in the main app: the shared entity card continues every truncatable category.
// EXACT_E2E_PAGED_MAIN_APP_URL serves the paged fixture contexts with alternate definitions
// admitted (prepare_main_app_package.py --extra-category alternate_definitions);
// EXACT_E2E_PAGED_SOURCE is the source ontology version of that fixture.
const app = process.env.EXACT_E2E_PAGED_MAIN_APP_URL ?? null;
const ontology = process.env.EXACT_E2E_PAGED_SOURCE ?? null;
test.use({ screenshot: "off", trace: "off", viewport: { width: 1280, height: 900 } });
test.beforeEach(() => test.skip(!app || !ontology, "Set EXACT_E2E_PAGED_MAIN_APP_URL and EXACT_E2E_PAGED_SOURCE for the paged-facts fixture"));

test("main app: full context pages every category beyond its first page, without repeats or false completeness", async ({ page }) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${app}/browse/?so=${encodeURIComponent(ontology!)}&s=${encodeURIComponent("urn:source:paged")}&sk=class`);
  const browser = page.getByRole("region", { name: "Source ontology browser" });
  await expect(browser.locator(".focus-label")).toHaveText("Paged facts source");
  await exhaustBrowserParents(browser);
  await browser.getByRole("button", { name: "Open full context", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Full context" });
  await exhaustPagedCard(dialog.locator(".entity-card"), { alternate: true });
  // A continued parent is a working link like the first page's.
  await dialog.locator('[data-category="parents"] li').last().getByRole("button").click();
  await expect(dialog).toHaveCount(0);
  await expect(browser.locator(".focus-label")).toHaveText(/^Paged parent \d{3}$/);
  expect(new URL(page.url()).searchParams.get("sk")).toBe("class");
  expect(errors).toEqual([]);
});

for (const typed of TYPED)
  test(`main app: a ${typed.name}'s later-page superproperty opens as a ${typed.name} (R11)`, async ({ page }) => {
    const reads: { path: string; status: number }[] = [];
    page.on("response", (response) => {
      const url = new URL(response.url());
      if (url.pathname.startsWith("/api/v1/")) reads.push({ path: url.pathname + url.search, status: response.status() });
    });
    await page.goto(`${app}/browse/?so=${encodeURIComponent(ontology!)}&s=${encodeURIComponent(`urn:source:paged-${typed.slug}-property`)}&sk=${typed.kind}`);
    const browser = page.getByRole("region", { name: "Source ontology browser" });
    await expect(browser.locator(".focus-label")).toHaveText(typed.label);
    await browser.getByRole("button", { name: "Open full context", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Full context" });
    const iri = await chooseContinuedParent(dialog, typed.kind);
    await expect(dialog).toHaveCount(0);
    // Ontology, IRI and kind are preserved in the navigation and the parent's own reads succeed.
    const params = new URL(page.url()).searchParams;
    expect([params.get("so"), params.get("s"), params.get("sk")]).toEqual([ontology, iri, typed.kind]);
    await expect(browser.locator(".focus-label")).toHaveText(typed.parent);
    await expect(browser.getByText(`Focused ${typed.name}`)).toBeVisible();
    await browser.getByRole("button", { name: "Open full context", exact: true }).click();
    await expect(page.getByRole("dialog", { name: "Full context" }).locator(".entity-title")).toHaveText(typed.parent);
    const parentReads = reads.filter((item) => new URLSearchParams(item.path.split("?")[1]).get("iri") === iri);
    expect(parentReads.some((item) => item.path.startsWith("/api/v1/entity-context") && item.status === 200 && item.path.includes(`kind=${typed.kind}`))).toBe(true);
    expect(parentReads.filter((item) => item.path.includes("kind=class"))).toEqual([]);
  });
