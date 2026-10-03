import { expect, test } from "@playwright/test";

import { exhaustBrowserParents, exhaustPagedCard } from "./factPagination";

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
  expect(errors).toEqual([]);
});
