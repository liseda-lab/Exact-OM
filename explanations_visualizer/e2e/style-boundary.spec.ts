import { expect, test } from "@playwright/test";

import { MAX_STYLED_ICON_PX, StyleDiagnostics } from "./styleDiagnostics";

// HR1 check validity (spec 24 §3): the boundary style check must fail on the incident's symptom.
// SYNTHETIC FAULT INJECTION, not a reproduction of the original incident: the application
// stylesheet transfer is aborted for one document, which renders browser-default text and unsized
// icons as in the retained failure screenshot. Uses the exploration service
// (EXACT_E2E_REVIEW_MAIN_URL); no study session is created.
const main = process.env.EXACT_E2E_REVIEW_MAIN_URL ?? null;
test.beforeEach(() => test.skip(!main, "Set EXACT_E2E_REVIEW_MAIN_URL"));

test("the style check rejects a document whose stylesheet transfer failed, and records the failure (injected)", async ({ page }) => {
  test.setTimeout(120_000);
  const styles = await StyleDiagnostics.attach(page, "style-boundary-injected");
  await page.goto(`${main}/browse/`);
  await expect(page.getByRole("heading", { level: 1, name: "Browse both ontologies" })).toBeVisible();
  await styles.expectStyled("styled-baseline");

  await page.route("**/_next/static/css/**", (route) => route.abort("connectionreset"));
  await styles.beforeNavigation("injected stylesheet failure");
  await page.reload();
  await expect(page.getByRole("heading", { level: 1, name: "Browse both ontologies" })).toBeVisible();
  const probe = await styles.probe("injected");
  expect(probe.styled).toBe(false);
  expect(probe.token).toBe("");
  expect(probe.max_icon_px).toBeGreaterThan(MAX_STYLED_ICON_PX);
  await expect(styles.expectStyled("injected", 2_000)).rejects.toThrow(/Application styling missing at "injected"/);
  expect(styles.summary().css_failed_not_canceled).toBeGreaterThan(0);

  // The next ordinary reload is styled again; the record keeps both documents.
  await page.unroute("**/_next/static/css/**");
  await styles.beforeNavigation("ordinary reload");
  await page.reload();
  await styles.expectStyled("after-recovery");
  await styles.finish(test.info());
});
