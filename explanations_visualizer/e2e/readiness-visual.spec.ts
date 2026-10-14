import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

import { settledDescriptions } from "./reviewData";
import { harness, harnessForms, REVIEW_FORMS, seedToCase } from "./v2Harness";

// Spec 24 A24-07 captures (opt-in: EXACT_SCREENSHOT_DIR): the prepared-example selector, the
// curated hierarchy start and the study controls after a reload, in light and dark, at desktop
// width and at 375 px and 320 px with 200% text. Every capture is taken only after its content
// is asserted and settled; each state is also checked with Axe and for horizontal overflow.
const dir = process.env.EXACT_SCREENSHOT_DIR ?? null;
const main = process.env.EXACT_E2E_REVIEW_MAIN_URL ?? null;

async function checked(page: Page, name: string) {
  const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  expect(result.violations.map((item) => `${item.id}: ${item.nodes.map((node) => node.target.join(" ")).join(", ")}`), name).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), `${name}: no horizontal page scroll`).toBe(false);
  if (dir) await page.screenshot({ path: `${dir}/${name}.png`, fullPage: false });
}

const variants = [
  { name: "desktop-light", width: 1440, height: 1000, scheme: "light" as const, scale: 1 },
  { name: "desktop-dark", width: 1440, height: 1000, scheme: "dark" as const, scale: 1 },
  { name: "375-light", width: 375, height: 812, scheme: "light" as const, scale: 1 },
  { name: "375-200pct-light", width: 375, height: 812, scheme: "light" as const, scale: 2 },
  { name: "320-200pct-dark", width: 320, height: 720, scheme: "dark" as const, scale: 2 },
];

async function prepare(page: Page, variant: (typeof variants)[number]) {
  await page.setViewportSize({ width: variant.width, height: variant.height });
  await page.emulateMedia({ colorScheme: variant.scheme });
  await page.addInitScript(([scale, theme]) => {
    localStorage.setItem("exact.textScale", String(scale));
    localStorage.setItem("exact.theme", String(theme));
  }, [variant.scale, variant.scheme] as const);
}

test.describe("main app: prepared examples and the curated hierarchy start", () => {
  test.beforeEach(() => test.skip(!main, "Set EXACT_E2E_REVIEW_MAIN_URL"));
  for (const variant of variants)
    test(`${variant.name}`, async ({ page }) => {
      test.setTimeout(240_000);
      await prepare(page, variant);
      const rail = page.getByRole("complementary", { name: "Choose a source and candidate" });
      await page.goto(`${main}/`);
      await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible({ timeout: 60_000 });
      await settledDescriptions(page);
      await expect(rail.getByRole("radiogroup", { name: "Sources shown" }).getByRole("radio", { name: /Prepared examples/ })).toHaveAttribute("aria-checked", "true");
      await expect(rail.locator(".candidate-item.current .prepared-flag")).toBeVisible();
      await checked(page, `hr2-landing-${variant.name}`);
      if (variant.width >= 1000) {
        await rail.scrollIntoViewIfNeeded();
        await rail.getByRole("radio", { name: /All sources/ }).click();
        await expect(rail.getByRole("list", { name: "Sources in this run" })).toBeVisible();
        await checked(page, `hr2-all-sources-${variant.name}`);
      } else {
        // Narrow: the example collapses to the current one with a way to change it.
        await expect(rail.getByRole("button", { name: /^Change example/ })).toBeVisible();
        await rail.getByRole("button", { name: /^Change example/ }).click();
        await expect(rail.getByRole("list", { name: "Prepared examples" })).toBeVisible();
        await rail.getByRole("radiogroup", { name: "Sources shown" }).scrollIntoViewIfNeeded();
        await checked(page, `hr2-examples-open-${variant.name}`);
      }

      await page.goto(`${main}/browse/`);
      const target = page.getByRole("region", { name: "Target ontology browser" });
      await expect(target.locator(".focus-card")).toContainText("Suggested starting point", { timeout: 60_000 });
      await expect(target.locator(".browser-section").filter({ hasText: "Children" }).locator(".child-list li").first()).toBeVisible();
      await target.locator(".focus-card").scrollIntoViewIfNeeded();
      await checked(page, `hr4-suggested-start-${variant.name}`);
      await target.getByRole("button", { name: /Show all top-level classes/ }).click();
      await expect(target.locator(".entry-points .pill-suggested")).toHaveCount(1);
      await target.locator(".entry-points").scrollIntoViewIfNeeded();
      await checked(page, `hr4-entry-list-${variant.name}`);
    });

  test("catalogue failure notice (injected 503)", async ({ page }) => {
    await page.route("**/api/v1/review-bundle", (route) => route.fulfill({ status: 503, json: { code: "unavailable", message: "Unavailable", retryable: true } }));
    await page.goto(`${main}/`);
    await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible({ timeout: 60_000 });
    await settledDescriptions(page);
    await expect(page.locator(".catalogue-note")).toContainText("could not be loaded");
    await checked(page, "hr2-catalogue-failed-desktop-light");
  });
});

test.describe("study: controls after a reload", () => {
  test.beforeEach(() => test.skip(!harness || harnessForms !== REVIEW_FORMS, "Set EXACT_E2E_STUDY_V2_CONFIG to a forms/3 review publication"));
  for (const variant of variants.filter((item) => item.width < 1000 || item.scheme === "dark"))
    test(`case view after reload, ${variant.name}`, async ({ page }) => {
      test.setTimeout(240_000);
      await prepare(page, variant);
      // Seeded only to reach a case for this capture; the acceptance journey seeds nothing.
      await seedToCase(page);
      await page.reload();
      // Wide: a "Candidates and your answer" sidebar; narrow: "Candidates" and "Your answer" regions.
      await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible({ timeout: 60_000 });
      await expect(page.getByRole("button", { name: /^Add .* as rank 1$/ }).first()).toBeEnabled();
      // Styled (HR1): the stylesheet token applies and icons are sized by it.
      const styled = await page.evaluate(() => ({ token: getComputedStyle(document.documentElement).getPropertyValue("--paper").trim(), widest: Math.max(0, ...Array.from(document.querySelectorAll("svg.icon")).map((icon) => icon.getBoundingClientRect().width)) }));
      expect(styled.token).not.toBe("");
      expect(styled.widest).toBeLessThanOrEqual(64);
      await checked(page, `study-case-after-reload-${variant.name}`);
    });
});
