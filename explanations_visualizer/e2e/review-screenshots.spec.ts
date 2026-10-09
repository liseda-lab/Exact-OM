import { mkdirSync } from "node:fs";
import { join } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { harness, seedCondition, seedToCase, seedToTutorial } from "./v2Harness";

// Review screenshots for the product-review passes (20 Q10). Not an assertion suite: it runs
// only when EXACT_SCREENSHOT_DIR is set, against the served main app (EXACT_E2E_URL) and the
// real v2 study service (EXACT_E2E_STUDY_V2_CONFIG). Lesson positions are set through the
// participant API so the same script captures any tutorial design.
const out = process.env.EXACT_SCREENSHOT_DIR ?? null;
test.use({ screenshot: "off", trace: "off" });
test.beforeEach(() => test.skip(!out, "Set EXACT_SCREENSHOT_DIR to capture review screenshots"));

const SCHEMES = ["light", "dark"] as const;
type Scheme = (typeof SCHEMES)[number];

async function shot(page: Page, name: string, scheme: Scheme, fullPage = false) {
  mkdirSync(out!, { recursive: true });
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(out!, `${name}-${scheme}.png`), fullPage });
}

async function mainReady(page: Page) {
  await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible();
  await expect(page.locator(".source-list button").first()).not.toContainText("Loading label");
}

for (const scheme of SCHEMES) {
  test.describe(`main app · ${scheme}`, () => {
    test.use({ colorScheme: scheme, viewport: { width: 1440, height: 900 } });
    test("compare, decision trace, graph and browse", async ({ page }) => {
      await page.goto("/");
      await mainReady(page);
      await shot(page, "main-compare", scheme);
      await shot(page, "main-compare-full", scheme, true);
      await page.getByRole("button", { name: /decision trace/i }).first().click();
      await shot(page, "main-trace-after-click", scheme);
      await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
      await expect(page.locator(".graph-canvas canvas").first()).toBeVisible();
      await page.locator(".graph-wrap").scrollIntoViewIfNeeded();
      await page.waitForTimeout(800);
      await page.locator(".details").screenshot({ path: join(out!, `main-graph-${scheme}.png`) });
      const feature = page.locator('.graph-items [data-tour="graph-feature"]').first();
      if (await feature.count()) {
        await feature.click();
        await page.waitForTimeout(400);
        await page.locator(".details").screenshot({ path: join(out!, `main-graph-selected-${scheme}.png`) });
      }
      await page.getByRole("tab", { name: "Decision trace", exact: true }).click();
      await page.waitForTimeout(400);
      await page.locator(".details").screenshot({ path: join(out!, `main-trace-${scheme}.png`) });
      await page.goto("/browse/");
      await page.waitForTimeout(1200);
      await shot(page, "main-browse", scheme);
    });
  });

  test.describe(`study · ${scheme}`, () => {
    test.use({ colorScheme: scheme, viewport: { width: 1440, height: 900 } });
    test.beforeEach(() => test.skip(!harness, "Set EXACT_E2E_STUDY_V2_CONFIG"));

    test("tutorial lessons", async ({ page }) => {
      test.setTimeout(120_000);
      const api = await seedToTutorial(page);
      const { tutorial } = await api.state();
      const at = async (lessonId: string) => {
        await api.write("PUT", "/tutorial/progress", { tutorial_version: tutorial.version, position: { view: "lesson", lesson_id: lessonId, question_id: null } });
        await page.reload();
        await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
        await page.waitForTimeout(800);
      };
      await at(tutorial.lessons[0].lesson_id);
      await shot(page, "tutorial-lesson1", scheme);
      await shot(page, "tutorial-lesson1-full", scheme, true);
      const showMe = page.locator(".coach").getByRole("button", { name: "Show me" });
      if (await showMe.count()) {
        await showMe.click();
        await page.waitForTimeout(700);
        await shot(page, "tutorial-lesson1-showme", scheme);
      }
      await at("context");
      await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
      await page.waitForTimeout(600);
      await shot(page, "tutorial-lesson2-hierarchy", scheme);
      const search = page.getByRole("region", { name: "Source ontology browser" }).getByRole("combobox");
      await search.fill("crate");
      await page.getByRole("listbox").getByRole("option", { name: /^crate/ }).first().click();
      await page.waitForTimeout(600);
      await page.getByRole("region", { name: "Source ontology browser" }).scrollIntoViewIfNeeded();
      await shot(page, "tutorial-lesson2-away", scheme);
      await at("graph");
      await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
      await expect(page.locator(".graph-canvas canvas").first()).toBeVisible();
      await page.locator(".graph-wrap").scrollIntoViewIfNeeded();
      await page.waitForTimeout(800);
      await shot(page, "tutorial-graph", scheme);
    });

    test("scored explanation case", async ({ page }) => {
      test.setTimeout(120_000);
      const api = await seedToCase(page);
      await seedCondition(page, api, "explanation");
      await page.reload();
      await expect(page.getByRole("button", { name: /^Add .* as rank 1$/ }).first()).toBeEnabled({ timeout: 30_000 });
      await shot(page, "study-case", scheme);
    });
  });
}

test.describe("narrow", () => {
  test.use({ viewport: { width: 375, height: 812 } });
  test("tutorial lesson 1 at 375px", async ({ page }) => {
    test.skip(!harness, "Set EXACT_E2E_STUDY_V2_CONFIG");
    await seedToTutorial(page);
    await page.reload();
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await page.waitForTimeout(800);
    await shot(page, "tutorial-lesson1-narrow", "light");
  });
});
