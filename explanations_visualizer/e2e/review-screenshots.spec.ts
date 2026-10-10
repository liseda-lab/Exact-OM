import { mkdirSync } from "node:fs";
import { join } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { atLeast, cls, EX, exactly, factId, only, some, syntheticGraph, type SyntheticFeature } from "./graphFixtures";
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
      await page.getByRole("tab", { name: "Scores", exact: true }).click();
      await page.waitForTimeout(300);
      await page.locator(".details").screenshot({ path: join(out!, `main-scores-${scheme}.png`) });
      await page.locator(".scores-panel").getByRole("button", { name: "About these scores" }).click();
      await page.locator(".scores-panel").getByRole("button", { name: "What this score means" }).first().click();
      await page.waitForTimeout(300);
      await page.locator(".details").screenshot({ path: join(out!, `main-scores-open-${scheme}.png`) });
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

    test("assessment answer kept across offline Next/Previous (S0-R2)", async ({ page, context }) => {
      test.setTimeout(120_000);
      const api = await seedToTutorial(page);
      const { tutorial } = await api.state();
      const question = tutorial.assessment[0];
      await api.write("PUT", "/tutorial/progress", { tutorial_version: tutorial.version, position: { view: "assessment", lesson_id: null, question_id: question.question_id } });
      await page.reload();
      await expect(page.locator(".assessment-item h2")).toContainText("1 ·");
      const answer = page.getByRole("radio", { name: question.options[0].label, exact: true });
      await context.setOffline(true);
      await answer.check();
      await page.getByRole("button", { name: "Next question", exact: true }).click();
      await page.getByRole("button", { name: "Previous question", exact: true }).click();
      await expect(answer).toBeChecked();
      await shot(page, "assessment-offline-back", scheme);
      await context.setOffline(false);
      await expect.poll(async () => (await api.state()).tutorial_progress.assessment_drafts?.[question.question_id]?.choice).toBe(question.options[0].code);
      await expect(answer).toBeChecked();
      await shot(page, "assessment-reconnected", scheme);
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

// Synthetic graph shapes (S0 follow-up): invented example.org records over the real compared pair.
const SHAPES: Record<string, { features: SyntheticFeature[]; labels: Record<string, string> }> = {
  shared: {
    features: [
      { evidenceId: "rel-some", side: "source", channel: "relational", facts: [{ id: factId("some"), superClass: some("holds", cls("Syndrome")) }] },
      { evidenceId: "rel-only", side: "source", channel: "relational", facts: [{ id: factId("only"), superClass: only("holds", cls("Syndrome")) }] },
      { evidenceId: "rel-min", side: "source", channel: "relational", facts: [{ id: factId("min"), superClass: atLeast(2, "hasPart", cls("Syndrome")) }] },
      { evidenceId: "hier-parent", side: "source", channel: "hierarchy", facts: [{ id: factId("parent"), superClass: cls("Syndrome") }] },
      { evidenceId: "rel-exact", side: "target", channel: "relational", facts: [{ id: factId("exact"), superClass: exactly(1, "locatedIn", cls("Site")) }] },
      { evidenceId: "t-parent", side: "target", channel: "hierarchy", facts: [{ id: factId("tparent"), superClass: cls("Site") }] },
    ],
    labels: { [`${EX}Syndrome`]: "Syndrome X", [`${EX}holds`]: "holds", [`${EX}hasPart`]: "has part", [`${EX}locatedIn`]: "located in", [`${EX}Site`]: "Anatomical site Y" },
  },
  grouped: {
    features: [
      { evidenceId: "hier", side: "source", channel: "hierarchy", facts: [{ id: factId("hier"), superClass: cls("Parent") }] },
      ...["alpha label", "beta synonym", "gamma synonym", "delta synonym", "epsilon synonym", "zeta synonym"].map((text, index): SyntheticFeature => ({ evidenceId: `lex-${index}`, side: "source", channel: "lexical", facts: [{ id: factId(`lex-${index}`), annotation: { property: "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym", text } }] })),
      { evidenceId: "unsupported", side: "target", channel: "relational", facts: [{ id: factId("only-unsupported"), unsupported: true }] },
    ],
    labels: { [`${EX}Parent`]: "Parent P" },
  },
  labels: {
    features: [
      { evidenceId: "u-some", side: "source", channel: "relational", facts: [{ id: factId("u-some"), superClass: some("holds", cls("Syndrome")) }] },
      { evidenceId: "u-only", side: "source", channel: "relational", facts: [{ id: factId("u-only"), superClass: only("holds", cls("Syndrome")) }] },
      { evidenceId: "l-some", side: "source", channel: "relational", facts: [{ id: factId("l-some"), superClass: some("longRel", cls("Site")) }] },
      { evidenceId: "l-only", side: "source", channel: "relational", facts: [{ id: factId("l-only"), superClass: only("longRel", cls("Site")) }] },
      { evidenceId: "x-min", side: "source", channel: "relational", facts: [{ id: factId("x-min"), superClass: atLeast(3, "longerRel", cls("Site")) }] },
    ],
    // `holds` deliberately has no label: the line draws its local name, the list its full IRI.
    labels: {
      [`${EX}Syndrome`]: "Syndrome X",
      [`${EX}Site`]: "Site Y",
      [`${EX}longRel`]: "has a remarkably long relation name that wraps",
      [`${EX}longerRel`]: "has an even longer relation name that the line must clip before its qualifier",
    },
  },
  dense: {
    features: [
      ...Array.from({ length: 56 }, (_, index): SyntheticFeature => ({ evidenceId: `d${index}`, side: "source", channel: index % 3 ? "hierarchy" : "relational", facts: [{ id: factId(`d${index}`), superClass: index % 3 ? cls(`Dense${index}`) : some("hasFinding", cls(`Dense${index}`)) }] })),
      ...Array.from({ length: 10 }, (_, index): SyntheticFeature => ({ evidenceId: `t${index}`, side: "target", channel: index % 2 ? "relational" : "hierarchy", facts: [{ id: factId(`t${index}`), superClass: index % 2 ? only("holds", cls("Shared")) : cls(`TParent${index}`) }] })),
    ],
    labels: Object.fromEntries([
      [`${EX}holds`, "holds"],
      [`${EX}hasFinding`, "has finding site"],
      [`${EX}Shared`, "Shared filler concept"],
      ...Array.from({ length: 56 }, (_, index) => [`${EX}Dense${index}`, index % 4 ? `Synthetic parent ${index}` : `An exceptionally long synthetic parent concept name number ${index} that needs wrapping`]),
      ...Array.from({ length: 10 }, (_, index) => [`${EX}TParent${index}`, `Target parent ${index}`]),
    ]),
  },
};

async function graphShot(page: Page, name: string) {
  // An element taller than the viewport is captured while scrolled; keep the sticky header out of it.
  await page.locator(".app-header").evaluate((header) => ((header as HTMLElement).style.position = "static"));
  await page.locator(".graph-wrap").scrollIntoViewIfNeeded();
  await page.waitForTimeout(500);
  await page.locator(".details").screenshot({ path: join(out!, `${name}.png`) });
}

for (const scheme of SCHEMES) {
  test.describe(`synthetic graphs · ${scheme}`, () => {
    test.use({ colorScheme: scheme, viewport: { width: 1440, height: 900 } });
    const open = async (page: Page, shape: keyof typeof SHAPES) => {
      await syntheticGraph(page, SHAPES[shape].features, SHAPES[shape].labels);
      await page.goto("/?details=graph");
      await mainReady(page);
      await expect(page.locator(".graph-canvas canvas").first()).toBeVisible();
      await page.waitForTimeout(800);
    };
    test("shared targets, selected relation", async ({ page }) => {
      await open(page, "shared");
      await graphShot(page, `graph-shared-${scheme}`);
      await page.locator('.graph-items [data-tour="graph-feature"]').filter({ hasText: "holds only" }).click();
      await graphShot(page, `graph-shared-selected-only-${scheme}`);
    });
    test("grouped record revealed from the list; undrawn record", async ({ page }) => {
      await open(page, "grouped");
      await graphShot(page, `graph-grouped-${scheme}`);
      await page.locator('.graph-items [data-tour="graph-feature"]').filter({ hasText: "Parent P" }).click();
      await page.getByRole("tab", { name: "Evidence", exact: true }).click();
      await page.locator('[data-evidence-id="lex-4"]').getByRole("button", { name: "Show in graph", exact: true }).click();
      await page.waitForTimeout(600);
      await graphShot(page, `graph-grouped-revealed-${scheme}`);
      await page.getByRole("tab", { name: "Evidence", exact: true }).click();
      await page.locator('[data-evidence-id="unsupported"]').getByRole("button", { name: "Show in graph", exact: true }).click();
      await page.waitForTimeout(600);
      await graphShot(page, `graph-undrawn-${scheme}`);
    });
    test("unlabelled and long relation names", async ({ page }) => {
      await open(page, "labels");
      await page.locator('.graph-items [data-tour="graph-feature"]').filter({ hasText: "holds only" }).click();
      await graphShot(page, `graph-labels-${scheme}`);
    });
    test("dense with long labels; Fit all", async ({ page }) => {
      await open(page, "dense");
      await graphShot(page, `graph-dense-${scheme}`);
      await page.locator('.graph-items [data-tour="graph-feature"]').filter({ hasText: "holds only" }).first().click();
      await graphShot(page, `graph-dense-selected-${scheme}`);
      await page.getByRole("button", { name: "Fit all", exact: true }).click();
      await graphShot(page, `graph-dense-fit-${scheme}`);
    });
    test("expanded parents and 200% text on the real fixture", async ({ page }) => {
      await page.goto("/?details=graph");
      await mainReady(page);
      await page.waitForTimeout(800);
      await page.locator('.graph-items [data-tour="graph-feature"]').first().click();
      const add = page.locator(".graph-inspector").getByRole("button", { name: /^Add parents of / });
      if (await add.count()) await add.click();
      await page.waitForTimeout(800);
      await graphShot(page, `graph-expanded-selected-${scheme}`);
      for (let index = 0; index < 5; index += 1) await page.getByRole("button", { name: "Increase text size" }).click();
      await page.waitForTimeout(800);
      await graphShot(page, `graph-text200-selected-${scheme}`);
    });
  });
}
