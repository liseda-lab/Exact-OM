import { mkdirSync } from "node:fs";
import { join } from "node:path";

import { expect, test, type Locator, type Page } from "@playwright/test";

import { artifactFor, inspectCandidate, openMainPair, reviewCases, reviewMain, settledDescriptions, studyCaseInExplanation, type Artifact, type CaseDef } from "./reviewData";
import { harnessForms, REVIEW_FORMS, seedCondition, seedToCase, seedToTutorial } from "./v2Harness";

// Visual acceptance captures on the active review publication (23 C05, review R5): the main
// app, a study explanation case and the tutorial; real graphs (not exported, the sparsest, the
// densest and the longest labels) including narrow study panels; every description variant;
// partial ELK; Decision trace reveal; light/dark, 1280×720, 1440×900, 375/320 px and 200% text.
// Variants are chosen from the served presentation metadata, and a description is captured
// only after its attribution, exact claim text and quotation styling are asserted with no
// loading placeholder left. Runs only with EXACT_SCREENSHOT_DIR, EXACT_E2E_REVIEW_MAIN_URL and
// a forms/3 harness. Study states are reached through the participant API (captures only; the
// unseeded journey is study-review-journey.spec.ts).
const out = process.env.EXACT_SCREENSHOT_DIR ?? null;
test.use({ screenshot: "off", trace: "off" });
test.beforeEach(() => test.skip(!out || !reviewMain || harnessForms !== REVIEW_FORMS, "Set EXACT_SCREENSHOT_DIR, EXACT_E2E_REVIEW_MAIN_URL and a forms/3 EXACT_E2E_STUDY_V2_CONFIG"));

const SCHEMES = ["light", "dark"] as const;

async function shot(page: Page, name: string, fullPage = false) {
  mkdirSync(out!, { recursive: true });
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(out!, `${name}.png`), fullPage });
}

/** Element capture without the sticky app header stitched into it. */
async function elementShot(page: Page, locator: Locator, name: string) {
  mkdirSync(out!, { recursive: true });
  const header = page.locator(".app-header");
  if (await header.count()) await header.evaluate((element) => ((element as HTMLElement).style.position = "static"));
  await locator.scrollIntoViewIfNeeded();
  await locator.screenshot({ path: join(out!, `${name}.png`) });
  if (await header.count()) await header.evaluate((element) => ((element as HTMLElement).style.position = ""));
}

async function openTab(page: Page, name: string) {
  const tab = page.getByRole("tab", { name, exact: true });
  if ((await tab.getAttribute("aria-selected")) !== "true") await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true");
}

const GRAPHS: [string, string, number][] = [
  ["not-exported", "review-04", 1],
  ["sparsest", "review-18", 3],
  ["dense", "review-02", 4],
  ["long-labels", "review-13", 1],
];

const cats = (artifact: Artifact) => new Set(artifact.claims.map((claim) => claim.category));
interface Variant {
  name: string;
  select: (artifact: Artifact) => boolean;
  title: string;
  tag: string;
  quote: boolean;
}
const VARIANTS: Variant[] = [
  { name: "reviewed-summary", select: (a) => a.presentation?.kind === "reviewed_comparison" && a.support_status === "independently_reviewed" && [...cats(a)].every((c) => c === "meaning"), title: "How the two descriptions compare", tag: "Generated · reviewed", quote: false },
  { name: "reviewed-descriptions-side-by-side", select: (a) => a.presentation?.kind === "paired_context" && a.support_status === "independently_reviewed", title: "Descriptions side by side", tag: "Generated · reviewed", quote: false },
  { name: "reviewed-grouped", select: (a) => a.presentation?.kind === "reviewed_comparison" && cats(a).has("difference") && cats(a).has("agreement"), title: "How the two descriptions compare", tag: "Generated · reviewed", quote: false },
  { name: "extract-definitions-side-by-side", select: (a) => a.presentation?.kind === "paired_context" && a.preparation_kind === "fact_extract", title: "Definitions side by side", tag: "Original excerpts", quote: true },
  { name: "extract-one-sided", select: (a) => (a.presentation?.kind === "source_only_profile" || a.presentation?.kind === "target_only_profile") && a.preparation_kind === "fact_extract", title: "One-sided information", tag: "Original excerpts", quote: true },
];

interface Found {
  variant: Variant;
  item: CaseDef;
  position: number;
  artifact: Artifact;
}

/** The first current comparison of each variant, from served metadata (never assumed positions). */
async function findVariants(page: Page): Promise<Found[]> {
  const found = new Map<string, Found>();
  for (const item of reviewCases())
    for (const candidate of item.candidates) {
      if (found.size === VARIANTS.length) break;
      const artifact = await artifactFor(page, "pair_comparison", item.source, candidate.entity);
      const variant = artifact && VARIANTS.find((entry) => !found.has(entry.name) && entry.select(artifact));
      if (variant && artifact) found.set(variant.name, { variant, item, position: candidate.display_position, artifact });
    }
  expect([...found.keys()].sort()).toEqual(VARIANTS.map((variant) => variant.name).sort());
  return [...found.values()];
}

/** The comparison shows exactly this variant (R1/R5): attribution, wording and quotation. */
async function expectVariant(page: Page, found: Found) {
  await settledDescriptions(page);
  const panel = page.locator(".comparison");
  await expect(panel.getByRole("heading").first()).toHaveText(found.variant.title);
  await expect(panel.locator(".origin-tag")).toHaveText(found.variant.tag);
  for (const claim of found.artifact.claims) await expect(panel.locator(found.variant.quote ? ".claim-quote" : ".claim-text").filter({ hasText: claim.text })).toBeVisible();
  if (!found.variant.quote) await expect(panel.locator(".claim-quote")).toHaveCount(0);
  return panel;
}

for (const scheme of SCHEMES) {
  test.describe(`main app · ${scheme}`, () => {
    test.use({ colorScheme: scheme });

    test("graphs on real pairs at 1280×720 and 1440×900", async ({ page }) => {
      test.setTimeout(300_000);
      for (const [width, height] of [[1280, 720], [1440, 900]]) {
        await page.setViewportSize({ width, height });
        for (const [name, caseId, position] of GRAPHS) {
          await openMainPair(page, reviewCases().find((item) => item.case_id === caseId)!, position);
          await openTab(page, "Evidence graph");
          await expect(page.locator(".graph-canvas canvas, .graph-empty").first()).toBeVisible({ timeout: 60_000 });
          await page.waitForTimeout(1500);
          await elementShot(page, page.getByRole("tabpanel"), `main-graph-${name}-${width}-${scheme}-panel`);
          await page.locator(".graph-canvas, .graph-empty").first().scrollIntoViewIfNeeded();
          await shot(page, `main-graph-${name}-${width}-${scheme}`);
        }
      }
    });

    test("every description variant, the referenced-label citation, ELK and Decision trace", async ({ page }) => {
      test.setTimeout(600_000);
      const variants = await findVariants(page);
      for (const [label, width, height] of [["1440", 1440, 900], ["375", 375, 812]] as [string, number, number][]) {
        if (label === "375" && scheme === "dark") continue;
        await page.setViewportSize({ width, height });
        for (const found of variants) {
          await openMainPair(page, found.item, found.position);
          const panel = await expectVariant(page, found);
          await elementShot(page, panel, `main-description-${found.variant.name}-${label}-${scheme}`);
        }
      }
      await page.setViewportSize({ width: 1440, height: 900 });
      // Profiles: a typed necessary-conditions template and a reviewed profile citing a referenced label.
      const cases = reviewCases();
      const review01 = cases.find((item) => item.case_id === "review-01")!;
      for (const [name, position] of [["necessary-conditions-template", 1], ["referenced-label", 4]] as [string, number][]) {
        await openMainPair(page, review01, position);
        const card = page.locator(".entity-card-target");
        const profile = await artifactFor(page, "entity_profile", review01.candidates.find((candidate) => candidate.display_position === position)!.entity);
        for (const claim of profile!.claims) await expect(card.locator(".claim-text").filter({ hasText: claim.text })).toBeVisible();
        await expect(card.locator(".origin-tag").first()).toHaveText(name === "referenced-label" ? "Generated · reviewed" : "From original axioms");
        await elementShot(page, card, `main-profile-${name}-${scheme}`);
      }
      // The citation names the referenced entity (R2): small intestine cancer, not ileum cancer.
      await page.locator(".entity-card-target").getByRole("button", { name: "Open the cited original record: Label of DOID:10154" }).click();
      await expect(page.getByRole("dialog").locator('[data-subject-iri="http://purl.obolibrary.org/obo/DOID_10154"]')).toBeVisible();
      await shot(page, `main-citation-referenced-label-${scheme}`);
      await page.keyboard.press("Escape");
      await openMainPair(page, cases.find((item) => item.case_id === "review-02")!, 4);
      await page.getByRole("button", { name: "Open decision trace" }).click();
      await expect(page.getByRole("heading", { name: "How this pair was decided" })).toBeFocused();
      await shot(page, `main-trace-reveal-1440-${scheme}`);
      const ontologies = Object.fromEntries((await (await page.request.get(`${reviewMain}/api/v1/ontologies`)).json()).items.map((item: { name: string; ontology_version_id: string }) => [item.name, item.ontology_version_id]));
      await page.goto(`${reviewMain}/browse/?${new URLSearchParams({ so: ontologies.NCIT, s: "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C9305", sk: "class", to: ontologies.DOID, tk: "class" })}`);
      const source = page.getByRole("region", { name: "Source ontology browser" });
      await expect(source.locator(".focus-label")).toHaveText("Malignant Neoplasm");
      await source.getByRole("radio", { name: "ELK inferred · partial" }).click();
      await source.getByRole("button", { name: "About ELK partial results" }).click();
      await expect(source.locator(".child-list li").first()).toBeVisible();
      await shot(page, `main-browse-elk-entry-points-${scheme}`);
    });
  });

  test.describe(`study and tutorial · ${scheme}`, () => {
    test.use({ colorScheme: scheme });

    test("description variants in study explanation cases, wide and narrow", async ({ page }) => {
      test.setTimeout(900_000);
      const variants = await findVariants(page);
      for (const found of variants) {
        for (const [label, width, height] of [["1440", 1440, 900], ["375", 375, 812]] as [string, number, number][]) {
          if (label === "375" && scheme === "dark" && found.variant.quote) continue;
          await page.setViewportSize({ width, height });
          await studyCaseInExplanation(page, found.item.case_id);
          await inspectCandidate(page, found.position);
          const panel = await expectVariant(page, found);
          await elementShot(page, panel, `study-description-${found.variant.name}-${label}-${scheme}`);
        }
      }
    });

    test("narrow study graph: context outside the canvas is named and reachable", async ({ page }) => {
      test.setTimeout(300_000);
      for (const [label, width, height, scale] of [["375", 375, 812, 1], ["320-200pct", 320, 720, 2]] as [string, number, number, number][]) {
        await page.addInitScript((value) => localStorage.setItem("exact.textScale", String(value)), scale);
        await page.setViewportSize({ width, height });
        await studyCaseInExplanation(page, "review-02");
        await inspectCandidate(page, 4);
        await openTab(page, "Evidence graph");
        await expect(page.locator(".graph-outside")).toBeVisible();
        await page.waitForTimeout(1200);
        await page.locator(".graph-outside").scrollIntoViewIfNeeded();
        await shot(page, `study-graph-narrow-${label}-${scheme}`);
      }
    });

    test("explanation case, baseline case and graph lesson side by side, desktop, phone and 200% text", async ({ page }) => {
      test.setTimeout(420_000);
      for (const [label, width, height, scale] of [["1280", 1280, 720, 1], ["1440", 1440, 900, 1], ["375", 375, 812, 1], ["320-200pct", 320, 720, 2]] as [string, number, number, number][]) {
        await page.addInitScript((value) => localStorage.setItem("exact.textScale", String(value)), scale);
        await page.setViewportSize({ width, height });
        // Tutorial: the graph lesson, with its adjacent task and Show me.
        const api = await seedToTutorial(page);
        const tutorial = (await api.state()).tutorial;
        await api.write("PUT", "/tutorial/progress", { tutorial_version: tutorial.version, position: { view: "lesson", lesson_id: "graph", question_id: null } });
        await page.reload();
        await expect(page.getByRole("heading", { level: 1, name: /^Lesson 4 of 6/ })).toBeVisible();
        await openTab(page, "Evidence graph");
        await expect(page.locator(".graph-canvas canvas").first()).toBeVisible();
        await page.waitForTimeout(1500);
        await shot(page, `tutorial-graph-lesson-${label}-${scheme}`);
        // Study: both conditions in a new test session, in the order the schedule assigns them.
        const caseApi = await seedToCase(page);
        const first = (await caseApi.currentCase()).condition as "explanation" | "ontology_baseline";
        for (const condition of first === "explanation" ? (["explanation", "ontology_baseline"] as const) : (["ontology_baseline", "explanation"] as const)) {
          await seedCondition(page, caseApi, condition);
          await page.reload();
          // The rail is a landmark only on wide layouts; the case question is there at every width.
          await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible({ timeout: 60_000 });
          if (condition === "ontology_baseline") {
            await shot(page, `study-baseline-case-${label}-${scheme}`);
            continue;
          }
          await settledDescriptions(page);
          await shot(page, `study-explanation-case-${label}-${scheme}`);
          if (label !== "1440") continue;
          await openTab(page, "Hierarchy");
          await page.getByRole("region", { name: "Source ontology browser" }).getByRole("radio", { name: "ELK inferred · partial" }).click();
          await shot(page, `study-explanation-elk-${label}-${scheme}`);
          await openTab(page, "Evidence graph");
          await expect(page.locator(".graph-canvas canvas, .graph-empty").first()).toBeVisible({ timeout: 60_000 });
          await shot(page, `study-explanation-graph-${label}-${scheme}`);
        }
      }
    });
  });
}
