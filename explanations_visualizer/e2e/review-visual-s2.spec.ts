import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { harness, harnessForms, REVIEW_FORMS, seedCondition, seedToCase, seedToTutorial } from "./v2Harness";

// S2 visual acceptance captures on the corrected review publication (23 C05): the main app,
// a study explanation case and the tutorial side by side; real graphs (not exported, the
// sparsest, the densest and the longest labels); description kinds; partial ELK; Decision
// trace reveal; light/dark, 1280×720, 1440×900, 375/320 px and 200% text. Not an assertion
// suite beyond reaching each view: it runs only with EXACT_SCREENSHOT_DIR, the review main app
// (EXACT_E2E_REVIEW_MAIN_URL) and a forms/3 harness. Study states are seeded through the
// participant API (screenshots only; the unseeded journey is study-review-journey.spec.ts).
const out = process.env.EXACT_SCREENSHOT_DIR ?? null;
const main = process.env.EXACT_E2E_REVIEW_MAIN_URL ?? null;
test.use({ screenshot: "off", trace: "off" });
test.beforeEach(() => test.skip(!out || !main || harnessForms !== REVIEW_FORMS, "Set EXACT_SCREENSHOT_DIR, EXACT_E2E_REVIEW_MAIN_URL and a forms/3 EXACT_E2E_STUDY_V2_CONFIG"));

const SCHEMES = ["light", "dark"] as const;
type Scheme = (typeof SCHEMES)[number];

async function shot(page: Page, name: string, fullPage = false) {
  mkdirSync(out!, { recursive: true });
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(out!, `${name}.png`), fullPage });
}

interface CaseDef {
  case_id: string;
  source: { iri: string };
  candidates: { display_position: number; entity: { iri: string } }[];
}

/** The main app's run and pair for a review case candidate, resolved through the main API. */
async function pairOf(page: Page, caseId: string, position: number) {
  const definition = JSON.parse(readFileSync(harness!.publication, "utf8")).definition as { cases: CaseDef[] };
  const item = definition.cases.find((entry) => entry.case_id === caseId)!;
  const target = item.candidates.find((candidate) => candidate.display_position === position)!.entity.iri;
  const run = (await (await page.request.get(`${main}/api/v1/runs`)).json()).items[0].run_id as string;
  let cursor: string | null = null;
  for (let guard = 0; guard < 20; guard += 1) {
    const params: Record<string, string> = { source: item.source.iri, source_kind: "class", limit: "100", ...(cursor ? { cursor } : {}) };
    const page_ = await (await page.request.get(`${main}/api/v1/runs/${encodeURIComponent(run)}/candidates?${new URLSearchParams(params)}`)).json();
    const found = page_.items.find((candidate: { target: { iri: string } }) => candidate.target.iri === target);
    if (found) return { run, source: item.source.iri, pair: found.pair_id as string };
    cursor = page_.next_cursor;
    if (!cursor) break;
  }
  throw new Error(`No saved pair for ${caseId} candidate ${position}`);
}

async function openPair(page: Page, caseId: string, position: number) {
  const pair = await pairOf(page, caseId, position);
  await page.goto(`${main}/?${new URLSearchParams({ run: pair.run, source: pair.source, pair: pair.pair })}`);
  await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible({ timeout: 60_000 });
}

const GRAPHS: [string, string, number][] = [
  ["not-exported", "review-04", 1],
  ["sparsest", "review-18", 3],
  ["dense", "review-02", 4],
  ["long-labels", "review-13", 1],
];

const DESCRIPTIONS: [string, string, number][] = [
  ["reviewed-comparison", "review-01", 1],
  ["source-only-necessary-conditions", "review-01", 2],
  ["paired-excerpts", "review-01", 3],
  ["target-only", "review-12", 1],
];

for (const scheme of SCHEMES) {
  test.describe(`main app · ${scheme}`, () => {
    test.use({ colorScheme: scheme });

    test("graphs on real pairs at 1280×720 and 1440×900", async ({ page }) => {
      test.setTimeout(300_000);
      for (const [width, height] of [[1280, 720], [1440, 900]]) {
        await page.setViewportSize({ width, height });
        for (const [name, caseId, position] of GRAPHS) {
          // Tabs toggle: open the graph by one click, not by URL plus click.
          await openPair(page, caseId, position);
          await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
          await expect(page.locator(".graph-canvas canvas, .graph-empty").first()).toBeVisible({ timeout: 60_000 });
          await page.waitForTimeout(1500);
          // The whole graph panel (canvas, legend and item list), then the canvas in the viewport.
          const panel = page.getByRole("tabpanel");
          // Element captures scroll the page: keep the sticky app header out of the stitched image.
          await page.locator(".app-header").evaluate((header) => ((header as HTMLElement).style.position = "static"));
          await panel.screenshot({ path: join(out!, `main-graph-${name}-${width}-${scheme}-panel.png`) });
          await page.locator(".app-header").evaluate((header) => ((header as HTMLElement).style.position = ""));
          await page.locator(".graph-canvas, .graph-empty").first().scrollIntoViewIfNeeded();
          await shot(page, `main-graph-${name}-${width}-${scheme}`);
        }
      }
    });

    test("description kinds, ELK hierarchy and Decision trace", async ({ page }) => {
      test.setTimeout(300_000);
      await page.setViewportSize({ width: 1440, height: 900 });
      for (const [name, caseId, position] of DESCRIPTIONS) {
        await openPair(page, caseId, position);
        await page.locator(".comparison").scrollIntoViewIfNeeded();
        await page.locator(".comparison").screenshot({ path: join(out!, `main-description-${name}-comparison-${scheme}.png`) });
        await page.locator(".card-pair").screenshot({ path: join(out!, `main-description-${name}-cards-${scheme}.png`) });
      }
      await openPair(page, "review-02", 4);
      await page.getByRole("button", { name: "Open decision trace" }).click();
      await expect(page.getByRole("heading", { name: "How this pair was decided" })).toBeFocused();
      await shot(page, `main-trace-reveal-1440-${scheme}`);
      await page.setViewportSize({ width: 375, height: 812 });
      await page.reload();
      await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible({ timeout: 60_000 });
      await page.getByRole("button", { name: "Open decision trace" }).click();
      await expect(page.getByRole("heading", { name: "How this pair was decided" })).toBeFocused();
      await shot(page, `main-trace-reveal-375-${scheme}`);
      // Browse: partial ELK on a real class with two pages of ELK children.
      await page.setViewportSize({ width: 1440, height: 900 });
      const ontologies = Object.fromEntries((await (await page.request.get(`${main}/api/v1/ontologies`)).json()).items.map((item: { name: string; ontology_version_id: string }) => [item.name, item.ontology_version_id]));
      await page.goto(`${main}/browse/?${new URLSearchParams({ so: ontologies.NCIT, s: "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C9305", sk: "class", to: ontologies.DOID, tk: "class" })}`);
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
        await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
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
          await expect(page.locator(".entity-card-source .entity-title")).not.toBeEmpty();
          await shot(page, `study-explanation-case-${label}-${scheme}`);
          if (label !== "1440") continue;
          await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
          await page.getByRole("region", { name: "Source ontology browser" }).getByRole("radio", { name: "ELK inferred · partial" }).click();
          await shot(page, `study-explanation-elk-${label}-${scheme}`);
          await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
          await expect(page.locator(".graph-canvas canvas, .graph-empty").first()).toBeVisible({ timeout: 60_000 });
          await shot(page, `study-explanation-graph-${label}-${scheme}`);
        }
      }
    });
  });
}
