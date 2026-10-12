import { expect, test, type Locator, type Page, type Route } from "@playwright/test";

import { artifactFor, inspectCandidate, openMainPair, reviewCases, reviewMain, settledDescriptions, studyCaseInExplanation, type Artifact, type CaseDef, type EntityRef } from "./reviewData";
import { harness, harnessForms, REVIEW_FORMS, scopedPath } from "./v2Harness";

// Regressions for the independent S2/Luna review (docs/verification/explanation-s2-luna-
// independent-review.md, R1–R4 and R6) on the real Luna publication, in the main app and in
// the scoped study workspace. Targets are chosen from current presentation metadata. Failure
// states (R3, R4) and synthetic cycles/chains (R6) are injected into otherwise real responses
// and are labelled as such. Requires EXACT_E2E_REVIEW_MAIN_URL and a forms/3 harness.
test.use({ viewport: { width: 1440, height: 900 }, screenshot: "off", trace: "off" });
test.beforeEach(() => test.skip(!reviewMain || !harness || harnessForms !== REVIEW_FORMS, "Set EXACT_E2E_REVIEW_MAIN_URL and a forms/3 EXACT_E2E_STUDY_V2_CONFIG"));

const categories = (artifact: Artifact) => new Set(artifact.claims.map((claim) => claim.category));
/** R1's two shapes: reviewed comparisons whose only claims are meaning prose, and reviewed per-concept prose. */
const misattributed = (artifact: Artifact) =>
  artifact.support_status === "independently_reviewed" &&
  ((artifact.presentation?.kind === "reviewed_comparison" && [...categories(artifact)].every((category) => category === "meaning" || category === "key_fact")) || artifact.presentation?.kind === "paired_context");
const extract = (artifact: Artifact) => artifact.preparation_kind === "fact_extract";

interface Target {
  item: CaseDef;
  position: number;
  artifact: Artifact;
}

async function comparisons(page: Page, select: (artifact: Artifact) => boolean): Promise<Target[]> {
  const targets: Target[] = [];
  for (const item of reviewCases())
    for (const candidate of item.candidates) {
      const artifact = await artifactFor(page, "pair_comparison", item.source, candidate.entity);
      if (artifact && select(artifact)) targets.push({ item, position: candidate.display_position, artifact });
    }
  return targets;
}

/** The comparison shows the artifact's approved wording with accurate attribution (R1). */
async function expectGeneratedComparison(panel: Locator, artifact: Artifact) {
  await expect(panel.locator(".origin-tag")).toHaveText("Generated · reviewed");
  await expect(panel.getByRole("heading").first()).toHaveText(artifact.presentation?.kind === "paired_context" ? "Descriptions side by side" : "How the two descriptions compare");
  for (const claim of artifact.claims) await expect(panel.locator(".claim-text").filter({ hasText: claim.text })).toBeVisible();
  await expect(panel.locator(".claim-quote")).toHaveCount(0);
  await expect(panel.getByText(/Quoted from each ontology/)).toHaveCount(0);
  await expect(panel.getByText(/^Nothing recorded for/)).toHaveCount(0);
  await expect(panel.getByRole("button", { name: /Original statements it quotes/ })).toHaveCount(0);
}

async function expectExtract(panel: Locator, artifact: Artifact) {
  await expect(panel.locator(".origin-tag")).toHaveText("Original excerpts");
  for (const claim of artifact.claims) await expect(panel.locator(".claim-quote").filter({ hasText: claim.text })).toBeVisible();
}

test.describe("R1 generated comparisons are shown as reviewed prose", () => {
  test("main app: all nine affected comparisons, and genuine extracts keep quotation", async ({ page }) => {
    test.setTimeout(600_000);
    const targets = await comparisons(page, misattributed);
    // The Luna publication: five meaning-only reviewed comparisons and four reviewed paired descriptions.
    expect(targets).toHaveLength(9);
    for (const target of targets) {
      await openMainPair(page, target.item, target.position);
      await expectGeneratedComparison(page.locator(".comparison"), target.artifact);
    }
    const extracts = await comparisons(page, extract);
    expect(extracts.length).toBeGreaterThan(0);
    for (const target of extracts) {
      await openMainPair(page, target.item, target.position);
      await expectExtract(page.locator(".comparison"), target.artifact);
    }
  });

  test("study workspace: the same nine in explanation cases", async ({ page }) => {
    test.setTimeout(900_000);
    const targets = await comparisons(page, misattributed);
    const byCase = new Map<string, Target[]>();
    for (const target of targets) byCase.set(target.item.case_id, [...(byCase.get(target.item.case_id) ?? []), target]);
    for (const [caseId, list] of byCase) {
      await studyCaseInExplanation(page, caseId);
      for (const target of list) {
        await inspectCandidate(page, target.position);
        await expectGeneratedComparison(page.locator(".comparison"), target.artifact);
      }
    }
    expect([...byCase.values()].flat()).toHaveLength(9);
  });
});

interface Referenced {
  item: CaseDef;
  position: number;
  entity: EntityRef;
  labels: { factId: string; subject: EntityRef }[];
}

async function referencedLabelProfiles(page: Page): Promise<Referenced[]> {
  const found: Referenced[] = [];
  const seen = new Set<string>();
  for (const item of reviewCases())
    for (const candidate of item.candidates) {
      const key = `${candidate.entity.ontology_version_id}|${candidate.entity.iri}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const artifact = await artifactFor(page, "entity_profile", candidate.entity);
      const references = artifact?.support_packet?.selection?.referenced_labels ?? [];
      const cited = new Set(artifact?.claims.flatMap((claim) => claim.fact_ids) ?? []);
      const labels = references.filter((fact) => cited.has(fact.fact_id) && fact.subject.iri !== candidate.entity.iri).map((fact) => ({ factId: fact.fact_id, subject: fact.subject }));
      if (labels.length) found.push({ item, position: candidate.display_position, entity: candidate.entity, labels });
    }
  return found;
}

/** Open a "Label of …" citation and assert the inspector's About entity, kind, ontology, literal and AST. */
async function expectReferencedLabel(page: Page, scope: Locator, label: { factId: string; subject: EntityRef }, axiomUrl: string) {
  const curie = label.subject.iri.replace(/^.*\/obo\//, "").replace("_", ":");
  await scope.getByRole("button", { name: `Open the cited original record: Label of ${curie}` }).first().click();
  const dialog = page.getByRole("dialog");
  const about = dialog.locator(`[data-subject-iri="${label.subject.iri}"]`);
  await expect(about).toHaveAttribute("data-subject-kind", label.subject.kind);
  await expect(about).toHaveAttribute("data-subject-ontology", label.subject.ontology_version_id);
  await expect(dialog.locator("[data-subject-iri]")).toHaveCount(1);
  const original = await (await page.request.get(axiomUrl, { headers: harnessHeaders(axiomUrl) })).json();
  await expect(dialog.locator(".inspector-literal")).toContainText(original.ast.value.lexical_form);
  await dialog.getByRole("button", { name: "Show full IRIs" }).click();
  await expect(dialog.locator(".fs-code")).toContainText(`<${label.subject.iri}>`);
  await page.keyboard.press("Escape");
}

let studySession: string | null = null;
const harnessHeaders = (url: string): Record<string, string> => (studySession && url.includes("/study/workspace/") ? { "X-Study-Session": studySession } : {});

test.describe("R2 referenced-label citations name the referenced entity", () => {
  test("main app: every referenced-label citation in the 33 profiles", async ({ page }) => {
    test.setTimeout(900_000);
    const profiles = await referencedLabelProfiles(page);
    expect(profiles).toHaveLength(33);
    // ileum cancer (DOID_10153) cites the label of small intestine cancer (DOID_10154).
    expect(profiles.some((profile) => profile.entity.iri.endsWith("DOID_10153") && profile.labels.some((label) => label.subject.iri.endsWith("DOID_10154")))).toBe(true);
    let checked = 0;
    for (const profile of profiles) {
      await openMainPair(page, profile.item, profile.position);
      const card = page.locator(".entity-card-target");
      for (const label of profile.labels) {
        await expectReferencedLabel(page, card, label, `${reviewMain}/api/v1/axioms/${encodeURIComponent(label.factId)}?ontology_version_id=${encodeURIComponent(label.subject.ontology_version_id)}`);
        checked += 1;
      }
    }
    expect(checked).toBe(35);
  });

  test("study workspace: the same citations in an explanation case resolve through the scope", async ({ page }) => {
    test.setTimeout(600_000);
    const profiles = (await referencedLabelProfiles(page)).filter((profile) => profile.item.case_id === "review-01");
    expect(profiles.map((profile) => profile.entity.iri.replace(/^.*\//, ""))).toEqual(expect.arrayContaining(["DOID_10153"]));
    const { current } = await studyCaseInExplanation(page, "review-01");
    studySession = (await (await page.request.get(`${harness!.origin}/api/v1/study/state`)).json()).session_id;
    const scope = current.workspace.scope_id as string;
    for (const profile of profiles) {
      await inspectCandidate(page, profile.position);
      for (const label of profile.labels)
        await expectReferencedLabel(page, page.locator(".entity-card-target"), label, `${harness!.origin}${scopedPath(scope)}/axioms/${encodeURIComponent(label.factId)}?ontology_version_id=${encodeURIComponent(label.subject.ontology_version_id)}`);
    }
  });
});

/** Open a detail tab without toggling it closed when it is already open (tabs toggle). */
async function openTab(page: Page, name: string) {
  const tab = page.getByRole("tab", { name, exact: true });
  if ((await tab.getAttribute("aria-selected")) !== "true") await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true");
}

/** The cytoscape view of the open graph. */
async function graphView(page: Page) {
  return page.locator(".graph-canvas").evaluate((element) => {
    const cy = (element as unknown as { _cyreg: { cy: { zoom: () => number; pan: () => { x: number; y: number }; edges: () => { map: <T>(fn: (item: { id: () => string }) => T) => T[] } } } })._cyreg.cy;
    return { zoom: cy.zoom(), pan: cy.pan(), edges: cy.edges().map((edge) => edge.id()) };
  });
}

/** Fail the first matching request (injected diagnostic), then pass the rest through. */
async function failOnce(page: Page, pattern: string | RegExp, status = 503) {
  let failed = 0;
  await page.route(pattern, async (route: Route) => {
    if (failed++ === 0) return route.fulfill({ status, contentType: "application/json", body: JSON.stringify({ detail: "Injected comparison-data failure" }) });
    return route.continue();
  });
  return () => failed;
}

async function expectGraphRecovery(page: Page) {
  const graph = page.locator(".graph-items");
  const comparisons = graph.locator("section").filter({ has: page.getByRole("heading", { name: "Recorded comparisons" }) });
  await expect(comparisons.getByRole("alert")).toContainText("Recorded comparisons");
  await expect(comparisons.getByText(/No comparison (record is available|was recorded)/)).toHaveCount(0);
  // Features stay usable while comparison data is unavailable.
  const feature = graph.locator('[data-tour="graph-feature"]').first();
  await expect(feature).toBeVisible();
  await page.getByRole("button", { name: "Zoom in" }).click();
  await feature.click();
  await expect(feature).toHaveAttribute("aria-pressed", "true");
  const before = await graphView(page);
  expect(before.edges.some((id) => id.startsWith("bridge:") || id.startsWith("corr:"))).toBe(false);
  await comparisons.getByRole("button", { name: "Try again" }).click();
  await expect(comparisons.getByRole("alert")).toHaveCount(0);
  await expect(comparisons.locator("li").first()).toBeVisible();
  const after = await graphView(page);
  expect(after.edges.some((id) => id.startsWith("bridge:") || id.startsWith("corr:"))).toBe(true);
  expect(after.zoom).toBeCloseTo(before.zoom, 5);
  expect(after.pan.x).toBeCloseTo(before.pan.x, 1);
  expect(after.pan.y).toBeCloseTo(before.pan.y, 1);
  await expect(feature).toHaveAttribute("aria-pressed", "true");
}

test.describe("R3 a failed comparison-data read is a failure with Retry, not an absence", () => {
  const dense = () => reviewCases().find((item) => item.case_id === "review-02")!;

  test("main app: failure, then Retry restores recorded comparisons in the same view and selection", async ({ page }) => {
    await openMainPair(page, dense(), 4);
    // The decision trace already read the pair record; the graph's comparison read is the next one.
    const failures = await failOnce(page, /\/api\/v1\/runs\/[^/]+\/pair\?/);
    await openTab(page, "Evidence graph");
    await expectGraphRecovery(page);
    expect(failures()).toBeGreaterThan(1);
  });

  test("study workspace: the scoped read fails once and recovers; a refused scope offers no Retry", async ({ page }) => {
    test.setTimeout(300_000);
    await studyCaseInExplanation(page, "review-02");
    await inspectCandidate(page, 4);
    await failOnce(page, /\/evidence-metadata\?/);
    await openTab(page, "Evidence graph");
    await expectGraphRecovery(page);
    // An unauthorized read stays inaccessible: the message, and no Retry (injected 403).
    await page.unrouteAll({ behavior: "ignoreErrors" });
    await page.route(/\/evidence-metadata\?/, (route) => route.fulfill({ status: 403, contentType: "application/json", body: JSON.stringify({ detail: "This scope is not available" }) }));
    await inspectCandidate(page, 3);
    await openTab(page, "Evidence graph");
    const comparisons = page.locator(".graph-items section").filter({ has: page.getByRole("heading", { name: "Recorded comparisons" }) });
    await expect(comparisons.getByRole("alert")).toContainText("Recorded comparisons");
    await expect(comparisons.getByRole("button", { name: "Try again" })).toHaveCount(0);
    await expect(page.locator('.graph-items [data-tour="graph-feature"]').first()).toBeVisible();
  });
});

const NCIT_MALIGNANT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C9305";
const MAIN_ROUTES = { hierarchy: /\/api\/v1\/hierarchy\?/, equivalents: /\/api\/v1\/ontologies\/[^/]+\/equivalents\?/ };
const STUDY_ROUTES = { hierarchy: /\/api\/v1\/study\/workspace\/[^/]+\/hierarchy\?/, equivalents: /\/api\/v1\/study\/workspace\/[^/]+\/equivalents\?/ };
async function versions(page: Page): Promise<Record<string, string>> {
  const body = await (await page.request.get(`${reviewMain}/api/v1/ontologies`)).json();
  return Object.fromEntries(body.items.map((item: { name: string; ontology_version_id: string }) => [item.name, item.ontology_version_id]));
}

/** Injected 25-member equivalence group of `focus`: page 1 (20 incl. the focus), page 2 fails once, then 6 (one duplicate). */
async function equivalenceGroup(page: Page, focus: EntityRef, routes: typeof MAIN_ROUTES) {
  const member = (index: number) => ({ ontology_version_id: focus.ontology_version_id, iri: `urn:test:equivalent-${index}`, kind: "class" });
  const requests: string[] = [];
  let laterFailures = 0;
  await page.route(routes.hierarchy, async (route) => {
    const url = new URL(route.request().url());
    if (url.searchParams.get("basis") !== "elk_inferred_partial" || url.searchParams.get("iri") !== focus.iri) return route.continue();
    const response = await route.fetch();
    const body = await response.json();
    return route.fulfill({ response, json: { ...body, entity_semantics: { ...(body.entity_semantics ?? {}), equivalent_entity_count: 25 } } });
  });
  await page.route(routes.equivalents, async (route) => {
    const url = new URL(route.request().url());
    requests.push(url.search);
    const cursor = url.searchParams.get("cursor");
    const page_ = (items: unknown[], next: string | null) => ({ items, returned_count: items.length, total_count: 25, next_cursor: next, truncated: Boolean(next), status: "partial", reason: "Partial equivalence; absence is unknown" });
    if (!cursor) return route.fulfill({ json: page_([{ ...focus }, ...Array.from({ length: 19 }, (_, index) => member(index + 1))], "page-2") });
    if (laterFailures++ === 0) return route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Injected later-page failure" }) });
    return route.fulfill({ json: page_([member(19), ...Array.from({ length: 5 }, (_, index) => member(index + 20))], null) });
  });
  return requests;
}

async function expectEquivalents(page: Page, browser: Locator, requests: string[]) {
  await browser.getByRole("radio", { name: "ELK inferred · partial" }).click();
  const group = browser.locator(".equivalents");
  await expect(group.locator(".section-head .meta")).toHaveText("19 of 24 shown");
  await expect(group.locator(".child-list li")).toHaveCount(19);
  await group.getByRole("button", { name: "Load more equivalent classes" }).click();
  await expect(group.getByRole("alert")).toBeVisible();
  await expect(group.locator(".child-list li")).toHaveCount(19);
  await group.getByRole("button", { name: "Try again" }).click();
  await expect(group.locator(".child-list li")).toHaveCount(24);
  await expect(group.locator(".section-head .meta")).toHaveText("24 other classes");
  const ids = await group.locator(".child-list .iri").allTextContents();
  expect(new Set(ids).size).toBe(24);
  // A basis change drops the group; returning reads its first page again, never the old cursor.
  const before = requests.length;
  await browser.getByRole("radio", { name: "Asserted" }).click();
  await expect(group).toHaveCount(0);
  await browser.getByRole("radio", { name: "ELK inferred · partial" }).click();
  await expect(group.locator(".section-head .meta")).toHaveText("19 of 24 shown");
  expect(requests.slice(before).every((search) => !search.includes("cursor="))).toBe(true);
  // Members are typed focus links.
  await group.locator(".child-list .tree-label").first().click();
  await expect(browser.locator(".focus-card .iri")).toHaveText("urn:test:equivalent-1");
}

test.describe("R4 equivalence groups page, count, recover and navigate (injected group)", () => {
  test("main app: a 25-member group", async ({ page }) => {
    const ids = await versions(page);
    const requests = await equivalenceGroup(page, { ontology_version_id: ids.NCIT, iri: NCIT_MALIGNANT, kind: "class" }, MAIN_ROUTES);
    await page.goto(`${reviewMain}/browse/?${new URLSearchParams({ so: ids.NCIT, s: NCIT_MALIGNANT, sk: "class", to: ids.DOID, tk: "class" })}`);
    await expectEquivalents(page, page.getByRole("region", { name: "Source ontology browser" }), requests);
  });

  test("study workspace: the same group through the scope", async ({ page }) => {
    test.setTimeout(300_000);
    const source = reviewCases().find((item) => item.case_id === "review-02")!.source;
    const requests = await equivalenceGroup(page, source, STUDY_ROUTES);
    await studyCaseInExplanation(page, "review-02");
    await openTab(page, "Hierarchy");
    await expectEquivalents(page, page.getByRole("region", { name: "Source ontology browser" }), requests);
  });
});

/** Injected ancestry of `focus`: a cycle back to it and to an intermediate class, a 15-level chain and a punned parent. */
async function ancestry(page: Page, focus: EntityRef, hierarchy: RegExp) {
  const node = (iri: string, kind = "class") => ({ ontology_version_id: focus.ontology_version_id, iri, kind });
  const parents: Record<string, ReturnType<typeof node>[]> = {
    [focus.iri]: [node("urn:test:A"), node("urn:test:chain-1"), node("urn:test:pun")],
    "urn:test:A": [node("urn:test:B")],
    "urn:test:B": [node(focus.iri), node("urn:test:C")],
    "urn:test:C": [node("urn:test:A")],
    "urn:test:pun": [node("urn:test:pun", "individual")],
  };
  for (let level = 1; level < 16; level += 1) parents[`urn:test:chain-${level}`] = [node(`urn:test:chain-${level + 1}`)];
  await page.route(hierarchy, async (route) => {
    const url = new URL(route.request().url());
    if (url.searchParams.get("direction") !== "parents" || url.searchParams.get("basis") !== "literal_asserted") return route.continue();
    const child = node(url.searchParams.get("iri")!, url.searchParams.get("kind") ?? "class");
    const items = (child.kind === "class" ? parents[child.iri] ?? [] : []).map((parent, index) => ({ id: `test:${child.iri}>${parent.iri}|${parent.kind}|${index}`, child, parent, basis: "literal_asserted", direct: true, relation: "subclass", axiom_id: null, origins: [], interpretation: { kind: "asserted" } }));
    return route.fulfill({ json: { items, returned_count: items.length, total_count: items.length, next_cursor: null, truncated: false, status: items.length ? "available" : "absent_in_scope", reason: null, basis: "literal_asserted", nodes: [] } });
  });
}

async function expectAncestry(page: Page, browser: Locator) {
  await expect(browser.locator(".focus-label")).not.toHaveText(/Loading label|No label in scope/);
  const focusName = (await browser.locator(".focus-label").textContent())!.trim();
  const expand = async (name: string) => browser.getByRole("button", { name: `Show parents of ${name}` }).first().click();
  // A → B → (focus: cycle) and B → C → (A: cycle).
  await expand("urn:test:A");
  await expand("urn:test:B");
  const toFocus = browser.locator(".tree-cycle").filter({ hasText: focusName });
  await expect(toFocus).toContainText("already in this path");
  await toFocus.getByRole("button", { name: "Show where it appears" }).click();
  await expect(browser.locator(".focus-card")).toBeFocused();
  await expand("urn:test:C");
  const toA = browser.locator(".tree-cycle").filter({ hasText: "urn:test:A" });
  await toA.getByRole("button", { name: "Show where it appears" }).click();
  await expect(browser.getByRole("button", { name: "urn:test:A", exact: true })).toBeFocused();
  expect(await browser.getByRole("button", { name: /^Show parents of / }).evaluateAll((nodes) => nodes.some((node) => (node as HTMLButtonElement).disabled))).toBe(false);
  // A punned parent (same IRI, another kind) is a distinct identity, not a cycle.
  await expand("urn:test:pun");
  await expect(browser.locator(".tree-cycle").filter({ hasText: "urn:test:pun" })).toHaveCount(0);
  // A legitimate 15-level chain: the limit is stated and continuing refocuses on that class.
  for (let level = 1; level <= 12; level += 1) await expand(`urn:test:chain-${level}`);
  const limit = browser.locator(".tree-limit");
  await expect(limit).toContainText("12 levels are shown under one focus.");
  await limit.getByRole("button", { name: "Continue from urn:test:chain-13" }).click();
  await expect(browser.locator(".focus-card .iri")).toHaveText("urn:test:chain-13");
  await expect(browser.getByRole("button", { name: "Show parents of urn:test:chain-14" })).toBeEnabled();
}

test.describe("R6 hierarchy cycles and depth limits are explained and navigable (injected ancestry)", () => {
  test("main app: cycle return links, a continuation at the depth limit, punned identities", async ({ page }) => {
    const ids = await versions(page);
    await ancestry(page, { ontology_version_id: ids.NCIT, iri: NCIT_MALIGNANT, kind: "class" }, MAIN_ROUTES.hierarchy);
    await page.goto(`${reviewMain}/browse/?${new URLSearchParams({ so: ids.NCIT, s: NCIT_MALIGNANT, sk: "class", to: ids.DOID, tk: "class" })}`);
    await expectAncestry(page, page.getByRole("region", { name: "Source ontology browser" }));
  });

  test("study workspace: the same through the scope", async ({ page }) => {
    test.setTimeout(300_000);
    const source = reviewCases().find((item) => item.case_id === "review-02")!.source;
    await ancestry(page, source, STUDY_ROUTES.hierarchy);
    await studyCaseInExplanation(page, "review-02");
    await openTab(page, "Hierarchy");
    await expectAncestry(page, page.getByRole("region", { name: "Source ontology browser" }));
  });
});

test.describe("Graph context outside a narrow study canvas is announced and reachable", () => {
  for (const [width, height, scale, scheme] of [[375, 812, 1, "light"], [320, 720, 2, "dark"]] as [number, number, number, "light" | "dark"][])
    test(`${width} px, ${scale * 100}% text, ${scheme}`, async ({ page }) => {
      test.setTimeout(300_000);
      await page.emulateMedia({ colorScheme: scheme });
      await page.addInitScript((value) => localStorage.setItem("exact.textScale", String(value)), scale);
      await page.setViewportSize({ width, height });
      await studyCaseInExplanation(page, "review-02");
      await inspectCandidate(page, 4);
      await openTab(page, "Evidence graph");
      const status = page.locator(".graph-outside");
      await expect(status).toContainText(/\d+ of \d+ items are outside this view; the item list has every item\./);
      const right = status.getByRole("button", { name: /^Show \d+ items? to the right$/ });
      await expect(right).toBeVisible();
      const before = await graphView(page);
      await right.click();
      const after = await graphView(page);
      expect(after.pan.x).toBeLessThan(before.pan.x);
      expect(after.zoom).toBeCloseTo(before.zoom, 5);
      // The previously hidden side is now reachable the other way; the list still names everything.
      await expect(status.getByRole("button", { name: /to the left$/ })).toBeVisible();
      await expect(page.locator('.graph-items [data-tour="graph-feature"]').first()).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)).toBe(false);
    });
});
