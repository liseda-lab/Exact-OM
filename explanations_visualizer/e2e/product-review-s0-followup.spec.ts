import { expect, test, type Page } from "@playwright/test";

import { atLeast, cls, contrast, drawnGraph, EX, exactly, factId, only, some, syntheticGraph, type SyntheticFeature } from "./graphFixtures";
import { harness, seedToTutorial, type ParticipantApi } from "./v2Harness";

// Regressions for the independent S0 review (S0-R1–R4, S0-C1). The review's three diagnostics
// asserted the defects; these assert the corrected behaviour. Main-app graph tests override only
// the pair-evidence/axiom/label READ responses with invented example.org records (see
// graphFixtures.ts); assessment tests run on the real v2 study service (HTTPS, PostgreSQL,
// compiled frontend, normal CSP) with sessions API-seeded to the assessment only.
test.use({ screenshot: "off", trace: "off", viewport: { width: 1440, height: 900 } });

const RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label";
const SYNONYM = "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym";

async function openGraph(page: Page) {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible();
  await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
  await expect(page.locator(".graph-canvas canvas").first()).toBeVisible();
  await expect.poll(async () => (await drawnGraph(page)).elements.length).toBeGreaterThan(1);
}

const feature = (page: Page, text: string) => page.locator('.graph-items [data-tour="graph-feature"]').filter({ hasText: text });

/** Whether the selected line and both its ends are inside the graph canvas. */
const selectionInView = (page: Page) =>
  page.locator(".graph-canvas").evaluate((element) => {
    type Collection = { empty: () => boolean; union: (other: Collection) => Collection; connectedNodes: () => Collection; renderedBoundingBox: () => { x1: number; x2: number; y1: number; y2: number } };
    const cy = (element as unknown as { _cyreg: { cy: { width: () => number; height: () => number; $: (selector: string) => Collection } } })._cyreg.cy;
    const selected = cy.$("edge:selected");
    if (selected.empty()) return false;
    const box = selected.union(selected.connectedNodes()).renderedBoundingBox();
    return box.x1 >= 0 && box.y1 >= 0 && box.x2 <= cy.width() && box.y2 <= cy.height();
  });

test.describe("S0-R1 relationships sharing a target keep their own meaning", () => {
  const shared: SyntheticFeature[] = [
    { evidenceId: "rel-some", side: "source", channel: "relational", facts: [{ id: factId("some"), superClass: some("holds", cls("Syndrome")) }] },
    { evidenceId: "rel-only", side: "source", channel: "relational", facts: [{ id: factId("only"), superClass: only("holds", cls("Syndrome")) }] },
    { evidenceId: "rel-min", side: "source", channel: "relational", facts: [{ id: factId("min"), superClass: atLeast(2, "hasPart", cls("Syndrome")) }] },
    { evidenceId: "rel-exact", side: "source", channel: "relational", facts: [{ id: factId("exact"), superClass: exactly(1, "locatedIn", cls("Site")) }] },
    { evidenceId: "hier-parent", side: "source", channel: "hierarchy", facts: [{ id: factId("parent"), superClass: cls("Syndrome") }] },
  ];
  const labels = { [`${EX}Syndrome`]: "Syndrome X", [`${EX}holds`]: "holds", [`${EX}hasPart`]: "has part", [`${EX}locatedIn`]: "located in", [`${EX}Site`]: "Site Y" };

  test("graph, item list, inspector and original record agree before and after selection, tab and theme changes", async ({ page }) => {
    await syntheticGraph(page, shared, labels);
    await openGraph(page);
    const expected: Record<string, string> = { some: "holds some", only: "holds only", min: "has part at least 2", parent: "subclass of", exact: "located in exactly 1" };
    const check = async () => {
      const graph = await drawnGraph(page);
      const target = graph.elements.filter((item) => item.node && item.label === "Syndrome X");
      expect(target, "one typed node for the shared entity, labelled with the entity only").toHaveLength(1);
      for (const [name, relation] of Object.entries(expected)) {
        const edge = graph.elements.find((item) => !item.node && item.id.endsWith(factId(name)))!;
        expect(edge.label).toBe(relation);
        expect(edge.drawnLabel, `the ${name} line draws its own relation`).toBe(relation);
      }
      expect(graph.elements.filter((item) => !item.node && item.target === target[0].id)).toHaveLength(4);
    };
    await check();
    await feature(page, "holds only").click();
    await expect(page.locator(".graph-inspector")).toContainText("holds only Syndrome X");
    let graph = await drawnGraph(page);
    const selected = graph.elements.filter((item) => item.selected);
    expect(selected.map((item) => item.id)).toEqual([`e:rel-only:${factId("only")}`]);
    expect(selected[0].drawnLabel).toBe("holds only");
    await check();
    await page.locator(".graph-inspector").getByRole("button", { name: "Original record" }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toContainText("only");
    await expect(dialog).not.toContainText(/\bsome\b/);
    await page.keyboard.press("Escape");
    // Tab round trip and theme change keep the same meanings and selection.
    await page.getByRole("tab", { name: "Evidence", exact: true }).click();
    await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
    await expect.poll(async () => (await drawnGraph(page)).elements.find((item) => item.selected)?.id).toBe(`e:rel-only:${factId("only")}`);
    await page.emulateMedia({ colorScheme: "dark" });
    await check();
    graph = await drawnGraph(page);
    expect(graph.elements.find((item) => item.selected)?.drawnLabel).toBe("holds only");
  });

  test("an unlabelled property is drawn by its local name; parallel and wrapped labels never overlap each other or a node", async ({ page }) => {
    // The review's diagnostic shape: some/only on a property with no label (its full IRI otherwise).
    const long = "has a remarkably long relation name that wraps";
    const longer = "has an even longer relation name that the line must clip before its qualifier";
    await syntheticGraph(
      page,
      [
        { evidenceId: "u-some", side: "source", channel: "relational", facts: [{ id: factId("u-some"), superClass: some("holds", cls("Syndrome")) }] },
        { evidenceId: "u-only", side: "source", channel: "relational", facts: [{ id: factId("u-only"), superClass: only("holds", cls("Syndrome")) }] },
        { evidenceId: "l-some", side: "source", channel: "relational", facts: [{ id: factId("l-some"), superClass: some("longRel", cls("Site")) }] },
        { evidenceId: "l-only", side: "source", channel: "relational", facts: [{ id: factId("l-only"), superClass: only("longRel", cls("Site")) }] },
        { evidenceId: "x-min", side: "source", channel: "relational", facts: [{ id: factId("x-min"), superClass: atLeast(3, "longerRel", cls("Site")) }] },
      ],
      { [`${EX}Syndrome`]: "Syndrome X", [`${EX}Site`]: "Site Y", [`${EX}longRel`]: long, [`${EX}longerRel`]: longer },
    );
    await openGraph(page);
    const graph = await drawnGraph(page);
    const drawn = (name: string) => graph.elements.find((item) => !item.node && item.id.endsWith(factId(name)))!.drawnLabel;
    expect(drawn("u-some")).toBe("holds some");
    expect(drawn("u-only")).toBe("holds only");
    expect(drawn("l-only")).toBe(`${long} only`);
    expect(drawn("x-min"), "a clipped name keeps its qualifier").toBe(`${longer.slice(0, 47)}… at least 3`);
    const boxes = await page.locator(".graph-canvas").evaluate((element) => {
      type Box = { x1: number; x2: number; y1: number; y2: number };
      type Item = { id: () => string; isNode: () => boolean; style: (key: string) => string; boundingBox: (options: object) => Box };
      const cy = (element as unknown as { _cyreg: { cy: { elements: () => { map: <T>(fn: (item: Item) => T) => T[] } } } })._cyreg.cy;
      return cy.elements().map((item) => ({
        id: item.id(),
        node: item.isNode(),
        labelled: !item.isNode() && Boolean(item.style("label")),
        box: item.isNode() ? item.boundingBox({ includeLabels: false }) : item.boundingBox({ includeNodes: false, includeEdges: false, includeLabels: true }),
      }));
    });
    const overlap = (a: { x1: number; x2: number; y1: number; y2: number }, b: typeof a) => a.x1 < b.x2 && b.x1 < a.x2 && a.y1 < b.y2 && b.y1 < a.y2;
    const labelBoxes = boxes.filter((item) => item.labelled);
    expect(labelBoxes).toHaveLength(5);
    for (const label of labelBoxes) {
      for (const other of labelBoxes) if (other !== label) expect(overlap(label.box, other.box), `${label.id} and ${other.id} labels`).toBe(false);
      for (const node of boxes.filter((item) => item.node)) expect(overlap(label.box, node.box), `${label.id} label and node ${node.id}`).toBe(false);
    }
    // The list, inspector and original record keep the property's full identity.
    await expect(feature(page, `${EX}holds only Syndrome X`)).toHaveCount(1);
    await feature(page, `${EX}holds only Syndrome X`).click();
    await expect(page.locator(".graph-inspector")).toContainText(`${EX}holds only Syndrome X`);
    // Chosen from the list, the line is panned into view if it was not (the zoom is kept).
    await expect.poll(() => selectionInView(page)).toBe(true);
  });
});

test.describe("S0-R3 Show in graph reveals the exact requested record", () => {
  const texts = ["alpha label", "beta synonym", "gamma synonym", "delta synonym", "epsilon synonym", "zeta synonym"];
  const features: SyntheticFeature[] = [
    { evidenceId: "hier", side: "source", channel: "hierarchy", facts: [{ id: factId("hier"), superClass: cls("Parent") }] },
    ...texts.map((text, index): SyntheticFeature => ({ evidenceId: `lex-${index}`, side: "source", channel: "lexical", facts: [{ id: factId(`lex-${index}`), annotation: { property: index ? SYNONYM : RDFS_LABEL, text } }] })),
    { evidenceId: "multi", side: "target", channel: "relational", facts: [{ id: factId("multi-drawn"), superClass: some("holds", cls("Thing")) }, { id: factId("multi-unsupported"), unsupported: true }] },
    { evidenceId: "unsupported", side: "target", channel: "relational", facts: [{ id: factId("only-unsupported"), unsupported: true }] },
    { evidenceId: "gone", side: "target", channel: "lexical", facts: [], status: "unresolved" },
  ];

  test("every grouped record, a multi-record feature, and undrawable records, after a previous selection and a tab restore", async ({ page }) => {
    await syntheticGraph(page, features, { [`${EX}Parent`]: "Parent P", [`${EX}Thing`]: "Thing T", [`${EX}holds`]: "holds" });
    await openGraph(page);
    await expect(page.locator(".graph-items")).toContainText(/Show \d+ more records/);
    const showFromList = async (evidenceId: string) => {
      await page.getByRole("tab", { name: "Evidence", exact: true }).click();
      await page.locator(`[data-evidence-id="${evidenceId}"]`).getByRole("button", { name: "Show in graph", exact: true }).click();
      await expect(page.locator(".graph-inspector")).toBeInViewport();
    };
    for (const [index, text] of texts.entries()) {
      // A different previous selection each time.
      await feature(page, "subclass of Parent P").click();
      await expect(page.locator(".graph-inspector")).toContainText("subclass of Parent P");
      await showFromList(`lex-${index}`);
      await expect(page.locator(".graph-inspector")).toContainText(text);
      await expect(page.locator(".graph-inspector")).not.toContainText("Parent P");
      await expect.poll(async () => (await drawnGraph(page)).elements.find((item) => item.selected)?.id).toBe(`e:lex-${index}:${factId(`lex-${index}`)}`);
      await expect(feature(page, text)).toHaveAttribute("aria-pressed", "true");
    }
    await expect(page.locator(".graph-items")).not.toContainText(/Show \d+ more records/);
    // Restoring the tab keeps the revealed record.
    await page.getByRole("tab", { name: "Hierarchy", exact: true }).click();
    await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
    await expect(page.locator(".graph-inspector")).toContainText(texts.at(-1)!);
    // A feature with a drawn and an undrawn record: the drawn one is shown, the other is named.
    await showFromList("multi");
    await expect(page.locator(".graph-inspector")).toContainText("holds some Thing T");
    await expect(page.locator(".graph-records")).toContainText("has no graph form");
    // Undrawable records: no stale previous selection, a specific state and a route to the record.
    await feature(page, "holds some Thing T").click();
    await showFromList("unsupported");
    await expect(page.locator(".graph-inspector")).toContainText("This feature is not drawn");
    await expect(page.locator(".graph-inspector")).not.toContainText("Thing T");
    expect((await drawnGraph(page)).elements.filter((item) => item.selected)).toEqual([]);
    await page.locator(".graph-inspector").getByRole("button", { name: "Original record" }).click();
    await expect(page.getByRole("dialog")).toContainText("Original record");
    await page.keyboard.press("Escape");
    await showFromList("gone");
    await expect(page.locator(".graph-inspector")).toContainText("not available in this view");
  });
});

test.describe("S0-R4 graph labels stay readable during selection", () => {
  const features: SyntheticFeature[] = [
    { evidenceId: "h1", side: "source", channel: "hierarchy", facts: [{ id: factId("h1"), superClass: cls("Parent") }] },
    { evidenceId: "r1", side: "source", channel: "relational", facts: [{ id: factId("r1"), superClass: some("holds", cls("Thing")) }] },
    { evidenceId: "t1", side: "target", channel: "hierarchy", facts: [{ id: factId("t1"), superClass: cls("Other") }] },
  ];
  for (const scheme of ["light", "dark"] as const)
    for (const scale of [1, 2])
      test(`${scheme}, text size ${scale * 100}%: every label keeps at least 4.5:1 when a feature is selected`, async ({ page }) => {
        await page.emulateMedia({ colorScheme: scheme });
        await page.addInitScript((value) => localStorage.setItem("exact.textScale", String(value)), scale);
        await syntheticGraph(page, features, { [`${EX}Parent`]: "Parent P", [`${EX}Thing`]: "Thing T", [`${EX}Other`]: "Other O", [`${EX}holds`]: "holds" });
        await openGraph(page);
        await feature(page, "subclass of Parent P").click();
        const graph = await drawnGraph(page);
        expect(graph.elements.some((item) => item.faded), "the rest is de-emphasised").toBe(true);
        const focal = graph.elements.filter((item) => item.role === "focal");
        expect(focal).toHaveLength(2);
        for (const item of graph.elements.filter((element) => element.label && (element.node || element.drawnLabel))) {
          expect(item.opacity, `${item.id} is not faded as a whole`).toBe(1);
          expect(item.textOpacity).toBe(1);
          expect(contrast(item.color, item.background), `${item.id} label contrast`).toBeGreaterThanOrEqual(4.5);
        }
        expect(parseFloat(focal[0].fontSize)).toBe(15 * scale);
      });

  test("each larger text size keeps the zoom and the selected relation in view, and both compared concepts whenever they fit (21 F01)", async ({ page }) => {
    await syntheticGraph(page, features, { [`${EX}Parent`]: "Parent P", [`${EX}Thing`]: "Thing T", [`${EX}Other`]: "Other O", [`${EX}holds`]: "holds" });
    await openGraph(page);
    await feature(page, "subclass of Parent P").click();
    const view = () =>
      page.locator(".graph-canvas").evaluate((element) => {
        type Box = { x1: number; x2: number; y1: number; y2: number; w: number; h: number };
        type Collection = { union: (other: Collection) => Collection; connectedNodes: () => Collection; renderedBoundingBox: () => Box };
        const cy = (element as unknown as { _cyreg: { cy: { zoom: () => number; width: () => number; height: () => number; $: (selector: string) => Collection } } })._cyreg.cy;
        const within = (box: Box) => box.x1 >= 0 && box.y1 >= 0 && box.x2 <= cy.width() && box.y2 <= cy.height();
        const selected = cy.$("edge:selected");
        const selection = selected.union(selected.connectedNodes());
        const pair = cy.$("node[role = 'focal']");
        const both = selection.union(pair).renderedBoundingBox();
        const fits = both.w <= cy.width() - 48 && both.h <= cy.height() - 48;
        return { zoom: Math.round(cy.zoom() * 1000) / 1000, selection: within(selection.renderedBoundingBox()), pairWhenItFits: !fits || within(pair.renderedBoundingBox()), fits };
      });
    const before = await view();
    expect(before).toEqual({ zoom: before.zoom, selection: true, pairWhenItFits: true, fits: true });
    const focalSize = async () => parseFloat((await drawnGraph(page)).elements.find((item) => item.role === "focal")!.fontSize);
    for (let size = await focalSize(); size < 30; ) {
      await page.getByRole("button", { name: "Increase text size" }).click();
      const previous = size;
      await expect.poll(async () => (size = await focalSize())).toBeGreaterThan(previous);
      await expect.poll(async () => ({ ...(await view()), fits: undefined }), `focal text ${size}px`).toEqual({ zoom: before.zoom, selection: true, pairWhenItFits: true, fits: undefined });
    }
    expect(await focalSize()).toBe(30);
    // A tab round trip keeps that view.
    await page.getByRole("tab", { name: "Evidence", exact: true }).click();
    await page.getByRole("tab", { name: "Evidence graph", exact: true }).click();
    await expect.poll(async () => ({ ...(await view()), fits: undefined })).toEqual({ zoom: before.zoom, selection: true, pairWhenItFits: true, fits: undefined });
  });
});

test.describe("S0-R1/R4 rich synthetic shapes render readably", () => {
  test("dense columns with long labels: no overlaps, readable start zoom, full text in the list", async ({ page }) => {
    const long = (index: number) => `An exceptionally long synthetic parent concept name number ${index} that needs wrapping`;
    const features: SyntheticFeature[] = [
      ...Array.from({ length: 56 }, (_, index): SyntheticFeature => ({ evidenceId: `d${index}`, side: "source", channel: "hierarchy", facts: [{ id: factId(`d${index}`), superClass: cls(`Dense${index}`) }] })),
      ...Array.from({ length: 8 }, (_, index): SyntheticFeature => ({ evidenceId: `t${index}`, side: "target", channel: index % 2 ? "relational" : "hierarchy", facts: [{ id: factId(`t${index}`), superClass: index % 2 ? only("holds", cls("Shared")) : cls(`TParent${index}`) }] })),
    ];
    const labels: Record<string, string> = { [`${EX}holds`]: "holds", [`${EX}Shared`]: "Shared filler" };
    for (let index = 0; index < 56; index += 1) labels[`${EX}Dense${index}`] = long(index);
    await syntheticGraph(page, features, labels);
    await openGraph(page);
    const graph = await drawnGraph(page);
    expect(graph.zoom).toBeGreaterThanOrEqual(0.7);
    const nodes = graph.elements.filter((item) => item.node);
    expect(nodes.length).toBe(2 + 56 + 4 + 1);
    for (const side of [-1, 1]) {
      const column = nodes.filter((item) => item.role === "context" && Math.sign(item.x) === side).sort((a, b) => a.y - b.y);
      for (let index = 1; index < column.length; index += 1) expect(column[index].y - column[index].height / 2).toBeGreaterThanOrEqual(column[index - 1].y + column[index - 1].height / 2);
    }
    expect(graph.elements.filter((item) => !item.node && item.label === "holds only")).toHaveLength(4);
    await expect(feature(page, long(55))).toHaveCount(1);
  });
});

test.describe("S0-C1 score caveats are on request", () => {
  test("trace and scores panel show concise labels; meanings open from accessible disclosures", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible();
    await page.getByRole("button", { name: "Open decision trace", exact: true }).click();
    const trace = page.locator(".trace");
    await expect(trace.getByText("Matching score", { exact: true }).first()).toBeVisible();
    await expect(trace).not.toContainText("not a probability");
    const disclosure = trace.getByRole("button", { name: "What this score means" }).first();
    await expect(disclosure).toHaveAttribute("aria-expanded", "false");
    await disclosure.focus();
    await page.keyboard.press("Enter");
    await expect(trace.getByRole("button", { name: "Hide score meaning" }).first()).toHaveAttribute("aria-expanded", "true");
    await expect(trace).toContainText("not a probability");
    await page.getByRole("tab", { name: "Scores", exact: true }).click();
    const scores = page.locator(".scores-panel");
    await expect(scores).not.toContainText("not probabilities");
    await expect(scores).not.toContainText("not a probability");
    await scores.getByRole("button", { name: "About these scores" }).click();
    await expect(scores).toContainText("not probabilities of being correct");
    await scores.getByRole("button", { name: "What this score means" }).first().click();
    await expect(scores).toContainText(/calibration:/);
  });
});

// ---------------------------------------------------------------------------------------------
// S0-R2 on the real study service.

async function atQuestion(page: Page, api: ParticipantApi, index: number) {
  const state = await api.state();
  const question = state.tutorial.assessment[index];
  await api.write("PUT", "/tutorial/progress", { tutorial_version: state.tutorial.version, position: { view: "assessment", lesson_id: null, question_id: question.question_id } });
  await page.reload();
  await expect(page.locator(".assessment-item h2")).toContainText(`${index + 1} ·`);
  return { state, question };
}

const savedDraft = async (api: ParticipantApi, questionId: string) => (await api.state()).tutorial_progress.assessment_drafts?.[questionId];
const next = (page: Page) => page.getByRole("button", { name: "Next question", exact: true }).click();
const previous = (page: Page) => page.getByRole("button", { name: "Previous question", exact: true }).click();

test.describe("S0-R2 visible assessment drafts survive navigation and recovery", () => {
  test.beforeEach(() => test.skip(!harness, "Set EXACT_E2E_STUDY_V2_CONFIG"));

  test("offline: answer, Next, Previous keeps the answer; after reconnecting it is saved and still shown", async ({ page, context }) => {
    const api = await seedToTutorial(page);
    const { question } = await atQuestion(page, api, 0);
    const answer = page.getByRole("radio", { name: question.options[0].label, exact: true });
    await context.setOffline(true);
    await answer.check();
    await next(page);
    await previous(page);
    await expect(answer).toBeChecked();
    await context.setOffline(false);
    await expect.poll(async () => (await savedDraft(api, question.question_id))?.choice).toBe(question.options[0].code);
    await expect(answer).toBeChecked();
  });

  test("immediate Next/Back before the debounce, and a delayed save, keep a multi-part answer", async ({ page }) => {
    const api = await seedToTutorial(page);
    const { question } = await atQuestion(page, api, 2);
    await page.route("**/api/v1/study/tutorial/progress", async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      await route.continue().catch(() => undefined);
    });
    const rows = page.locator(".assessment-item fieldset.match-row");
    await rows.nth(0).getByRole("radio", { name: "None of these" }).check();
    await rows.nth(1).getByRole("radio", { name: "Insufficient information" }).check();
    await next(page);
    await previous(page);
    await expect(rows.nth(0).getByRole("radio", { name: "None of these" })).toBeChecked();
    await expect(rows.nth(1).getByRole("radio", { name: "Insufficient information" })).toBeChecked();
    await next(page);
    await previous(page);
    await expect.poll(async () => Object.keys((await savedDraft(api, question.question_id))?.matches ?? {}).length, { timeout: 20_000 }).toBe(2);
    await expect(rows.nth(1).getByRole("radio", { name: "Insufficient information" })).toBeChecked();
  });

  test("reload with queued work shows the queued answer and saves it", async ({ page }) => {
    const api = await seedToTutorial(page);
    const { question } = await atQuestion(page, api, 1);
    // Writes fail as network errors (retried, kept in this tab's outbox) until released.
    await page.route("**/api/v1/study/tutorial/progress", (route) => (route.request().method() === "PUT" ? route.abort("internetdisconnected") : route.continue()));
    const answer = page.getByRole("radio", { name: question.options[1].label, exact: true });
    await answer.check();
    await expect(page.locator(".save-indicator")).toContainText(/Offline|Saving/);
    await page.reload();
    await expect(page.locator(".assessment-item h2")).toContainText("2 ·");
    await expect(answer).toBeChecked();
    await page.unroute("**/api/v1/study/tutorial/progress");
    await expect.poll(async () => (await savedDraft(api, question.question_id))?.choice, { timeout: 30_000 }).toBe(question.options[1].code);
    await expect(answer).toBeChecked();
  });

  test("competing tabs: each tab's conflicting draft stays visible, is saved, and does not overwrite the other's", async ({ page, context }) => {
    const api = await seedToTutorial(page);
    const { question, state } = await atQuestion(page, api, 0);
    const q2 = state.tutorial.assessment[1];
    const other = await context.newPage();
    await other.goto(`${harness!.origin}/participate/`);
    await expect(other.locator(".assessment-item h2")).toContainText("1 ·");
    // The other tab moves on first, so this tab's next write is a stale-revision conflict (409).
    await next(other);
    await expect(other.locator(".assessment-item h2")).toContainText("2 ·");
    await expect.poll(async () => (await api.state()).tutorial_progress.position.question_id).toBe(q2.question_id);
    await page.getByRole("radio", { name: question.options[0].label, exact: true }).check();
    await expect.poll(async () => (await savedDraft(api, question.question_id))?.choice, { timeout: 20_000 }).toBe(question.options[0].code);
    // That tab followed the server's position after the conflict; its draft is still its own.
    await page.locator(".question-nav button").nth(0).click();
    await expect(page.getByRole("radio", { name: question.options[0].label, exact: true })).toBeChecked();
    // Now the other tab is stale: its draft conflicts, survives and is saved too.
    await other.getByRole("radio", { name: q2.options[1].label, exact: true }).check();
    await expect.poll(async () => (await savedDraft(api, q2.question_id))?.choice, { timeout: 20_000 }).toBe(q2.options[1].code);
    // After its conflict that tab followed the server's position (question 1); its draft is kept.
    await other.locator(".question-nav button").nth(1).click();
    await expect(other.getByRole("radio", { name: q2.options[1].label, exact: true })).toBeChecked();
    await other.locator(".question-nav button").nth(0).click();
    await expect(other.getByRole("radio", { name: question.options[0].label, exact: true })).toBeChecked();
    expect((await savedDraft(api, question.question_id))?.choice, "the first tab's draft is not overwritten").toBe(question.options[0].code);
    await other.close();
  });
});
