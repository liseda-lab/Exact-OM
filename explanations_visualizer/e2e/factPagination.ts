// Exhausts every truncatable category of the shared entity card for the synthetic
// `urn:source:paged` entity (fixture option --paged-facts 26: 26 facts per category and 55
// parents). Used by the main app and the study, which render the same component (19 F16).

import { expect, type Locator } from "@playwright/test";

export const PAGED = { facts: 26, parents: 55 };

async function loadAll(scope: Locator, name: string) {
  for (let guard = 0; guard < 12; guard += 1) {
    const button = scope.getByRole("button", { name, exact: true });
    if (!(await button.count())) return;
    await button.click();
    await expect(scope.getByRole("button", { name: "Loading…", exact: true })).toHaveCount(0);
  }
  throw new Error(`${name} never finished`);
}

async function ids(scope: Locator, selector: string, attribute = "data-fact-id") {
  return scope.locator(selector).evaluateAll((nodes, name) => nodes.map((node) => node.getAttribute(name as string)), attribute);
}

const expectedTexts = (prefix: string) => Array.from({ length: PAGED.facts }, (_, index) => `${prefix} ${String(index).padStart(3, "0")}`).sort();

/** Page every category to its end and prove each item is reachable exactly once, with honest counts. */
export async function exhaustPagedCard(card: Locator, { alternate }: { alternate: boolean }) {
  // Definitions: no completeness claim while a continuation exists.
  const definitions = card.locator('[data-category="definitions"]');
  await expect(definitions.locator(".continuation .meta").first()).toHaveText(`20 of ${PAGED.facts} definitions shown`);
  await expect(card).not.toContainText("all are shown");
  const firstPage = await ids(definitions, "[data-fact-id]");
  await loadAll(definitions, "Load more definitions");
  const allDefinitions = await ids(definitions, "[data-fact-id]");
  expect(allDefinitions).toHaveLength(PAGED.facts);
  expect(new Set(allDefinitions).size).toBe(PAGED.facts);
  expect(allDefinitions.slice(0, 20)).toEqual(firstPage);
  expect((await definitions.locator(".definition-text").allTextContents()).sort()).toEqual(expectedTexts("Paged definition"));
  await expect(definitions).toContainText(`${PAGED.facts} definitions are asserted; all are shown.`);

  // Alternate definitions exist only where the policy admits them.
  const alternates = card.locator('[data-category="alternate_definitions"]');
  if (alternate) {
    await expect(alternates.locator(".continuation .meta").first()).toHaveText(`20 of ${PAGED.facts} alternate definitions shown`);
    await loadAll(alternates, "Load more alternate definitions");
    expect(new Set(await ids(alternates, "[data-fact-id]")).size).toBe(PAGED.facts);
    await expect(alternates.locator(".continuation")).toHaveCount(0);
  } else await expect(alternates).toHaveCount(0);

  // Synonyms: continue, then expand the local six-chip view.
  const synonyms = card.locator('[data-category="synonyms"]');
  await expect(synonyms.locator(".continuation .meta").first()).toHaveText(`20 of ${PAGED.facts} synonyms shown`);
  await loadAll(synonyms, "Load more synonyms");
  await synonyms.getByRole("button", { name: /^Show \d+ more$/ }).click();
  const chips = await ids(synonyms, ".chip[data-fact-id]");
  expect(chips).toHaveLength(PAGED.facts);
  expect(new Set(chips).size).toBe(PAGED.facts);
  await expect(synonyms.locator(".continuation")).toHaveCount(0);

  // Defining facts: each restriction is an original axiom block.
  const restrictions = card.locator('[data-category="restrictions"]');
  await expect(restrictions.locator(".continuation .meta").first()).toHaveText(`20 of ${PAGED.facts} restrictions shown`);
  await loadAll(restrictions, "Load more restrictions");
  await expect(restrictions.locator(".fact-block[data-fact-id]")).toHaveCount(PAGED.facts);
  expect(new Set(await ids(restrictions, ".fact-block[data-fact-id]")).size).toBe(PAGED.facts);

  // Parents continue through the hierarchy route and are deduplicated by edge.
  const parentsSection = card.locator('[data-section="parents"]');
  await expect(parentsSection.locator('[data-category="parents"] li')).toHaveCount(50);
  await expect(parentsSection.locator(".continuation .meta")).toHaveText(`50 of ${PAGED.parents} parents shown`);
  await loadAll(parentsSection, "Load more parents");
  const edges = await ids(parentsSection, '[data-category="parents"] li', "data-edge-id");
  expect(edges).toHaveLength(PAGED.parents);
  expect(new Set(edges).size).toBe(PAGED.parents);
  await expect(card.locator('.card-section:has([data-section="parents"]) .section-head .meta')).toHaveText(`${PAGED.parents} asserted · multiple inheritance`);

  // Other recorded facts: comments.
  await card.getByRole("button", { name: /^Other recorded facts/ }).click();
  const comments = card.locator(".more-category").filter({ hasText: "Comments" });
  await expect(comments.locator("h4 .meta")).toHaveText(`20 of ${PAGED.facts}`);
  await loadAll(comments, "Load more comments");
  await expect(comments.locator("h4 .meta")).toHaveText(`${PAGED.facts} of ${PAGED.facts}`);
  expect(new Set(await ids(comments, "[data-fact-id]")).size).toBe(PAGED.facts);
  await expect(card).not.toContainText("more exist");
}

/** The hierarchy browser's own parent list continues too, deduplicated by edge. */
export async function exhaustBrowserParents(browser: Locator) {
  const parentsMeta = browser.locator(".browser-section .section-head .meta").first();
  await expect(parentsMeta).toHaveText(`50 of ${PAGED.parents} · multiple inheritance`);
  await loadAll(browser.locator(".browser-section").first(), "Load more parents");
  await expect(parentsMeta).toHaveText(`${PAGED.parents} · multiple inheritance`);
  const labels = await browser.locator(".browser-section").first().locator(":scope > .tree > .tree-item > .tree-row .tree-label").allTextContents();
  expect(labels).toHaveLength(PAGED.parents);
  expect(new Set(labels).size).toBe(PAGED.parents);
}

export const TYPED = [
  { slug: "object", kind: "object_property", label: "Paged object property", parent: /^Paged object parent \d{3}$/, name: "object property" },
  { slug: "data", kind: "data_property", label: "Paged data property", parent: /^Paged data parent \d{3}$/, name: "data property" },
] as const;

/**
 * Page a property's superproperties past the first page in its full-context card and choose a
 * later-page parent (19 F21). Returns the chosen parent's IRI; every listed parent keeps its kind.
 */
export async function chooseContinuedParent(dialog: Locator, kind: string): Promise<string> {
  const items = dialog.locator('[data-category="parents"] li');
  await expect(items).toHaveCount(50);
  await dialog.getByRole("button", { name: "Load more parents", exact: true }).click();
  await expect(items).toHaveCount(PAGED.parents);
  const kinds = await items.evaluateAll((nodes) => nodes.map((node) => node.getAttribute("data-kind")));
  expect(new Set(kinds)).toEqual(new Set([kind]));
  const chosen = items.last();
  const iri = (await chosen.getAttribute("data-iri"))!;
  await chosen.getByRole("button").click();
  return iri;
}

