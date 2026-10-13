import { expect, test, type Page, type Route } from "@playwright/test";

import { settledDescriptions } from "./reviewData";

// Spec 24 HR2 and HR4 on the real review package's exploration service (EXACT_E2E_REVIEW_MAIN_URL):
// the first visit opens a catalogue-backed prepared example; All sources keeps the run's full
// source/candidate pool; explicit links and deliberate choices win; the curated DOID starting
// point is selected and labelled without losing the other roots. Identities come from the
// service (catalogue, run, entry pages), never from constants. Negative states (failed or late
// catalogue reads, candidate paging, older entry contracts) are route interceptions, labelled
// "injected" in each title; the successful paths use the actual catalogue and data.
const main = process.env.EXACT_E2E_REVIEW_MAIN_URL ?? null;
test.use({ viewport: { width: 1440, height: 1000 } });
test.beforeEach(() => test.skip(!main, "Set EXACT_E2E_REVIEW_MAIN_URL to the review package's exploration service"));

interface Ref {
  ontology_version_id: string;
  iri: string;
  kind: string;
}
interface Case {
  case_id: string;
  source: Ref;
  targets: Ref[];
}
const same = (a: Ref, b: Ref) => a.ontology_version_id === b.ontology_version_id && a.iri === b.iri && a.kind === b.kind;
const curieOf = (iri: string) => iri.replace(/^.*[#/]/, "").replace("_", ":");

async function json<T>(page: Page, path: string): Promise<T> {
  const response = await page.request.get(`${main}${path}`);
  expect(response.ok(), path).toBe(true);
  return (await response.json()) as T;
}
const catalogue = (page: Page) => json<{ source_run_id: string; cases: Case[] }>(page, "/api/v1/review-bundle");
async function sourcesInOrder(page: Page, run: string): Promise<{ entity: Ref }[]> {
  const items: { entity: Ref }[] = [];
  let cursor: string | null = null;
  for (let guard = 0; guard < 20; guard += 1) {
    const query = new URLSearchParams({ limit: "100", ...(cursor ? { cursor } : {}) });
    const page_: { items: { entity: Ref }[]; next_cursor: string | null } = await json(page, `/api/v1/runs/${encodeURIComponent(run)}/sources?${query}`);
    items.push(...page_.items);
    cursor = page_.next_cursor;
    if (!cursor) break;
  }
  return items;
}
async function candidatesOf(page: Page, run: string, source: Ref): Promise<{ pair_id: string; target: Ref }[]> {
  const items: { pair_id: string; target: Ref }[] = [];
  let cursor: string | null = null;
  for (let guard = 0; guard < 20; guard += 1) {
    const query = new URLSearchParams({ source: source.iri, source_kind: source.kind, limit: "100", ...(cursor ? { cursor } : {}) });
    const page_: { items: { pair_id: string; target: Ref }[]; next_cursor: string | null } = await json(page, `/api/v1/runs/${encodeURIComponent(run)}/candidates?${query}`);
    items.push(...page_.items);
    cursor = page_.next_cursor;
    if (!cursor) break;
  }
  return items;
}

/** Tabs toggle: open one only when it is not already open. */
async function openTab(page: Page, name: string) {
  const tab = page.getByRole("tab", { name, exact: true });
  if ((await tab.getAttribute("aria-selected")) !== "true") await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true");
}

const rail = (page: Page) => page.getByRole("complementary", { name: "Choose a source and candidate" });
const modes = (page: Page) => rail(page).getByRole("radiogroup", { name: "Sources shown" });
const params = (page: Page) => new URL(page.url()).searchParams;
async function pairOpen(page: Page) {
  await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible({ timeout: 60_000 });
  await settledDescriptions(page);
}
/** The current candidate is the first prepared one in the order the rail shows (HR2 §2). */
async function expectFirstPreparedOpen(page: Page, item: Case, run: string) {
  const candidates = await candidatesOf(page, run, item.source);
  const opened = candidates.find((candidate) => candidate.pair_id === params(page).get("pair"));
  expect(opened, "the opened pair is one of the source's candidates").toBeTruthy();
  expect(item.targets.some((target) => same(target, opened!.target)), "the opened pair is prepared for the example").toBe(true);
  const rows = rail(page).locator(".candidate-list > li");
  const flags = await rows.evaluateAll((items) => items.map((row) => ({ current: Boolean(row.querySelector(".candidate-item.current")), prepared: Boolean(row.querySelector(".prepared-flag")) })));
  expect(flags.findIndex((row) => row.prepared)).toBe(flags.findIndex((row) => row.current));
  expect(flags.filter((row) => row.prepared)).toHaveLength(item.targets.length);
}

test.describe("HR2 prepared examples on first entry", () => {
  test("a fresh visit opens the first prepared example; All sources keeps every source and candidate", async ({ page }) => {
    test.setTimeout(180_000);
    const about = await catalogue(page);
    const [first] = about.cases;
    await page.goto(`${main}/`);
    await expect(modes(page).getByRole("radio", { name: /Prepared examples/ })).toHaveAttribute("aria-checked", "true");
    await pairOpen(page);
    expect(params(page).get("source")).toBe(first.source.iri);
    await expectFirstPreparedOpen(page, first, about.source_run_id);
    // Prepared content, not the unprepared first matcher source of the earlier landing.
    await expect(page.locator(".entity-card")).toHaveCount(2);
    await expect(page.getByText("No generated description was prepared.")).toHaveCount(0);
    await expect(page.locator(".comparison")).not.toContainText("No comparison was prepared");
    await expect(rail(page).getByRole("list", { name: "Prepared examples" }).getByRole("listitem")).toHaveCount(about.cases.length);
    await expect(rail(page).getByRole("list", { name: "Prepared examples" }).locator("[aria-current=true]")).toContainText(curieOf(first.source.iri));

    // All sources: the run's whole source pool (not filtered), the same candidates in the same order.
    const order = await rail(page).locator(".candidate-item .iri").allTextContents();
    const pair = params(page).get("pair");
    const sources = await sourcesInOrder(page, about.source_run_id);
    await modes(page).getByRole("radio", { name: /All sources/ }).click();
    await expect(modes(page).getByRole("radio", { name: /All sources/ })).toHaveAttribute("aria-checked", "true");
    await expect(modes(page).getByRole("radio", { name: /All sources/ })).toContainText(sources.length.toLocaleString());
    await expect(rail(page).locator(".source-list > li").first()).toContainText(curieOf(sources[0].entity.iri));
    expect(await rail(page).locator(".candidate-item .iri").allTextContents()).toEqual(order);
    expect(params(page).get("pair")).toBe(pair);
    // Keyboard: the switch is one tab stop; arrows move between the two choices.
    await modes(page).getByRole("radio", { name: /All sources/ }).focus();
    await page.keyboard.press("ArrowLeft");
    await expect(modes(page).getByRole("radio", { name: /Prepared examples/ })).toBeFocused();
    await expect(modes(page).getByRole("radio", { name: /Prepared examples/ })).toHaveAttribute("aria-checked", "true");
  });

  test("an example beyond the first source page stays selected; Back, Forward and reload keep each choice", async ({ page }) => {
    test.setTimeout(240_000);
    const about = await catalogue(page);
    const sources = await sourcesInOrder(page, about.source_run_id);
    const far = about.cases.find((item) => sources.findIndex((source) => same(source.entity, item.source)) >= 100);
    expect(far, "a catalogue source outside the first page of 100 sources").toBeTruthy();
    await page.goto(`${main}/`);
    await pairOpen(page);
    const firstUrl = page.url();
    await rail(page).getByRole("list", { name: "Prepared examples" }).getByRole("button", { name: new RegExp(curieOf(far!.source.iri)) }).click();
    await expect.poll(() => params(page).get("source")).toBe(far!.source.iri);
    await pairOpen(page);
    await expectFirstPreparedOpen(page, far!, about.source_run_id);
    const farUrl = page.url();
    // In All sources the selected source is shown although its page is not loaded.
    await modes(page).getByRole("radio", { name: /All sources/ }).click();
    await expect(rail(page).locator(".outside-loaded")).toContainText(curieOf(far!.source.iri));
    expect(params(page).get("source")).toBe(far!.source.iri);
    await page.reload();
    await pairOpen(page);
    expect(params(page).get("source")).toBe(far!.source.iri);
    await page.goBack();
    await expect.poll(() => page.url()).toBe(farUrl);
    await page.goBack();
    await expect.poll(() => page.url()).toBe(firstUrl);
    await pairOpen(page);
    await expect(page.locator(".candidate-item.current .prepared-flag")).toHaveCount(1);
    await page.goForward();
    await expect.poll(() => params(page).get("source")).toBe(far!.source.iri);
    await pairOpen(page);
  });

  test("a deliberate unprepared link is kept and labelled honestly; a deliberate choice is remembered for this package and run", async ({ page, context }) => {
    test.setTimeout(240_000);
    const about = await catalogue(page);
    const sources = await sourcesInOrder(page, about.source_run_id);
    const plain = sources.find((source) => !about.cases.some((item) => same(item.source, source.entity)))!.entity;
    const [candidate, other] = await candidatesOf(page, about.source_run_id, plain);
    await page.goto(`${main}/?${new URLSearchParams({ run: about.source_run_id, source: plain.iri, pair: candidate.pair_id })}`);
    await pairOpen(page);
    expect(params(page).get("pair")).toBe(candidate.pair_id);
    await expect(modes(page).getByRole("radio", { name: /All sources/ })).toHaveAttribute("aria-checked", "true");
    await expect(page.getByText("No generated description was prepared.").first()).toBeVisible();
    await page.reload();
    await pairOpen(page);
    expect(params(page).get("pair")).toBe(candidate.pair_id);
    // A deliberate choice in the rail is remembered; a new visit to / in this browser returns to it.
    await rail(page).locator(".candidate-item").filter({ hasText: curieOf(other.target.iri) }).first().click();
    await expect.poll(() => params(page).get("pair")).toBe(other.pair_id);
    const again = await context.newPage();
    await again.goto(`${main}/`);
    await pairOpen(again);
    expect(params(again).get("source")).toBe(plain.iri);
    expect(params(again).get("pair")).toBe(other.pair_id);
    await again.close();
  });

  test("an explicit pair that is not a candidate is never replaced silently; a pair-only link finds its source", async ({ page }) => {
    test.setTimeout(180_000);
    const about = await catalogue(page);
    const [first] = about.cases;
    const missing = `sha256:${"0".repeat(64)}`;
    await page.goto(`${main}/?${new URLSearchParams({ run: about.source_run_id, source: first.source.iri, pair: missing })}`);
    const alert = page.getByRole("alert").filter({ hasText: "The linked pair is not among" });
    await expect(alert).toBeVisible({ timeout: 60_000 });
    expect(params(page).get("pair")).toBe(missing);
    await alert.getByRole("button", { name: "Open a prepared example" }).click();
    await pairOpen(page);
    await expectFirstPreparedOpen(page, first, about.source_run_id);
    // Pair only: the run's record names the source.
    const [candidate] = await candidatesOf(page, about.source_run_id, about.cases[1].source);
    await page.goto(`${main}/?${new URLSearchParams({ run: about.source_run_id, pair: candidate.pair_id })}`);
    await pairOpen(page);
    expect(params(page).get("source")).toBe(about.cases[1].source.iri);
    expect(params(page).get("pair")).toBe(candidate.pair_id);
  });

  test("Browse and back returns to the same pair, tab and each side's hierarchy focus", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto(`${main}/`);
    await pairOpen(page);
    await openTab(page, "Hierarchy");
    // The side with recorded parents (the first example's source has none asserted).
    await page.getByRole("region", { name: "Target ontology browser" }).locator(".tree .tree-label").first().click();
    await expect.poll(() => params(page).get("ht")).not.toBeNull();
    const before = Object.fromEntries(["source", "pair", "details", "ht", "htk"].map((key) => [key, params(page).get(key)]));
    const run = params(page).get("run") ?? (await json<{ items: { run_id: string }[] }>(page, "/api/v1/runs")).items[0].run_id;
    await page.getByRole("button", { name: "Open both in the full ontology browser" }).click();
    await expect(page.getByRole("heading", { level: 1, name: "Browse both ontologies" })).toBeVisible();
    await page.getByRole("link", { name: "Back to the comparison" }).click();
    await pairOpen(page);
    for (const [key, value] of Object.entries(before)) expect(params(page).get(key), key).toBe(value);
    expect(params(page).get("run") ?? run).toBe(run);
    await expect(page.getByRole("tab", { name: "Hierarchy", exact: true })).toHaveAttribute("aria-selected", "true");
  });

  test.describe("catalogue failures stay visible (injected responses)", () => {
    const cases: { title: string; fulfil: (route: Route) => Promise<void>; notice: RegExp | null; examples: boolean }[] = [
      { title: "declared absence (404) is ordinary browsing", fulfil: (route) => route.fulfill({ status: 404, json: { code: "review_bundle_not_available", message: "No development review catalogue is prepared", retryable: false } }), notice: null, examples: false },
      { title: "invalid catalogue (409)", fulfil: (route) => route.fulfill({ status: 409, json: { code: "review_bundle_invalid", message: "Review catalogue failed validation", retryable: false } }), notice: /failed validation/, examples: false },
      { title: "incompatible schema", fulfil: (route) => route.fulfill({ status: 200, json: { schema_version: "exact-review-about/9", source_run_id: "x", cases: [] } }), notice: /format this page does not read/, examples: false },
      { title: "denied (403)", fulfil: (route) => route.fulfill({ status: 403, json: { code: "forbidden", message: "Forbidden", retryable: false } }), notice: /not available to this session/, examples: false },
      { title: "network failure", fulfil: (route) => route.abort("connectionreset"), notice: /not responding/, examples: false },
    ];
    for (const item of cases)
      test(`${item.title}: the first source opens and the reason is shown`, async ({ page }) => {
        test.setTimeout(120_000);
        const about = await catalogue(page);
        const [firstSource] = await sourcesInOrder(page, about.source_run_id);
        await page.route("**/api/v1/review-bundle", item.fulfil);
        await page.goto(`${main}/`);
        await pairOpen(page);
        expect(params(page).get("source")).toBe(firstSource.entity.iri);
        if (item.notice) await expect(rail(page).locator(".catalogue-note")).toContainText(item.notice);
        else await expect(rail(page).locator(".catalogue-note")).toHaveCount(0);
        await expect(modes(page)).toHaveCount(0);
        if (item.notice) {
          // Retry after the failure clears: examples become available; the current source stays.
          await page.unroute("**/api/v1/review-bundle");
          await rail(page).locator(".catalogue-note").getByRole("button", { name: "Try again" }).click();
          await expect(modes(page).getByRole("radio", { name: /Prepared examples/ })).toBeVisible();
          expect(params(page).get("source")).toBe(firstSource.entity.iri);
        }
      });

    test("a catalogue for another run offers no examples and says so (injected)", async ({ page }) => {
      await page.route("**/api/v1/review-bundle", async (route) => {
        const response = await route.fetch();
        return route.fulfill({ response, json: { ...(await response.json()), source_run_id: "sha256:another-run" } });
      });
      await page.goto(`${main}/`);
      await pairOpen(page);
      await expect(rail(page).locator(".catalogue-note")).toContainText("belong to a saved run that is not in this bundle");
    });
  });

  test("a late catalogue never overrides a choice made while it loads (injected delay)", async ({ page }) => {
    test.setTimeout(120_000);
    const about = await catalogue(page);
    const sources = await sourcesInOrder(page, about.source_run_id);
    const chosen = sources.slice(0, 100).find((source) => !about.cases.some((item) => same(item.source, source.entity)) && source !== sources[0])!.entity;
    let release: () => void = () => undefined;
    const released = new Promise<void>((resolve) => (release = resolve));
    await page.route("**/api/v1/review-bundle", async (route) => {
      await released;
      return route.continue();
    });
    await page.goto(`${main}/`);
    await expect(rail(page).getByText("Loading prepared examples…")).toBeVisible();
    await modes(page).getByRole("radio", { name: /All sources/ }).click();
    await rail(page).locator(".source-list .source-item").filter({ hasText: curieOf(chosen.iri) }).first().click();
    await expect.poll(() => params(page).get("source")).toBe(chosen.iri);
    release();
    await expect(modes(page).getByRole("radio", { name: /Prepared examples/ })).toContainText(String(about.cases.length));
    await pairOpen(page);
    expect(params(page).get("source")).toBe(chosen.iri);
    await expect(modes(page).getByRole("radio", { name: /All sources/ })).toHaveAttribute("aria-checked", "true");
  });

  test("a catalogue slower than the bounded wait leaves ordinary browsing in place (injected delay)", async ({ page }) => {
    test.setTimeout(120_000);
    const about = await catalogue(page);
    const [firstSource] = await sourcesInOrder(page, about.source_run_id);
    await page.route("**/api/v1/review-bundle", async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 12_000));
      return route.continue().catch(() => undefined);
    });
    await page.goto(`${main}/`);
    await expect.poll(() => params(page).get("source"), { timeout: 15_000 }).toBe(firstSource.entity.iri);
    await expect(modes(page).getByRole("radio", { name: /Prepared examples/ })).toContainText(String(about.cases.length), { timeout: 20_000 });
    expect(params(page).get("source")).toBe(firstSource.entity.iri);
  });

  test("a linked pair beyond the first candidate pages is found and stays selected (injected paging)", async ({ page }) => {
    test.setTimeout(180_000);
    const about = await catalogue(page);
    const item = about.cases[0];
    const all = await candidatesOf(page, about.source_run_id, item.source);
    const PAGE = 10;
    const target = all[Math.min(all.length - 1, 80)];
    // The real candidates, re-paged by 10 with labelled synthetic cursors: the first load (5 pages)
    // holds 50, so the linked pair needs further pages.
    await page.route(/\/api\/v1\/runs\/[^/]+\/candidates\?/, async (route) => {
      const url = new URL(route.request().url());
      if (url.searchParams.get("source") !== item.source.iri) return route.continue();
      const index = Number((url.searchParams.get("cursor") ?? "injected-0").replace("injected-", ""));
      const slice = all.slice(index * PAGE, index * PAGE + PAGE);
      const more = (index + 1) * PAGE < all.length;
      return route.fulfill({ json: { items: slice, returned_count: slice.length, total_count: all.length, next_cursor: more ? `injected-${index + 1}` : null, truncated: more, scope: {}, status: "available", reason: null } });
    });
    await page.goto(`${main}/?${new URLSearchParams({ run: about.source_run_id, source: item.source.iri, pair: target.pair_id })}`);
    await pairOpen(page);
    expect(params(page).get("pair")).toBe(target.pair_id);
    await expect(rail(page).locator(".candidate-item.current")).toContainText(curieOf(target.target.iri));
  });
});

// HR4 — the curated hierarchy start on Browse, and its limits.
async function ontologyIds(page: Page): Promise<Record<string, string>> {
  const body = await json<{ items: { name: string; ontology_version_id: string }[] }>(page, "/api/v1/ontologies");
  return Object.fromEntries(body.items.map((item) => [item.name, item.ontology_version_id]));
}
interface EntryPage {
  items: { entity: Ref }[];
  next_cursor: string | null;
  total_count: number;
  default_entity: Ref | null;
  selection?: { kind: string };
}
/** The ontology whose first asserted class entry page declares a curated default, as served. */
async function curatedSide(page: Page): Promise<{ ontology: string; entries: EntryPage; other: string }> {
  const ids = Object.values(await ontologyIds(page));
  for (const ontology of ids) {
    const page_ = await json<EntryPage>(page, `/api/v1/ontologies/${encodeURIComponent(ontology)}/entry-points?limit=20`);
    if (page_.selection?.kind === "curated_default") return { ontology, entries: page_, other: ids.find((id) => id !== ontology)! };
  }
  throw new Error("No ontology declares a curated default");
}
const target = (page: Page) => page.getByRole("region", { name: "Target ontology browser" });
const sourceSide = (page: Page) => page.getByRole("region", { name: "Source ontology browser" });

async function browse(page: Page, query: Record<string, string> = {}) {
  const { ontology, entries, other } = await curatedSide(page);
  await page.goto(`${main}/browse/?${new URLSearchParams({ to: ontology, so: other, ...query })}`);
  return { ontology, entries, other };
}

test.describe("HR4 curated hierarchy start", () => {
  test("a fresh Browse opens the curated default, labelled; every other root stays listed and pageable", async ({ page }) => {
    test.setTimeout(180_000);
    const { entries } = await browse(page);
    const start = entries.default_entity!;
    const side = target(page);
    await expect(side.locator(".focus-card .iri")).toHaveText(start.iri);
    await expect(side.locator(".focus-card")).toContainText("Suggested starting point");
    await expect(side.locator(".browser-section").filter({ hasText: "Children" }).locator(".child-list li").first()).toBeVisible();
    expect(params(page).get("t"), "a suggestion is not written as a deliberate focus").toBeNull();
    // The other side has many ordinary roots: still a list, never a focus.
    await expect(sourceSide(page).locator(".focus-card")).toHaveCount(0);
    await expect(sourceSide(page).locator(".entry-points .child-list li").first()).toBeVisible();
    // All roots stay reachable: the list with its continuation, the suggestion marked once.
    await side.getByRole("button", { name: /Show all top-level classes/ }).click();
    const list = side.locator(".entry-points");
    await expect(list.locator(".continuation .meta")).toContainText(`1 of ${entries.total_count} top-level entries shown`);
    await expect(list.locator(".pill-suggested")).toHaveCount(1);
    await list.getByRole("button", { name: "Load more top-level entries" }).click();
    await expect(list.locator(".child-list li")).toHaveCount(21);
    const shown = await list.locator(".child-list li .iri").allTextContents();
    expect(new Set(shown).size).toBe(shown.length);
    // Another root: a deliberate focus (URL), no longer the suggestion; reload keeps it.
    const another = list.locator(".child-list li").nth(1);
    const anotherIri = await another.locator(".iri").textContent();
    await another.getByRole("button").first().click();
    await expect.poll(() => params(page).get("t")).not.toBeNull();
    await expect(side.locator(".focus-card .iri")).not.toHaveText(start.iri);
    await expect(side.locator(".focus-card")).not.toContainText("Suggested starting point");
    await page.reload();
    await expect(side.locator(".focus-card")).toContainText(anotherIri!.replace(":", "_"));
  });

  test("an explicit focus is never replaced, and a choice made while the default loads wins (injected delay)", async ({ page }) => {
    test.setTimeout(120_000);
    const { ontology, entries } = await curatedSide(page);
    const children = await json<{ items: { child: Ref }[] }>(page, `/api/v1/hierarchy?${new URLSearchParams({ ontology_version_id: ontology, iri: entries.default_entity!.iri, kind: "class", direction: "children", limit: "5" })}`);
    const explicit = children.items[0].child;
    await browse(page, { t: explicit.iri, tk: "class" });
    await expect(target(page).locator(".focus-card .iri")).toHaveText(explicit.iri);
    await expect(target(page).locator(".focus-card")).not.toContainText("Suggested starting point");

    // Delay the first entry page; search and choose before it arrives.
    let release: () => void = () => undefined;
    const released = new Promise<void>((resolve) => (release = resolve));
    const fresh = await page.context().newPage();
    await fresh.route(/\/entry-points\?/, async (route) => {
      await released;
      return route.continue();
    });
    await fresh.goto(`${main}/browse/?${new URLSearchParams({ to: ontology })}`);
    const side = target(fresh);
    await side.getByRole("combobox").fill("cancer");
    await side.getByRole("listbox").getByRole("option").first().click();
    await expect.poll(() => new URL(fresh.url()).searchParams.get("t")).not.toBeNull();
    const chosen = new URL(fresh.url()).searchParams.get("t");
    release();
    await fresh.waitForTimeout(1500);
    await expect(side.locator(".focus-card .iri")).toHaveText(chosen!);
    expect(new URL(fresh.url()).searchParams.get("t")).toBe(chosen);
    await fresh.close();
  });

  test("older entry pages without selection, and a missing default, keep the list (injected contract variants)", async ({ page }) => {
    test.setTimeout(120_000);
    const { ontology, entries } = await curatedSide(page);
    for (const variant of [{ name: "no selection field", strip: ["selection"] }, { name: "curated without a default", strip: ["default_entity"] }]) {
      const fresh = await page.context().newPage();
      await fresh.route(/\/entry-points\?/, async (route) => {
        try {
          const response = await route.fetch();
          const body = await response.json();
          if (body.scope?.ontology_version_id === ontology) for (const key of variant.strip) body[key] = key === "default_entity" ? null : undefined;
          await route.fulfill({ response, json: body });
        } catch {
          /* the page closed while a read was intercepted */
        }
      });
      await fresh.goto(`${main}/browse/?${new URLSearchParams({ to: ontology })}`);
      const list = target(fresh).locator(".entry-points");
      await expect(list.locator(".continuation .meta"), variant.name).toContainText(`1 of ${entries.total_count} top-level entries shown`);
      await expect(target(fresh).locator(".focus-card"), variant.name).toHaveCount(0);
      await fresh.close();
    }
  });

  test("partial ELK keeps the current focus but declares no suggestion", async ({ page }) => {
    test.setTimeout(120_000);
    const { entries } = await browse(page);
    const side = target(page);
    await expect(side.locator(".focus-card")).toContainText("Suggested starting point");
    await side.getByRole("radiogroup", { name: "Relationships shown" }).getByRole("radio", { name: "ELK inferred · partial" }).click();
    await expect(side).toContainText("A missing link is unknown, not absent.");
    await expect(side.locator(".focus-card .iri")).toHaveText(entries.default_entity!.iri);
    await expect(side.locator(".focus-card")).not.toContainText("Suggested starting point");
  });

  test("the compared entity stays the focus in the pair workspace; the suggestion is only labelled in the list", async ({ page }) => {
    test.setTimeout(180_000);
    const { ontology, entries } = await curatedSide(page);
    const runs = await json<{ items: { source_ontology_version_id: string; target_ontology_version_id: string }[] }>(page, "/api/v1/runs");
    await page.goto(`${main}/`);
    await pairOpen(page);
    await openTab(page, "Hierarchy");
    const side = runs.items[0].target_ontology_version_id === ontology ? target(page) : sourceSide(page);
    const compared = await side.locator(".focus-card .iri").textContent();
    expect(compared).not.toBe(entries.default_entity!.iri);
    await side.getByRole("button", { name: "Top-level classes" }).click();
    await expect(side.locator(".entry-points .pill-suggested")).toHaveCount(1);
    await expect(side.locator(".focus-card .iri")).toHaveText(compared!);
    await expect(side.locator(".focus-card")).not.toContainText("Suggested starting point");
  });
});
