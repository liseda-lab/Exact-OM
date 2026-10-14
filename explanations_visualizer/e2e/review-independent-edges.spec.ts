import { mkdir } from "node:fs/promises";
import { join } from "node:path";

import { expect, test, type Page, type Route } from "@playwright/test";

import type { Candidate, EntityRef, Page as CandidatePage } from "../src/lib/types";
import type { ReviewCase } from "../src/lib/reviewCatalogue";

// Promoted from the 14 October independent review's two retained reproductions. Boundary
// responses below are explicitly injected; underlying sources, pairs, scores and prose come
// from the actual bound review package. These do not replace the unmodified acceptance journey.
const main = process.env.EXACT_E2E_REVIEW_MAIN_URL;
const captures = process.env.EXACT_E2E_STYLE_DIAGNOSTICS;
/** Public exploration states only; fixed names contain no URL or session information. */
async function captureBoundary(page: Page, name: string) {
  if (!captures) return;
  await mkdir(captures, { recursive: true, mode: 0o700 });
  await page.screenshot({ path: join(captures, `hr2-${name}.png`), fullPage: false });
}
test.use({ viewport: { width: 1440, height: 1000 } });
test.beforeEach(() => test.skip(!main, "Set EXACT_E2E_REVIEW_MAIN_URL to the review exploration service"));
const rail = (page: Page) => page.getByRole("complementary", { name: "Choose a source and candidate" });
const params = (page: Page) => new URL(page.url()).searchParams;
const same = (a: EntityRef, b: EntityRef) => a.ontology_version_id === b.ontology_version_id && a.iri === b.iri && a.kind === b.kind;
const curie = (iri: string) => iri.replace(/^.*[#/]/, "").replace("_", ":");
const cataloguePath = "**/api/v1/review-bundle";
const candidatesPath = /\/api\/v1\/runs\/[^/]+\/candidates\?/;
type Catalogue = Record<string, unknown> & { source_run_id: string; cases: ReviewCase[] };

async function read<T>(page: Page, path: string): Promise<T> {
  const response = await page.request.get(`${main}${path}`);
  expect(response.ok(), path).toBe(true);
  return await response.json() as T;
}

async function fixture(page: Page) {
  const about = await read<Catalogue>(page, "/api/v1/review-bundle");
  const item = about.cases[0];
  let cursor: string | null = null;
  let template: CandidatePage<Candidate> | null = null;
  const rows: Candidate[] = [];
  for (let n = 0; n < 20; n++) {
    const q: URLSearchParams = new URLSearchParams({ source: item.source.iri, source_kind: item.source.kind, limit: "100", ...(cursor ? { cursor } : {}) });
    template = await read<CandidatePage<Candidate>>(page, `/api/v1/runs/${encodeURIComponent(about.source_run_id)}/candidates?${q}`);
    rows.push(...template!.items);
    cursor = template!.next_cursor;
    if (!cursor) break;
  }
  expect(cursor, "bounded test setup must completely read this single source").toBeNull();
  const admitted = rows.filter((row) => item.targets.some((target) => same(target, row.target)));
  const others = rows.filter((row) => !item.targets.some((target) => same(target, row.target)));
  expect(admitted.length).toBeGreaterThan(0);
  expect(others.length).toBeGreaterThanOrEqual(5);
  return { about, item, template: template!, admitted, others };
}

type Fixture = Awaited<ReturnType<typeof fixture>>;
async function repage(page: Page, data: Fixture, rows: Candidate[], intercept?: (route: Route, index: number) => Promise<boolean>) {
  const requested: number[] = [];
  await page.route(candidatesPath, async (route) => {
    const url = new URL(route.request().url());
    if (url.searchParams.get("source") !== data.item.source.iri) return route.continue();
    const index = Number((url.searchParams.get("cursor") ?? "edge-0").replace("edge-", ""));
    requested.push(index);
    if (await intercept?.(route, index)) return;
    const items = rows.slice(index, index + 1);
    const next = index + 1 < rows.length ? `edge-${index + 1}` : null;
    await route.fulfill({ json: { ...data.template, items, returned_count: items.length, total_count: rows.length, next_cursor: next, truncated: Boolean(next) } });
  });
  return requested;
}
const exampleUrl = (data: Fixture) => `${main}/?${new URLSearchParams({ run: data.about.source_run_id, source: data.item.source.iri, sk: data.item.source.kind, list: "examples" })}`;
async function pairOpen(page: Page) {
  await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible({ timeout: 60_000 });
}
const unresolved = (page: Page) => page.getByRole("alert").filter({ hasText: "No unprepared candidate has been selected automatically" });

// Original independent reproduction: retain both its heading and source-list assertions.
test("review: an explicit examples view falls back to source browsing when the catalogue is absent (injected)", async ({ page }) => {
  await page.route(cataloguePath, (route) => route.fulfill({ status: 404, json: { code: "review_bundle_not_available", message: "No development review catalogue is prepared", retryable: false } }));
  await page.goto(`${main}/?list=examples`);
  await pairOpen(page);
  await expect(rail(page).getByRole("list", { name: "Sources in this run" })).toBeVisible({ timeout: 5000 });
  await expect(rail(page).locator(".catalogue-note")).toContainText("All sources remain available");
  await captureBoundary(page, "absent-catalogue-source-browsing");
});

// Original independent reproduction: five unprepared pages before any catalogue target.
test("review: a prepared example resolves its target beyond the initial candidate pages (injected paging)", async ({ page }) => {
  test.setTimeout(120_000);
  const data = await fixture(page);
  const requested = await repage(page, data, [...data.others.slice(0, 5), ...data.admitted, ...data.others.slice(5)]);
  await page.goto(exampleUrl(data));
  await pairOpen(page);
  await expect(rail(page).locator(".candidate-item.current .prepared-flag")).toBeVisible({ timeout: 5000 });
  expect(requested).toEqual([0, 1, 2, 3, 4, 5]);
});

for (const variant of ["empty", "unavailable run", "failed"] as const) {
  test(`HR2-A: an explicit examples URL with a ${variant} catalogue keeps sources and the deliberate pair (injected)`, async ({ page }) => {
    test.setTimeout(120_000);
    const data = await fixture(page);
    const chosen = data.others[0];
    await page.route(cataloguePath, (route) => variant === "failed"
      ? route.fulfill({ status: 503, json: { code: "unavailable", message: "Injected unavailable catalogue", retryable: true } })
      : route.fulfill({ json: { ...data.about, ...(variant === "empty" ? { cases: [] } : { source_run_id: "sha256:injected-unavailable-run" }) } }));
    await page.goto(`${exampleUrl(data)}&pair=${encodeURIComponent(chosen.pair_id)}`);
    await pairOpen(page);
    await expect(rail(page).getByRole("list", { name: "Sources in this run" })).toBeVisible();
    await expect(rail(page).locator(".catalogue-note")).toContainText(variant === "empty" ? "No prepared examples are listed" : variant === "unavailable run" ? "not in this bundle" : "could not be loaded");
    expect(params(page).get("pair")).toBe(chosen.pair_id);
    await page.unroute(cataloguePath);
    await rail(page).locator(".catalogue-note").getByRole("button", { name: "Try again" }).click();
    await expect(rail(page).getByRole("list", { name: "Prepared examples" })).toBeVisible();
    expect(params(page).get("pair")).toBe(chosen.pair_id);
    await expect(rail(page).locator(".candidate-item.current .prepared-flag")).toHaveCount(0);
  });
}

test("HR2-A: a source chosen while catalogue recovery is pending wins over its late response (injected)", async ({ page }) => {
  test.setTimeout(120_000);
  const data = await fixture(page);
  let release: () => void = () => undefined;
  const released = new Promise<void>((resolve) => { release = resolve; });
  let attempt = 0;
  await page.route(cataloguePath, async (route) => {
    if (++attempt === 1) return route.fulfill({ status: 503, json: { code: "unavailable", message: "Injected unavailable catalogue", retryable: true } });
    await released;
    return route.fulfill({ json: data.about });
  });
  await page.goto(`${main}/?list=examples`);
  await pairOpen(page);
  await rail(page).locator(".catalogue-note").getByRole("button", { name: "Try again" }).click();
  // While the catalogue reloads, the visitor explicitly keeps ordinary source browsing.
  await rail(page).getByRole("radio", { name: /All sources/ }).click();
  await rail(page).getByRole("list", { name: "Sources in this run" }).locator(".source-item").nth(1).click();
  await pairOpen(page);
  const chosen = { source: params(page).get("source"), pair: params(page).get("pair") };
  expect(chosen.pair).not.toBeNull();
  release();
  await expect(rail(page).getByRole("radio", { name: /Prepared examples/ })).toContainText(String(data.about.cases.length));
  expect(params(page).get("source")).toBe(chosen.source);
  expect(params(page).get("pair")).toBe(chosen.pair);
  await expect(rail(page).getByRole("radio", { name: /All sources/ })).toHaveAttribute("aria-checked", "true");
});

for (const variant of ["absent target", "wrong typed identity"] as const) {
  test(`HR2-B: ${variant} exhausts honestly without automatic substitution (injected)`, async ({ page }) => {
    test.setTimeout(120_000);
    const data = await fixture(page);
    if (variant === "wrong typed identity") {
      await page.route(cataloguePath, (route) => route.fulfill({ json: { ...data.about, cases: data.about.cases.map((item, index) => index ? item : { ...item, targets: item.targets.map((target) => ({ ...target, kind: "individual" })) }) } }));
      expect(data.admitted.every((item) => item.target.kind === "class")).toBe(true);
    }
    const rows = variant === "absent target" ? data.others.slice(0, 6) : data.admitted;
    const requested = await repage(page, data, rows);
    await page.goto(exampleUrl(data));
    await expect(unresolved(page)).toContainText("None of this example's prepared targets is present");
    expect(params(page).get("pair")).toBeNull();
    await expect(rail(page).locator(".candidate-item.current")).toHaveCount(0);
    expect(requested).toHaveLength(rows.length);
    if (variant === "absent target") await captureBoundary(page, "prepared-targets-exhausted");
    // The fallback is an explicit visitor choice, never an automatic prepared example.
    await unresolved(page).getByRole("button", { name: "Browse all sources" }).click();
    await pairOpen(page);
    await expect(rail(page).getByRole("list", { name: "Sources in this run" })).toBeVisible();
  });
}

test("HR2-B: the page budget pauses and explicit continuation can resolve a later target (injected paging)", async ({ page }) => {
  test.setTimeout(120_000);
  const data = await fixture(page);
  expect(data.others.length).toBeGreaterThanOrEqual(26);
  const requested = await repage(page, data, [...data.others.slice(0, 26), ...data.admitted]);
  await page.goto(exampleUrl(data));
  await expect(unresolved(page)).toContainText("search paused after 20 further pages");
  expect(requested).toHaveLength(25);
  expect(params(page).get("pair")).toBeNull();
  await expect(rail(page).locator(".candidate-item.current")).toHaveCount(0);
  await captureBoundary(page, "prepared-target-search-paused");
  await unresolved(page).getByRole("button", { name: "Continue prepared-target search" }).click();
  await pairOpen(page);
  await expect(rail(page).locator(".candidate-item.current .prepared-flag")).toBeVisible();
  expect(requested).toHaveLength(27);
});

test("HR2-B: a failed candidate continuation keeps the reason and resumes only on retry (injected failure)", async ({ page }) => {
  test.setTimeout(120_000);
  const data = await fixture(page);
  let failures = 0;
  const requested = await repage(page, data, [...data.others.slice(0, 5), ...data.admitted], async (route, index) => {
    if (index !== 5 || failures) return false;
    failures++;
    await route.fulfill({ status: 503, json: { code: "unavailable", message: "Injected candidate-page failure", retryable: true } });
    return true;
  });
  await page.goto(exampleUrl(data));
  await expect(unresolved(page)).toContainText("a candidate page failed to load");
  expect(params(page).get("pair")).toBeNull();
  expect(requested).toEqual([0, 1, 2, 3, 4, 5]);
  await captureBoundary(page, "failed-candidate-page-retry");
  await unresolved(page).getByRole("button", { name: "Retry prepared-target search" }).click();
  await pairOpen(page);
  await expect(rail(page).locator(".candidate-item.current .prepared-flag")).toBeVisible();
  expect(requested).toEqual([0, 1, 2, 3, 4, 5, 5]);
});

for (const choice of ["candidate", "another example"] as const) {
  test(`HR2-B: an intervening ${choice} choice cancels target resolution and defeats its late response (injected delay)`, async ({ page }) => {
    test.setTimeout(120_000);
    const data = await fixture(page);
    let release: () => void = () => undefined;
    const released = new Promise<void>((resolve) => { release = resolve; });
    let entered: () => void = () => undefined;
    const pending = new Promise<void>((resolve) => { entered = resolve; });
    const requested = await repage(page, data, [...data.others.slice(0, 5), ...data.admitted], async (_route, index) => {
      if (index === 5) { entered(); await released; }
      return false;
    });
    await page.goto(exampleUrl(data));
    await pending;
    expect(params(page).get("pair")).toBeNull();
    const aborted = page.waitForEvent("requestfailed", { predicate: (request) => {
      const url = new URL(request.url());
      return url.pathname.endsWith("/candidates") && url.searchParams.get("source") === data.item.source.iri && url.searchParams.get("cursor") === "edge-5" && /ABORTED/i.test(request.failure()?.errorText ?? "");
    } });
    if (choice === "candidate") await rail(page).locator(".candidate-item").filter({ has: page.getByText(curie(data.others[0].target.iri), { exact: true }) }).click();
    else await rail(page).getByRole("list", { name: "Prepared examples" }).locator(".source-item").nth(1).click();
    release();
    await aborted;
    await pairOpen(page);
    const chosen = { source: params(page).get("source"), pair: params(page).get("pair") };
    if (choice === "candidate") {
      expect(chosen.pair).toBe(data.others[0].pair_id);
      await expect(rail(page).locator(".candidate-item.current .prepared-flag")).toHaveCount(0);
    } else {
      expect(chosen.source).toBe(data.about.cases[1].source.iri);
      await expect(rail(page).locator(".candidate-item.current .prepared-flag")).toBeVisible();
    }
    expect(requested).toEqual([0, 1, 2, 3, 4, 5]);
    // A reload keeps the explicit final choice; the old source's continuation cannot reselect it.
    await page.reload();
    await pairOpen(page);
    expect(params(page).get("source")).toBe(chosen.source);
    expect(params(page).get("pair")).toBe(chosen.pair);
  });
}
