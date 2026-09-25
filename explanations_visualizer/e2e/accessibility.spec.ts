import AxeBuilder from "@axe-core/playwright";
import { readFileSync, writeFileSync } from "node:fs";
import { expect, test, type Page, type TestInfo } from "@playwright/test";

type StudyConfig = { origin: string; researcher_token: string; study_revision: string };
const study: StudyConfig | null = process.env.EXACT_E2E_STUDY_CONFIG
  ? JSON.parse(readFileSync(process.env.EXACT_E2E_STUDY_CONFIG, "utf8")) : null;
const fixture = process.env.EXACT_E2E_FIXTURE_URL;
const profiles = [
  { name: "desktop", width: 1440, scale: 1 },
  { name: "mobile enlarged text", width: 375, scale: 2 },
] as const;
test.use({ screenshot: "off", trace: "off" });

async function preferences(page: Page, scale: number, theme: string) {
  await page.addInitScript(({ scale, theme }) => {
    localStorage.setItem("exact.textScale", String(scale));
    localStorage.setItem("exact.theme", theme);
  }, { scale, theme });
}
async function comparisonReady(page: Page) {
  await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible();
  await expect(page.getByRole("heading", { level: 1, name: /^Does / })).not.toContainText("Loading label");
  await expect(page.locator(".entity-card")).toHaveCount(2);
}
async function welcome(page: Page) {
  const issued = await page.request.post(`${study!.origin}/api/v1/admin/studies/${encodeURIComponent(study!.study_revision)}/invitations`, {
    headers: { Authorization: `Bearer ${study!.researcher_token}` }, data: { count: 1, test: true },
  });
  expect(issued.ok()).toBe(true);
  const invitation = (await issued.json()).invitations[0].invitation as string;
  const secret = new URLSearchParams(invitation.split("#")[1]).get("invite");
  const exchanged = await page.request.post(`${study!.origin}/api/v1/study/session`, {
    headers: { Origin: study!.origin }, data: { secret },
  });
  expect(exchanged.ok()).toBe(true);
  await page.goto(`${study!.origin}/participate/`);
  await expect(page.getByRole("heading", { level: 1, name: /^Welcome,/ })).toBeVisible();
  await expect(page.getByRole("region", { name: "Participant information and consent text" })).toBeVisible();
}
async function audit(page: Page, info: TestInfo) {
  await page.evaluate(() => document.fonts.ready);
  const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
  // Keep no request bodies, cookies, invitation links, or researcher credentials.
  const summarize = (items: typeof result.violations) => items.map(({ id, impact, description, helpUrl, nodes }) => ({
    id, impact, description, helpUrl,
    nodes: nodes.map(({ target, html, failureSummary }) => ({ target, html, failureSummary })),
  }));
  const report = {
    test: info.titlePath, scope: "Automated axe browser checks only; no screen-reader/conformance claim.",
    axe_version: result.testEngine.version, viewport: page.viewportSize(),
    preferences: await page.evaluate(() => ({ theme: document.documentElement.dataset.theme, text_scale: getComputedStyle(document.documentElement).getPropertyValue("--text-scale").trim() })),
    violations: summarize(result.violations), incomplete: summarize(result.incomplete),
    horizontal_overflow: await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1),
  };
  const output = info.outputPath("accessibility-results.json");
  writeFileSync(output, JSON.stringify(report, null, 2));
  await info.attach("accessibility-results", { path: output, contentType: "application/json" });
  expect(report.preferences.theme).toBe(info.titlePath.find((part) => part.endsWith(", dark")) ? "dark" : "light");
  expect(Number(report.preferences.text_scale)).toBe(page.viewportSize()?.width === 375 ? 2 : 1);
  expect(report.violations).toEqual([]);
  expect(report.horizontal_overflow).toBe(false);
}

for (const profile of profiles) for (const theme of ["light", "dark"]) {
  test.describe(`${profile.name}, ${theme}`, () => {
    test.use({ viewport: { width: profile.width, height: 1000 } });
    test.beforeEach(async ({ page }) => preferences(page, profile.scale, theme));
    test("comparison accessibility", async ({ page }, info) => {
      await page.goto("/");
      await comparisonReady(page);
      await page.getByRole("tab", { name: "Evidence", exact: true }).click();
      await expect(page.getByRole("tabpanel")).toBeVisible();
      await audit(page, info);
    });
    test("ontology browsing accessibility", async ({ page }, info) => {
      await page.goto("/browse/");
      const search = page.getByRole("combobox", { name: "Search the source ontology" });
      await search.fill("myopathy");
      await expect(page.getByRole("listbox").getByRole("option").first()).toBeVisible();
      await search.press("ArrowDown");
      await search.press("Enter");
      await expect(page.getByRole("region", { name: "Source ontology browser" }).locator(".focus-card")).toBeVisible();
      await audit(page, info);
    });
    test("bundle library accessibility", async ({ page }, info) => {
      test.skip(!fixture, "Requires the isolated local import fixture URL");
      await page.goto(`${fixture}/library/`);
      await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
      await expect(page.getByLabel("Choose file")).toBeAttached();
      await audit(page, info);
    });
    test("study welcome accessibility", async ({ page }, info) => {
      test.skip(!study, "Requires the private synthetic study harness");
      await welcome(page);
      await audit(page, info);
    });
    test("researcher sign-in accessibility", async ({ page }, info) => {
      test.skip(!study, "Requires the private synthetic study harness");
      await page.goto(`${study!.origin}/admin/`);
      await expect(page.getByRole("heading", { level: 1, name: "Sign in with the researcher token" })).toBeVisible();
      await audit(page, info);
    });
  });
}

test("dialog keyboard focus wraps and returns to its trigger", async ({ page }) => {
  await page.goto("/");
  await comparisonReady(page);
  const help = page.getByRole("button", { name: "How to read this screen", exact: true });
  await help.focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog", { name: "How to read this screen" });
  await expect(dialog).toBeVisible();
  const close = dialog.getByRole("button", { name: "Close", exact: true });
  await expect(close).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(close).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(close).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(help).toBeFocused();
});

test("detail tabs support arrow-key navigation and activation", async ({ page }) => {
  await page.goto("/");
  await comparisonReady(page);
  const hierarchy = page.getByRole("tab", { name: "Hierarchy", exact: true });
  const evidence = page.getByRole("tab", { name: "Evidence", exact: true });
  await hierarchy.focus();
  await page.keyboard.press("ArrowRight");
  await expect(evidence).toBeFocused();
  await expect(evidence).toHaveAttribute("tabindex", "0");
  await expect(hierarchy).toHaveAttribute("tabindex", "-1");
  await expect(evidence).toHaveAttribute("aria-selected", "false");
  expect(new URL(page.url()).searchParams.get("details")).toBeNull();
  await page.keyboard.press("End");
  await expect(page.getByRole("tab", { name: "Scores", exact: true })).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await expect(hierarchy).toBeFocused();
  await page.keyboard.press("ArrowLeft");
  await expect(page.getByRole("tab", { name: "Scores", exact: true })).toBeFocused();
  await page.keyboard.press("Home");
  await expect(hierarchy).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Enter");
  await expect(evidence).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("tabpanel")).toHaveAttribute("aria-labelledby", "tab-evidence");
  expect(new URL(page.url()).searchParams.get("details")).toBe("evidence");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("tabpanel")).toBeFocused();
});
