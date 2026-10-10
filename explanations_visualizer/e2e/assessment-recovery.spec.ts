import { mkdirSync } from "node:fs";
import { join } from "node:path";

import { expect, test, type Page } from "@playwright/test";

import { harness, seedToTutorial, type ParticipantApi } from "./v2Harness";

// Assessment draft recovery (S0-FU-R1, S0-FU-R2), converted from the independent review's fault
// diagnostics into expected behaviour. Real v2 study service (HTTPS, PostgreSQL, compiled
// frontend, normal CSP); synthetic test sessions API-seeded to the assessment. Requests are held,
// aborted or delayed in the browser only; every conflict is a real 409 from the server.
test.use({ screenshot: "off", trace: "off", viewport: { width: 1440, height: 900 } });
test.beforeEach(() => test.skip(!harness, "Set EXACT_E2E_STUDY_V2_CONFIG"));

const PROGRESS = "**/api/v1/study/tutorial/progress";

/** Opt-in review evidence (EXACT_SCREENSHOT_DIR), in light and dark. Never an assertion. */
async function evidence(page: Page, name: string) {
  const out = process.env.EXACT_SCREENSHOT_DIR;
  if (!out) return;
  mkdirSync(out, { recursive: true });
  for (const scheme of ["light", "dark"] as const) {
    await page.emulateMedia({ colorScheme: scheme });
    await page.locator(".assessment-item").scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);
    await page.screenshot({ path: join(out, `${name}-${scheme}.png`) });
  }
  await page.emulateMedia({ colorScheme: "light" });
}
type Option = { code: string; label: string };

async function atQuestion(page: Page, api: ParticipantApi, index: number) {
  const state = await api.state();
  const question = state.tutorial.assessment[index];
  await api.write("PUT", "/tutorial/progress", { tutorial_version: state.tutorial.version, position: { view: "assessment", lesson_id: null, question_id: question.question_id } });
  await page.reload();
  await expect(page.locator(".assessment-item h2")).toContainText(`${index + 1} ·`);
  return { state, question };
}

const saved = async (api: ParticipantApi, questionId: string) => (await api.state()).tutorial_progress.assessment_drafts?.[questionId];

function gate() {
  let open!: () => void;
  const opened = new Promise<void>((resolve) => {
    open = resolve;
  });
  return { open, opened };
}

/** Every answered draft write for one question from this page: status, answer and expected revision. */
function draftWrites(page: Page, questionId: string) {
  const writes: { status: number; response: Record<string, unknown>; expected: number }[] = [];
  page.on("response", (response) => {
    const request = response.request();
    if (!request.url().endsWith("/tutorial/progress") || request.method() !== "PUT") return;
    const body = request.postDataJSON();
    if (body.assessment_draft?.question_id === questionId) writes.push({ status: response.status(), response: body.assessment_draft.response, expected: body.expected_revision });
  });
  return writes;
}

/** The assessment responses this page submitted with Check answer. */
function checkedResponses(page: Page) {
  const responses: Record<string, unknown>[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/tutorial/assessment") && request.method() === "POST") responses.push(request.postDataJSON().response);
  });
  return responses;
}

/** Assessment drafts persisted in this tab's outbox. */
const queuedDrafts = (page: Page) =>
  page.evaluate(() => {
    const key = Object.keys(sessionStorage).find((name) => name.startsWith("exact.study.outbox.") && !name.endsWith(".recovery"));
    return key ? JSON.parse(sessionStorage.getItem(key)!).entries.filter((entry: { body: { assessment_draft?: unknown } }) => entry.body.assessment_draft).length : 0;
  });

const radio = (page: Page, option: Option) => page.getByRole("radio", { name: option.label, exact: true });
const rows = (page: Page) => page.locator(".assessment-item fieldset.match-row");
const conflictPanel = (page: Page) => page.locator(".assessment-conflict");
const checkButton = (page: Page) => page.getByRole("button", { name: "Check answer", exact: true });

async function secondTab(page: Page, index: number) {
  const other = await page.context().newPage();
  await other.goto(`${harness!.origin}/participate/`);
  await expect(other.locator(".assessment-item h2")).toContainText(`${index + 1} ·`);
  return other;
}

/** Hold this page's first draft write for the question until released. */
async function holdFirstWrite(page: Page, questionId: string) {
  const release = gate();
  const seen = gate();
  let held = false;
  await page.route(PROGRESS, async (route) => {
    if (!held && route.request().postDataJSON().assessment_draft?.question_id === questionId) {
      held = true;
      seen.open();
      await release.opened;
    }
    await route.continue().catch(() => undefined);
  });
  return { release: release.open, seen: seen.opened };
}

test.describe("S0-FU-R1 an older same-question answer never overwrites a newer saved one", () => {
  test("single choice: the saved answer stays authoritative; this tab's answer is kept, survives reload, and is dropped by choice", async ({ page }) => {
    const api = await seedToTutorial(page);
    const { question } = await atQuestion(page, api, 0);
    const [mine, theirs] = question.options as Option[];
    const other = await secondTab(page, 0);
    const writes = draftWrites(page, question.question_id);
    const hold = await holdFirstWrite(page, question.question_id);
    try {
      await radio(page, mine).check();
      await hold.seen;
      await radio(other, theirs).check();
      await expect.poll(async () => (await saved(api, question.question_id))?.choice).toBe(theirs.code);
      hold.release();
      await expect.poll(() => writes.map((write) => write.status)).toEqual([409]);
      await page.waitForTimeout(2500);
      expect(writes.map((write) => write.status), "no automatic re-save").toEqual([409]);
      expect((await saved(api, question.question_id))?.choice, "the newer saved answer stays authoritative").toBe(theirs.code);
      // The saved answer is shown; this tab's answer is kept and offered, never written by itself.
      await expect(conflictPanel(page)).toContainText(mine.label);
      await expect(radio(page, theirs)).toBeChecked();
      await expect(radio(page, mine)).not.toBeChecked();
      await expect(checkButton(page)).toBeDisabled();
      await expect(page.locator(".question-nav")).toContainText("choose which answer to keep");
      await expect(page.getByRole("alert").filter({ hasText: "changed in another tab or device" })).toBeVisible();
      await evidence(page, "fu-r1-conflict-single");
      // The choice is still offered after reloading this tab.
      await page.reload();
      await expect(conflictPanel(page)).toContainText(mine.label);
      await expect(radio(page, theirs)).toBeChecked();
      await page.getByRole("button", { name: "Keep the saved answer", exact: true }).click();
      await expect(conflictPanel(page)).toHaveCount(0);
      await expect(checkButton(page)).toBeEnabled();
      await page.reload();
      await expect(page.locator(".assessment-item h2")).toContainText("1 ·");
      await expect(radio(page, theirs)).toBeChecked();
      await expect(conflictPanel(page)).toHaveCount(0);
      expect(writes.map((write) => write.status)).toEqual([409]);
      expect((await saved(api, question.question_id))?.choice).toBe(theirs.code);
      await expect(radio(other, theirs)).toBeChecked();
    } finally {
      hold.release();
      await other.close();
    }
  });

  test("multi-part: this tab's answer replaces the newer one only when the participant chooses it; the choice survives reload in both tabs", async ({ page }) => {
    const api = await seedToTutorial(page);
    const { question } = await atQuestion(page, api, 3);
    expect(question.kind).toBe("match_and_single");
    const code = (label: string) => (question.row_options as Option[]).find((option) => option.label === label)!.code;
    const [first, second, third] = question.rows as { row_id: string; label: string }[];
    const yes = (question.part_b.options as Option[])[0];
    const mine = { matches: { [first.row_id]: code("Original ontology fact"), [second.row_id]: code("Matcher evidence") }, part_b: yes.code };
    const theirs = { matches: { [first.row_id]: code("Generated text"), [third.row_id]: code("Matcher evidence") } };
    const other = await secondTab(page, 3);
    const writes = draftWrites(page, question.question_id);
    const hold = await holdFirstWrite(page, question.question_id);
    try {
      await rows(page).nth(0).getByRole("radio", { name: "Original ontology fact" }).check();
      await rows(page).nth(1).getByRole("radio", { name: "Matcher evidence" }).check();
      await rows(page).nth(3).getByRole("radio", { name: yes.label, exact: true }).check();
      await hold.seen;
      await rows(other).nth(0).getByRole("radio", { name: "Generated text" }).check();
      await rows(other).nth(2).getByRole("radio", { name: "Matcher evidence" }).check();
      await expect.poll(() => saved(api, question.question_id)).toEqual(theirs);
      hold.release();
      await expect.poll(() => writes.filter((write) => write.status === 409).length).toBeGreaterThan(0);
      await page.waitForTimeout(2500);
      expect(writes.every((write) => write.status === 409), "no automatic re-save").toBe(true);
      expect(await saved(api, question.question_id), "the newer saved answer stays authoritative").toEqual(theirs);
      await expect(conflictPanel(page)).toContainText("Original ontology fact");
      await expect(rows(page).nth(0).getByRole("radio", { name: "Generated text" })).toBeChecked();
      await expect(rows(page).nth(2).getByRole("radio", { name: "Matcher evidence" })).toBeChecked();
      await evidence(page, "fu-r1-conflict-multipart");
      // A deliberate resolution writes this tab's whole answer on top of the one it was shown.
      await page.getByRole("button", { name: "Use this tab’s answer", exact: true }).click();
      await expect.poll(() => saved(api, question.question_id)).toEqual(mine);
      expect(writes.at(-1)).toMatchObject({ status: 200, response: mine });
      await expect(conflictPanel(page)).toHaveCount(0);
      await expect(rows(page).nth(0).getByRole("radio", { name: "Original ontology fact" })).toBeChecked();
      await evidence(page, "fu-r1-resolved-this-tab");
      for (const tab of [page, other]) {
        await tab.reload();
        await expect(tab.locator(".assessment-item h2")).toContainText("4 ·");
        await expect(rows(tab).nth(0).getByRole("radio", { name: "Original ontology fact" })).toBeChecked();
        await expect(rows(tab).nth(1).getByRole("radio", { name: "Matcher evidence" })).toBeChecked();
        await expect(rows(tab).nth(3).getByRole("radio", { name: yes.label, exact: true })).toBeChecked();
        await expect(conflictPanel(tab)).toHaveCount(0);
      }
    } finally {
      hold.release();
      await other.close();
    }
  });

  test("independent questions: an answer whose saved value is unchanged is saved again automatically, beside the other tab's answer", async ({ page }) => {
    const api = await seedToTutorial(page);
    const { question, state } = await atQuestion(page, api, 0);
    const q2 = state.tutorial.assessment[1];
    const other = await secondTab(page, 0);
    const writes = draftWrites(page, question.question_id);
    const hold = await holdFirstWrite(page, question.question_id);
    try {
      await radio(page, question.options[0]).check();
      await hold.seen;
      await other.getByRole("button", { name: "Next question", exact: true }).click();
      await expect(other.locator(".assessment-item h2")).toContainText("2 ·");
      await radio(other, q2.options[1]).check();
      await expect.poll(async () => (await saved(api, q2.question_id))?.choice).toBe(q2.options[1].code);
      hold.release();
      await expect.poll(() => writes.map((write) => write.status)).toEqual([409, 200]);
      await expect.poll(async () => (await saved(api, question.question_id))?.choice).toBe(question.options[0].code);
      expect((await saved(api, q2.question_id))?.choice, "the other tab's answer is kept").toBe(q2.options[1].code);
      await page.locator(".question-nav button").nth(0).click();
      await expect(radio(page, question.options[0])).toBeChecked();
      await expect(conflictPanel(page)).toHaveCount(0);
      await page.locator(".question-nav button").nth(1).click();
      await expect(radio(page, q2.options[1])).toBeChecked();
    } finally {
      hold.release();
      await other.close();
    }
  });
});

test.describe("S0-FU-R2 the latest answer stays visible throughout restored outbox replay", () => {
  /**
   * The first edit goes out but cannot reach the server (aborted, retried); the later edits queue
   * behind it. Then the page reloads with the restored writes held, each until released by name.
   */
  async function interruptedThenReloaded(page: Page, questionId: string, key: (response: Record<string, any>) => string, first: () => Promise<void>, later: () => Promise<void>, expected: number) {
    let mode: "abort" | "hold" | "open" = "abort";
    let blocked = 0;
    const gates = new Map<string, ReturnType<typeof gate>>();
    const held: string[] = [];
    const gateFor = (name: string) => gates.get(name) ?? gates.set(name, gate()).get(name)!;
    await page.route(PROGRESS, async (route) => {
      const draft = route.request().postDataJSON().assessment_draft;
      if (draft?.question_id !== questionId || mode === "open") return route.continue();
      if (mode === "abort") {
        blocked += 1;
        return route.abort("internetdisconnected");
      }
      const name = key(draft.response);
      held.push(name);
      await gateFor(name).opened;
      await route.continue().catch(() => undefined);
    });
    await first();
    await expect.poll(() => blocked).toBeGreaterThan(0);
    await later();
    await expect.poll(() => queuedDrafts(page)).toBe(expected);
    mode = "hold";
    await page.reload();
    return {
      held,
      release: (name: string) => gateFor(name).open(),
      /** Release everything held, and let every later write through. */
      releaseAll: () => {
        mode = "open";
        gates.forEach((value) => value.open());
      },
    };
  }

  test("saved A, in-flight B, latest A, reload: A stays visible while B is acknowledged, and Check answer submits A", async ({ page }) => {
    const api = await seedToTutorial(page);
    const { question } = await atQuestion(page, api, 0);
    const [a, b] = question.options as Option[];
    await radio(page, a).check();
    await expect.poll(async () => (await saved(api, question.question_id))?.choice).toBe(a.code);
    const checks = checkedResponses(page);
    const replay = await interruptedThenReloaded(
      page,
      question.question_id,
      (response) => response.choice,
      () => radio(page, b).check(),
      () => radio(page, a).check(),
      2,
    );
    try {
      await expect(radio(page, a)).toBeChecked();
      await expect.poll(() => replay.held).toEqual([b.code]);
      await expect(radio(page, a)).toBeChecked();
      replay.release(b.code);
      await expect.poll(async () => (await saved(api, question.question_id))?.choice).toBe(b.code);
      await expect.poll(() => replay.held).toEqual([b.code, a.code]);
      // The intermediate acknowledgement is not the participant's latest answer.
      await expect(radio(page, a)).toBeChecked();
      await expect(radio(page, b)).not.toBeChecked();
      await evidence(page, "fu-r2-latest-visible-while-replaying");
      await checkButton(page).click();
      replay.release(a.code);
      await expect.poll(async () => (await saved(api, question.question_id))?.choice).toBe(a.code);
      await expect.poll(() => checks.length).toBe(1);
      expect(checks[0]).toEqual({ choice: a.code });
      await expect.poll(async () => (await api.state()).tutorial_progress.attempts?.at(-1)?.response).toEqual({ choice: a.code });
      await expect(radio(page, a)).toBeChecked();
    } finally {
      replay.releaseAll();
    }
  });

  test("multi-part with distinct A, B and C: C stays visible while B is acknowledged and is saved last", async ({ page }) => {
    const api = await seedToTutorial(page);
    const { question } = await atQuestion(page, api, 2);
    const [first, second] = question.rows as { row_id: string; label: string }[];
    const code = (label: string) => (question.row_options as Option[]).find((option) => option.label === label)!.code;
    const row = (index: number, label: string) => rows(page).nth(index).getByRole("radio", { name: label });
    await row(0, "None of these").check();
    await expect.poll(() => saved(api, question.question_id)).toEqual({ matches: { [first.row_id]: code("None of these") } });
    const replay = await interruptedThenReloaded(
      page,
      question.question_id,
      (response) => response.matches[first.row_id],
      () => row(0, "Insufficient information").check(),
      async () => {
        await row(0, "An empty, unsubmitted draft").check();
        await row(1, "None of these").check();
      },
      2,
    );
    const latest = { matches: { [first.row_id]: code("An empty, unsubmitted draft"), [second.row_id]: code("None of these") } };
    try {
      await expect(row(0, "An empty, unsubmitted draft")).toBeChecked();
      await expect(row(1, "None of these")).toBeChecked();
      await expect.poll(() => replay.held).toEqual([code("Insufficient information")]);
      replay.release(code("Insufficient information"));
      await expect.poll(() => saved(api, question.question_id)).toEqual({ matches: { [first.row_id]: code("Insufficient information") } });
      await expect.poll(() => replay.held.length).toBe(2);
      await expect(row(0, "An empty, unsubmitted draft")).toBeChecked();
      await expect(row(1, "None of these")).toBeChecked();
      await expect(row(0, "Insufficient information")).not.toBeChecked();
      replay.release(code("An empty, unsubmitted draft"));
      await expect.poll(() => saved(api, question.question_id)).toEqual(latest);
      await expect(row(0, "An empty, unsubmitted draft")).toBeChecked();
    } finally {
      replay.releaseAll();
    }
  });

  test("terminal replay failure after another question changed: the latest answer stays visible and is saved again safely", async ({ page }) => {
    const api = await seedToTutorial(page);
    const { question, state } = await atQuestion(page, api, 0);
    const [a, b, c] = question.options as Option[];
    await radio(page, a).check();
    await expect.poll(async () => (await saved(api, question.question_id))?.choice).toBe(a.code);
    const writes = draftWrites(page, question.question_id);
    const replay = await interruptedThenReloaded(
      page,
      question.question_id,
      (response) => response.choice,
      () => radio(page, b).check(),
      () => radio(page, c).check(),
      2,
    );
    try {
      await expect.poll(() => replay.held).toEqual([b.code]);
      // Elsewhere, another question is answered: the restored writes' revisions are now stale.
      const q2 = state.tutorial.assessment[1];
      await api.write("PUT", "/tutorial/progress", { tutorial_version: state.tutorial.version, assessment_draft: { question_id: q2.question_id, response: { choice: q2.options[2].code } } });
      replay.releaseAll();
      await expect.poll(() => writes.map((write) => write.status)).toEqual([409, 200]);
      expect(writes[1].response).toEqual({ choice: c.code });
      await expect.poll(async () => (await saved(api, question.question_id))?.choice).toBe(c.code);
      expect((await saved(api, q2.question_id))?.choice).toBe(q2.options[2].code);
      await expect(radio(page, c)).toBeChecked();
      await expect(conflictPanel(page)).toHaveCount(0);
    } finally {
      replay.releaseAll();
    }
  });

  test("terminal replay failure after the same question changed elsewhere: the saved answer is shown and the restored latest answer is kept for a choice", async ({ page }) => {
    const api = await seedToTutorial(page);
    const { question, state } = await atQuestion(page, api, 4);
    expect(question.kind).toBe("multiple");
    const options = question.options as Option[];
    const box = (option: Option) => page.getByRole("checkbox", { name: option.label, exact: true });
    await box(options[0]).check();
    await expect.poll(async () => (await saved(api, question.question_id))?.choices).toEqual([options[0].code]);
    const writes = draftWrites(page, question.question_id);
    const replay = await interruptedThenReloaded(
      page,
      question.question_id,
      (response) => String(response.choices.length),
      () => box(options[1]).check(),
      () => box(options[2]).check(),
      2,
    );
    const latest = [options[0].code, options[1].code, options[2].code];
    try {
      await expect(box(options[2])).toBeChecked();
      await expect.poll(() => replay.held).toEqual(["2"]);
      await api.write("PUT", "/tutorial/progress", { tutorial_version: state.tutorial.version, assessment_draft: { question_id: question.question_id, response: { choices: [options[3].code] } } });
      replay.releaseAll();
      await expect.poll(() => writes.map((write) => write.status)).toEqual([409]);
      await page.waitForTimeout(2500);
      expect(writes.map((write) => write.status), "no automatic re-save").toEqual([409]);
      expect((await saved(api, question.question_id))?.choices, "the newer saved answer stays authoritative").toEqual([options[3].code]);
      await expect(box(options[3])).toBeChecked();
      await expect(box(options[0])).not.toBeChecked();
      await expect(conflictPanel(page)).toContainText(options[2].label);
      await evidence(page, "fu-r2-replay-conflict");
      await page.getByRole("button", { name: "Use this tab’s answer", exact: true }).click();
      await expect.poll(async () => (await saved(api, question.question_id))?.choices).toEqual(latest);
      await page.reload();
      await expect(page.locator(".assessment-item h2")).toContainText("5 ·");
      for (const option of options.slice(0, 3)) await expect(box(option)).toBeChecked();
      await expect(box(options[3])).not.toBeChecked();
    } finally {
      replay.releaseAll();
    }
  });

  test("a delayed acknowledgement of an older answer never replaces the newer visible one; Check answer submits the newer one", async ({ page }) => {
    const api = await seedToTutorial(page);
    const { question } = await atQuestion(page, api, 1);
    const [a, b] = question.options as Option[];
    const release = gate();
    const reached = gate();
    let delayed = false;
    await page.route(PROGRESS, async (route) => {
      if (delayed || route.request().postDataJSON().assessment_draft?.question_id !== question.question_id) return route.continue();
      delayed = true;
      // The server stores A at once; only its acknowledgement reaches the page late.
      const response = await route.fetch();
      reached.open();
      await release.opened;
      await route.fulfill({ response });
    });
    const checks = checkedResponses(page);
    try {
      await radio(page, a).check();
      await reached.opened;
      await expect.poll(async () => (await saved(api, question.question_id))?.choice).toBe(a.code);
      await radio(page, b).check();
      await expect(radio(page, b)).toBeChecked();
      await checkButton(page).click();
      release.open();
      await expect.poll(async () => (await saved(api, question.question_id))?.choice).toBe(b.code);
      await expect(radio(page, b)).toBeChecked();
      await expect.poll(() => checks.length).toBe(1);
      expect(checks[0]).toEqual({ choice: b.code });
    } finally {
      release.open();
    }
  });
});
