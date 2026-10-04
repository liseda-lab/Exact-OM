/** Synthetic reload diagnostic against tools.review_study_disconnects.
 * node tools/review_study_reload.mjs PRIVATE_CONFIG FRONTEND_DIR CHROME OUTPUT [REPETITIONS] [standard|resource]
 * FRONTEND_DIR supplies installed @playwright/test. No credentials or query strings are output.
 * Consent/setup/background are seeded through the real API; this is not a participant acceptance journey.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { createHash, randomUUID } from "node:crypto";
import { resolve } from "node:path";
const [configPath, frontend, chrome, output, countArg = "20", mode = "standard"] = process.argv.slice(2);
if (!output) throw new Error("Expected PRIVATE_CONFIG FRONTEND_DIR CHROME OUTPUT [REPETITIONS] [standard|resource]");
const config = JSON.parse(readFileSync(configPath, "utf8"));
const publication = JSON.parse(readFileSync(config.publication, "utf8")).definition;
if (!publication.synthetic || !/^https:\/\/127\.0\.0\.1:\d+$/.test(config.origin)) throw new Error("Only a synthetic loopback HTTPS harness is allowed");
const { chromium } = createRequire(resolve(frontend, "package.json"))("@playwright/test");
const count = Number(countArg);
if (!Number.isInteger(count) || count < 1 || count > 100) throw new Error("REPETITIONS must be1..100");
if (!["standard", "resource"].includes(mode)) throw new Error("Mode must be standard or resource");
const report = { scenario: `synthetic-tutorial-reload-${mode}`, authenticated_resource_reload: 0, full_resource_checks: 0, full_state_checks: 0, repetitions: count, normal_reload: 0, reload_during_stream: 0, explicit_stream_abort: 0, interrupted_producer: 0, subsequent_health_checks: 0, csp_violations: 0, reload_ms: [], failures: [] };
const browser = await chromium.launch({ executablePath: chrome });
try {
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  const page = await context.newPage();
  page.on("console", (message) => { if (/Content Security Policy|Refused to/.test(message.text())) report.csp_violations++; });
  const admin = await page.request.post(`${config.origin}/api/v1/admin/studies/${encodeURIComponent(config.study_revision)}/invitations`, { headers: { Authorization: `Bearer ${config.researcher_token}` }, data: { count: 1, test: true } });
  if (!admin.ok()) throw new Error("Synthetic invitation issue failed");
  const invitation = (await admin.json()).invitations[0].invitation;
  await page.goto(config.origin + invitation, { timeout: 15000 });
  await page.getByRole("button", { name: "I agree to take part", exact: true }).waitFor({ timeout: 15000 });
  const state = async () => {
    const response = await page.request.get(`${config.origin}/api/v1/study/state`);
    if (!response.ok()) throw new Error(`State HTTP${response.status()}`);
    return response.json();
  };
  const mutate = async (path, data) => {
    const current = await state();
    const response = await page.request.put(`${config.origin}/api/v1/study/${path}`, { headers: { Origin: config.origin, "X-Study-Session": current.session_id }, data: { idempotency_key: randomUUID().replaceAll("-", ""), expected_revision: current.revision, ...data } });
    if (!response.ok()) throw new Error(`Preparation HTTP${response.status()}`);
    return response.json();
  };
  await mutate("consent", { information_version: publication.information_version, accepted: true });
  await mutate("setup", { setup_version: "setup/2", instructions_acknowledged: true, external_inspection_optional_understood: true, resource_access: "available", submitted: true });
  const current = await state();
  const answers = {};
  for (const q of current.forms.background) if (q.required && !q.show_if) {
    const selected = q.option_order.includes("prefer_not_to_say") ? "prefer_not_to_say" : q.option_order[0];
    answers[q.id] = q.multiple ? [selected] : selected;
  }
  await mutate("questionnaires/background", { form_version: current.forms.version, answers, submitted: true });
  const reload = async () => {
    const started = Date.now();
    await page.reload({ waitUntil: "load", timeout: 15000 });
    await page.getByRole("heading", { level: 1, name: /^Lesson 1 of / }).waitFor({ timeout: 15000 });
    report.reload_ms.push(Date.now() - started);
    if (!(await page.request.get(config.origin + "/api/health")).ok()) throw new Error("Subsequent health request failed");
    report.subsequent_health_checks++;
  };
  const resourceUrl = config.origin + "/api/v1/study/resources/" + encodeURIComponent(publication.tutorial.case.explanation_refs[0]);
  const reference = await page.request.get(resourceUrl);
  if (!reference.ok()) throw new Error("Authenticated resource unavailable");
  const resourceBytes = await reference.body();
  const resourceHash = createHash("sha256").update(resourceBytes).digest("hex");
  for (let i = 0; i < count; i++) {
    if (mode === "resource") {
      if (i === 0) await reload();
      const firstBytes = await page.evaluate(async (url) => {
        const response = await fetch(url, { headers: { "X-Review-Delay-Body": "1" } });
        if (!response.ok) throw new Error("Real resource did not authorize");
        window.__reviewReader = response.body.getReader();
        const first = await window.__reviewReader.read();
        window.__reviewReader.read().catch(() => {});
        return first.value.length;
      }, resourceUrl);
      if (firstBytes !== 1024) throw new Error("Original resource was not interrupted at expected ASGI chunk");
      await reload(); report.authenticated_resource_reload++;
      const fullResource = await page.request.get(resourceUrl);
      const bytes = await fullResource.body();
      if (!fullResource.ok() || bytes.length !== Number(fullResource.headers()["content-length"]) || createHash("sha256").update(bytes).digest("hex") !== resourceHash) throw new Error("Subsequent original resource bytes/length mismatch");
      report.full_resource_checks++;
      const fullState = await page.request.get(config.origin + "/api/v1/study/state");
      const stateBytes = await fullState.body();
      if (!fullState.ok() || stateBytes.length !== Number(fullState.headers()["content-length"]) || JSON.parse(stateBytes).stage !== "tutorial") throw new Error("Subsequent state body/length mismatch");
      report.full_state_checks++;
      continue;
    }
    await reload(); report.normal_reload++;
    // Actual fetch and first body chunk are received before navigation cancels the page.
    await page.evaluate(async () => {
      const response = await fetch("/_review/slow");
      window.__reviewReader = response.body.getReader();
      await window.__reviewReader.read();
      window.__reviewReader.read().catch(() => {});
    });
    await reload(); report.reload_during_stream++;
    const aborted = await page.evaluate(async () => {
      const control = new AbortController();
      const response = await fetch("/_review/slow", { signal: control.signal });
      const reader = response.body.getReader();
      await reader.read(); control.abort();
      try { await reader.read(); return false; } catch (error) { return error.name === "AbortError"; }
    });
    if (!aborted) throw new Error("Expected explicit fetch cancellation");
    report.explicit_stream_abort++;
    // The intentionally failed producer remains failed; a complete8-byte response is forbidden.
    const interrupted = await page.evaluate(async () => {
      try { const response = await fetch("/_review/cancel"); await response.arrayBuffer(); return false; }
      catch { return true; }
    });
    if (!interrupted) throw new Error("Interrupted producer was falsely acknowledged complete");
    report.interrupted_producer++;
    if (!(await page.request.get(config.origin + "/api/health")).ok()) throw new Error("Health after cancellation failed");
    report.subsequent_health_checks++;
  }
  if (report.csp_violations) throw new Error("CSP violation observed");
} catch (error) {
  report.failures.push(error.name);
  process.exitCode = 1;
} finally {
  await browser.close();
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { mode: 0o600 });
  console.log(JSON.stringify(report));
}
