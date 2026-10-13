/** Bounded real-browser read diagnostics. No URL, payload, cookie or participant logging. */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
const [environmentRoot, modulesRoot, output, phase = "normal"] = process.argv.slice(2);
if (!environmentRoot || !modulesRoot || !output) throw new Error("Expected environment, modules, new receipt path, phase");
const require = createRequire(path.join(modulesRoot, "package.json"));
const { chromium } = require("@playwright/test");
const config = JSON.parse(fs.readFileSync(path.join(environmentRoot, "environment.json")));
const bundle = JSON.parse(fs.readFileSync(config.package));
const origin = `https://127.0.0.1:${config.ports.exploration}`;
const profile = Object.entries(bundle.explanations).map(([id, loc]) => [id, JSON.parse(fs.readFileSync(path.resolve(path.dirname(config.package), loc)))]).find(([, r]) => r.task === "entity_profile" && r.entities[0].iri === "http://purl.obolibrary.org/obo/DOID_7894");
if (!profile) throw new Error("Retained incident subject absent");
const [id, record] = profile;
const subject = record.entities[0];
const query = new URLSearchParams(subject);
const urls = [
  ["profile_discovery", `/api/v1/explanations?${query}&task=entity_profile`],
  ["profile", `/api/v1/explanations/${encodeURIComponent(id)}`],
  ["comparisons", `/api/v1/explanations?${query}&task=pair_comparison`],
  ["context", `/api/v1/entity-context?${query}`],
];
const browser = await chromium.launch({ executablePath: process.env.EXACT_CHROMIUM_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true });
const context = await browser.newContext({ ignoreHTTPSErrors: true });
const page = await context.newPage();
const report = { schema: "exact-browser-read-probe/1", phase, started_at: new Date().toISOString(), status: "running", normal: [], aborts: [], reloads: [], streams: [], request_failures: [] };
let mode = "startup";
page.on("requestfailed", request => {
  const raw = request.failure()?.errorText || "unknown";
  report.request_failures.push({ mode, error: /^net::ERR_[A-Z_]+$/.test(raw) ? raw : "other_browser_error" });
});
async function readBatch() {
  return page.evaluate(async entries => Promise.all(entries.map(async ([category, url]) => {
    const started = performance.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(url, { signal: controller.signal });
      const bytes = await response.arrayBuffer();
      const body = JSON.parse(new TextDecoder().decode(bytes));
      const length = response.headers.get("content-length");
      return { category, status: response.status, bytes: bytes.byteLength, declared_length: length === null ? null : Number(length), read_id: response.headers.get("x-exact-read-id"), csp_present: Boolean(response.headers.get("content-security-policy")), body_shape_valid: category === "profile" ? Array.isArray(body.claims) && body.claims.length > 0 : typeof body === "object", elapsed_ms: Math.round(performance.now() - started) };
    } catch (e) { return { category, error_type: e.name, elapsed_ms: Math.round(performance.now() - started) }; }
    finally { clearTimeout(timer); }
  })), urls);
}
try {
  const document = await page.goto(origin, { waitUntil: "domcontentloaded", timeout: 30000 });
  report.document_csp = document.headers()["content-security-policy"] || "";
  if (!report.document_csp.includes("connect-src 'self'") || !report.document_csp.includes("object-src 'none'") || !report.document_csp.includes("script-src 'self' 'sha256-")) throw new Error("Served document CSP missing");
  for (let i = 0; i < 20; i++) {
    mode = i === 0 ? "first_reads" : "normal";
    report.normal.push(...await readBatch());
  }
  mode = "intentional_abort";
  for (let i = 0; i < 10; i++) {
    report.aborts.push(await page.evaluate(async url => {
      const c = new AbortController(); const timer = setTimeout(() => c.abort(), 1);
      try { const r = await fetch(url, { signal: c.signal }); await r.arrayBuffer(); return "completed_before_abort"; }
      catch (e) { return e.name; } finally { clearTimeout(timer); }
    }, urls[1][1]));
    report.normal.push(...await readBatch());
  }
  mode = "intentional_reload";
  for (let i = 0; i < 10; i++) {
    await page.evaluate(entries => { window.__probeReads = Promise.all(entries.map(([, u]) => fetch(u).then(r => r.arrayBuffer()).catch(e => e.name))); }, urls);
    const started = performance.now();
    await page.reload({ waitUntil: "domcontentloaded", timeout: 30000 });
    report.reloads.push({ completed: true, elapsed_ms: Math.round(performance.now() - started) });
    report.normal.push(...await readBatch());
  }
  const axiom = record.support_packet.facts.find(f => f.axiom_ref)?.axiom_ref;
  if (axiom) {
    mode = "intentional_stream_cancel";
    const stream = `/api/v1/axioms/${encodeURIComponent(axiom)}/stream?${query}`;
    for (let i = 0; i < 10; i++) {
      report.streams.push(await page.evaluate(async url => {
        const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 15000);
        try {
          const r = await fetch(url, { signal: controller.signal }); const reader = r.body.getReader(); const first = await reader.read(); await reader.cancel();
          return { status: r.status, first_bytes: first.value?.length ?? 0, done_on_first: first.done, read_id: r.headers.get("x-exact-read-id"), interpretation: "client cancellation attempt; not proof of incomplete server transmission" };
        } catch (e) { return { error_type: e.name }; } finally { clearTimeout(timer); }
      }, stream));
      report.normal.push(...await readBatch());
    }
  }
  const unexpected = report.request_failures.filter(x => !["intentional_abort", "intentional_reload", "intentional_stream_cancel"].includes(x.mode));
  if (report.normal.some(r => r.status !== 200 || !r.read_id || !r.body_shape_valid || (r.declared_length !== null && r.declared_length !== r.bytes)) || report.aborts.some(x => !["AbortError", "completed_before_abort"].includes(x)) || report.streams.some(x => x.status !== 200) || unexpected.length) throw new Error("Read invariant failed; inspect bounded receipt");
  report.status = "passed";
} catch (e) { report.status = "failed"; report.failure_type = e.name; process.exitCode = 1; }
finally { report.finished_at = new Date().toISOString(); await browser.close(); fs.writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { mode: 0o600, flag: "wx" }); }
console.log(JSON.stringify({ status: report.status, phase, completed_reads: report.normal.length, aborts: report.aborts.length, reloads: report.reloads.length, streams: report.streams.length }));
