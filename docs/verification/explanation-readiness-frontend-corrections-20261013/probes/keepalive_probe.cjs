// HR1 diagnostic (labelled; not a reproduction of the incident): fetch the application CSS from
// the study origin after idle gaps around uvicorn's 5 s keep-alive timeout, in real Chrome, and
// record transfer outcomes. Paths only; no session, invitation or page content is used.
const { chromium } = require(require.resolve("@playwright/test", { paths: [process.env.FE] }));
(async () => {
  const origin = process.env.ORIGIN, css = process.env.CSS;
  const browser = await chromium.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" });
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  const failures = [];
  cdp.on("Network.loadingFailed", (e) => failures.push({ error: e.errorText, canceled: e.canceled ?? false }));
  await page.goto(`${origin}/participate/`);
  const gaps = [4800, 4900, 4950, 5000, 5025, 5050, 5100, 5200];
  const results = [];
  for (let i = 0; i < Number(process.env.N || 32); i += 1) {
    const gap = gaps[i % gaps.length];
    await page.waitForTimeout(gap);
    const before = failures.length;
    const outcome = await page.evaluate(async (url) => {
      try { const r = await fetch(url, { cache: "no-store" }); const b = await r.arrayBuffer(); return { status: r.status, bytes: b.byteLength }; }
      catch (e) { return { error: String(e).slice(0, 80) }; }
    }, `${origin}${css}`);
    results.push({ i, gap, ...outcome, net: failures.slice(before).map((f) => f.error) });
  }
  const bad = results.filter((r) => r.error || r.bytes !== Number(process.env.BYTES));
  console.log(JSON.stringify({ attempts: results.length, failures: bad.length, bad, sample: results.slice(0, 3) }));
  await browser.close();
})();
