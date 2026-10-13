// Does Chrome reuse the cached application CSS on reload, with the certificate allowance vs a trusted SPKI?
const { chromium } = require(require.resolve("@playwright/test", { paths: [process.env.FE] }));
(async () => {
  for (const mode of ["ignoreHTTPSErrors", "trusted-spki"]) {
    const browser = await chromium.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", args: mode === "trusted-spki" ? [`--ignore-certificate-errors-spki-list=${process.env.SPKI}`] : [] });
    const context = await browser.newContext({ ignoreHTTPSErrors: mode === "ignoreHTTPSErrors" });
    const page = await context.newPage();
    const cdp = await context.newCDPSession(page);
    await cdp.send("Network.enable");
    const seen = [];
    const ids = new Map();
    cdp.on("Network.requestWillBeSent", (e) => { if (e.request.url.includes("/_next/static/css/")) ids.set(e.requestId, { cache: "network" }); });
    cdp.on("Network.requestServedFromCache", (e) => { if (ids.has(e.requestId)) ids.get(e.requestId).cache = "memory"; });
    cdp.on("Network.responseReceived", (e) => { if (ids.has(e.requestId) && e.response.fromDiskCache) ids.get(e.requestId).cache = "disk"; });
    cdp.on("Network.loadingFinished", (e) => { if (ids.has(e.requestId)) seen.push(ids.get(e.requestId).cache); });
    try {
      await page.goto("https://127.0.0.1:19655/browse/");
      await page.waitForLoadState("load");
      for (let i = 0; i < 2; i += 1) { await page.reload(); await page.waitForLoadState("load"); }
      console.log(mode, "css loads:", seen.join(","));
    } catch (error) { console.log(mode, "error:", String(error).split("\n")[0]); }
    await browser.close();
  }
})();
