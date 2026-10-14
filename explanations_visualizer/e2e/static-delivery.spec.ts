// Actual compiled export, normal CSP and a trusted loopback TLS certificate.
// No intercepted responses; a cached response keeps the original server read ID.
import { expect, test, type Page } from "@playwright/test";
import { StyleDiagnostics } from "./styleDiagnostics";
import { harness } from "./v2Harness";

const diagnosticsByPage = new WeakMap<Page, StyleDiagnostics>();
test.afterEach(({ page }, info) => {
  // The runner knows the final outcome here, including the trailing hash assertions.
  diagnosticsByPage.get(page)?.write(info);
  diagnosticsByPage.delete(page);
});

for (const profile of ["exploration", "study"] as const) {
  test(`verified public CSS is complete and reused on ${profile} reload`, async ({ page, baseURL }) => {
    test.skip(!harness || !process.env.EXACT_E2E_CERT_SPKI, "Managed TLS binding required for actual browser cache verification");
    const styles = await StyleDiagnostics.attach(page, `static-cache-${profile}`, harness!.identity);
    diagnosticsByPage.set(page, styles);
    const url = profile === "study" ? `${harness!.origin}/participate/` : baseURL!;
    try {
      await page.goto(url);
      await styles.expectStyled("initial");
      await expect.poll(() => styles.network.filter((row) => row.path.startsWith("/_next/static/css/") && row.outcome === "finished").length).toBeGreaterThan(0);
      const count = styles.network.length;
      await styles.beforeNavigation("normal-reload");
      await page.reload();
      await styles.expectStyled("reload");
      await expect.poll(() => styles.network.slice(count).filter((row) => row.path.startsWith("/_next/static/css/") && row.outcome === "finished" && ["memory", "disk"].includes(row.from_cache)).length).toBeGreaterThan(0);
    } finally {
      await styles.finish();
    }
    const css = styles.network.filter((row) => row.path.startsWith("/_next/static/css/"));
    expect(css.length).toBeGreaterThanOrEqual(2);
    for (const row of css) {
      expect(row.status).toBe(200);
      expect(row.mime).toBe("text/css");
      expect(row.outcome).toBe("finished");
      expect(row.matches_export).toBe(true);
      expect(row.body_sha256).toBe(row.server_asset_sha256);
      expect(row.server_read_id).toMatch(/^[a-f0-9]{32}$/);
    }
    expect(styles.summary().failed_checks).toBe(0);
  });
}
