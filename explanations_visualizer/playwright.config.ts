import { defineConfig } from "@playwright/test";

// Trust only the managed local certificate when testing real browser cache behavior.
// ignoreHTTPSErrors alone bypasses certificate errors but Chrome can disable caching.
// Set NODE_EXTRA_CA_CERTS to the same certificate for Playwright API-request helpers.
const certificateSpki = process.env.EXACT_E2E_CERT_SPKI;
if (certificateSpki && !/^[A-Za-z0-9+/]{43}=$/.test(certificateSpki)) throw new Error("Invalid local certificate SPKI hash");

// Serve a prepared package with exact-inspect first; see the verification guide.
// This suite never starts matching, generation, or real participant sessions.
export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["json", { outputFile: "test-results/results.json" }]],
  use: {
    baseURL: process.env.EXACT_E2E_URL ?? "http://127.0.0.1:18765",
    viewport: { width: 1440, height: 1000 },
    ignoreHTTPSErrors: !certificateSpki,
    // Study invitations/tokens must not be embedded in retained traces/videos.
    trace: "off",
    screenshot: "only-on-failure",
    launchOptions: {
      ...(process.env.EXACT_CHROMIUM_PATH ? { executablePath: process.env.EXACT_CHROMIUM_PATH } : {}),
      ...(certificateSpki ? { args: [`--ignore-certificate-errors-spki-list=${certificateSpki}`] } : {}),
    },
  },
});
