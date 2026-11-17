import { defineConfig } from '@playwright/test';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';

const port = Number(process.env.EXACT_UI_PORT || 19818);
const archive = process.env.EXACT_UI_BUNDLE || join(tmpdir(), `exact-ui-${randomUUID()}.zip`);
const python = process.env.EXACT_UI_PYTHON || 'python';
const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`;
process.env.EXACT_UI_BUNDLE = archive;
export default defineConfig({
  testDir: './e2e',
  workers: 1,
  timeout: 30_000,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    channel: process.env.EXACT_UI_BROWSER || undefined,
    viewport: { width: 1440, height: 1000 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `${quote(python)} ../tests/inspect_ui_fixture.py --port ${port} --archive ${quote(archive)}`,
    url: `http://127.0.0.1:${port}/api/v1/health`,
    reuseExistingServer: false,
    gracefulShutdown: { signal: 'SIGTERM', timeout: 5000 },
  },
});
