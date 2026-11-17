import { expect, test } from '@playwright/test';

const archive = process.env.EXACT_UI_BUNDLE;
if (!archive) throw new Error('Set EXACT_UI_BUNDLE to a valid inspection ZIP.');

test('comparison, trace, scores, evidence and browse round trip work offline', async ({ page, baseURL }) => {
  const errors: string[] = [];
  const external: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.origin !== new URL(baseURL!).origin) {
      external.push(url.href);
      return route.abort();
    }
    return route.continue();
  });
  await page.goto('/');
  await expect(page.locator('.entity-card').first()).toContainText('Definition');
  await page.getByRole('button', { name: 'Open decision trace', exact: true }).click();
  await expect(page.locator('.trace-heading')).toBeInViewport();
  await expect(page.getByRole('tab', { name: 'Decision trace', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.getByRole('tab', { name: 'Scores', exact: true }).click();
  await expect(page.getByRole('tabpanel')).toContainText('0.81');
  await page.getByRole('tab', { name: 'Evidence', exact: true }).click();
  await page.getByRole('button', { name: 'Show original axiom', exact: true }).first().click();
  await expect(page.getByRole('tabpanel')).toContainText('SubClassOf');
  await page.getByRole('button', { name: 'Show in graph', exact: true }).first().click();
  await expect(page.getByRole('img', { name: /^Evidence graph with/ })).toBeVisible();
  await expect(page.locator('.graph-canvas canvas').first()).toBeVisible();
  await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
  await page.getByLabel('Colour theme').selectOption('dark');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.getByRole('button', { name: 'Fit all', exact: true }).click();
  await expect(page.locator('.graph-outside')).toHaveCount(0);
  await page.screenshot({ path: 'test-results/local-graph.png', fullPage: true });
  await page.getByRole('tab', { name: 'Hierarchy', exact: true }).click();
  const comparison = new URL(page.url()).searchParams;
  await page.getByRole('button', { name: 'Open both in the full ontology browser' }).click();
  await expect(page.getByRole('heading', { name: 'Browse both ontologies' })).toBeVisible();
  const search = page.getByRole('combobox', { name: 'Search the source ontology' });
  await search.fill('Source root');
  await page.getByRole('listbox').getByRole('option').first().click();
  await expect(page.locator('.focus-card').first()).toContainText('Source root');
  await page.getByRole('link', { name: 'Back to the comparison' }).click();
  await expect.poll(() => new URL(page.url()).searchParams.get('pair')).toBe(comparison.get('pair'));
  await expect(page.getByRole('button', { name: 'Open decision trace', exact: true })).toBeVisible();
  await page.screenshot({ path: 'test-results/local-comparison.png', fullPage: true });
  expect(errors).toEqual([]);
  expect(external).toEqual([]);
});

test('standalone browse starts at a prepared root and fits a phone', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('/browse/');
  await expect(page.locator('.focus-card').first()).toContainText('root');
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
});

test('library validates, imports and opens a portable bundle', async ({ page }) => {
  await page.goto('/library/');
  await page.getByLabel('Choose file').setInputFiles({ name: 'invalid.zip', mimeType: 'application/zip', buffer: Buffer.from('not a ZIP') });
  await page.getByRole('button', { name: 'Import this bundle', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Try again', exact: true })).toBeVisible();
  await page.getByLabel('Choose file').setInputFiles(archive!);
  await page.getByRole('button', { name: 'Import this bundle', exact: true }).click();
  await page.getByRole('button', { name: 'Open this bundle', exact: true }).first().click();
  await expect(page.getByRole('button', { name: 'Open decision trace', exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Open decision trace', exact: true })).toBeVisible();
});


test('punned sources keep their kind through comparison and Browse', async ({ page }) => {
  await page.goto('/');
  await page.locator('button.source-item').filter({ hasText: 'individual' }).click();
  await expect.poll(() => new URL(page.url()).searchParams.get('sk')).toBe('individual');
  await expect(page.locator('.entity-card-source')).toContainText('Source individual');
  await expect(page.locator('button.source-item.current')).toHaveCount(1);
  await expect(page.locator('button.source-item.current')).toContainText('individual');
  await page.getByRole('tab', { name: 'Hierarchy', exact: true }).click();
  await page.getByRole('button', { name: 'Open both in the full ontology browser' }).click();
  await expect(page.getByRole('heading', { name: 'Browse both ontologies' })).toBeVisible();
  const params = new URL(page.url()).searchParams;
  for (const key of ['sk', 'tk', 'pin_sk', 'pin_tk', 'from_sk']) expect(params.get(key)).toBe('individual');
  await expect(page.locator('.focus-card').first()).toHaveAttribute('aria-label', /^Focused individual:/);
  await page.getByRole('link', { name: 'Back to the comparison' }).click();
  await expect(page.locator('.entity-card-source')).toContainText('Source individual');
});
