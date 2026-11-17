import { expect, test, type APIRequestContext } from '@playwright/test';

type Entity = { iri: string; kind: string };
type Source = { entity: Entity; query_id?: number; candidate_count?: number };
type Candidate = { pair_id: string; source: Entity; target: Entity; scores: { name: string; value: number }[]; ranking?: { query_id: number; rank: number } };

async function data(request: APIRequestContext) {
  const run = (await (await request.get('/api/v1/runs')).json()).items[0];
  const prefix = `/api/v1/runs/${encodeURIComponent(run.run_id)}`;
  const sources: Source[] = (await (await request.get(`${prefix}/sources?limit=100`)).json()).items;
  const rows = [];
  for (const source of sources) {
    const params: Record<string, string | number> = { source: source.entity.iri, source_kind: source.entity.kind, limit: 100 };
    if (source.query_id !== undefined) params.query_id = source.query_id;
    const response = await request.get(`${prefix}/candidates`, { params });
    expect(response.ok()).toBeTruthy();
    rows.push({ source, candidates: (await response.json()).items as Candidate[] });
  }
  return { run, prefix, rows };
}

function link(run: string, source: Source, pair?: string, includeQuery = true) {
  const params = new URLSearchParams({ run, source: source.entity.iri, sk: source.entity.kind });
  if (pair) params.set('pair', pair);
  if (includeQuery && source.query_id !== undefined) params.set('q', String(source.query_id));
  return `/?${params}`;
}

test('real staged output opens scores, trace and ontology evidence without outside requests', async ({ page, request, baseURL }) => {
  const { run, prefix, rows } = await data(request);
  let selected: { source: Source; candidate: Candidate } | undefined;
  for (const row of rows) {
    for (const candidate of row.candidates) {
      const evidence = await (await request.get(`${prefix}/pair-evidence`, { params: { pair_id: candidate.pair_id } })).json();
      if (evidence.items.some((item: { status: string; fact_ids: string[] }) => item.status === 'available' && item.fact_ids.length)) {
        selected = { source: row.source, candidate };
        break;
      }
    }
    if (selected) break;
  }
  expect(selected, 'Real native output should retain inspectable ontology evidence').toBeTruthy();
  const { source, candidate } = selected!;
  const errors: string[] = [], external: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => {
    if (new URL(route.request().url()).origin !== new URL(baseURL!).origin) {
      external.push(route.request().url()); return route.abort();
    }
    return route.continue();
  });
  await page.goto(link(run.run_id, source, candidate.pair_id));
  await expect(page.locator('.entity-card-source')).toBeVisible();
  await page.getByRole('button', { name: 'Open decision trace', exact: true }).click();
  await expect(page.locator('.trace-heading')).toBeInViewport();
  await expect(page.getByRole('tabpanel')).toContainText(candidate.ranking ? 'Candidate ranking' : /Saved alignment|saved alignment/);
  await page.getByRole('tab', { name: 'Scores', exact: true }).click();
  await expect(page.getByRole('tabpanel')).toContainText(candidate.ranking ? 'Ranking score' : /Final score|Matching score|Final matching score/i);
  await page.getByRole('tab', { name: 'Evidence', exact: true }).click();
  await page.getByRole('button', { name: 'Show original axiom', exact: true }).first().click();
  await expect(page.getByRole('tabpanel')).toContainText(/SubClassOf|AnnotationAssertion/);
  await page.getByRole('button', { name: 'Show in graph', exact: true }).first().click();
  await expect(page.getByRole('img', { name: /^Evidence graph with/ })).toBeVisible();
  await expect(page.locator('.graph-canvas canvas').first()).toBeVisible();
  await page.getByRole('tab', { name: 'Hierarchy', exact: true }).click();
  await page.getByRole('button', { name: 'Open both in the full ontology browser' }).click();
  await expect(page.getByRole('heading', { name: 'Browse both ontologies' })).toBeVisible();
  await page.getByRole('link', { name: 'Back to the comparison' }).click();
  await expect.poll(() => new URL(page.url()).searchParams.get('pair')).toBe(candidate.pair_id);
  if (source.query_id !== undefined) await expect.poll(() => new URL(page.url()).searchParams.get('q')).toBe(String(source.query_id));
  await expect(page.locator('.entity-card-source')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Open decision trace', exact: true })).toBeVisible();
  await expect(page.locator('.entity-card .skeleton')).toHaveCount(0);
  await expect(page.getByText('Loading label…', { exact: true })).toHaveCount(0);
  await expect(page.locator('.matcher-bar')).toContainText(candidate.ranking ? `Ranked ${candidate.ranking.rank}` : /Selected|Not selected/);
  await page.screenshot({ path: `test-results/staged-${candidate.ranking ? 'local' : 'global'}.png`, fullPage: true });
  expect(errors).toEqual([]); expect(external).toEqual([]);
});

test('repeated and empty ranking queries survive selection, links and reload', async ({ page, request }) => {
  const { run, rows } = await data(request);
  test.skip(rows.every(row => row.source.query_id === undefined), 'Global alignment has no query identities');
  const repeated = rows.filter(row => rows.some(other => other.source.entity.iri === row.source.entity.iri && other.source.query_id !== row.source.query_id));
  expect(repeated.length).toBeGreaterThanOrEqual(2);
  for (const row of repeated) {
    expect(row.candidates.length).toBeGreaterThan(0);
    const candidate = row.candidates[0];
    // Deliberately omit q: the pair must recover its own query, not the first source match.
    await page.goto(link(run.run_id, row.source, candidate.pair_id, false));
    await expect.poll(() => new URL(page.url()).searchParams.get('q')).toBe(String(row.source.query_id));
    await expect(page.locator('button.source-item.current')).toHaveCount(1);
    await expect(page.locator('button.source-item.current')).toContainText(`Query ${row.source.query_id}`);
    await page.reload();
    await expect(page.getByRole('button', { name: 'Open decision trace', exact: true })).toBeVisible();
  }
  const empty = rows.find(row => row.source.query_id !== undefined && !row.candidates.length);
  expect(empty).toBeTruthy();
  await page.locator('button.source-item').filter({ hasText: `Query ${empty!.source.query_id}` }).click();
  await expect.poll(() => new URL(page.url()).searchParams.get('q')).toBe(String(empty!.source.query_id));
  await expect(page.getByRole('button', { name: 'Open decision trace', exact: true })).toHaveCount(0);
  await expect(page.locator('#main')).toContainText(/No candidates|No recorded candidates|No candidate/i);
});

test('real production export can be imported through the local library', async ({ page }) => {
  await page.goto('/library/');
  await page.getByLabel('Choose file').setInputFiles(process.env.EXACT_UI_BUNDLE!);
  await page.getByRole('button', { name: 'Import this bundle', exact: true }).click();
  await page.getByRole('button', { name: 'Open this bundle', exact: true }).first().click();
  await expect(page.locator('button.source-item').first()).toBeVisible();
  await page.reload();
  await expect(page.locator('button.source-item').first()).toBeVisible();
});
