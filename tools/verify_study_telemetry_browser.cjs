/* Browser acceptance for an isolated study service; never runs against a public host. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { randomUUID } = require('node:crypto');
const { chromium } = require(path.resolve('explanations_visualizer/node_modules/playwright'));
const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, item, i, all) => { if (i % 2 === 0) pairs.push([item.replace(/^--/, ''), all[i + 1]]); return pairs; }, []));
const origin = args.origin;
assert(['127.0.0.1', 'localhost'].includes(new URL(origin).hostname), 'Only an isolated loopback service may be tested');
const sessions = JSON.parse(fs.readFileSync(args.sessions));
const prefix = (sid) => `exact.study.observations.v1.${encodeURIComponent(sid)}.`;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
(async () => {
  const browser = await chromium.launch();
  const journeys = [], network = [], errors = [];
  let faultTests = null;
  try {
    for (const [journey, s] of sessions.entries()) {
      const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
      await context.addCookies([{ name: 'exact_study_session', value: s.cookie, domain: new URL(origin).hostname, path: '/api/v1/study', httpOnly: true, sameSite: 'Strict' }]);
      let page = await context.newPage();
      const watch = (p) => { p.on('pageerror', (e) => errors.push(e.message)); p.on('response', (r) => { if (/\/study\/(events|timing)$/.test(r.url())) network.push({ status: r.status(), kind: r.url().endsWith('/events') ? 'events' : 'timing' }); }); };
      watch(page);
      const state = async () => { const r = await page.request.get(origin + '/api/v1/study/state'); assert(r.ok()); const value = await r.json(); assert(value.synthetic, 'Refusing to operate a real participant session'); return value; };
      const current = async () => { const r = await page.request.get(origin + '/api/v1/study/cases/current'); assert(r.ok()); return r.json(); };
      const outbox = () => page.evaluate((p) => Object.keys(localStorage).filter((k) => k.startsWith(p + 'record.')).map((k) => JSON.parse(localStorage.getItem(k))), prefix(s.session));
      const drain = async () => {
        await page.evaluate(() => document.dispatchEvent(new Event('visibilitychange')));
        for (let i = 0; i < 100; i++) { if (!(await outbox()).some((r) => !r.rejected)) return; if (i > 0 && i % 10 === 0) await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange"))); await sleep(100); }
        throw Error('Observation outbox did not drain');
      };
      const usable = async () => {
        await page.getByRole('button', { name: 'Keep initial order', exact: true }).waitFor({ timeout: 60000 });
        const dialog = page.getByRole('dialog', { name: 'Welcome back', exact: true });
        if (await dialog.count()) { await dialog.getByRole('radio', { name: 'Not sure', exact: true }).check(); await dialog.getByRole('button', { name: 'Continue', exact: true }).click(); await dialog.waitFor({ state: 'hidden' }); }
        await page.waitForFunction(() => [...document.querySelectorAll('button')].some((b) => b.textContent.trim() === 'Keep initial order' && !b.disabled), null, { timeout: 60000 });
      };
      await page.goto(origin + '/participate/'); await state(); await usable(); await drain();
      const seen = [];
      for (let index = 0; index < 10; index++) {
        await usable(); const c = await current(); seen.push({ case_id: c.case_id, condition: c.condition });
        if (c.condition === 'explanation') {
          for (const name of ['Hierarchy', 'Evidence', 'Decision trace', 'Scores', 'Graph']) {
            const tab = page.getByRole('tab', { name, exact: true }); if (await tab.count()) await tab.click();
          }
          await page.getByRole('button', { name: 'Inspect ' + c.candidates[1].label, exact: true }).first().click();
          await page.getByRole('tab', { name: 'Decision trace', exact: true }).click();
          await page.getByRole('heading', { name: 'How this pair was decided', exact: true }).waitFor();
          await page.getByRole('tab', { name: 'Scores', exact: true }).click();
          await page.locator('.score-rows').first().waitFor();
        } else {
          assert.equal(await page.getByRole('tab', { name: 'Decision trace', exact: true }).count(), 0);
          assert.equal(await page.getByRole('tab', { name: 'Scores', exact: true }).count(), 0);
        }
        await drain();
        if (!faultTests && c.condition === 'explanation') {
          const p = prefix(s.session), id = randomUUID(), timingId = randomUUID();
          await page.evaluate(({ p, c, id, timingId }) => {
            const event = { event_id: id, page_instance_id: 'invalid-record-test', sequence: 0, case_id: c.case_id, presentation_id: c.presentation_id, type: 'tab_open', component_id: 'details', element_id: 'unknown-tab', client_monotonic_ms: 1, build_version: 'exact-explain-ui-1.2' };
            const timing = { segment_id: timingId, page_instance_id: 'invalid-record-test', stage: 'case', case_id: c.case_id, presentation_id: c.presentation_id, monotonic_start_ms: 0, monotonic_end_ms: 600000 };
            for (const [recordId, kind, payload] of [[id, 'event', event], [timingId, 'timing', timing]]) localStorage.setItem(p + 'record.' + recordId, JSON.stringify({ id: recordId, kind, payload, queuedAt: Date.now() }));
          }, { p, c, id, timingId });
          await page.getByRole('tab', { name: 'Decision trace', exact: true }).click(); await drain();
          const rejected = (await outbox()).filter((r) => r.rejected);
          assert.deepEqual(new Set(rejected.map((r) => r.id)), new Set([id, timingId]));
          await page.getByRole('status', { name: 'Activity recording status' }).waitFor();
          // Remove only the deliberately invalid test records; the implementation retains them.
          await page.evaluate(({ p, id, timingId }) => { localStorage.removeItem(p + 'record.' + id); localStorage.removeItem(p + 'record.' + timingId); }, { p, id, timingId });
          let unavailable = true, acknowledgedButLost = false; const accepted = new Set(); let lostBody = null;
          await context.route('**/api/v1/study/events', async (route) => {
            if (unavailable) return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Controlled temporary outage' }) });
            if (!acknowledgedButLost) { const data = route.request().postDataJSON(); lostBody = data; const response = await route.fetch(); assert(response.ok()); const receipt = await response.json(); for (const value of receipt.acknowledged_event_ids) accepted.add(value); acknowledgedButLost = true; return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Controlled lost acknowledgement' }) }); }
            return route.continue();
          });
          await page.getByRole('tab', { name: 'Scores', exact: true }).click(); await page.evaluate(() => document.dispatchEvent(new Event('visibilitychange'))); await sleep(500);
          const beforeReload = (await outbox()).filter((r) => !r.rejected && r.kind === 'event'); assert(beforeReload.length > 0);
          await page.reload(); await usable();
          const afterReload = await outbox(); for (const r of beforeReload) assert(afterReload.some((v) => v.id === r.id && v.payload.page_instance_id === r.payload.page_instance_id));
          unavailable = false; await drain();
          // The first delivery after reconnection was committed but its response was lost.
          if ((await outbox()).some((r) => !r.rejected)) await drain();
          assert(acknowledgedButLost && accepted.size > 0 && lostBody);
          await context.unroute('**/api/v1/study/events'); await drain();
          assert.equal((await outbox()).length, 0);
          faultTests = { bad_event_isolated: true, bad_timing_isolated: true, warning_visible: true, reload_retained_ids_and_clocks: true, outage_recovered: true, committed_ack_loss_retried: true, replay_event_ids: [...accepted] };
        }
        await page.getByRole('button', { name: 'Keep initial order', exact: true }).click();
        await page.getByRole('button', { name: 'Submit answer', exact: true }).click();
        await page.waitForFunction(() => ![...document.querySelectorAll('button')].some((b) => b.textContent.trim() === 'Submitting…'));
        for (let i = 0; i < 100; i++) { const next = await state(); if (next.stage !== 'case' || next.current_presentation_id !== c.presentation_id) break; if (i === 99) throw Error('Submission did not advance'); await sleep(100); }
        let next = await state();
        if (next.stage === 'consultation') { await page.getByRole('radio', { name: 'No', exact: true }).check(); await page.getByRole('button', { name: /^Save and (go to case|continue)/ }).click(); for (let i = 0; i < 100; i++) { next = await state(); if (next.stage !== 'consultation') break; await sleep(100); } }
      }
      assert.equal(new Set(seen.map((c) => c.case_id)).size, 10);
      assert.equal(seen.filter((c) => c.condition === 'explanation').length, 5);
      const final = await state(); assert.equal(final.stage, 'final');
      for (const q of final.forms.final) {
        if (!q.required || q.show_if) continue;
        const block = page.locator('#q-' + q.id);
        if (q.matrix) { const rows = block.locator('table tbody tr'); for (let i = 0; i < await rows.count(); i++) await rows.nth(i).getByRole('radio').first().check(); }
        else if (q.options) await block.locator('input').first().check();
        else await block.locator('textarea,input').first().fill('Isolated acceptance check');
      }
      await page.getByRole('button', { name: 'Finish the study', exact: true }).click();
      await page.getByRole('heading', { name: 'Thank you. Your responses are recorded.', exact: true }).waitFor({ timeout: 20000 });
      await drain(); const remaining = await outbox(); if (remaining.length) fs.writeFileSync(args.output + '.remaining.json', JSON.stringify(remaining, null, 2), {mode:0o600}); assert.equal(remaining.length, 0); assert.equal((await state()).stage, 'completed');
      journeys.push({ session: s.session, first_condition: seen[0].condition, cases: seen, completed: true });
      await context.close();
    }
    assert.equal(errors.length, 0, errors.join(';'));
    const report = { journeys, faultTests, network_statuses: network.reduce((o, r) => { const key = `${r.kind}:${r.status}`; o[key] = (o[key] || 0) + 1; return o; }, {}), browser_errors: errors };
    fs.writeFileSync(args.output, JSON.stringify(report, null, 2), { mode: 0o600 });
    console.log(JSON.stringify({ complete_journeys: journeys.length, cases: journeys.length * 10, fault_recovery_passed: Boolean(faultTests), browser_errors: 0 }));
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
