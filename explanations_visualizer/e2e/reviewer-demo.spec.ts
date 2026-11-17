import {test, expect} from '@playwright/test';

test.skip(process.env.EXACT_REVIEWER_DEMO !== '1', 'Requires the fixed ten-source reviewer demo');
test("recovered decisions show original ranking and scoring without accepted-alignment claims", async ({ page }) => {
  await page.goto("/?run=reviewer-demo-20261117&source=http%3A%2F%2Fncicb.nci.nih.gov%2Fxml%2Fowl%2FEVS%2FThesaurus.owl%23C118847");
  await expect(page.getByRole("heading", { level: 1, name: /^Does / })).toBeVisible();
  await expect(page.locator(".matcher-bar")).toContainText("Ranked 1 of 100 in the original run");
  await expect(page.locator(".matcher-bar")).not.toContainText("saved alignment unknown");
  await page.getByRole("button", { name: "Open decision trace", exact: true }).click();
  await expect(page.locator(".trace-heading")).toBeInViewport();
  await expect(page.locator(".trace")).toContainText("Pair scores recorded from the ontology evidence");
  await expect(page.locator(".trace")).toContainText("Ranked within the original candidate set");
  await expect(page.locator(".trace")).toContainText("Candidate ranking");
  await expect(page.locator(".trace")).not.toContainText("Not recorded in this bundle");
  await page.getByRole("tab", { name: "Scores", exact: true }).click();
  await expect(page.locator(".scores-panel")).toContainText("Base pair score");
  await expect(page.locator(".scores-panel")).toContainText("Ranking utility");
  await expect(page.locator(".scores-panel")).not.toContainText("Membership in the saved alignment");
  await page.setViewportSize({ width: 375, height: 900 });
  await page.getByRole("tab", { name: "Decision trace", exact: true }).click();
  await page.locator(".trace").getByRole("button", { name: "Technical details", exact: true }).first().click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
});
test('fixed demo pool, full comparison tabs, theme and responsive browse', async ({page}) => {
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');
  await expect(page.getByRole('heading',{level:1,name:/^Does /})).toBeVisible();
  await expect(page.locator('.source-list button')).toHaveCount(10);
  await expect(page.getByRole('link',{name:'Library',exact:true})).toHaveCount(0);
  for(const name of ['Hierarchy','Evidence','Decision trace','Evidence graph','Scores']){
    await page.getByRole('tab',{name,exact:true}).click();
    await expect(page.getByRole('tabpanel')).toBeVisible();
    if(name==='Evidence graph')await expect(page.locator('.graph-canvas canvas').first()).toBeVisible();
  }
  await page.getByLabel('Colour theme').selectOption('dark');
  await expect(page.locator('html')).toHaveAttribute('data-theme','dark');
  await page.screenshot({path:'test-results/reviewer-demo-dark.png',fullPage:true});
  await page.getByLabel('Colour theme').selectOption('light');
  await page.goto('/browse/');
  await expect(page.getByRole('heading',{level:1})).toBeVisible();
  await page.screenshot({path:'test-results/reviewer-demo-browse.png',fullPage:true});
  await page.setViewportSize({width:375,height:900});
  await page.goto('/');
  await expect(page.getByRole('heading',{level:1,name:/^Does /})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  expect(errors).toEqual([]);
});
