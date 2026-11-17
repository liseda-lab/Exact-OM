import {test, expect} from '@playwright/test';

test.skip(process.env.EXACT_REVIEWER_DEMO !== '1', 'Requires the fixed ten-source reviewer demo');
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
