import {chromium} from 'playwright';
const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:900}});await page.goto('http://127.0.0.1:8797');await page.waitForFunction(()=>window.neuronMap);await page.evaluate(()=>neuronMap.face(12));await page.waitForTimeout(1900);
for(const variant of ['strength','direction','tiles','ytan']){await page.evaluate(v=>neuronMap.gridVariant(v),variant);await page.waitForTimeout(500);await page.screenshot({path:`shots/grid-options/${variant}.png`});}
await browser.close();
