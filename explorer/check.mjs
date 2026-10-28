import { chromium } from './node_modules/playwright/index.mjs';
import { mkdir,writeFile } from 'node:fs/promises';
const dir=new URL('./shots/',import.meta.url).pathname;await mkdir(dir,{recursive:true});
const browser=await chromium.launch({headless:true});const checks=[];
for(const [width,height]of [[1440,900],[1920,1080]]){
 const page=await browser.newPage({viewport:{width,height},deviceScaleFactor:1});const errors=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('http://127.0.0.1:8797',{waitUntil:'domcontentloaded',timeout:60000});await page.waitForFunction(()=>window.neuronMap);await page.waitForTimeout(1400);
 const shot=async name=>page.screenshot({path:`${dir}${width}x${height}-${name}.png`});
 await shot('tower');await page.keyboard.press('Enter');await page.waitForTimeout(1500);
 for(const l of [2,12,22]){await page.evaluate(l=>window.neuronMap.face(l),l);await page.waitForTimeout(800);await shot(`L${l}`);checks.push({size:width,layer:l,title:await page.locator('#layer-title').innerText(),facts:await page.locator('#facts .metric span').innerText()});}
 await page.keyboard.press('Meta+k');await page.locator('#query').fill('death');await page.waitForSelector('.result');await page.waitForTimeout(300);await shot('semantic-death');checks.push({size:width,search:await page.locator('#results').innerText()});await page.keyboard.press('Escape');
 await page.evaluate(()=>window.neuronMap.select(17,5244));await page.waitForSelector('#favourite');await page.waitForTimeout(1000);await shot('L17-N5244');await page.locator('#favourite').click();await page.locator('#dose').fill('0.8');await page.locator('#dose').dispatchEvent('change');await page.locator('[data-sign="1"]').click();await page.keyboard.press('Meta+k');await page.locator('[data-tab="favourites"]').click();await page.locator('#query').fill('');await page.waitForTimeout(400);await shot('favourites');checks.push({size:width,favourites:await page.locator('#results').innerText()});await page.locator('.result').first().click();await page.waitForSelector('#favourite');checks.push({size:width,restoredDose:await page.locator('#dose').inputValue(),restoredSign:await page.locator('[data-sign="1"]').getAttribute('aria-pressed')});
 const pos=await page.evaluate(()=>window.neuronMap.project(17,5244));await page.mouse.click(pos.x,pos.y);await page.waitForTimeout(250);checks.push({size:width,clicked:await page.evaluate(()=>window.neuronMap.stats().selected),overflow:await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),errors});
 if(await page.evaluate(()=>window.neuronMap.stats().reachable)){
 await page.locator('#add-steer').click();await page.locator('#message').fill('What is a neuron? One sentence.');await page.locator('#send').click();await page.waitForFunction(()=>!document.querySelector('#send').disabled,null,{timeout:120000});await shot('chat');checks.push({size:width,chat:await page.locator('#chat-log').innerText()});
 }else{checks.push({size:width,chat:'Model unavailable; controls hidden'});await shot('chat-unavailable');}
 await page.goto('about:blank');await page.goto('http://127.0.0.1:8797/#L17-N5244',{waitUntil:'domcontentloaded',timeout:60000});await page.waitForFunction(()=>window.neuronMap?.stats().selected?.n===5244);await page.waitForSelector('#favourite');checks.push({size:width,hash:page.url(),persistedFavourites:await page.evaluate(()=>window.neuronMap.stats().favourites)});await page.close();
}
await writeFile(`${dir}checks.json`,JSON.stringify(checks,null,2));await browser.close();console.log(JSON.stringify(checks,null,2));
