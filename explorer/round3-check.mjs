import {chromium} from 'playwright';
import {readFile,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const browser=await chromium.launch({headless:true});const report={sizes:{},errors:[],chatRequests:[]};
for(const [width,height]of [[1440,900],[1920,1080]]){
 const context=await browser.newContext({viewport:{width,height}}),page=await context.newPage();page.on('pageerror',e=>report.errors.push(e.message));
 page.on('request',req=>{if(req.url().endsWith('/api/chat')){const b=req.postDataJSON();report.chatRequests.push({width,pushes:b.pushes,turns:b.history.steered.length,message:b.message});}});
 await page.goto('http://127.0.0.1:8797');await page.waitForFunction(()=>window.neuronMap?.stats().reachable);await page.waitForTimeout(1900);
 const shot=async name=>page.screenshot({path:`shots/round3/${width}x${height}-${name}.png`});
 await shot('tower');await page.evaluate(()=>neuronMap.face(12));await page.waitForTimeout(1900);await shot('L12');
 assert.equal(await page.locator('#rail button').count(),24);assert.equal((await page.evaluate(()=>neuronMap.gridStats())).unique,6144);
 await page.evaluate(()=>neuronMap.select(17,5244));await page.locator('#favourite').waitFor();await shot('L17-N5244');
 await page.locator('#dose').fill('0.7');await page.locator('#dose').dispatchEvent('change');await page.locator('[data-sign="1"]').click();await page.locator('#favourite').click();await page.locator('[data-tab="favourites"]').click();await page.locator('#results .result').waitFor();await shot('favourites');
 await page.locator('#results .result').click();await page.locator('#dose').waitFor();assert.equal(await page.locator('#dose').inputValue(),'0.7');assert.equal(await page.locator('[data-sign="1"]').getAttribute('aria-pressed'),'true');
 await page.locator('[data-tab="semantic"]').click();
 for(const q of ['death','pirate']){await page.locator('#query').fill(q);await page.waitForFunction(()=>document.querySelector('#search-status').textContent==='Meaning × strength vs random'&&document.querySelectorAll('#results .result').length>0);await shot(`search-${q}`);}
 await page.evaluate(()=>neuronMap.select(17,5244));await page.locator('#add-steer').waitFor();await page.locator('#add-steer').click();await page.locator('#chat-expand').click();await page.locator('#message').fill('Say hello in one short sentence.');await page.locator('#send').click();await page.waitForFunction(()=>!document.querySelector('#send').disabled,{},{timeout:240000});
 assert.equal(await page.locator('.chat-column').count(),3);assert.equal(await page.locator('.column-log .assistant').count(),3);await shot('chat-three');
 await page.locator('[aria-label="Show minus column"]').click();assert.equal(await page.locator('.chat-column').count(),2);await shot('chat-minus-hidden');
 await page.locator('[aria-label="Show minus column"]').click();await page.locator('[aria-label="Move plus left"]').click();assert.deepEqual(await page.locator('.chat-column').evaluateAll(cs=>cs.map(c=>c.dataset.column)),['minus','plus','normal']);await shot('chat-reordered');
 if(width===1440){await page.locator('#message').fill('What did I ask you to say?');await page.locator('#send').click();await page.waitForFunction(()=>!document.querySelector('#send').disabled,{},{timeout:240000});assert.equal(await page.locator('.column-log .assistant').count(),6);await shot('chat-multiturn');}
 const bounds=await page.evaluate(()=>{const rect=id=>{const {x,y,width,height}=document.querySelector(id).getBoundingClientRect();return {x,y,width,height};};return {sidebar:rect('#palette'),rail:rect('#rail'),chat:rect('#chat'),overflow:document.documentElement.scrollWidth>innerWidth};});assert.equal(bounds.overflow,false);assert.ok(bounds.sidebar.x+bounds.sidebar.width<=bounds.rail.x);assert.ok(Object.values(bounds.chat).every(Number.isInteger));report.sizes[width]={bounds,grid:await page.evaluate(()=>neuronMap.gridStats())};
 await page.reload();await page.waitForFunction(()=>window.neuronMap?.stats().reachable);await page.locator('#chat-open').click();assert.deepEqual(await page.locator('.chat-column').evaluateAll(cs=>cs.map(c=>c.dataset.column)),['minus','plus','normal']);await context.close();
}
assert.equal(report.errors.length,0);await writeFile('shots/round3/checks.json',JSON.stringify(report,null,2));await browser.close();console.log('PASS desktop, semantic search, favourites, real three-column chat, multi-turn, persistence, bounds');
