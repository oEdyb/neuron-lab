import {chromium} from 'playwright'
import assert from 'node:assert/strict'
const browser=await chromium.launch({headless:true}),page=await browser.newPage(),calls=[]
await page.route('**/model/api/help',route=>route.fulfill({status:200,contentType:'application/json',body:'{"ok":true}'}))
await page.route('**/model/api/chat',async route=>{calls.push(route.request().postDataJSON());await route.fulfill({status:200,contentType:'application/x-ndjson',body:'{"ok":true,"done":true,"text":"Compatibility reply.","stopped":"eos"}\n'})})
await page.goto('http://127.0.0.1:8792/')
await page.evaluate(()=>localStorage.setItem('neuron-next-chats',JSON.stringify([{id:'old',title:'Legacy test',neuron:{layer:17,n:5244},dose:.6,pushes:[{layer:12,n:1339,dose:.4}],columns:['minus','normal','plus'].map(id=>({id,dose:0,hidden:false,messages:[]})),timestamp:Date.now()}])))
await page.reload();await page.getByRole('button',{name:'Open chat Legacy test',exact:true}).click();assert.equal(await page.locator('.steer-row').count(),2)
async function send(){await page.locator('#message').fill('Compatibility check');await page.getByRole('button',{name:'Send message',exact:true}).click();await page.waitForFunction(()=>document.querySelectorAll('.markdown').length>=3);await page.getByRole('button',{name:'Send message',exact:true}).waitFor()}
await send();assert.equal(calls.length,3);assert.deepEqual(calls[1].pushes,[]);assert.equal(calls[0].pushes[1].dose,-.4);assert.equal(calls[2].pushes[1].dose,.4)
await page.getByRole('button',{name:'Mute L12 N1339',exact:true}).click();await send();await page.getByText('muted L12 N1339',{exact:true}).waitFor();assert.equal(calls.at(-1).pushes.length,1)
await page.getByRole('button',{name:'Show minus column',exact:true}).click();await page.getByRole('button',{name:'Move plus left',exact:true}).click();await page.reload();await page.getByRole('button',{name:'Open chat Compatibility check',exact:true}).click();assert.deepEqual(await page.locator('.chat-column').evaluateAll(es=>es.map(e=>e.dataset.column)),['plus','normal'])
console.log('PASS: legacy conversion, sequential old-server fallback, per-turn mute divider, hidden/order persistence');await browser.close()
