import assert from 'node:assert/strict'
import {mkdir,writeFile,readFile} from 'node:fs/promises'
import {chromium} from 'playwright'
await mkdir('shots/v3',{recursive:true})
const browser=await chromium.launch({headless:true})
const page=await browser.newPage({viewport:{width:1440,height:900}})
const requests=[],errors=[]
page.on('pageerror',e=>errors.push(e.message))
await page.route('**/model/api/compare',async route=>{
 const body=route.request().postDataJSON();body.tokens=60;requests.push(body)
 await route.continue({postData:JSON.stringify(body)})
})
const saved=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('neuron-next-chats'))[0])
const button=name=>page.getByRole('button',{name,exact:true})
const shot=name=>page.screenshot({path:`shots/v3/${name}.png`})
async function send(text){await page.getByRole('textbox',{name:'Message',exact:true}).fill(text);await button('Send message').click();await button('Stop generating').waitFor();assert.equal(await button('Clear chat').isDisabled(),true);await button('Send message').waitFor({timeout:180000});assert.equal(await page.locator('.chat-error').count(),0)}
try{
 await page.goto('http://127.0.0.1:8792');await page.waitForFunction(()=>window.neuronNext)
 await page.evaluate(()=>window.neuronNext.select({layer:17,n:5244}));await button('Compare replies').click();await button('Expand chat').click()
 await send('Say hello in five words.');let chat=await saved();assert.ok(chat.columns.every(c=>c.messages[1].content));assert.equal(await page.locator('.setup-divider').count(),0);await shot('01-first-turn')
 await button('Increase Push dose').click();await button('Increase Push dose').click()
 await send('Name one fruit.');chat=await saved();assert.equal(chat.columns[0].messages[0].setup.dose,.6);assert.equal(chat.columns[0].messages[2].setup.dose,.8);assert.equal(await page.locator('.setup-divider').count(),1);assert.match(await page.locator('.setup-divider').innerText(),/dose 0.6 → 0.8/);await shot('02-dose-divider')
 const oldId=chat.columns[0].messages[2].id,firstReplies=chat.columns.map(c=>c.messages.slice(0,2))
 await button('Increase Push dose').click();await button('Rerun').click();await button('Stop generating').waitFor();await button('Send message').waitFor({timeout:180000});chat=await saved();assert.equal(await page.locator('.chat-error').count(),0)
 assert.ok(chat.columns.every(c=>c.messages.length===4&&c.messages[2].id!==oldId&&c.messages[2].setup.dose===.9&&c.messages[3].content));chat.columns.forEach((c,i)=>assert.deepEqual(c.messages.slice(0,2),firstReplies[i]));assert.equal(requests[2].columns[0].history.length,2);assert.equal(requests[2].message,'Name one fruit.');await shot('03-rerun')
 await button('Show minus column').click();await button('Move plus left').click();const before=await saved()
 await button('Clear chat').click();assert.ok((await saved()).columns.every(c=>c.messages.length===0));await shot('04-cleared');await button('Undo').click();assert.deepEqual((await saved()).columns,before.columns);await shot('05-undo')
 const downloadPromise=page.waitForEvent('download');await button('Export Markdown').click();const download=await downloadPromise;await download.saveAs('shots/v3/chat.md');const md=await readFile('shots/v3/chat.md','utf8');assert.ok(md.includes('Setup: dose 0.6'));assert.ok(md.includes('Setup: dose 0.9'))
 await page.evaluate(()=>{window.copiedText='';if(navigator.clipboard){const original=navigator.clipboard.writeText.bind(navigator.clipboard);navigator.clipboard.writeText=text=>{window.copiedText=text;return original(text)}}else{const original=document.execCommand.bind(document);document.execCommand=(...args)=>{window.copiedText=document.activeElement.value;return original(...args)}}});await button('Copy share text').click();await page.getByText('Copied',{exact:true}).waitFor();assert.ok((await page.evaluate(()=>window.copiedText)).includes('Setup: dose 0.9'))
 await page.reload();await page.waitForFunction(()=>window.neuronNext);await button('Open chat Say hello in five words.').click();assert.deepEqual((await saved()).columns,before.columns);assert.match(await page.locator('.setup-divider').innerText(),/dose 0.6 → 0.9/);await button('Expand chat').click();await shot('06-reopened')
 assert.deepEqual(errors,[]);assert.equal(requests.length,3);assert.ok(requests.every(r=>r.tokens<=60));await writeFile('shots/v3/checks.json',JSON.stringify({checks:['send','dose divider','rerun replaces last replies and excludes old last turn from history','clear all columns including hidden','undo preserves setup/order/visibility','Markdown export','copy','reload and reopen'],requests,errors},null,2));console.log('PASS: real-model chat flow, 3 requests at 60 tokens, no runtime errors')
}finally{await browser.close()}
