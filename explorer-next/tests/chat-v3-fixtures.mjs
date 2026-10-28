import assert from 'node:assert/strict'
import {chromium} from 'playwright'
const browser=await chromium.launch({headless:true})
const page=await browser.newPage({viewport:{width:1440,height:900}})
const errors=[];page.on('pageerror',e=>errors.push(e.message))
const saved=()=>page.evaluate(()=>JSON.parse(localStorage.getItem('neuron-next-chats'))[0])
const button=name=>page.getByRole('button',{name,exact:true})
const shot=name=>page.screenshot({path:`shots/v3/${name}.png`})
try{
 await page.goto('http://127.0.0.1:8792');await page.waitForFunction(()=>window.neuronNext)
 const seed={id:'compact-v3',title:'Compact chat',neuron:{layer:17,n:5244},dose:.6,pushes:[],timestamp:Date.now(),columns:['minus','normal','plus'].map(id=>({id,dose:id==='normal'?0:id==='minus'?-.6:.6,hidden:false,messages:[0,1].flatMap(i=>[{id:`compact-${i}`,sentAt:i+1,role:'user',content:`Question ${i+1}`,setup:{dose:.6,pushes:[]}},{id:`compact-${i}`,sentAt:i+1,role:'assistant',content:'Short reply.',setup:{dose:.6,pushes:[]},stopped:'eos'}])}))}
 await page.evaluate(c=>localStorage.setItem('neuron-next-chats',JSON.stringify([c])),seed);await page.reload();await page.waitForFunction(()=>window.neuronNext);await button('Open chat Compact chat').click();await button('Expand chat').click()
 await button('Collapse chat').click();await shot('07-compact')
 const original=await saved()
 const legacy=structuredClone(original);legacy.id='legacy-v3';legacy.title='Legacy chat';legacy.columns.forEach(c=>c.messages.forEach(m=>{delete m.setup;delete m.sentAt;delete m.id}))
 await page.evaluate(c=>localStorage.setItem('neuron-next-chats',JSON.stringify([c])),legacy);await page.reload();await page.waitForFunction(()=>window.neuronNext);await button('Open chat Legacy chat').click();assert.equal(await page.locator('.chat-turn').count(),2);assert.equal(await page.locator('.setup-divider').count(),0);assert.ok((await saved()).columns[0].messages[0].content);await shot('08-legacy')
 const fixture=structuredClone(original);fixture.id='extras-v3';fixture.title='Extra neuron history';fixture.columns.forEach(c=>{c.hidden=false;c.messages=[];for(let i=0;i<4;i++){const setup={dose:.6,pushes:i===1||i===2?[{layer:12,n:1339,dose:.4}]:[]};c.messages.push({id:`fixture-${i}`,sentAt:i+1,role:'user',content:`Question ${i+1}`,setup},{id:`fixture-${i}`,sentAt:i+1,role:'assistant',content:'Fixture reply.',stopped:'eos',setup})}})
 await page.evaluate(c=>localStorage.setItem('neuron-next-chats',JSON.stringify([c])),fixture);await page.reload();await page.waitForFunction(()=>window.neuronNext);await button('Open chat Extra neuron history').click();await button('Expand chat').click();assert.equal(await page.locator('.setup-divider').count(),2);assert.equal(await page.locator('.setup-divider .meaning-plus').innerText(),'+ L12 N1339 +0.4');assert.equal(await page.locator('.setup-divider .meaning-minus').innerText(),'− L12 N1339');await shot('09-extra-dividers')
 await page.evaluate(c=>localStorage.setItem('neuron-next-chats',JSON.stringify([c])),original)
 assert.deepEqual(errors,[]);console.log('PASS: compact layout, legacy saved chat, extra push dividers, unchanged setup')
}finally{await browser.close()}
