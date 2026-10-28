import {chromium} from 'playwright'
import assert from 'node:assert/strict'
import {mkdir,writeFile} from 'node:fs/promises'
await mkdir('shots/multi',{recursive:true})
const browser=await chromium.launch({headless:true}),page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[],requests=[]
page.on('pageerror',e=>errors.push(e.message))
await page.route('**/model/api/compare',async route=>{const body=route.request().postDataJSON();body.tokens=60;requests.push(body);await route.continue({postData:JSON.stringify(body)})})
const neuron={layer:17,n:5244},pair=[{...neuron,dose:.6,muted:false},{layer:12,n:1339,dose:.4,muted:false}]
const chat={id:'multi-check',title:'Multi neuron check',neuron,dose:.6,pushes:[],steerSet:pair,columns:['minus','normal','plus'].map(id=>({id,dose:id==='normal'?0:id==='minus'?-.6:.6,hidden:false,messages:[]})),timestamp:Date.now()}
await page.goto('http://127.0.0.1:8792/')
await page.evaluate(chat=>localStorage.setItem('neuron-next-chats',JSON.stringify([chat])),chat)
await page.reload();await page.getByRole('button',{name:'Open chat Multi neuron check',exact:true}).click()
await page.getByRole('button',{name:'Expand chat',exact:true}).click()
const shots=[]
for(const count of [2,3]){if(count===3){await page.evaluate(chat=>{chat.steerSet.push({layer:13,n:3956,dose:-.3,muted:false});localStorage.setItem('neuron-next-chats',JSON.stringify([chat]))},chat);await page.reload();await page.getByRole('button',{name:'Open chat Multi neuron check',exact:true}).click();await page.getByRole('button',{name:'Expand chat',exact:true}).click()}
for(const size of [{width:1440,height:900},{width:1920,height:1080}]){await page.setViewportSize(size);await page.waitForTimeout(600);const path=`shots/multi/${count}-neurons-${size.width}x${size.height}.png`;await page.screenshot({path});shots.push(path)}}
await page.getByRole('button',{name:'Remove L13 N3956',exact:true}).click()
await page.locator('#message').fill('Who are you and what do you want?');await page.getByRole('button',{name:'Send message',exact:true}).click();await page.getByRole('button',{name:'Stop generating',exact:true}).waitFor();await page.getByRole('button',{name:'Send message',exact:true}).waitFor({timeout:240000})
assert.equal(await page.locator('.chat-error').count(),0,(await page.locator('.chat-error').allTextContents()).join(' '))
const replies=await page.evaluate(()=>JSON.parse(localStorage.getItem('neuron-next-chats'))[0].columns.map(c=>({id:c.id,text:c.messages.at(-1).content})))
assert.deepEqual(requests[0].columns.find(c=>c.id==='normal').pushes,[])
assert.deepEqual(requests[0].columns.find(c=>c.id==='minus').pushes,pair.map(({layer,n,dose})=>({layer,n,dose:-dose})))
const plain=await page.evaluate(async()=>{const r=await fetch('/model/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pushes:[],history:{steered:[]},message:'Who are you and what do you want?',compare:false,tokens:60,sample:false})});return (await r.text()).trim().split('\n').map(x=>JSON.parse(x)).find(e=>e.done)?.text})
assert.equal(replies.find(c=>c.id==='normal').text,plain)
assert.notEqual(replies.find(c=>c.id==='plus').text,plain)
await page.getByRole('button',{name:'Mute L12 N1339',exact:true}).click();await page.getByRole('button',{name:'Clear chat',exact:true}).click();await page.locator('#message').fill('Who are you and what do you want?');await page.getByRole('button',{name:'Send message',exact:true}).click();await page.getByRole('button',{name:'Stop generating',exact:true}).waitFor();await page.getByRole('button',{name:'Send message',exact:true}).waitFor({timeout:240000})
assert.equal(requests[1].columns.find(c=>c.id==='plus').pushes.length,1)
const mutedReplies=await page.evaluate(()=>JSON.parse(localStorage.getItem('neuron-next-chats'))[0].columns.map(c=>({id:c.id,text:c.messages.at(-1).content})))
await writeFile('shots/multi/checks.json',JSON.stringify({replies,plain,mutedReplies,requests,shots,errors},null,2));assert.deepEqual(errors,[]);console.log(JSON.stringify({replies,plain,mutedReplies,shots,errors},null,2));await browser.close()
