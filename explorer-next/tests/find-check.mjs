import { chromium } from 'playwright'
import assert from 'node:assert/strict'
import { mkdir,writeFile } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
const shots='shots/find';await mkdir(shots,{recursive:true})
const baseline=JSON.parse(execFileSync('uv',['run','modal','app','list','--json'],{cwd:'..',encoding:'utf8'}));await writeFile(`${shots}/modal-before.json`,JSON.stringify(baseline,null,2))
const browser=await chromium.launch({headless:true}),page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[]
page.on('pageerror',e=>errors.push(e.message))
async function shot(name){for(const size of [{width:1440,height:900},{width:1920,height:1080}]){await page.setViewportSize(size);await page.screenshot({path:`${shots}/${name}-${size.width}x${size.height}.png`})}await page.setViewportSize({width:1440,height:900})}
let appId=null
const start=Date.now()
const guard=setInterval(()=>{try{const apps=JSON.parse(execFileSync('uv',['run','modal','app','list','--json'],{cwd:'..',encoding:'utf8'}));const added=apps.filter(a=>!baseline.some(b=>b.app_id===a.app_id)&&a.description==='neuron-lab');if(!appId&&added.length===1){appId=added[0].app_id;console.log('Find app',appId)}if(Date.now()-start>480000&&appId){execFileSync('uv',['run','modal','app','stop',appId,'-y'],{cwd:'..'});console.log('Budget guard stopped Find');clearInterval(guard)}}catch(e){console.log('Guard:',e.message)}},10000)
try{
 await page.goto('http://127.0.0.1:8792/');await page.getByRole('radio',{name:'Find',exact:true}).click();await page.getByRole('button',{name:'Custom',exact:true}).waitFor();await shot('empty')
 await page.getByRole('button',{name:'Custom',exact:true}).click();await page.getByRole('menuitem',{name:'Evil (8 pairs)',exact:true}).click();await page.getByRole('textbox',{name:'Want 8',exact:true}).waitFor();assert.equal(await page.getByRole('textbox',{name:/^Want /}).count(),8)
 for(let i=8;i>3;i--)await page.getByRole('button',{name:`Remove pair ${i}`,exact:true}).click()
 await page.getByRole('spinbutton',{name:'Top neurons',exact:true}).fill('10')
 const request=page.waitForRequest(r=>r.url().endsWith('/model/api/find')&&r.method()==='POST');await page.getByRole('button',{name:'Run Find',exact:true}).click();const body=(await request).postDataJSON();assert.equal(body.where,'modal');assert.equal(body.model,'Qwen/Qwen3.5-2B');assert.equal(body.pairs.length,3);assert.equal(body.top,10)
 await page.getByText(/no live stage updates/).waitFor();await shot('running');await page.reload();await page.getByRole('radio',{name:'Find',exact:true}).click();await page.getByText(/no live stage updates/).waitFor();console.log('Running state survived reload')
 await Promise.race([page.locator('.find-results').waitFor({timeout:480000}),page.locator('.find-error').waitFor({timeout:480000}).then(async()=>{throw Error(await page.locator('.find-error').innerText())})]);assert.equal(await page.locator('.find-error').count(),0)
 const state=await page.evaluate(()=>JSON.parse(localStorage.getItem('neuron-next-find')));await writeFile(`${shots}/real-result.json`,JSON.stringify({...state.result,elapsed_seconds:(Date.now()-start)/1000,app_id:appId},null,2));await shot('results')
 assert.equal(state.result.rows.length,10);const row=state.result.rows[0],name=`L${row.layer} N${row.n}`
 await page.getByRole('button',{name:`Star ${name}`,exact:true}).click();assert.equal(await page.getByRole('button',{name:`Star ${name}`,exact:true}).getAttribute('aria-pressed'),'true')
 await page.getByRole('button',{name:'Layer',exact:true}).click();assert.equal(await page.locator('.find-table th').first().getAttribute('aria-sort'),'ascending')
 await page.getByRole('button',{name:`Add ${name} to steer set`,exact:true}).click();await page.getByRole('tab',{name:'Chat',exact:true}).waitFor();assert.ok((await page.evaluate(()=>JSON.parse(localStorage.getItem('neuron-next-chats')))).some(c=>c.steerSet.some(n=>n.layer===row.layer&&n.n===row.n)))
 await page.getByRole('button',{name:`Open ${name} in Plane`,exact:true}).click();assert.equal(await page.getByRole('radio',{name:'Plane',exact:true}).getAttribute('data-state'),'on')
 await page.getByRole('radio',{name:'Find',exact:true}).click();await page.getByRole('button',{name:`Open ${name} in Web`,exact:true}).click();await page.getByRole('region',{name:'Neuron web',exact:true}).waitFor()
 await page.reload();await page.getByRole('radio',{name:'Find',exact:true}).click();assert.equal(await page.locator('.find-table tbody tr').count(),10);assert.equal(await page.getByRole('textbox',{name:/^Want /}).count(),3)
 for(const size of [{width:1440,height:900},{width:1920,height:1080}]){await page.setViewportSize(size);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false)}
 assert.deepEqual(errors,[]);console.log('PASS: real Modal, reload during run, results reload, sorting, star, steer set, Plane, Web, no page overflow/errors');console.log(JSON.stringify(state.result.rows.slice(0,5)));console.log('elapsed', (Date.now()-start)/1000)
}catch(e){await page.screenshot({path:`${shots}/failure.png`});console.error(e);process.exitCode=1}finally{clearInterval(guard);await browser.close();if(appId){try{execFileSync('uv',['run','modal','app','stop',appId,'-y'],{cwd:'..',stdio:'pipe'})}catch(e){if(!String(e.stderr).includes('already stopped'))throw e}}const after=execFileSync('uv',['run','modal','app','list','--json'],{cwd:'..',encoding:'utf8'});await writeFile(`${shots}/modal-after.json`,after);console.log('Modal after',after)}
