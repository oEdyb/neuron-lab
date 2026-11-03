// Run with EXPLORER_URL pointing at server.py --no-model with NEURON_LAB_MAPS set.
// MAP_STATE=none: an empty folder. MAP_STATE=core: a Ministral view folder without Meaning or Web data.
import { chromium } from 'playwright'
import assert from 'node:assert/strict'
const base=process.env.EXPLORER_URL||'http://127.0.0.1:8799',ministral='mistralai/Ministral-3-3B-Instruct-2512'
const browser=await chromium.launch(),page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[]
page.on('pageerror',e=>errors.push(e.message))
try{
 if(process.env.MAP_STATE==='none'){
  await page.goto(base);await page.getByRole('alert').waitFor()
  assert.match(await page.getByRole('alert').innerText(),/Map isn't downloaded\. Run: uv run python map_data\.py --model/)
  assert.ok(await page.getByRole('button',{name:'Switch model',exact:true}).isEnabled())
  await page.goto(`${base}/?model=${encodeURIComponent(ministral)}`);await page.getByRole('alert').waitFor()
  assert.match(await page.getByRole('alert').innerText(),/No complete map for mistralai\/Ministral-3-3B-Instruct-2512\. Run: uv run python map_data\.py/)
  await page.getByRole('button',{name:'Switch model',exact:true}).click();await page.getByRole('menuitem',{name:'Gemma 4 E2B'}).waitFor()
 }else{
  const gemma={ok:true,model:'google/gemma-4-E2B-it',where:'local',layers:'3-34',dose:.6,pairs:[['a','b']],rows:[{layer:34,n:12000,sign:1,guess:.8,real:.6,null:.2,beats_null:true,words:['gemma']}]}
  await page.addInitScript(result=>localStorage.setItem('neuron-next-find:google/gemma-4-E2B-it',JSON.stringify({pairs:[{want:'a',opposite:'b'}],preset:'Custom',lo:3,hi:34,where:'local',top:20,job:null,result})),gemma)
  await page.goto(base);await page.getByRole('radio',{name:'Find',exact:true}).click();await page.locator('.find-view').waitFor()
  assert.equal(await page.locator('.find-table').count(),0,"another model's Find result is on screen")
  assert.equal(await page.getByRole('spinbutton',{name:'Last layer',exact:true}).inputValue(),'23')
  await page.getByPlaceholder('Word, meaning, or L17 N5244').fill('the')
  await page.getByText("Meaning search isn't built for this map. Use Exact.").waitFor({timeout:30000})
  await page.getByRole('tab',{name:'Exact',exact:true}).click();await page.locator('.result-row').first().waitFor({timeout:30000})
  await page.locator('.result-row .result').first().dispatchEvent('click');await page.locator('.find-view').waitFor({state:'hidden'});await page.getByRole('radio',{name:'Web',exact:true}).click({timeout:90000})
  await page.getByText('Connection data unavailable. Build it with connections.py.').waitFor({timeout:30000})
 }
 assert.deepEqual(errors,[]);console.log(`missing-map ${process.env.MAP_STATE}: ok`)
}finally{await browser.close()}
