import {chromium} from 'playwright'
import assert from 'node:assert/strict'
const base=process.env.EXPLORER_URL||'http://127.0.0.1:8800'
const browser=await chromium.launch({headless:true}),page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[]
page.on('pageerror',e=>errors.push(e.message))
const frames=()=>page.evaluate(()=>window.neuronNext.stats().frames)
const during=async ms=>{const start=await frames();await page.waitForTimeout(ms);return await frames()-start}
try{
 await page.goto(base);await page.waitForFunction(()=>window.neuronNext,null,{timeout:180000})
 assert.ok(await during(1500)>0,'Tower renders while visible')
 await page.getByRole('radio',{name:'Library'}).click();await page.waitForTimeout(300)
 assert.equal(await during(1500),0,'Map renders nothing while hidden')
 await page.getByRole('radio',{name:'Tower'}).click();await page.waitForTimeout(300)
 assert.ok(await during(1500)>0,'Tower renders again when shown')
 assert.deepEqual(errors,[]);console.log('PASS map rendering stops while hidden and resumes when shown')
}finally{await browser.close()}
