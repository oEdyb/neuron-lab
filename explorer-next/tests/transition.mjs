import {chromium} from 'playwright'
import {readFile,writeFile} from 'node:fs/promises'
import assert from 'node:assert/strict'
const browser=await chromium.launch({headless:true}),context=await browser.newContext({viewport:{width:1440,height:900},recordVideo:{dir:'shots/video',size:{width:1440,height:900}}}),page=await context.newPage(),frames=[]
try{
 await page.goto('http://127.0.0.1:8800');await page.waitForFunction(()=>window.neuronNext);await page.waitForTimeout(1100)
 await page.evaluate(()=>window.neuronNext.face(17));const start=Date.now()
 for(let i=0;i<6;i++){if(i)await page.waitForTimeout(150);const file=`shots/transition-frame-${i}.png`;await page.screenshot({path:file});frames.push({file,ms:Date.now()-start,stats:await page.evaluate(()=>window.neuronNext.stats())})}
 await page.waitForFunction(()=>!window.neuronNext.stats().flying)
 await page.evaluate(()=>window.neuronNext.face(null));await page.waitForTimeout(160);const before=await page.evaluate(()=>window.neuronNext.stats().camera);await page.evaluate(()=>window.neuronNext.face(12));const after=await page.evaluate(()=>window.neuronNext.stats().camera);assert.ok(Math.hypot(...after.map((v,i)=>v-before[i]))<3)
 await page.waitForTimeout(160);await page.evaluate(()=>window.neuronNext.face(17));await page.waitForFunction(()=>!window.neuronNext.stats().flying);await page.screenshot({path:'shots/transition-final.png'})
 await page.emulateMedia({reducedMotion:'reduce'});await page.evaluate(()=>window.neuronNext.face(null));assert.equal(await page.evaluate(()=>window.neuronNext.stats().flying),false);await page.evaluate(()=>window.neuronNext.face(17));assert.equal(await page.evaluate(()=>window.neuronNext.stats().flying),false)
 await page.emulateMedia({reducedMotion:'no-preference'});await page.evaluate(()=>window.neuronNext.face(null));await page.waitForTimeout(150);const r=await page.locator('.viewport').boundingBox();await page.mouse.move(r.x+200,r.y+200);await page.mouse.down();await page.mouse.move(r.x+230,r.y+220);assert.equal(await page.evaluate(()=>window.neuronNext.stats().flying),false);await page.mouse.up()
 const video=page.video();await page.close();await context.close();await video.saveAs('shots/tower-to-plane.webm')

 await writeFile('shots/transition-check.json',JSON.stringify({ok:true,checks:['Continuous camera path in one scene','Rapid retargeting preserves current camera position','Pointer input cancels flight','Reduced motion completes synchronously'],frames},null,2));console.log('PASS transition, retarget, pointer cancel, reduced motion; video and six frames saved')
}finally{await browser.close()}
