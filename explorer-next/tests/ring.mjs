import {chromium} from 'playwright'
import {writeFile} from 'node:fs/promises'
import assert from 'node:assert/strict'
const browser=await chromium.launch({headless:true}),page=await browser.newPage({viewport:{width:1440,height:900}}),measurements=[]
try{await page.goto('http://127.0.0.1:8800/#L17-N5244');await page.waitForFunction(()=>window.neuronNext&&!window.neuronNext.stats().flying);await page.waitForTimeout(200)
 for(let zoom=0;zoom<2;zoom++){
  const center=await page.evaluate(()=>{const r=document.querySelector('.viewport').getBoundingClientRect(),p=window.neuronNext.project(17,5244);return{x:r.left+p.x,y:r.top+p.y}})
  const png=(await page.screenshot()).toString('base64')
  const m=await page.evaluate(async({png,center})=>{const img=new Image();img.src='data:image/png;base64,'+png;await img.decode();const c=document.createElement('canvas');c.width=img.width;c.height=img.height;const ctx=c.getContext('2d');ctx.drawImage(img,0,0);const d=ctx.getImageData(0,0,c.width,c.height).data,points=[];for(let y=Math.floor(center.y)-14;y<center.y+14;y++)for(let x=Math.floor(center.x)-14;x<center.x+14;x++){const radius=Math.hypot(x-center.x,y-center.y),i=(y*c.width+x)*4;if(radius>6&&radius<13&&d[i]>185&&d[i+1]>185&&d[i+2]>185)points.push({x,y})}return{center,pixels:points.length,x:(Math.min(...points.map(p=>p.x))+Math.max(...points.map(p=>p.x)))/2,y:(Math.min(...points.map(p=>p.y))+Math.max(...points.map(p=>p.y)))/2,width:Math.max(...points.map(p=>p.x))-Math.min(...points.map(p=>p.x))+1,height:Math.max(...points.map(p=>p.y))-Math.min(...points.map(p=>p.y))+1}},{png,center})
  assert.ok(m.pixels>20);assert.ok(Math.abs(m.x-center.x)<=1);assert.ok(Math.abs(m.y-center.y)<=1);measurements.push(m);await page.screenshot({path:`shots/ring-zoom-${zoom}.png`});await page.mouse.move(center.x,center.y);await page.mouse.wheel(0,-120);await page.waitForTimeout(300)
 }
 assert.ok(Math.abs(measurements[0].width-measurements[1].width)<=2);await writeFile('shots/ring-check.json',JSON.stringify({ok:true,measurements},null,2));console.log('PASS selection ring centered within 1px at both zoom levels')
}finally{await browser.close()}
