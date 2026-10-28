import { config } from './config.js';
import { gridShape } from './layout.js';
import { detail } from './data.js';
export function createPlane(canvas,scene,onSelect,onHover){
 let layer=0,selected=null,cells=[],geometry,rowsData,loadVersion=0,variant='strength';
 const ctx=canvas.getContext('2d');
 function draw(){
  const r=canvas.getBoundingClientRect();if(!r.width||!r.height)return;
  const dpr=Math.min(devicePixelRatio,2),count=config.counts[layer],{columns,rows}=gridShape(count);canvas.width=Math.round(r.width*dpr);canvas.height=Math.round(r.height*dpr);ctx.scale(dpr,dpr);
  const step=Math.floor(Math.min((r.width-config.planePadding*2)/columns,(r.height-config.planePadding*2)/rows)*dpr)/dpr,x=Math.round((r.width-step*(columns-1))/2*dpr)/dpr,y=Math.round((r.height-step*(rows-1))/2*dpr)/dpr;
  geometry={x,y,step,columns,rows};ctx.clearRect(0,0,r.width,r.height);cells=[];
  const positions=scene.positions(layer),order=Array.from({length:count},(_,n)=>n).sort((a,b)=>Math.round((positions[a*3+2]+config.depth/2)/config.depth*(rows-1))-Math.round((positions[b*3+2]+config.depth/2)/config.depth*(rows-1))||positions[a*3]-positions[b*3]);
  for(let i=0;i<count;i++)cells[i]=order[i];
  for(let i=0;i<count;i++){
   const n=cells[i],col=i%columns,row=Math.floor(i/columns),px=x+col*step,py=y+row*step,score=scene.scores[config.offsets[layer]+n],t=Math.min(1,Math.log1p(score)/Math.log(13));
   const negative=rowsData&&rowsData.rows[n].kl[1]/rowsData.p95[1]>rowsData.rows[n].kl[0]/rowsData.p95[0];
   ctx.fillStyle=variant==='direction'&&negative?`rgb(${Math.round(40+100*t)},${Math.round(55+160*t)},${Math.round(65+185*t)})`:`rgb(${Math.round(45+210*t)},${Math.round(38+190*t)},${Math.round(28+160*t)})`;
   if(variant==='tiles'){ctx.fillRect(px-step*.37,py-step*.37,step*.74,step*.74);}
   else if(variant==='ytan'){
    const size=step*(.12+.5*t),level=Math.min(3,Math.floor(t*4));
    if(level===0){ctx.fillRect(px-.5,py-.5,1,1);}else if(level===1){ctx.fillRect(px-size/2,py-.5,size,1);}else if(level===2){ctx.fillRect(px-size/2,py-.5,size,1);ctx.fillRect(px-.5,py-size/2,1,size);}else{ctx.strokeStyle=ctx.fillStyle;ctx.lineWidth=1;ctx.strokeRect(Math.round(px-size/2)+.5,Math.round(py-size/2)+.5,Math.round(size),Math.round(size));}
   }else{ctx.beginPath();ctx.arc(px,py,Math.max(.55,step*(.09+.21*t)),0,Math.PI*2);ctx.fill();}
   if(n===selected){ctx.strokeStyle='#ffe2b3';ctx.lineWidth=1;ctx.strokeRect(Math.round(px-step*.6)+.5,Math.round(py-step*.6)+.5,Math.round(step*1.2),Math.round(step*1.2));}
  }
 }
 function hit(e){if(!geometry)return null;const r=canvas.getBoundingClientRect(),{x,y,step,columns,rows}=geometry,col=Math.round((e.clientX-r.left-x)/step),row=Math.round((e.clientY-r.top-y)/step);return col>=0&&col<columns&&row>=0&&row<rows?cells[row*columns+col]:null;}
 canvas.onclick=e=>{const n=hit(e);if(n!==null&&n!==undefined)onSelect(layer,n);};canvas.onpointermove=e=>onHover(layer,hit(e),e);canvas.onpointerleave=()=>onHover(layer,null);
 canvas.onkeydown=e=>{const offsets={ArrowLeft:-1,ArrowRight:1,ArrowUp:-geometry.columns,ArrowDown:geometry.columns};if(!(e.key in offsets))return;e.preventDefault();e.stopPropagation();const cell=selected===null?0:cells.indexOf(selected);onSelect(layer,cells[Math.max(0,Math.min(cells.length-1,cell+offsets[e.key]))]);};
 new ResizeObserver(draw).observe(canvas);
 return {show(l,n=null){layer=l;selected=n;rowsData=null;const version=++loadVersion;draw();if(variant==='direction')detail(l).then(d=>{if(version===loadVersion){rowsData=d;draw();}});},draw,variant(value){variant=value;this.show(layer,selected);},project(n){const i=cells.indexOf(n),r=canvas.getBoundingClientRect();return{x:r.left+geometry.x+i%geometry.columns*geometry.step,y:r.top+geometry.y+Math.floor(i/geometry.columns)*geometry.step};},stats:()=>({cells:cells.length,unique:new Set(cells).size,...geometry})};
}
