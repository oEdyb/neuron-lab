import { config } from './config.js';
import { $ } from './panels.js';
export async function connectModel(expectedModel) {
 let base;
 for(const candidate of (new URLSearchParams(location.search).has('server')?[config.server]:['./model',config.server])){
  try{const response=await fetch(`${candidate}/api/state`,{signal:AbortSignal.timeout(config.probeTimeout)}),state=await response.json();if(response.ok&&state.ok&&state.model===expectedModel){base=candidate;break;}}catch{}
 }
 if(!base)return {reachable:false,add(){}};
 let settings;try{settings=JSON.parse(localStorage.getItem('neuron-chat-columns'));}catch{}
 const ids=['minus','normal','plus'];
 let order=Array.isArray(settings?.order)&&settings.order.length===3&&new Set(settings.order).size===3&&settings.order.every(id=>ids.includes(id))?settings.order:[...ids];
 let hidden=new Set((settings?.hidden||[]).filter(id=>ids.includes(id)));if(hidden.size===3)hidden.clear();
 const histories=Object.fromEntries(ids.map(id=>[id,[]])),pushes=[],failures={};
 let busy=false;
 const label=id=>id==='normal'?'Normal':`${id==='minus'?'−':'+'}${Math.abs(pushes[0]?.dose??config.dose)}`;
 function persist(){localStorage.setItem('neuron-chat-columns',JSON.stringify({order,hidden:[...hidden]}));}
 function render(){
  $('#chat-controls').replaceChildren();$('#chat-log').replaceChildren();
  for(const id of order){
   const toggle=document.createElement('button');toggle.textContent=label(id);toggle.setAttribute('aria-pressed',String(!hidden.has(id)));toggle.setAttribute('aria-label',`Show ${id} column`);toggle.onclick=()=>{if(hidden.has(id))hidden.delete(id);else if(hidden.size<2)hidden.add(id);persist();render();};$('#chat-controls').append(toggle);
   if(hidden.has(id))continue;
   const column=document.createElement('section');column.className='chat-column';column.dataset.column=id;
   const head=document.createElement('div');head.className='column-head';const title=document.createElement('strong');title.textContent=label(id);head.append(title);
   for(const [offset,symbol]of [[-1,'←'],[1,'→']]){const b=document.createElement('button');b.textContent=symbol;b.setAttribute('aria-label',`Move ${id} ${offset<0?'left':'right'}`);b.disabled=order.indexOf(id)+offset<0||order.indexOf(id)+offset>=order.length||busy;b.onclick=()=>{const index=order.indexOf(id);[order[index],order[index+offset]]=[order[index+offset],order[index]];persist();render();};head.append(b);}
   const log=document.createElement('div');log.className='column-log';column.append(head,log);
   for(const turn of histories[id]){const p=document.createElement('p');p.className=turn.role;p.textContent=turn.content;log.append(p);}
   if(failures[id]){const error=document.createElement('p');error.className='chat-error';error.textContent=failures[id];log.append(error);}$('#chat-log').append(column);
  }
 }
 function open(){$('#selection').hidden=true;$('#facts').hidden=true;$('#chat').hidden=false;render();}
 $('#chat-open').hidden=false;$('#chat-open').onclick=open;
 $('#chat-close').onclick=()=>{$('#chat').hidden=true;document.body.classList.remove('chat-expanded');window.neuronMap.restoreInspector();};
 $('#chat-expand').onclick=()=>{const expanded=document.body.classList.toggle('chat-expanded');$('#chat-expand').textContent=expanded?'Collapse ↙':'Expand ↗';};
 $('#composer').onsubmit=async event=>{
  event.preventDefault();const message=$('#message').value.trim();if(!message||busy)return;
  busy=true;$('#send').disabled=true;$('#message').value='';$('#chat-controls').inert=true;
  const visible=order.filter(id=>!hidden.has(id));
  for(const id of visible){
   delete failures[id];const history=histories[id].map(t=>({...t}));histories[id].push({role:'user',content:message},{role:'assistant',content:'Waiting…'});render();
   const turn=histories[id].at(-1);let complete=false;
   try{
    const signed=id==='normal'?[]:pushes.map(p=>({...p,dose:Math.abs(p.dose)*(id==='minus'?-1:1)}));
    const response=await fetch(`${base}/api/chat`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pushes:signed,history:{steered:history},message,compare:false,tokens:120,sample:false})});
    if(!response.ok)throw new Error((await response.json()).error||'Chat failed');
    const reader=response.body.pipeThrough(new TextDecoderStream()).getReader();let pending='';turn.content='';
    function chunk(line){if(!line.trim())return;const c=JSON.parse(line);if(c.ok===false)throw new Error(c.error);if(c.delta)turn.content+=c.delta;if(c.done){turn.content=c.text;complete=true;}const log=$(`[data-column="${id}"] .column-log`);if(log){log.lastChild.textContent=turn.content;log.scrollTop=log.scrollHeight;}}
    while(true){const {value,done}=await reader.read();pending+=value||'';const lines=pending.split('\n');pending=lines.pop();lines.forEach(chunk);if(done){chunk(pending);break;}}
    if(!complete)throw new Error('Response interrupted. Send again to retry.');
   }catch(error){histories[id].splice(-2);failures[id]=error.message;render();}
  }
  busy=false;$('#send').disabled=false;$('#chat-controls').inert=false;$('#chat-log').querySelectorAll('button').forEach(b=>{const index=order.indexOf(b.closest('[data-column]').dataset.column);b.disabled=index+(b.textContent==='←'?-1:1)<0||index+(b.textContent==='←'?-1:1)>=order.length;});
 };
 return {reachable:true,add(layer,n,dose){if(busy)return;pushes.splice(0,pushes.length,{layer,n,dose});$('#pushes').textContent=`L${layer} N${n}`;open();},state:()=>({order,hidden:[...hidden],histories})};
}
