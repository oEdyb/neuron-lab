import { z } from 'zod'
import type { Chat, Column, ColumnId } from './storage'
import { ColumnIdSchema,columnPushes } from './storage'
const EventSchema=z.object({ok:z.boolean(),col:ColumnIdSchema.optional(),delta:z.string().optional(),done:z.boolean().optional(),text:z.string().optional(),stopped:z.string().optional(),error:z.string().optional(),fix:z.string().optional()})
export type StreamEvent=z.infer<typeof EventSchema>
let modelBase=(new URLSearchParams(location.search).get('server')||'.').replace(/\/$/,'')
export async function probe(model:string){
 const explicit=new URLSearchParams(location.search).get('server')
 for(const candidate of explicit?[explicit]:['.']){
  try{const r=await fetch(`${candidate.replace(/\/$/,'')}/api/state`,{signal:AbortSignal.timeout(2000)});const state=z.object({ok:z.boolean(),model:z.string().nullable(),configured_model:z.string().optional()}).parse(await r.json());if(r.ok&&state.ok&&(state.model??state.configured_model)===model){modelBase=candidate.replace(/\/$/,'');return true}}catch{}
 }
 return false
}
async function stream(response:Response,onEvent:(event:StreamEvent)=>void,signal:AbortSignal){
 if(!response.ok){const body=await response.text();let message=body;try{const e=EventSchema.parse(JSON.parse(body));message=[e.error,e.fix].filter(Boolean).join(' ')}catch{}throw Error(message||`Model error ${response.status}`)}
 if(!response.body)throw Error('The model returned no stream')
 const reader=response.body.pipeThrough(new TextDecoderStream()).getReader();let pending=''
 function line(text:string){if(!text.trim())return;const e=EventSchema.parse(JSON.parse(text));if(!e.ok)throw Error([e.error,e.fix].filter(Boolean).join(' ')||'Model request failed');onEvent(e)}
 try{while(!signal.aborted){const {value,done}=await reader.read();pending+=value||'';const lines=pending.split('\n');pending=lines.pop()||'';lines.forEach(line);if(done){line(pending);return}}}finally{await reader.cancel().catch(()=>{})}
}
export async function compare(chat:Chat,message:string,signal:AbortSignal,onEvent:(id:ColumnId,event:StreamEvent)=>void){
 const visible=chat.columns.filter(c=>!c.hidden)
 const request=(path:string,body:unknown)=>fetch(`${modelBase}/api/${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal})
 const help=await fetch(`${modelBase}/api/help`,{signal}).then(r=>r.text()).catch(()=>''),supports=help.includes('column pushes')
 const response=supports?await request('compare',{neuron:chat.neuron,pushes:chat.pushes,columns:visible.map(c=>({id:c.id,pushes:columnPushes(chat,c.id),history:c.messages.map(({role,content})=>({role,content}))})),message,sample:false,tokens:300}):null
 if(response&&response.status!==404){const completed=new Set<ColumnId>();await stream(response,e=>{if(e.col){onEvent(e.col,e);if(e.done)completed.add(e.col)}},signal);if(visible.some(c=>!completed.has(c.id))&&!signal.aborted)throw Error('The response ended early. Partial replies have been saved.');return}
 for(const c of visible){if(signal.aborted)return;let complete=false;const r=await request('chat',{pushes:columnPushes(chat,c.id),history:{steered:c.messages.map(({role,content})=>({role,content}))},message,compare:false,tokens:300,sample:false});await stream(r,e=>{if(e.done)complete=true;onEvent(c.id,e.done?{...e,stopped:e.stopped??'complete'}:e)},signal);if(!complete&&!signal.aborted)throw Error('The response ended early. Partial replies have been saved.')}
}
