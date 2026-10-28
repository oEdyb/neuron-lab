import type { Chat,Setup } from './storage'
export function chatTurns(chat:Chat){
 const turns:{key:string;text:string;setup?:Setup;sentAt?:number}[]=[]
 for(const column of chat.columns){
  const users=column.messages.flatMap((m,i)=>m.role==='user'?[{key:m.id??`legacy-${i}`,text:m.content,setup:m.setup,sentAt:m.sentAt}]:[])
  users.forEach((turn,i)=>{if(turns.some(t=>t.key===turn.key))return;const next=users.slice(i+1).find(t=>turns.some(v=>v.key===t.key)),position=next?turns.findIndex(t=>t.key===next.key):turns.length;turns.splice(position,0,turn)})
 }
 return turns.sort((a,b)=>(a.sentAt??0)-(b.sentAt??0))
}
export function beforeTurn(chat:Chat,key:string):Chat{return {...chat,columns:chat.columns.map(c=>{const i=c.messages.findIndex((m,i)=>m.role==='user'&&(m.id??`legacy-${i}`)===key);return {...c,messages:i<0?c.messages:c.messages.slice(0,i)}})}}
export function setupChanges(previous:Setup|undefined,current:Setup|undefined){
 if(!previous||!current)return []
 const changes:{text:string;meaning?:'plus'|'minus'}[]=[]
 if(previous.council&&current.council){
 for(const col of current.council){const old=previous.council.find(c=>c.id===col.id);if(!old){changes.push({text:`+ ${col.name??col.id}`});continue}if(old.hidden!==col.hidden)changes.push({text:`${col.hidden?'muted':'unmuted'} ${col.name??col.id}`});for(const p of col.pushes){const before=old.pushes.find(n=>n.layer===p.layer&&n.n===p.n);if(before&&before.dose!==p.dose)changes.push({text:`${col.name??col.id}: L${p.layer} N${p.n} ${before.dose} → ${p.dose>=0?'+':''}${p.dose}`,meaning:p.dose<0?'minus':'plus'})}}
 return changes
 }
 const before=previous.steerSet??previous.pushes,after=current.steerSet??current.pushes
 for(const p of before)if(!after.some(n=>n.layer===p.layer&&n.n===p.n))changes.push({text:`− L${p.layer} N${p.n}`})
 for(const p of after){const old=before.find(n=>n.layer===p.layer&&n.n===p.n),name=`L${p.layer} N${p.n}`,dose=`${p.dose>=0?'+':'−'}${Math.abs(p.dose).toFixed(1)}`;if(!old)changes.push({text:`+ ${name} ${dose}`,meaning:p.dose<0?'minus':'plus'});else{if(old.dose!==p.dose)changes.push({text:`${name} ${old.dose.toFixed(1)} → ${dose}`,meaning:p.dose<0?'minus':'plus'});if(('muted' in old&&old.muted)!==('muted' in p&&p.muted))changes.push({text:`${'muted' in p&&p.muted?'muted':'unmuted'} ${name}`})}}
 return changes
}
