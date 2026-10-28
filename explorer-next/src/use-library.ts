import { useCallback,useEffect,useRef,useState } from 'react'
import { z } from 'zod'
import { NeuronSchema,sameNeuron,type Neuron } from './data'
import { FavouritesSchema,newId,read } from './storage'
const PushSchema=NeuronSchema.extend({dose:z.number().min(-2).max(2)})
const Metadata=z.object({id:z.string(),model:z.string(),name:z.string(),note:z.string(),tags:z.array(z.string()),source:z.enum(['you','agent']),added:z.string()})
export const LibraryEntrySchema=z.union([Metadata.extend({pushes:z.array(PushSchema).min(1).max(32)}),Metadata.extend({layer:z.number().int().nonnegative(),n:z.number().int().nonnegative(),dose:z.number().min(-2).max(2).nullable()})])
export type SingleEntry=Extract<z.infer<typeof LibraryEntrySchema>,{layer:number}>
export function isSingle(e:LibraryEntry):e is SingleEntry{return 'layer' in e}
export function entryPushes(e:LibraryEntry){return isSingle(e)?[{layer:e.layer,n:e.n,dose:e.dose??.6}]:e.pushes}
export function entryIdentity(e:LibraryEntry){return JSON.stringify(isSingle(e)?[e.model,e.layer,e.n,e.dose===null?null:Math.sign(e.dose)]:[e.model,[...e.pushes].sort((a,b)=>a.layer-b.layer||a.n-b.n)])}
export type LibraryChange={name?:string;note?:string;tags?:string[];dose?:number|null;pushes?:z.infer<typeof PushSchema>[]}
export type LibraryEntry=z.infer<typeof LibraryEntrySchema>
const Entries=z.array(LibraryEntrySchema),key='neuron-next-library',legacyKey='neuron-next-favourites',pendingKey='neuron-next-library-pending',deletedKey='neuron-next-library-deleted'
class LibraryHttpError extends Error{constructor(public status:number){super(`Library request failed (${status})`)}}
async function request(path='',method='GET',value?:unknown){const base=(new URLSearchParams(location.search).get('server')||'.').replace(/\/$/,'');const r=await fetch(`${base}/api/library${path}`,{method,headers:{'Content-Type':'application/json'},signal:AbortSignal.timeout(4000),...(value===undefined?{}:{body:JSON.stringify(value)})});if(!r.ok)throw new LibraryHttpError(r.status);const raw:unknown=await r.json();return raw}
function input({id,added,...entry}:LibraryEntry){void id;void added;return entry}
function localEntry(n:Neuron,model:string):LibraryEntry{return {...n,id:newId(),model,dose:null,name:`L${n.layer} N${n.n}`,note:'',tags:[],source:'you',added:new Date().toISOString()}}
export function useLibrary(model:string){
 const [entries,setEntries]=useState<LibraryEntry[]>(()=>read(key,Entries,[])),[error,setError]=useState('')
 const online=useRef(false),queue=useRef(Promise.resolve()),rows=useRef(entries)
 const store=useCallback((next:LibraryEntry[])=>{if(JSON.stringify(next)===JSON.stringify(rows.current))return;rows.current=next;setEntries(next);try{localStorage.setItem(key,JSON.stringify(next))}catch{}},[])
 const refresh=useCallback(async()=>{
  try{
   let result=z.object({ok:z.literal(true),entries:Entries}).parse(await request())
   online.current=true
   const legacy=read(legacyKey,FavouritesSchema,[]),pending=read(pendingKey,Entries,[]),deleted=read(deletedKey,z.array(z.string()),[])
   if(legacy.length||pending.length||deleted.length){
    for(const id of deleted)if(result.entries.some(e=>e.id===id))await request(`/${id}`,'DELETE')
    for(const e of pending.length?pending:legacy.map(n=>localEntry(n,model)))await request('','POST',input(e))
    localStorage.removeItem(legacyKey);localStorage.removeItem(pendingKey);localStorage.removeItem(deletedKey)
    result=z.object({ok:z.literal(true),entries:Entries}).parse(await request())
   }
   store(result.entries)
  }catch{
   online.current=false
   if(!rows.current.length){
    const legacy=read(legacyKey,FavouritesSchema,[]).map(n=>localEntry(n,model))
    if(legacy.length)store(legacy)
    else try{const r=await fetch('./council.json');if(r.ok){const presets=z.array(NeuronSchema.extend({name:z.string(),dose:z.number().min(-2).max(2),why:z.string().optional()})).parse(await r.json());store(presets.map(p=>({...localEntry(p,model),id:`preset-${p.layer}-${p.n}-${p.dose}`,dose:p.dose,name:p.name,note:p.why??'',source:'agent'})))}}catch{}
   }
  }
 },[model,store])
 useEffect(()=>{if(!model)return;let active=true;const check=()=>{queue.current=queue.current.then(async()=>{if(active)await refresh()})};check();const timer=setInterval(check,3000);return()=>{active=false;clearInterval(timer)}},[model,refresh])
 function mutate(action:()=>Promise<void>){const result=queue.current.then(action);queue.current=result.catch(e=>setError(e instanceof Error?e.message:'Library change failed')).then(()=>refresh());return result}
 function offline(next:LibraryEntry[],deleted?:string){store(next);try{localStorage.setItem(pendingKey,JSON.stringify(next));if(deleted)localStorage.setItem(deletedKey,JSON.stringify([...read(deletedKey,z.array(z.string()),[]),deleted]));localStorage.setItem(legacyKey,JSON.stringify(next.filter(isSingle).filter(e=>e.model===model).map(({layer,n})=>({layer,n}))))}catch{}}
 async function perform(remote:()=>Promise<unknown>,local:()=>void){if(online.current){try{await remote();return}catch(e){if(e instanceof LibraryHttpError&&![404,502,503,504].includes(e.status))throw e;online.current=false}}local()}
 function add(entry:LibraryEntry){return mutate(async()=>{setError('');await perform(()=>request('','POST',input(entry)),()=>offline([...rows.current.filter(e=>entryIdentity(e)!==entryIdentity(entry)),entry]))})}
 function patch(id:string,change:LibraryChange){return mutate(async()=>{setError('');await perform(()=>request(`/${id}`,'PATCH',change),()=>offline(rows.current.map(e=>e.id===id?{...e,...change}:e)))})}
 function remove(entry:LibraryEntry){return mutate(async()=>{setError('');await perform(()=>request(`/${entry.id}`,'DELETE'),()=>offline(rows.current.filter(e=>e.id!==entry.id),entry.id))})}
 function star(n:Neuron){const found=rows.current.filter(e=>isSingle(e)&&e.model===model&&e.dose===null&&sameNeuron(e,n));if(found.length)for(const e of found)void remove(e).catch(()=>{});else void add(localEntry(n,model)).catch(()=>{})}
 return {entries:entries.filter(e=>e.model===model),add,patch,remove,star,error,online:online.current}
}
