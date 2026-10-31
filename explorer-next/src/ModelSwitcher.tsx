import { useState } from 'react'
import { Check, ChevronsUpDown, Loader2 } from 'lucide-react'
import { Button } from './components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from './components/ui/dropdown-menu'
export const mappedModels=[{id:'Qwen/Qwen3.5-2B',name:'Qwen3.5 2B',logo:'qwen'},{id:'google/gemma-4-E2B-it',name:'Gemma 4 E2B',logo:'gemma'}]
export async function switchModel(model:string,keepSteer=false){
 const base=(new URLSearchParams(location.search).get('server')||'.').replace(/\/$/,'')
 const response=await fetch(`${base}/api/load`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model})})
 const data=await response.json()
 if(!response.ok||!data.ok)throw Error(data.error||'Model load failed')
 const url=new URL(location.href);url.searchParams.set('model',model);url.hash=''
 if(!keepSteer){url.searchParams.delete('steer');url.searchParams.delete('title')}
 location.assign(url)
}
export function ModelSwitcher({model,busy,onBusy}:{model:string;busy:boolean;onBusy:(value:boolean)=>void}){
 const [loading,setLoading]=useState(false),[error,setError]=useState('')
 const current=mappedModels.find(m=>m.id===model)??mappedModels[0]
 return <div><DropdownMenu><DropdownMenuTrigger asChild><Button variant="ghost" className="w-full justify-start h-12" aria-label="Switch model" disabled={busy||loading}><img width="24" height="24" src={`./logos/${current.logo}.svg`} alt=""/><span>{loading?'Loading model…':current.name}</span>{loading?<Loader2 className="ml-auto animate-spin"/>:<ChevronsUpDown className="ml-auto"/>}</Button></DropdownMenuTrigger><DropdownMenuContent side="top" align="start" className="w-56">{mappedModels.map(m=><DropdownMenuItem key={m.id} onSelect={()=>{if(m.id===model)return;setLoading(true);onBusy(true);setError('');void switchModel(m.id).catch(e=>{setError(String(e));setLoading(false);onBusy(false)})}}><img width="24" height="24" src={`./logos/${m.logo}.svg`} alt=""/>{m.name}{m.id===model&&<Check className="ml-auto"/>}</DropdownMenuItem>)}</DropdownMenuContent></DropdownMenu>{error&&<p role="alert">{error}</p>}</div>
}
