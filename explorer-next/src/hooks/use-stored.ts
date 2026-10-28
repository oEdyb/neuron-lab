import { useCallback,useEffect,useRef,useState } from 'react'
import { z } from 'zod'
import { read,write } from '../storage'
export function useStored<T>(key:string,schema:z.ZodType<T>,fallback:T){
 const [value,setValue]=useState(()=>read(key,schema,fallback)),[error,setError]=useState('')
 const unsaved=useRef(false),current=useRef(value);current.current=value
 useEffect(()=>{const sync=(e:StorageEvent)=>{if(!unsaved.current&&(e.key===key||e.key===null)){const next=read(key,schema,fallback);current.current=next;setValue(next)}};window.addEventListener('storage',sync);return()=>window.removeEventListener('storage',sync)},[key,schema])
 const update=useCallback((fn:(previous:T)=>T)=>{const next=fn(unsaved.current?current.current:read(key,schema,current.current));current.current=next;setValue(next);try{unsaved.current=true;write(key,next);unsaved.current=false;setError('')}catch{setError('Storage is full or unavailable. Export your chat before closing this page.')}},[key,schema])
 return [value,update,error] as const
}
