import { useEffect, useState } from 'react'
import { Minus,Plus } from 'lucide-react'
import { Button } from './components/ui/button'
const show=(v:number,signed:boolean)=>`${signed&&v>0?'+':''}${Number.isInteger(v*10)?v.toFixed(1):v.toFixed(2)}`
export function DoseStepper({value,onChange,signed=false,disabled=false,label='Push dose'}:{value:number;onChange:(v:number)=>void;signed?:boolean;disabled?:boolean;label?:string}){
 const min=signed?-2:0
 const [draft,setDraft]=useState(show(value,signed))
 useEffect(()=>setDraft(show(value,signed)),[value,signed])
 const set=(v:number)=>onChange(Math.max(min,Math.min(2,Math.round(v*100)/100)))
 const commit=()=>{const v=parseFloat(draft.replace(',','.').replace('−','-'));if(Number.isFinite(v))set(v);else setDraft(show(value,signed))}
 return <div className={`dose-stepper ${value<0?'meaning-minus':'meaning-plus'}`}><Button type="button" variant="ghost" size="icon-sm" aria-label={`Decrease ${label}`} disabled={disabled||value<=min} onClick={()=>set(value-.1)}><Minus/></Button><input role="spinbutton" aria-label={label} aria-valuemin={min} aria-valuemax={2} aria-valuenow={value} inputMode="decimal" disabled={disabled} value={draft} onChange={e=>setDraft(e.target.value)} onFocus={e=>e.target.select()} onBlur={commit} onKeyDown={e=>{if(e.key==='Enter'){e.currentTarget.blur()}else if(e.key==='Escape'){setDraft(show(value,signed));e.currentTarget.blur()}else if(e.key==='ArrowUp'||e.key==='ArrowDown'){e.preventDefault();set(value+(e.key==='ArrowUp'?.1:-.1))}}}/><Button type="button" variant="ghost" size="icon-sm" aria-label={`Increase ${label}`} disabled={disabled||value>=2} onClick={()=>set(value+.1)}><Plus/></Button></div>
}
