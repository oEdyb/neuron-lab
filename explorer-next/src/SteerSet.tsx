import { useEffect,useState } from 'react'
import { Volume2,VolumeX,X } from 'lucide-react'
import { Button } from './components/ui/button'
import { DoseStepper } from './DoseStepper'
import { detail,type Neuron,type Row } from './data'
import type { Chat } from './storage'
function Meaning({neuron}:{neuron:Chat['steerSet'][number]}){
 const [row,setRow]=useState<Row|null>(null)
 useEffect(()=>{let live=true;detail(neuron.layer).then(data=>{if(live)setRow(data.rows[neuron.n]??null)}).catch(()=>{});return()=>{live=false}},[neuron.layer,neuron.n])
 return <span className={`steer-words meaning-${neuron.dose<0?'minus':'plus'}`}>{(neuron.dose<0?row?.down:row?.up)?.slice(0,3).join(' · ')}</span>
}
export function SteerSet({chat,busy,selected,onAdd,onChange}:{chat:Chat;busy:boolean;selected:Neuron|null;onAdd:(n:Neuron)=>void;onChange:(set:Chat['steerSet'])=>void}){
 const change=(i:number,patch:Partial<Chat['steerSet'][number]>)=>onChange(chat.steerSet.map((n,j)=>j===i?{...n,...patch}:n))
 return <div className="steer-set">{chat.steerSet.map((n,i)=>{const name=`L${n.layer} N${n.n}`;return <div className={`steer-row ${n.muted?'steer-muted':''}`} key={name}><span className="steer-name">{name}</span><Meaning neuron={n}/><DoseStepper signed label={`${name} dose`} disabled={busy} value={n.dose} onChange={dose=>change(i,{dose})}/><Button variant="ghost" size="icon-sm" aria-label={`${n.muted?'Unmute':'Mute'} ${name}`} aria-pressed={n.muted} disabled={busy} onClick={()=>change(i,{muted:!n.muted})}>{n.muted?<VolumeX/>:<Volume2/>}</Button><Button variant="ghost" size="icon-sm" aria-label={`Remove ${name}`} disabled={busy} onClick={()=>onChange(chat.steerSet.filter((_,j)=>j!==i))}><X/></Button></div>})}<Button variant="outline" size="sm" disabled={busy||!selected||chat.steerSet.some(n=>n.layer===selected.layer&&n.n===selected.n)} onClick={()=>selected&&onAdd(selected)}>Add neuron</Button></div>
}
