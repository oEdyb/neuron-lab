import { useState } from 'react'
import { ArrowDownUp, Star, Plus, Layers, Network } from 'lucide-react'
import { Button } from './components/ui/button'
import { Badge } from './components/ui/badge'
import { Table,TableHeader,TableBody,TableRow,TableHead,TableCell } from './components/ui/table'
import { sameNeuron,type Neuron } from './data'
import type { FindResult,FindRow } from './find-data'
type Sort='layer'|'n'|'sign'|'guess'|'real'|'null'|'beats_null'|'words'
export type FindActions={favourites:Neuron[];onStar:(n:Neuron)=>void;onAdd:(n:Neuron)=>void;onOpen:(n:Neuron,view:'plane'|'web')=>void;canAdd:boolean}
export function FindResults({result,...actions}:FindActions&{result:FindResult}){
 const [sort,setSort]=useState<Sort>('real'),[ascending,setAscending]=useState(false)
 const rows=[...result.rows].sort((a,b)=>{const x=a[sort],y=b[sort],order=typeof x==='number'&&typeof y==='number'?x-y:typeof x==='boolean'&&typeof y==='boolean'?Number(x)-Number(y):String(x).localeCompare(String(y));return order*(ascending?1:-1)||a.layer-b.layer||a.n-b.n})
 const max=Math.max(...rows.flatMap(r=>[Math.abs(r.guess),Math.abs(r.real)]),.001)
 function order(key:Sort){if(key===sort)setAscending(v=>!v);else{setSort(key);setAscending(key==='layer'||key==='n')}}
 return <section className="find-results" aria-label="Find results"><div className="find-result-heading"><strong>Last result · {rows.length} neurons</strong><span>{result.model.split('/').at(-1)} · L{result.layers} · dose {result.dose} · {result.where}</span></div><Table className="find-table"><TableHeader><TableRow>{([['layer','Layer'],['n','Neuron'],['sign','Sign'],['guess','Guess'],['real','Real effect'],['null','Random'],['beats_null','Beats random'],['words','Top words']] satisfies [Sort,string][]).map(([key,label])=><TableHead key={key} aria-sort={sort===key?(ascending?'ascending':'descending'):'none'}><Button variant="ghost" size="sm" onClick={()=>order(key)}>{label}<ArrowDownUp/></Button></TableHead>)}<TableHead>Actions</TableHead></TableRow></TableHeader><TableBody>{rows.map(row=><ResultRow key={`${row.layer}:${row.n}`} row={row} max={max} {...actions}/>)}</TableBody></Table></section>
}
function ResultRow({row,max,favourites,onStar,onAdd,onOpen,canAdd}:FindActions&{row:FindRow;max:number}){
 const name=`L${row.layer} N${row.n}`,starred=favourites.some(n=>sameNeuron(n,row))
 return <TableRow className={row.beats_null?'':'find-faded'}><TableCell>{row.layer}</TableCell><TableCell>{row.n}</TableCell><TableCell className={row.sign>0?'meaning-plus':'meaning-minus'}>{row.sign>0?'+':'−'}</TableCell><TableCell><Effect value={row.guess} max={max} kind="guess"/></TableCell><TableCell><Effect value={row.real} max={max} kind="real"/></TableCell><TableCell>{row.null.toFixed(3)}</TableCell><TableCell><Badge variant={row.beats_null?'secondary':'outline'}>{row.beats_null?'Yes':'No'}</Badge></TableCell><TableCell className={`find-words ${row.sign>0?'meaning-plus':'meaning-minus'}`}>{row.words.join(' · ')}</TableCell><TableCell><div className="find-actions"><Button variant="ghost" size="icon-xs" title="Star neuron" aria-label={`Star ${name}`} aria-pressed={starred} onClick={()=>onStar(row)}><Star fill={starred?'currentColor':'none'}/></Button><Button variant="ghost" size="icon-xs" title="Add to chat" aria-label={`Add ${name} to chat`} disabled={!canAdd} onClick={()=>onAdd(row)}><Plus/></Button><Button variant="ghost" size="icon-xs" title="Open in Plane" aria-label={`Open ${name} in Plane`} onClick={()=>onOpen(row,'plane')}><Layers/></Button><Button variant="ghost" size="icon-xs" title="Open in Web" aria-label={`Open ${name} in Web`} onClick={()=>onOpen(row,'web')}><Network/></Button></div></TableCell></TableRow>
}
function Effect({value,max,kind}:{value:number;max:number;kind:string}){return <div className={`find-effect find-${kind}`}><span style={{width:`${Math.abs(value)/max*100}%`}}/><b>{value.toFixed(3)}</b></div>}
