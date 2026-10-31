import { z } from 'zod'
import { detail, neuronKey, type Neuron, type Row } from './data'
const Metadata=z.object({version:z.literal(1),model:z.string(),layers:z.number().int().positive(),neurons:z.number().int().positive(),counts:z.array(z.number().int().positive()).optional(),k:z.number().int().positive(),record_bytes:z.literal(6)})
const Link=z.object({layer:z.number().int().nonnegative(),n:z.number().int().nonnegative(),z:z.number().finite(),sign:z.union([z.literal(1),z.literal(-1)])})
export type Connection=z.infer<typeof Link>&{side:'upstream'|'downstream'}
export type Neighborhood={links:Connection[];words:Map<string,Row>}
let metadata:Promise<z.infer<typeof Metadata>>|undefined
const files=new Map<number,Promise<DataView>>()
async function meta(){
 if(!metadata){metadata=fetch('./data/connections/manifest.json').then(async r=>{if(!r.ok)throw Error('Connection data unavailable. Build it with connections.py.');return Metadata.parse(await r.json())});metadata.catch(()=>{metadata=undefined})}
 return metadata
}
function half(bits:number){const sign=bits&32768?-1:1,exponent=bits>>10&31,fraction=bits&1023;return sign*(exponent===0?fraction*2**-24:exponent===31?fraction?NaN:Infinity:(1+fraction/1024)*2**(exponent-15))}
async function file(layer:number,bytes:number){
 let pending=files.get(layer)
 if(!pending){pending=fetch(`./data/connections/L${layer}.bin`).then(async r=>{if(!r.ok)throw Error(`Connections for layer ${layer} unavailable`);const buffer=await r.arrayBuffer();if(buffer.byteLength!==bytes)throw Error('Connection file has the wrong size');return new DataView(buffer)});files.set(layer,pending);pending.catch(()=>files.delete(layer))}
 return pending
}
export async function neighborhood(neuron:Neuron,model:string):Promise<Neighborhood>{
 const m=await meta()
 if(m.model!==model||neuron.layer>=m.layers||neuron.n>=(m.counts?.[neuron.layer]??m.neurons))throw Error('Connections do not match this neuron map')
 const count=m.counts?.[neuron.layer]??m.neurons
 const data=await file(neuron.layer,2*count*m.k*m.record_bytes),links:Connection[]=[]
 for(const side of ['upstream','downstream'] satisfies Connection['side'][]){
  const offset=((side==='upstream'?0:1)*count+neuron.n)*m.k*m.record_bytes
  for(let i=0;i<m.k;i++){
   const at=offset+i*m.record_bytes,layer=data.getInt8(at)
   if(layer===-1)continue
   const link=Link.parse({layer,n:data.getInt16(at+1,true),z:half(data.getUint16(at+3,true)),sign:data.getInt8(at+5)})
   if(layer>=m.layers||link.n>=(m.counts?.[layer]??m.neurons)||(side==='upstream'?layer>=neuron.layer:layer<=neuron.layer))throw Error('Connection points outside its layer direction')
   links.push({...link,side})
  }
 }
 const words=new Map<string,Row>()
 const layers=await Promise.all([...new Set([neuron.layer,...links.map(l=>l.layer)])].map(async l=>({layer:l,data:await detail(l)})))
 for(const n of [neuron,...links]){const row=layers.find(l=>l.layer===n.layer)?.data.rows[n.n];if(row)words.set(neuronKey(n),row)}
 return {links,words}
}
