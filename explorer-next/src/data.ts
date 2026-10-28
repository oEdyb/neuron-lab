import { z } from 'zod'
import { configure } from './legacy/layout.js'
import { config } from './legacy/config.js'
import { detail as legacyDetail } from './legacy/data.js'
import { search as legacySearch } from './legacy/search.js'
export const NeuronSchema=z.object({layer:z.number().int().nonnegative(),n:z.number().int().nonnegative()})
export type Neuron=z.infer<typeof NeuronSchema>
export const RowSchema=z.object({n:z.number().optional(),up:z.array(z.string()),down:z.array(z.string()),persona:z.array(z.array(z.string())).optional(),kl:z.array(z.number()),spread:z.array(z.array(z.number()))})
const ManifestSchema=z.object({model:z.string(),dose:z.number(),layers:z.array(z.object({layer:z.number(),count:z.number()})),summaries:z.array(z.object({layer:z.number(),count:z.number(),above:z.number(),top:z.array(RowSchema)}))})
export type Manifest=z.infer<typeof ManifestSchema>
export type Row=z.infer<typeof RowSchema>
export const LayerSchema=z.object({rows:z.array(RowSchema),p95:z.array(z.number())})
export type Layer=z.infer<typeof LayerSchema>
export type Hit=Neuron&{matched:string[];rank?:number;score?:number}
const HitSchema=NeuronSchema.extend({matched:z.array(z.string()),rank:z.number().optional(),score:z.number().optional()})
export async function loadManifest(){const r=await fetch('./data/manifest.json');if(!r.ok)throw Error('Map data unavailable');const m=ManifestSchema.parse(await r.json());configure(config,m);return m}
const layers=new Map<number,Promise<Layer>>()
export function detail(layer:number):Promise<Layer>{const existing=layers.get(layer);if(existing)return existing;const pending=(async()=>LayerSchema.parse(await legacyDetail(layer)))();layers.set(layer,pending);pending.catch(()=>layers.delete(layer));return pending}
export async function search(query:string,mode:string){return z.array(HitSchema).parse(await legacySearch(query,mode,[]))}
export function neuronKey(n:Neuron){return `${n.layer}:${n.n}`}
export function sameNeuron(a:Neuron,b:Neuron){return a.layer===b.layer&&a.n===b.n}
export function kl(row:Row,p95:number[]){return Math.max(row.kl[0]/p95[0],row.kl[1]/p95[1])}
