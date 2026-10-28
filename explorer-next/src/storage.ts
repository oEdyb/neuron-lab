import { z } from 'zod'
import { NeuronSchema } from './data'
export const ColumnIdSchema=z.string().min(1)
export type ColumnId=z.infer<typeof ColumnIdSchema>
export const ids:ColumnId[]=['minus','normal','plus']
export const SteerSchema=NeuronSchema.extend({dose:z.number().min(-2).max(2),muted:z.boolean().default(false)})
export const CouncilSetupSchema=z.array(z.object({id:z.string(),name:z.string().optional(),hidden:z.boolean(),pushes:z.array(SteerSchema)}))
export const SetupSchema=z.object({council:CouncilSetupSchema.optional(),dose:z.number().min(0).max(2),pushes:z.array(NeuronSchema.extend({dose:z.number().min(-2).max(2)})),steerSet:z.array(SteerSchema).optional()})
export type Setup=z.infer<typeof SetupSchema>
export const TurnSchema=z.object({id:z.string().optional(),role:z.enum(['user','assistant']),content:z.string(),stopped:z.string().optional(),setup:SetupSchema.optional(),sentAt:z.number().optional()})
export type Turn=z.infer<typeof TurnSchema>
export const ColumnSchema=z.object({id:ColumnIdSchema,name:z.string().optional(),pushes:z.array(SteerSchema).optional(),dose:z.number().min(-2).max(2),hidden:z.boolean(),messages:z.array(TurnSchema)})
export type Column=z.infer<typeof ColumnSchema>
export const ChatSchema=z.object({id:z.string(),title:z.string(),council:z.boolean().default(false),neuron:NeuronSchema,dose:z.number().min(0).max(2),pushes:z.array(NeuronSchema.extend({dose:z.number().min(-2).max(2)})).default([]),steerSet:z.array(SteerSchema).optional(),columns:z.array(ColumnSchema).refine(cols=>cols.length>=1&&cols.length<=6&&new Set(cols.map(c=>c.id)).size===cols.length),timestamp:z.number().min(0).max(8640000000000000)}).transform(c=>({...c,columns:c.columns.map(col=>({...col,messages:col.messages.map(m=>m.setup&&!m.setup.steerSet?{...m,setup:{...m.setup,steerSet:[{...c.neuron,dose:m.setup.dose,muted:false},...m.setup.pushes.map(p=>({...p,muted:false}))]}}:m)})),steerSet:c.steerSet??[{...c.neuron,dose:c.dose,muted:false},...c.pushes.map(p=>({...p,muted:false}))]}))
export type Chat=z.infer<typeof ChatSchema>
export const ChatsSchema=z.array(z.unknown()).transform(rows=>rows.flatMap(row=>{const parsed=ChatSchema.safeParse(row);return parsed.success?[parsed.data]:[]}))
export const FavouritesSchema=z.array(z.unknown()).transform(rows=>rows.flatMap(row=>{const parsed=NeuronSchema.safeParse(row);return parsed.success?[parsed.data]:[]}))
export function read<T>(key:string,schema:z.ZodType<T>,fallback:T):T{try{const raw=localStorage.getItem(key);return raw?schema.parse(JSON.parse(raw)):fallback}catch{return fallback}}
export function write(key:string,value:unknown){localStorage.setItem(key,JSON.stringify(value))}
export function newId(){return Date.now().toString(36)+Math.random().toString(36).slice(2)}
export function newChat(neuron:z.infer<typeof NeuronSchema>,dose:number):Chat{return{council:false,id:newId(),title:`L${neuron.layer} N${neuron.n}`,neuron,dose,pushes:[],steerSet:[{...neuron,dose,muted:false}],columns:ids.map(id=>({id,dose:id==='normal'?0:id==='minus'?-dose:dose,hidden:id==='normal',messages:[]})),timestamp:Date.now()}}
export function chatMarkdown(chat:Chat){return `# ${chat.title}\n\nSteer set: ${setupLabel({dose:chat.dose,pushes:[],steerSet:chat.steerSet})}\nSaved: ${new Date(chat.timestamp).toISOString()}\n\n`+chat.columns.map(c=>`## ${columnLabel(c,chat)}${c.hidden?' (hidden)':''}\n\n`+c.messages.map(m=>`**${m.role==='user'?'You':'Model'}**\n\n${m.role==='user'&&m.setup?`*Setup: ${setupLabel(m.setup)}*\n\n`:''}${m.content}${m.stopped?`\n\n*stopped: ${m.stopped}*`:''}`).join('\n\n')).join('\n\n')}
export function columnLabel(c:Column,chat?:Chat){if(c.name)return c.name;if(c.id==='normal')return 'Normal';if(chat&&chat.steerSet.length!==1)return c.id==='minus'?'Flipped':'Steered';const dose=chat?chat.steerSet[0].dose*(c.id==='minus'?-1:1):c.dose;return `${dose<0?'−':'+'}${Math.abs(dose).toFixed(1)}`}
export function columnPushes(chat:Chat,id:ColumnId){const column=chat.columns.find(c=>c.id===id);if(chat.council)return (column?.pushes??[]).filter(p=>!p.muted).map(({layer,n,dose})=>({layer,n,dose}));return id==='normal'?[]:chat.steerSet.filter(p=>!p.muted).map(({layer,n,dose})=>({layer,n,dose:dose*(id==='minus'?-1:1)}))}

export function setupLabel(setup:Setup){if(setup.council)return setup.council.map(c=>`${c.name??c.id}${c.hidden?' (muted)':''}: ${c.pushes.map(p=>`L${p.layer} N${p.n} ${p.dose>=0?'+':''}${p.dose}`).join(' · ')}`).join('; ');return setup.steerSet?setup.steerSet.map(p=>`L${p.layer} N${p.n} ${p.dose>=0?'+':'−'}${Math.abs(p.dose).toFixed(1)}${p.muted?' (muted)':''}`).join(' · '):`dose ${setup.dose.toFixed(1)}${setup.pushes.map(p=>` · L${p.layer} N${p.n} ${p.dose>=0?'+':'−'}${Math.abs(p.dose).toFixed(1)}`).join('')}`}
