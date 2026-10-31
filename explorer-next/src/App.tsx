import { z } from 'zod'
import { switchModel } from './ModelSwitcher'
import { useCallback,useEffect,useRef,useState } from 'react'
import { SidebarProvider,SidebarTrigger } from './components/ui/sidebar'
import { TooltipProvider } from './components/ui/tooltip'
import { Tabs,TabsList,TabsTrigger } from './components/ui/tabs'
import { ToggleGroup,ToggleGroupItem } from './components/ui/toggle-group'
import { ResizableHandle,ResizablePanel,ResizablePanelGroup } from './components/ui/resizable'
import { Button } from './components/ui/button'
import { Library } from './Library'
import { useLibrary,entryPushes,isSingle,LibraryEntrySchema,type LibraryEntry } from './use-library'
import { MapView } from './MapView'
import { FindView } from './FindView'
import { WebView } from './WebView'
import { Inspector } from './Inspector'
import { columnLabel,newId,type Column } from './storage'
import { ChatPanel } from './ChatPanel'
import { SearchSidebar } from './SearchSidebar'
import { detail,kl,loadManifest,sameNeuron,type Manifest,type Neuron } from './data'
import { PROMPTS } from './prompts'
import { Dices, Moon, Sun } from 'lucide-react'
import { ChatsSchema,newChat,setChatModel,type Chat } from './storage'
import { probe,running } from './model'
import { useStored } from './hooks/use-stored'
import type { createScene } from './legacy/scene.js'
function storedLight(){try{return localStorage.getItem('neuron-next-theme')==='light'}catch{return false}}
function applyTheme(light:boolean){document.documentElement.classList.toggle('light',light);try{localStorage.setItem('neuron-next-theme',light?'light':'dark')}catch{}}
applyTheme(storedLight())
const linkParams=new URLSearchParams(location.search),modelOverride=linkParams.get('model')
const linkSteer=(linkParams.get('steer')||'').split(',').flatMap(part=>{const [layer,n,dose]=part.split(':').map(Number);return Number.isInteger(layer)&&Number.isInteger(n)&&Number.isFinite(dose)?[{layer,n,dose:Math.max(-2,Math.min(2,dose))}]:[]})
function columnChat(chat:Chat):Chat{
 if(chat.council)return chat
 return {...chat,council:true,columns:chat.columns.map(col=>({...col,name:columnLabel(col,chat),pushes:col.id==='normal'?[]:chat.steerSet.map(p=>({...p,dose:p.dose*(col.id==='minus'?-1:1)}))}))}
}
function Explorer(){
 const [switching,setSwitching]=useState(false)
 const [mapMissing,setMapMissing]=useState(false)
 const [manifest,setManifest]=useState<Manifest|null>(null),[error,setError]=useState(''),[layer,setLayer]=useState<number|null>(null),[selected,setSelected]=useState<Neuron|null>(null),[rightTab,setRightTab]=useState('inspect'),[expanded,setExpanded]=useState(false),[reachable,setReachable]=useState(false),[busy,setBusy]=useState(false),[dose,setDose]=useState(.6),[ready,setReady]=useState(false),[runningModel,setRunningModel]=useState<string|null>(null)
 const library=useLibrary(manifest?.model??'')
 const favourites=library.entries.filter(isSingle).filter(e=>e.dose===null)
 const [allChats,setChats,chatError]=useStored('neuron-next-chats',ChatsSchema,[]),[chatId,setChatId]=useState<string|null>(null)
 const chats=allChats.filter(c=>c.model===(modelOverride??manifest?.model??'Qwen/Qwen3.5-2B'))
 const [light,setLight]=useState(storedLight)
 const [libraryTab,setLibraryTab]=useState(false),[web,setWeb]=useState(false),[find,setFind]=useState(false),[localFind,setLocalFind]=useState(false),[rolled,setRolled]=useState<string|null>(null)
 const [sidebarOpen,setSidebarOpen]=useState(true)
 useEffect(()=>setSidebarOpen(!libraryTab),[libraryTab])
 const modelBusy=busy||localFind||switching,currentChat=chats.find(c=>c.id===chatId),chat=currentChat?columnChat(currentChat):null
 const scene=useRef<ReturnType<typeof createScene>|null>(null)
 const update=useCallback((id:string,fn:(c:Chat)=>Chat)=>setChats(all=>all.map(c=>c.id===id?fn(columnChat(c)):c)),[])
 useEffect(()=>{loadManifest().then(m=>{setChatModel(modelOverride??m.model);setManifest(m);setDose(m.dose);const h=location.hash.match(/^#L(\d+)-N(\d+)$/i);if(h&&+h[1]<m.layers.length&&+h[2]<m.layers[+h[1]].count){setLayer(+h[1]);setSelected({layer:+h[1],n:+h[2]})}}).catch(async()=>{const response=await fetch('./api/maps');const status=await response.json();setError(`Map isn't downloaded. Run: ${status.fix}`);setMapMissing(true);setManifest(null)})},[])
 useEffect(()=>{if(!manifest||!linkSteer.length)return;const title=new URLSearchParams(location.search).get('title')||linkSteer.map(p=>`L${p.layer} N${p.n} ${p.dose>=0?'+':''}${p.dose}`).join(' · ');const found=chats.find(c=>c.title===title);if(found){setChatId(found.id);setRightTab('chat');return}const c=newChat(linkSteer[0],Math.min(2,Math.abs(linkSteer[0].dose)));c.title=title;c.council=true;c.columns=[{id:'normal',name:'Normal',dose:0,hidden:false,messages:[]},{id:newId(),name:title,dose:0,hidden:false,messages:[],pushes:linkSteer.map(p=>({...p,muted:false}))}];setChats(all=>[c,...all]);setChatId(c.id);setRightTab('chat')},[manifest])
 useEffect(()=>{if(!manifest)return;let live=true;const check=()=>probe(modelOverride??manifest.model).then(ok=>{if(live){setReachable(ok);setRunningModel(ok?running.model:null)}});void check();const timer=setInterval(check,5000);return()=>{live=false;clearInterval(timer)}},[manifest])
 const select=useCallback((n:Neuron)=>{setSelected({layer:n.layer,n:n.n});setLayer(n.layer);setRightTab('inspect');setExpanded(false);history.replaceState(null,'',`#L${n.layer}-N${n.n}`)},[])
 const face=useCallback((l:number|null)=>{setLayer(l);setSelected(null);setExpanded(false);setRightTab('inspect');history.replaceState(null,'',location.pathname+location.search)},[])
 useEffect(()=>{if(!manifest)return;const raw=sessionStorage.getItem('neuron-pending-library');if(raw){sessionStorage.removeItem('neuron-pending-library');try{const pending=z.object({entries:z.array(LibraryEntrySchema),combine:z.boolean()}).parse(JSON.parse(raw));appendColumns(pending.entries,pending.combine)}catch{}}},[manifest])
 const onReady=useCallback((s:ReturnType<typeof createScene>)=>{scene.current=s;setReady(true);Object.assign(window,{neuronNext:{select,face,stats:()=>s.stats(),project:(l:number,n:number)=>s.project(l,n)}})},[select,face])
 function startChat(){if(modelBusy)return;if(!selected){setRightTab('chat');return}const c=newChat(selected,dose);c.council=true;c.columns=[{id:'normal',name:'Normal',dose:0,hidden:true,messages:[]},{id:newId(),name:c.title,dose:0,hidden:false,messages:[],pushes:[{...selected,dose,muted:false}]}];setChats(all=>[c,...all]);setChatId(c.id);setRightTab('chat');setExpanded(false)}
 function appendColumns(entries:LibraryEntry[],combine=false){
  if(modelBusy||!entries.length)return
  if(new Set(entries.map(e=>e.model)).size>1){setError("Choose Library entries from one model at a time.");return}
  if(entries[0].model!==manifest?.model){sessionStorage.setItem("neuron-pending-library",JSON.stringify({entries,combine}));setSwitching(true);void switchModel(entries[0].model).catch(e=>{setError(String(e));setSwitching(false)});return}
  const chosen=combine?[entries]:entries.map(e=>[e])
  const columns:Column[]=chosen.map(group=>{const pushes=new Map<string,Chat['steerSet'][number]>();const scale=1/group.length;for(const entry of group)for(const p of entryPushes(entry)){const key=`${p.layer}:${p.n}`,old=pushes.get(key);pushes.set(key,{...p,dose:Math.max(-2,Math.min(2,Math.round((p.dose*scale+(old?.dose??0))*20)/20)),muted:false})}return {id:newId(),name:group.map(e=>e.name).join(' + '),dose:0,hidden:false,messages:[],pushes:[...pushes.values()]}})
  let current=chat
  if(!current){const p=columns[0].pushes?.[0]??{layer:0,n:0};current=newChat(p,dose);current.council=true;current.title='Chat';current.columns=[{id:'normal',name:'Normal',dose:0,hidden:true,messages:[]}]}
  const count=current.columns.length+columns.length
  const existing=current.columns.filter(c=>!(count>6&&c.id==='normal'&&c.hidden&&!c.messages.length))
  if(existing.length+columns.length>6){setError('A chat holds up to 6 columns. Remove a column first.');return}
  const next=[...existing,...columns]
  const updated={...current,columns:next}
  setChats(all=>all.some(c=>c.id===updated.id)?all.map(c=>c.id===updated.id?updated:c):[updated,...all]);setChatId(updated.id);setRightTab('chat');setExpanded(false);setError('')
 }
 function addNeuron(n:Neuron){appendColumns([{...n,id:newId(),model:manifest?.model??'',dose,name:`L${n.layer} N${n.n}`,note:'',tags:[],source:'you',added:new Date().toISOString()}])}
 function openChat(id:string){const c=chats.find(v=>v.id===id);if(!c||modelBusy)return;const pushes=c.council?c.columns.flatMap(col=>col.pushes??[]):c.steerSet;if(!manifest||!modelOverride&&pushes.some(n=>!manifest.layers[n.layer]||n.n>=manifest.layers[n.layer].count)){setError('This saved chat contains a neuron outside this map. Choose another chat.');return}setError('');setChatId(id);setSelected(c.neuron);setLayer(c.neuron.layer);setDose(c.dose);setRightTab('chat');setExpanded(false)}
 async function roll(){if(!manifest||!reachable||modelBusy)return;const first=Math.min(3,manifest.layers.length-1);for(let tries=0;tries<6;tries++){const layer=first+Math.floor(Math.random()*(manifest.layers.length-first)),d=await detail(layer),pool=d.rows.flatMap((r,i)=>kl(r,d.p95)>1.5?[r.n??i]:[]);if(!pool.length)continue;const n={layer,n:pool[Math.floor(Math.random()*pool.length)]},c=columnChat(newChat(n,chat?.dose??dose));select(n);setChats(all=>[c,...all]);setChatId(c.id);setRolled(c.id);setRightTab('chat');setLibraryTab(false);setExpanded(false);return}}
 function saveColumn(column:Column){const pushes=column.pushes?.filter(p=>!p.muted).map(({layer,n,dose})=>({layer,n,dose}));if(!pushes?.length||!manifest)return;const base=column.name??'Combo',taken=library.entries.some(e=>e.name===base),fmt=(d:number)=>`${d>0?'+':''}${d}`,name=taken?`${base} ${pushes.map(p=>fmt(p.dose)).join(' ')}`:base,common={id:newId(),model:manifest.model,name,note:'',tags:[],source:'you' as const,added:new Date().toISOString()};void library.add(pushes.length===1?{...common,...pushes[0]}:{...common,pushes}).catch(()=>{})}
 if(!manifest)return <div className="boot" role="status">{error||'Loading map…'}</div>
 return <TooltipProvider delayDuration={300}><SidebarProvider className="app-shell" open={sidebarOpen} onOpenChange={setSidebarOpen}><SearchSidebar onModelBusy={setSwitching} saved={library.entries} onSaved={e=>{select(entryPushes(e)[0]);setLibraryTab(false);setFind(false);setWeb(false)}} onUnsave={e=>void library.remove(e).catch(()=>{})} selected={selected} onAdd={!modelBusy?addNeuron:undefined} onAddSaved={e=>appendColumns([e])} model={manifest.model} runningModel={runningModel} favourites={favourites} onStar={library.star} onSelect={n=>{select(n);setLibraryTab(false);setFind(false);setWeb(false)}} chats={chats} onChat={openChat} onRename={(id,title)=>update(id,c=>({...c,title}))} onDelete={id=>{setChats(all=>all.filter(c=>c.id!==id));if(chatId===id)setChatId(null)}} onNew={startChat} busy={modelBusy}/><main className="workspace"><header className="workspace-header"><SidebarTrigger className="mobile-sidebar"/><ToggleGroup type="single" size="sm" value={expanded&&rightTab==='chat'&&!libraryTab?'chat':libraryTab?'library':find?'find':web?'web':layer===null?'tower':'plane'} onValueChange={v=>{if(!v||modelBusy)return;if(v==='chat'){setLibraryTab(false);setRightTab('chat');setExpanded(true);setSidebarOpen(false);return}if(expanded){setExpanded(false);setSidebarOpen(v!=='library')}setLibraryTab(v==='library');setFind(v==='find');setWeb(v==='web');const back=find||web||libraryTab;if(v==='web'||v==='find'||v==='library')return;if(back&&(v==='tower')===(layer===null))return;face(v==='tower'?null:layer??Math.floor(manifest.layers.length/2))}}><ToggleGroupItem value="tower">Tower</ToggleGroupItem><ToggleGroupItem value="plane">Plane</ToggleGroupItem><ToggleGroupItem value="web">Web</ToggleGroupItem><ToggleGroupItem value="find">Find</ToggleGroupItem><ToggleGroupItem value="library">Library</ToggleGroupItem><ToggleGroupItem value="chat">Chat</ToggleGroupItem></ToggleGroup><span className="ml-auto"/><Button variant="ghost" size="sm" aria-label={light?'Dark mode':'Light mode'} onClick={()=>{applyTheme(!light);setLight(!light)}}>{light?<Moon/>:<Sun/>}</Button><Button variant="ghost" size="sm" aria-label="Random neuron" disabled={!reachable||modelBusy} onClick={()=>void roll()}><Dices/></Button><Button variant="ghost" size="sm" disabled={!selected||modelBusy||!reachable} onClick={startChat}>New chat</Button></header>
 {(error||library.error||chatError)&&<div role="alert" className="app-error">{error||library.error||chatError}</div>}
 <div className={`panels ${expanded&&(libraryTab||rightTab==='chat')?'chat-expanded':''}`}><ResizablePanelGroup orientation="horizontal"><ResizablePanel id="map" defaultSize="62%" minSize="25%"><div className="library-layout" hidden={!libraryTab}><Library entries={library.allEntries} onPatch={(id,change)=>void library.patch(id,change).catch(()=>{})} onRemove={library.remove} onRestore={e=>void library.add(e).catch(()=>{})} onPlane={e=>{if(e.model!==manifest.model){setSwitching(true);void switchModel(e.model).catch(e=>{setError(String(e));setSwitching(false)});return}const p=entryPushes(e)[0];select(p);setDose(Math.abs(p.dose));setLibraryTab(false);setFind(false);setWeb(false)}} onBatch={entries=>appendColumns(entries)} onCombine={entries=>appendColumns(entries,true)} busy={modelBusy}/></div><div className="map-surface" hidden={libraryTab||web||find}>{!mapMissing&&<MapView manifest={manifest} layer={layer} selected={selected} onLayer={face} onSelect={select} onReady={onReady}/>}</div>{!libraryTab&&web&&!find&&<WebView manifest={manifest} selected={selected} favourites={favourites} onSelect={select} onStar={library.star} onAdd={addNeuron} canAdd={!modelBusy&&reachable}/>}<div className="find-surface" hidden={libraryTab||!find}><FindView model={manifest.model} maxLayer={manifest.layers.length-1} chatBusy={busy} onLocalBusy={setLocalFind} favourites={favourites} onStar={library.star} onAdd={addNeuron} canAdd={!modelBusy} onOpen={(n,view)=>{select(n);setFind(false);setWeb(view==='web')}}/></div></ResizablePanel><ResizableHandle withHandle/><ResizablePanel id="inspector" defaultSize="38%" minSize="330px"><aside className="right-panel"><div className="right-tabs"><Tabs value={libraryTab?'chat':rightTab} onValueChange={v=>{setRightTab(v);if(v==='inspect'){setExpanded(false);if(libraryTab)setLibraryTab(false)}}}><TabsList><TabsTrigger value="inspect">Neuron</TabsTrigger><TabsTrigger value="chat">Chat</TabsTrigger></TabsList></Tabs></div><div className="right-content" hidden={libraryTab||rightTab!=='inspect'}><Inspector manifest={manifest} layer={layer} selected={selected} favourite={!!selected&&favourites.some(n=>sameNeuron(n,selected))} onStar={()=>{if(selected)library.star(selected)}} onSelect={select} dose={dose} onDose={setDose} onAdd={!modelBusy?()=>selected&&addNeuron(selected):undefined} onChat={startChat} reachable={reachable&&!localFind}/></div><div className="right-content chat-tab" hidden={!libraryTab&&rightTab!=='chat'}><ChatPanel key={chatId} model={runningModel} chat={chat} selected={selected} onAdd={addNeuron} update={update} expanded={expanded} onExpand={()=>setExpanded(v=>!v)} blocked={localFind||switching} onBusy={setBusy} autoSend={rolled===chatId?PROMPTS[0]:undefined} onRoll={()=>void roll()} onSaveColumn={saveColumn}/></div></aside></ResizablePanel></ResizablePanelGroup></div>
 {!mapMissing&&!ready&&!web&&!find&&!libraryTab&&<span className="loading-map">Loading layers…</span>}
 </main></SidebarProvider></TooltipProvider>
}

export default function App(){
 const [ready,setReady]=useState(false),[error,setError]=useState('')
 useEffect(()=>{let alive=true;async function boot(){const response=await fetch('./api/state');const state=await response.json();if(modelOverride&&state.model!==modelOverride){const r=await fetch('./api/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:modelOverride})});const result=await r.json();if(!r.ok)throw Error(result.error||'Model load failed')}if(alive)setReady(true)}void boot().catch(e=>{if(alive)setError(String(e))});return()=>{alive=false}},[])
 return ready?<Explorer/>:<div className="boot" role={error?'alert':'status'}>{error||'Loading model and map…'}</div>
}
