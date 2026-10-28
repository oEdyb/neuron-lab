import { useCallback,useEffect,useRef,useState } from 'react'
import { SidebarProvider,SidebarTrigger } from './components/ui/sidebar'
import { TooltipProvider } from './components/ui/tooltip'
import { Tabs,TabsList,TabsTrigger } from './components/ui/tabs'
import { ToggleGroup,ToggleGroupItem } from './components/ui/toggle-group'
import { ResizableHandle,ResizablePanel,ResizablePanelGroup } from './components/ui/resizable'
import { Button } from './components/ui/button'
import { Library } from './Library'
import { useLibrary,entryPushes,isSingle,type LibraryEntry } from './use-library'
import { MapView } from './MapView'
import { FindView } from './FindView'
import { WebView } from './WebView'
import { Inspector } from './Inspector'
import { newId,type Column } from './storage'
import { ChatPanel } from './ChatPanel'
import { SearchSidebar } from './SearchSidebar'
import { detail,kl,loadManifest,sameNeuron,type Manifest,type Neuron } from './data'
import { PROMPTS } from './prompts'
import { Dices } from 'lucide-react'
import { ChatsSchema,newChat,type Chat } from './storage'
import { probe } from './model'
import { useStored } from './hooks/use-stored'
import { useIsMobile } from './hooks/use-mobile'
import type { createScene } from './legacy/scene.js'
export default function App(){
 const [mapMissing,setMapMissing]=useState(false)
 const [manifest,setManifest]=useState<Manifest|null>(null),[error,setError]=useState(''),[layer,setLayer]=useState<number|null>(null),[selected,setSelected]=useState<Neuron|null>(null),[rightTab,setRightTab]=useState('inspect'),[expanded,setExpanded]=useState(false),[reachable,setReachable]=useState(false),[busy,setBusy]=useState(false),[dose,setDose]=useState(.6),[ready,setReady]=useState(false)
 const mobile=useIsMobile()
 const library=useLibrary(manifest?.model??'')
 const favourites=library.entries.filter(isSingle)
 const [chats,setChats,chatError]=useStored('neuron-next-chats',ChatsSchema,[]),[chatId,setChatId]=useState<string|null>(null)
 const [libraryTab,setLibraryTab]=useState(false),[web,setWeb]=useState(false),[find,setFind]=useState(false),[localFind,setLocalFind]=useState(false),[rolled,setRolled]=useState<string|null>(null)
 const [sidebarOpen,setSidebarOpen]=useState(true)
 useEffect(()=>setSidebarOpen(!libraryTab),[libraryTab])
 const modelBusy=busy||localFind,chat=chats.find(c=>c.id===chatId)??null
 const council=chat?.council?chat:chats.find(c=>c.council)??null
 const scene=useRef<ReturnType<typeof createScene>|null>(null)
 const update=useCallback((id:string,fn:(c:Chat)=>Chat)=>setChats(all=>all.map(c=>c.id===id?fn(c):c)),[])
 useEffect(()=>{loadManifest().then(m=>{setManifest(m);setDose(m.dose);const h=location.hash.match(/^#L(\d+)-N(\d+)$/i);if(h&&+h[1]<m.layers.length&&+h[2]<m.layers[+h[1]].count){setLayer(+h[1]);setSelected({layer:+h[1],n:+h[2]})}}).catch(async()=>{const response=await fetch('./api/maps');const status=await response.json();setError(`Map isn't downloaded. Run: ${status.fix}`);setMapMissing(true);setManifest({model:status.model,dose:.6,layers:Array.from({length:24},(_,layer)=>({layer,count:6144})),summaries:[]})})},[])
 useEffect(()=>{if(!manifest)return;let live=true;const check=()=>probe(manifest.model).then(ok=>{if(live)setReachable(ok)});void check();const timer=setInterval(check,5000);return()=>{live=false;clearInterval(timer)}},[manifest])
 const select=useCallback((n:Neuron)=>{setSelected({layer:n.layer,n:n.n});setLayer(n.layer);setRightTab('inspect');setExpanded(false);history.replaceState(null,'',`#L${n.layer}-N${n.n}`)},[])
 const face=useCallback((l:number|null)=>{setLayer(l);setSelected(null);setExpanded(false);setRightTab('inspect');history.replaceState(null,'',location.pathname+location.search)},[])
 const onReady=useCallback((s:ReturnType<typeof createScene>)=>{scene.current=s;setReady(true);Object.assign(window,{neuronNext:{select,face,stats:()=>s.stats(),project:(l:number,n:number)=>s.project(l,n)}})},[select,face])
 function startChat(){if(modelBusy)return;setLibraryTab(false);if(!selected){setRightTab('chat');return}const c=newChat(selected,dose);const template=chat&&!chat.council?chat:chats.find(c=>!c.council);if(template)c.columns=template.columns.map(col=>({...col,dose:col.id==='normal'?0:col.id==='minus'?-dose:dose,messages:[]}));setChats(all=>[c,...all]);setChatId(c.id);setRightTab('chat');setExpanded(false)}
 function appendColumns(entries:LibraryEntry[],combine=false){
  if(modelBusy||!entries.length)return
  const chosen=combine?[entries]:entries.map(e=>[e])
  const columns:Column[]=chosen.map(group=>{const pushes=new Map<string,Chat['steerSet'][number]>();const scale=1/group.length;for(const entry of group)for(const p of entryPushes(entry)){const key=`${p.layer}:${p.n}`,old=pushes.get(key);pushes.set(key,{...p,dose:Math.max(-2,Math.min(2,Math.round((p.dose*scale+(old?.dose??0))*20)/20)),muted:false})}return {id:newId(),name:group.map(e=>e.name).join(' + '),dose:0,hidden:false,messages:[],pushes:[...pushes.values()]}})
  let current=council
  if(!current){const p=columns[0].pushes?.[0]??{layer:0,n:0};current=newChat(p,dose);current.council=true;current.title='Library chat';current.columns=[{id:'normal',name:'Normal',dose:0,hidden:true,messages:[]}]}
  const existing=current.columns.filter(c=>c.id!=='normal'||!c.hidden)
  if(existing.length+columns.length>6){setError('A chat holds up to 6 columns. Remove a column first.');return}
  let next=[...existing,...columns]
  const normal=current.columns.find(c=>c.id==='normal'&&c.hidden)
  if(next.length<6&&normal)next=[normal,...next]
  const updated={...current,columns:next}
  setChats(all=>all.some(c=>c.id===updated.id)?all.map(c=>c.id===updated.id?updated:c):[updated,...all]);setChatId(updated.id);setLibraryTab(true);setError('')
 }
 function addCouncil(useSet=false){if(!selected)return;const pushes=useSet&&chat&&!chat.council?chat.steerSet.filter(p=>!p.muted).map(({layer,n,dose})=>({layer,n,dose})):[{...selected,dose}];if(!pushes.length)return;appendColumns([{id:newId(),model:manifest?.model??'',name:pushes.map(p=>`L${p.layer} N${p.n}`).join(' + '),note:'',tags:[],source:'you',added:new Date().toISOString(),pushes}])}
 function addNeuron(n:Neuron){if(modelBusy)return;if(chat?.council){appendColumns([{...n,id:newId(),model:manifest?.model??'',dose,name:`L${n.layer} N${n.n}`,note:'',tags:[],source:'you',added:new Date().toISOString()}]);return}if(!chat){const c=newChat(n,dose);setChats(all=>[c,...all]);setChatId(c.id);setRightTab('chat');return}update(chat.id,c=>c.steerSet.some(p=>sameNeuron(p,n))?c:{...c,steerSet:[...c.steerSet,{...n,dose:.4,muted:false}]});setRightTab('chat');setExpanded(false)}
 function openChat(id:string){const c=chats.find(v=>v.id===id);if(!c||modelBusy)return;const pushes=c.council?c.columns.flatMap(col=>col.pushes??[]):c.steerSet;if(!manifest||pushes.some(n=>!manifest.layers[n.layer]||n.n>=manifest.layers[n.layer].count)){setError('This saved chat contains a neuron outside this map. Choose another chat.');return}setError('');setChatId(id);setSelected(c.neuron);setLayer(c.neuron.layer);setDose(c.dose);setLibraryTab(c.council);setRightTab('chat');setExpanded(false)}
 async function roll(){if(!manifest||!reachable||modelBusy)return;const first=Math.min(3,manifest.layers.length-1);for(let tries=0;tries<6;tries++){const layer=first+Math.floor(Math.random()*(manifest.layers.length-first)),d=await detail(layer),pool=d.rows.flatMap((r,i)=>kl(r,d.p95)>1.5?[r.n??i]:[]);if(!pool.length)continue;const n={layer,n:pool[Math.floor(Math.random()*pool.length)]},c=newChat(n,chat?.dose??dose);select(n);setChats(all=>[c,...all]);setChatId(c.id);setRolled(c.id);setRightTab('chat');setLibraryTab(false);setExpanded(false);return}}
 function saveColumn(column:Column){const pushes=column.pushes?.filter(p=>!p.muted).map(({layer,n,dose})=>({layer,n,dose}));if(!pushes?.length||!manifest)return;const base=column.name??'Combo',taken=library.entries.some(e=>e.name===base),fmt=(d:number)=>`${d>0?'+':''}${d}`,name=taken?`${base} ${pushes.map(p=>fmt(p.dose)).join(' ')}`:base,common={id:newId(),model:manifest.model,name,note:'',tags:[],source:'you' as const,added:new Date().toISOString()};void library.add(pushes.length===1?{...common,...pushes[0]}:{...common,pushes}).catch(()=>{})}
 if(!manifest)return <div className="boot" role="status">{error||'Loading map…'}</div>
 return <TooltipProvider delayDuration={300}><SidebarProvider className="app-shell" open={sidebarOpen} onOpenChange={setSidebarOpen}><SearchSidebar saved={library.entries} onSaved={e=>{if(isSingle(e)){select(e);setLibraryTab(false);setFind(false);setWeb(false)}else appendColumns([e])}} onUnsave={e=>void library.remove(e).catch(()=>{})} selected={selected} onAdd={chat&&!modelBusy?addNeuron:undefined} model={manifest.model} favourites={favourites} onStar={library.star} onSelect={n=>{select(n);setLibraryTab(false);setFind(false);setWeb(false)}} chats={chats} onChat={openChat} onRename={(id,title)=>update(id,c=>({...c,title}))} onDelete={id=>{setChats(all=>all.filter(c=>c.id!==id));if(chatId===id)setChatId(null)}} onNew={startChat} busy={modelBusy}/><main className="workspace"><header className="workspace-header"><SidebarTrigger className="mobile-sidebar"/><ToggleGroup type="single" size="sm" value={libraryTab?'library':find?'find':web?'web':layer===null?'tower':'plane'} onValueChange={v=>{if(!v||modelBusy)return;setLibraryTab(v==='library');setFind(v==='find');setWeb(v==='web');const back=find||web||libraryTab;if(v==='web'||v==='find'||v==='library')return;if(back&&(v==='tower')===(layer===null))return;face(v==='tower'?null:layer??Math.floor(manifest.layers.length/2))}}><ToggleGroupItem value="tower">Tower</ToggleGroupItem><ToggleGroupItem value="plane">Plane</ToggleGroupItem><ToggleGroupItem value="web">Web</ToggleGroupItem><ToggleGroupItem value="find">Find</ToggleGroupItem><ToggleGroupItem value="library">Library</ToggleGroupItem></ToggleGroup><span className="ml-auto"/><Button variant="ghost" size="sm" aria-label="Random neuron" disabled={!reachable||modelBusy} onClick={()=>void roll()}><Dices/></Button><Button variant="ghost" size="sm" disabled={!selected||modelBusy||!reachable} onClick={startChat}>New chat</Button></header>
 {(error||library.error||chatError)&&<div role="alert" className="app-error">{error||library.error||chatError}</div>}
 {libraryTab?<div className={`library-layout ${expanded?'library-chat-expanded':''}`}><ResizablePanelGroup orientation={mobile?'vertical':'horizontal'}><ResizablePanel defaultSize="38%" minSize={mobile?'20%':'300px'}><Library entries={library.entries} onPatch={(id,change)=>void library.patch(id,change).catch(()=>{})} onRemove={library.remove} onRestore={e=>void library.add(e).catch(()=>{})} onPlane={e=>{const p=entryPushes(e)[0];select(p);setDose(Math.abs(p.dose));setLibraryTab(false);setFind(false);setWeb(false)}} onBatch={entries=>appendColumns(entries)} onCombine={entries=>appendColumns(entries,true)} busy={modelBusy}/></ResizablePanel><ResizableHandle withHandle/><ResizablePanel defaultSize="62%" minSize={mobile?'30%':'330px'}><div className="library-chat">{council?<ChatPanel key={council.id} chat={council} selected={selected} onAdd={addNeuron} update={update} expanded={expanded} onExpand={()=>setExpanded(v=>!v)} blocked={localFind} onBusy={setBusy} onRoll={()=>void roll()} onSaveColumn={saveColumn}/>:<div className="empty p-3">Select Library entries</div>}</div></ResizablePanel></ResizablePanelGroup></div>:<div className={`panels ${expanded&&rightTab==='chat'?'chat-expanded':''}`}><ResizablePanelGroup orientation="horizontal"><ResizablePanel id="map" defaultSize="62%" minSize="25%"><div className="map-surface" hidden={web||find}>{!mapMissing&&<MapView manifest={manifest} layer={layer} selected={selected} onLayer={face} onSelect={select} onReady={onReady}/>}</div>{web&&!find&&<WebView manifest={manifest} selected={selected} favourites={favourites} onSelect={select} onStar={library.star} onAdd={addNeuron} canAdd={!modelBusy&&reachable}/>}<div className="find-surface" hidden={!find}><FindView model={manifest.model} maxLayer={manifest.layers.length-1} chatBusy={busy} onLocalBusy={setLocalFind} favourites={favourites} onStar={library.star} onAdd={addNeuron} canAdd={!modelBusy} onOpen={(n,view)=>{select(n);setFind(false);setWeb(view==='web')}}/></div></ResizablePanel><ResizableHandle withHandle/><ResizablePanel id="inspector" defaultSize="38%" minSize="330px"><aside className="right-panel"><div className="right-tabs"><Tabs value={rightTab} onValueChange={v=>{setRightTab(v);if(v==='inspect')setExpanded(false)}}><TabsList><TabsTrigger value="inspect">Neuron</TabsTrigger><TabsTrigger value="chat">Chat</TabsTrigger></TabsList></Tabs></div><div className="right-content" hidden={rightTab!=='inspect'}><Inspector manifest={manifest} layer={layer} selected={selected} favourite={!!selected&&favourites.some(n=>sameNeuron(n,selected))} onStar={()=>{if(selected)library.star(selected)}} onSelect={select} dose={dose} onDose={setDose} onAdd={chat&&!modelBusy?()=>selected&&addNeuron(selected):undefined} onCouncil={!modelBusy?()=>addCouncil():undefined} onChat={startChat} reachable={reachable&&!localFind}/></div><div className="right-content chat-tab" hidden={rightTab!=='chat'}><Button variant="outline" size="sm" disabled={modelBusy||!chat||chat.council} onClick={()=>addCouncil(true)}>Add steer set to Library chat</Button><ChatPanel key={chatId} chat={chat?.council?null:chat} selected={selected} onAdd={addNeuron} update={update} expanded={expanded} onExpand={()=>setExpanded(v=>!v)} blocked={localFind} onBusy={setBusy} autoSend={rolled===chatId?PROMPTS[0]:undefined} onRoll={()=>void roll()}/></div></aside></ResizablePanel></ResizablePanelGroup></div>}
 {!mapMissing&&!ready&&!web&&!find&&!libraryTab&&<span className="loading-map">Loading layers…</span>}
 </main></SidebarProvider></TooltipProvider>
}
