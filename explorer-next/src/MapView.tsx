import { PlaneHover } from './PlaneHover'
import { useCallback, useEffect, useRef, useState } from 'react'
import { createScene } from './legacy/scene.js'
import type { Manifest, Neuron } from './data'
import { detail } from './data'
import { Button } from './components/ui/button'
import { ScrollArea } from './components/ui/scroll-area'
import { ChevronLeft, ChevronRight, RotateCcw } from 'lucide-react'
type Scene=ReturnType<typeof createScene>
export function MapView({manifest,layer,selected,onLayer,onSelect,onReady}:{manifest:Manifest;layer:number|null;selected:Neuron|null;onLayer:(l:number|null)=>void;onSelect:(n:Neuron)=>void;onReady:(scene:Scene)=>void}){
 const [loaded,setLoaded]=useState(false)
 const host=useRef<HTMLDivElement>(null),scene=useRef<Scene|null>(null),callbacks=useRef({onLayer,onSelect,onReady}),view=useRef({layer,selected})
 callbacks.current={onLayer,onSelect,onReady};view.current={layer,selected}
 useEffect(()=>{
  const el=host.current;if(!el)return
  let disposed=false;const s=createScene(el,manifest);scene.current=s
  let down:{x:number;y:number}|null=null
  const pointerDown=(e:PointerEvent)=>{down={x:e.clientX,y:e.clientY}}
  const pointerUp=(e:PointerEvent)=>{if(!down||Math.hypot(e.clientX-down.x,e.clientY-down.y)>5)return;const hit=s.pick(e);if(hit){if(view.current.layer===null)callbacks.current.onLayer(hit.layer);else callbacks.current.onSelect(hit)}}
  el.addEventListener('pointerdown',pointerDown);el.addEventListener('pointerup',pointerUp)
  Promise.all(manifest.layers.map(async l=>{const r=await fetch(`./data/L${l.layer}.bin`);if(!r.ok)throw Error(`Layer ${l.layer} unavailable`);const data=new Float32Array(await r.arrayBuffer());if(!disposed)s.add(l.layer,data);if(!disposed){const d=await detail(l.layer);if(!disposed)s.metrics(l.layer,d)}})).then(()=>{if(disposed)return;const v=view.current;if(v.selected)s.select(v.selected.layer,v.selected.n,null);else if(v.layer!==null)s.face(v.layer);else s.overview();setLoaded(true);callbacks.current.onReady(s)}).catch(e=>{if(!disposed)el.setAttribute('data-error',String(e))})
  return()=>{disposed=true;el.removeEventListener('pointerdown',pointerDown);el.removeEventListener('pointerup',pointerUp);s.dispose();scene.current=null}
 },[manifest])
 useEffect(()=>{const s=scene.current;if(!s||s.stats().layers!==manifest.layers.length)return;if(selected)s.select(selected.layer,selected.n,null);else if(layer!==null)s.face(layer);else s.overview()},[layer,selected,manifest])
 const hoverPick=useCallback((e:PointerEvent)=>view.current.layer===null?undefined:scene.current?.pick(e),[])
 const ring=useCallback(()=>{const n=view.current.selected;return n?scene.current?.project(n.layer,n.n)??null:null},[])
 return <section className="map-view" aria-label="Neuron map">
  <div className="map-toolbar"><span>{layer===null?`${manifest.layers.length} layers`: `Layer ${layer}`}</span><span className="text-muted-foreground">{(layer===null?manifest.layers.reduce((a,l)=>a+l.count,0):manifest.layers[layer].count).toLocaleString()} neurons</span><div className="ml-auto flex gap-1"><Button variant="ghost" size="icon-sm" aria-label="Previous layer" disabled={layer===0} onClick={()=>onLayer(Math.max(0,(layer??12)-1))}><ChevronLeft/></Button><Button variant="ghost" size="icon-sm" aria-label="Next layer" disabled={layer===manifest.layers.length-1} onClick={()=>onLayer(Math.min(manifest.layers.length-1,(layer??11)+1))}><ChevronRight/></Button><Button variant="ghost" size="icon-sm" aria-label="Reset map view" onClick={()=>{if(layer===null)scene.current?.overview();else scene.current?.face(layer)}}><RotateCcw/></Button></div></div>
  <div ref={host} className="viewport" tabIndex={0} aria-label={layer===null?'Tower. Select a layer.':'Layer plane. Drag to pan and scroll to zoom.'} onKeyDown={e=>{if(e.key==='Escape')onLayer(null);if(e.key==='ArrowUp'||e.key==='ArrowDown'){e.preventDefault();onLayer(Math.max(0,Math.min(manifest.layers.length-1,(layer??12)+(e.key==='ArrowUp'?1:-1))))}}}/>
  {loaded&&layer!==null&&<div className="plane-hover-overlay"><PlaneHover host={host.current} pick={hoverPick} ring={ring}/></div>}<span className="map-legend">Size + light: KL ×random · <span className="meaning-plus">+</span> / <span className="meaning-minus">−</span> stronger</span><ScrollArea className="layer-rail"><div className="flex flex-col gap-px">{[...manifest.layers].reverse().map(l=><Button key={l.layer} variant={layer===l.layer?'secondary':'ghost'} className="layer-button" aria-label={`Layer ${l.layer}`} aria-pressed={layer===l.layer} onClick={()=>onLayer(l.layer)}>{l.layer}</Button>)}</div></ScrollArea>
  {layer===null&&<Button className="enter-layer" variant="secondary" onClick={()=>onLayer(Math.floor(manifest.layers.length/2))}>Enter layer</Button>}
 </section>
}
