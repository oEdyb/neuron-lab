import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { gridShape } from './layout.js';
import { config } from './config.js';
import { vertex, fragment } from './shaders.js';
export function createScene(host, manifest) {
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(config.camera.fov, 1, .05, 500);
  const renderer = new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
  renderer.setClearColor(0x000000); renderer.setPixelRatio(Math.min(devicePixelRatio,config.pixelRatio)); host.append(renderer.domElement);
  const controls = new OrbitControls(camera,renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor=.085; controls.minDistance=2; controls.maxDistance=120; controls.zoomToCursor=true;
  const layers = new Map(), scores = new Float32Array(config.total);
  const marker = new THREE.Mesh(new THREE.RingGeometry(.09,.12,48),new THREE.MeshBasicMaterial({color:config.colors.accent,side:THREE.DoubleSide,depthTest:false}));
  marker.rotation.x=-Math.PI/2; marker.visible=false; marker.renderOrder=10; scene.add(marker);
  let tween, focused=null, selected=null, spread=null; let disposed=false;let lastTime=performance.now();
  const centerY = l => (l-(config.layers-1)/2)*config.spacing;
  function add(layer,data) {
    const count=data.length/3,{columns,rows}=gridShape(count), positions=new Float32Array(count*3), strengths=new Float32Array(count), ranks=new Float32Array(count);
    const order=Array.from({length:count},(_,i)=>i).sort((a,b)=>data[a*3+1]-data[b*3+1]||a-b), cells=new Int32Array(count);
    for(let r=0;r<rows;r++) order.slice(r*columns,(r+1)*columns).sort((a,b)=>data[a*3]-data[b*3]||a-b).forEach((id,c)=>cells[id]=r*columns+c);
    const ranked=Array.from({length:count},(_,i)=>i).sort((a,b)=>data[a*3+2]-data[b*3+2]);
    ranked.forEach((id,i)=>ranks[id]=i/count);
    for(let n=0;n<count;n++){
      const jitterX=((Math.sin(n*127.1+layer)*43758.5453)%1)*config.jitter, jitterZ=((Math.sin(n*311.7+layer)*13758.5453)%1)*config.jitter;
      positions.set([(cells[n]%columns)/(columns-1)*(config.width-.3)-(config.width-.3)/2+jitterX,centerY(layer),Math.floor(cells[n]/columns)/(rows-1)*(config.depth-.3)-(config.depth-.3)/2+jitterZ],n*3);
      strengths[n]=data[n*3+2];scores[config.offsets[layer]+n]=strengths[n];
    }
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('strength',new THREE.BufferAttribute(strengths,1));geometry.setAttribute('rank',new THREE.BufferAttribute(ranks,1));geometry.setAttribute('sign',new THREE.BufferAttribute(new Float32Array(count),1));
    const material=new THREE.ShaderMaterial({vertexShader:vertex,fragmentShader:fragment,transparent:true,depthWrite:false,depthTest:false,blending:THREE.AdditiveBlending,uniforms:{fade:{value:1},focus:{value:0},viewport:{value:new THREE.Vector2(host.clientWidth,host.clientHeight)},reveal:{value:new THREE.Vector2(...config.gridReveal)},sparseRange:{value:new THREE.Vector2(...config.sparse)},glow:{value:config.glow},pixelRatio:{value:renderer.getPixelRatio()},tint:{value:new THREE.Color(config.colors.light)}}});
    const points=new THREE.Points(geometry,material);scene.add(points);
    const outline=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-7,centerY(layer),-4.5),new THREE.Vector3(7,centerY(layer),-4.5),new THREE.Vector3(7,centerY(layer),4.5),new THREE.Vector3(-7,centerY(layer),4.5)]);
    const edge=new THREE.LineLoop(outline,new THREE.LineBasicMaterial({color:config.colors.outline,transparent:true,opacity:.18,depthWrite:false}));scene.add(edge);
    layers.set(layer,{points,edge,positions});
  }
  function metrics(layer,data){const obj=layers.get(layer);if(!obj)return;const strengths=obj.points.geometry.attributes.strength,sign=obj.points.geometry.attributes.sign;data.rows.forEach((r,n)=>{const plus=r.kl[0]/data.p95[0],minus=r.kl[1]/data.p95[1];strengths.setX(n,Math.max(plus,minus));sign.setX(n,plus>=minus?1:0);});strengths.needsUpdate=true;sign.needsUpdate=true;}
  function move(position,target){
    const from=camera.position.clone(),start=controls.target.clone(),rotation=camera.quaternion.clone();
    controls.enableDamping=false;controls.update();camera.position.copy(from);controls.target.copy(start);camera.quaternion.copy(rotation);
    const toRotation=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(position,target,camera.up));
    if(matchMedia('(prefers-reduced-motion: reduce)').matches){camera.position.copy(position);controls.target.copy(target);camera.quaternion.copy(toRotation);controls.update();tween=null;controls.enableDamping=true;return;}
    tween={from,to:position,start,target,rotation,toRotation,fromDistance:from.distanceTo(start),toDistance:position.distanceTo(target),time:performance.now()};
  }
  function paint(){
    const max=spread?Math.max(...spread,.0001):1;
    for(const [layer,obj]of layers){
      const isFocus=layer===focused;
      obj.fadeTarget=focused===null?1:isFocus?1:spread?.[layer]?Math.max(.07,spread[layer]/max*.4):.008;
      obj.focusTarget=+isFocus;
      obj.points.material.uniforms.tint.value.set(spread&&layer>=selected?.layer?config.colors.accent:config.colors.light);
      obj.edgeTarget=focused===null?.2:isFocus?.7:spread?.[layer]?Math.max(.04,spread[layer]/max*.55):.012;
      obj.edge.material.color.set(spread&&layer>=selected?.layer?config.colors.accent:config.colors.outline);
    }
  }
  function overview(){controls.enableRotate=true;controls.mouseButtons.LEFT=THREE.MOUSE.ROTATE;focused=null;spread=null;selected=null;marker.visible=false;move(new THREE.Vector3(...config.camera.overview).multiplyScalar(Math.max(1,config.camera.overviewAspect/camera.aspect,config.layers*config.spacing/22)),new THREE.Vector3(0,0,0));paint();}
  function face(layer){controls.enableRotate=false;controls.mouseButtons.LEFT=THREE.MOUSE.PAN;focused=layer;spread=null;selected=null;marker.visible=false;const target=new THREE.Vector3(0,centerY(layer),0);move(target.clone().add(new THREE.Vector3(0,Math.max(14.8,22.8/camera.aspect),.001)),target);paint();}
  function select(layer,n,values){controls.enableRotate=false;controls.mouseButtons.LEFT=THREE.MOUSE.PAN;focused=layer;selected={layer,n};spread=values;marker.position.fromArray(layers.get(layer).positions,n*3);marker.visible=true;const target=new THREE.Vector3(0,centerY(layer),0);move(target.clone().add(new THREE.Vector3(0,Math.max(14.8,22.8/camera.aspect),.001)),target);paint();}
  function updateSpread(values){spread=values;paint();}
  function clear(){selected=null;spread=null;marker.visible=false;paint();}
  function project(layer,n){const p=new THREE.Vector3().fromArray(layers.get(layer).positions,n*3).multiply(new THREE.Vector3(1,1,layers.get(layer).points.scale.z)).project(camera),rect=host.getBoundingClientRect();return{x:(p.x+1)*rect.width/2,y:(1-p.y)*rect.height/2,z:p.z};}
  const ray=new THREE.Raycaster(),pointer=new THREE.Vector2(),plane=new THREE.Plane(new THREE.Vector3(0,1,0)),hit=new THREE.Vector3();
  function pick(e){
    const rect=host.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,1-(e.clientY-rect.top)/rect.height*2);ray.setFromCamera(pointer,camera);
    let best, distance=8;
    for(const [layer,obj]of layers){
      if(focused!==null&&layer!==focused)continue;
      plane.constant=-centerY(layer);if(!ray.ray.intersectPlane(plane,hit)||Math.abs(hit.x)>7.1||Math.abs(hit.z)>4.6*obj.points.scale.z)continue;
      const worldRadius=camera.position.distanceTo(hit)*.012;
      for(let n=0;n<obj.positions.length/3;n++){
        if(Math.abs(obj.positions[n*3]-hit.x)>worldRadius||Math.abs(obj.positions[n*3+2]*obj.points.scale.z-hit.z)>worldRadius)continue;
        const p=project(layer,n);if(p.z>1||p.z< -1)continue;const d=Math.hypot(p.x-(e.clientX-rect.left),p.y-(e.clientY-rect.top));
        if(d<distance){distance=d;best={layer,n};}
      }
    }return best;
  }
  controls.addEventListener('start',()=>{tween=null;controls.enableDamping=true;});
  const observer=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);for(const obj of layers.values())obj.points.material.uniforms.viewport.value.set(w,h);});observer.observe(host);
  camera.position.set(...config.camera.overview);camera.lookAt(0,0,0);
  renderer.setAnimationLoop(time=>{
    if(disposed)return;
    if(tween){const t=Math.min(1,(time-tween.time)/(matchMedia('(prefers-reduced-motion: reduce)').matches?1:config.flight)),ease=t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;controls.target.lerpVectors(tween.start,tween.target,ease);camera.quaternion.slerpQuaternions(tween.rotation,tween.toRotation,ease);camera.position.set(0,0,1).applyQuaternion(camera.quaternion).multiplyScalar(THREE.MathUtils.lerp(tween.fromDistance,tween.toDistance,ease)).add(controls.target);if(t===1){tween=null;controls.enableDamping=true;}}
    if(!tween)controls.update();
    const dt=Math.min(64,time-lastTime);lastTime=time;const blend=matchMedia('(prefers-reduced-motion: reduce)').matches?1:1-Math.exp(-dt/110);
    for(const [layer,obj] of layers){
      const targetScale=layer===focused?Math.max(1,config.width/camera.aspect/config.depth*.9):1;
      obj.points.scale.z+=(targetScale-obj.points.scale.z)*blend;obj.edge.scale.z=obj.points.scale.z;
      obj.points.material.uniforms.fade.value+=( (obj.fadeTarget??1)-obj.points.material.uniforms.fade.value)*blend;
      obj.points.material.uniforms.focus.value+=((obj.focusTarget??0)-obj.points.material.uniforms.focus.value)*blend;
      obj.edge.material.opacity+=((obj.edgeTarget??.2)-obj.edge.material.opacity)*blend;
    }
    if(marker.visible&&selected){marker.position.z=layers.get(selected.layer).positions[selected.n*3+2]*layers.get(selected.layer).points.scale.z;const size=camera.position.distanceTo(marker.position)*2*Math.tan(camera.fov*Math.PI/360)/Math.max(1,host.clientHeight)*10;marker.scale.setScalar(size/.12);}
    renderer.render(scene,camera);
  });
  return {dispose(){disposed=true;observer.disconnect();renderer.setAnimationLoop(null);controls.dispose();scene.traverse(o=>{o.geometry?.dispose();if(o.material)o.material.dispose();});renderer.dispose();renderer.domElement.remove();},metrics,positions:layer=>layers.get(layer).positions,add,overview,face,select,clear,updateSpread,pick,project,scores,controls,stats:()=>({layers:layers.size,points:scores.length,focused,selected,flying:!!tween,camera:camera.position.toArray(),calls:renderer.info.render.calls,marker:selected?{point:project(selected.layer,selected.n),position:marker.position.toArray(),radius:marker.scale.x*.12}:null})};
}
