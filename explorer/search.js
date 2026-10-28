import { compressed,escape,load } from './data.js';
import { config } from './config.js';
let words,vectorWords,strengths,worker,sequence=0;
const pending=new Map();
function semantic(query){
 if(!worker){worker=new Worker(new URL('./semantic-worker.js',import.meta.url),{type:'module'});worker.onmessage=({data})=>{const job=pending.get(data.id);if(job){pending.delete(data.id);data.error?job.reject(new Error(data.error)):job.resolve(data.terms);}};worker.onerror=e=>{for(const job of pending.values())job.reject(new Error(e.message));pending.clear();worker.terminate();worker=null;};}
 return new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});worker.postMessage({id,query});});
}
export async function search(query,mode,scores){
 const q=query.toLocaleLowerCase().trim(),coordinate=q.match(/^l(\d+)[\s-]*n(\d+)$/i);
 if(coordinate)return +coordinate[1]<config.layers&&+coordinate[2]<config.counts[+coordinate[1]]?[{layer:+coordinate[1],n:+coordinate[2],matched:[]}]:[];
 words??=compressed('signed-search.json.gz').catch(e=>{words=null;throw e;});strengths??=load('signed-strength.bin').then(r=>r.arrayBuffer()).then(b=>new Float32Array(b));const [index,signedStrength]=await Promise.all([words,strengths]);
 let terms;
 if(mode==='semantic'){
  document.querySelector('#search-status').textContent='Loading meaning model…';
  vectorWords??=compressed('vector-words.json.gz');const [matches,vocabulary]=await Promise.all([semantic(q),vectorWords]);
  const cutoff=Math.max(.55,(matches[0]?.similarity||0)*.78);terms=matches.filter(t=>t.similarity>=cutoff).map(t=>({word:vocabulary[t.i],similarity:t.similarity}));
  document.querySelector('#search-status').textContent='Meaning × strength vs random';
 }else terms=Object.keys(index).filter(w=>w.trim().toLocaleLowerCase()===q).map(word=>({word,similarity:1}));
 const matches=new Map();
 for(const {word,similarity}of terms){
  for(const packed of index[word]||[]){const measurement=packed%signedStrength.length,id=Math.floor(measurement/2),signedScore=signedStrength[measurement]*(packed>=signedStrength.length?.35:1);let l=config.offsets.findLastIndex(offset=>offset<=id),n=id-config.offsets[l];if(l<0||n>=config.counts[l])continue;
   const rank=similarity**8*Math.log1p(signedScore);let hit=matches.get(id);
   if(!hit){hit={layer:l,n,matched:[],rank,score:signedScore,similarity};matches.set(id,hit);}
   if(rank>hit.rank){hit.rank=rank;hit.score=signedScore;}if(hit.matched.length<5)hit.matched.push(word.trim());
  }
 }
 const ranked=[...matches.values()].sort((a,b)=>b.rank-a.rank),counts=new Map();
 return ranked.filter(hit=>{const count=counts.get(hit.layer)||0;if(count>=3||hit.rank<(ranked[0]?.rank||0)*.25)return false;counts.set(hit.layer,count+1);return true;}).slice(0,config.results);
}
export function resultHTML(hit,words,suffix=''){
 return `<div class="result-meta"><strong>L${hit.layer} <span>N${hit.n}</span></strong><span>${escape(suffix)}</span></div><small>${(hit.matched?.length?hit.matched:words).slice(0,7).map(w=>`<mark>${escape(w)}</mark>`).join(' · ')}</small>`;
}
