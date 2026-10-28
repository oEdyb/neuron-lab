import { escape } from './data.js';
export const $ = selector => document.querySelector(selector);
export function spreadChart(values, label) {
  const max=Math.max(...values,.00001);
  return `<div class="spread" role="img" aria-label="${escape(label)}: ${values.map((v,i)=>`L${i} ${v.toFixed(4)}`).join(', ')}">${values.map((v,i)=>`<i title="L${i}: ${v.toFixed(4)}" style="height:${v/max*100}%"></i>`).join('')}</div><div class="spread-axis"><span>L0</span><span>L${Math.floor(values.length/2)}</span><span>L${values.length-1}</span></div><div class="spread-summary">peak ${max.toFixed(3)} · L${values.indexOf(max)}</div>`;
}
export function layerFacts(summary, select) {
  const percent=summary.above/summary.count*100;
  $('#facts').innerHTML=`<div class="metric"><strong>${percent.toFixed(1)}<small>%</small></strong><span>either sign > random p95 · ${summary.above.toLocaleString()} / ${summary.count.toLocaleString()}</span><div class="fraction"><i style="width:${percent}%"></i></div></div><h2>Highest KL · output change</h2>${summary.top.slice(0,3).map(row=>`<button class="top-neuron" data-n="${row.n}"><span>N${row.n}</span><div><strong>${(row.kl[0]>=row.kl[1]?row.up:row.down).slice(0,3).map(escape).join(' / ')}</strong><small>KL ${row.kl[0]>=row.kl[1]?'+':'−'}${Math.max(...row.kl).toFixed(3)}</small></div></button>`).join('')}<h2>Mean spread · relative change</h2>${spreadChart(summary.spread,'Layer mean spread')}`;
  $('#facts').hidden=false;
  $('#facts').querySelectorAll('[data-n]').forEach(button=>button.onclick=()=>select(summary.layer,+button.dataset.n));
}
export function neuronPanel(layer,n,row,p95,reachable,onSign,onSteer,onClose) {
  $('#selection').innerHTML=`<div class="panel-head"><span>NEURON</span><button id="close-selection" aria-label="Close neuron">×</button></div><h3>L${layer} <span style="color:var(--accent)">N${n}</span></h3><h2>+ words</h2><div class="words">${row.up.slice(0,10).map(w=>`<span>${escape(w)}</span>`).join('')}</div><h2>− words</h2><div class="words negative">${row.down.slice(0,10).map(w=>`<span>${escape(w)}</span>`).join('')}</div><h2>“I am a…” +</h2><div class="words">${(row.persona?.[0]?.slice(0,6)||[]).map(w=>`<span>${escape(w)}</span>`).join('')}</div><h2>“I am a…” −</h2><div class="words negative">${(row.persona?.[1]?.slice(0,6)||[]).map(w=>`<span>${escape(w)}</span>`).join('')}</div><div class="kl-row"><span>KL +${row.kl[0].toFixed(4)}</span><span>−${row.kl[1].toFixed(4)}</span></div><div class="kl-row"><span>Random p95 +${p95[0].toFixed(4)}</span><span>−${p95[1].toFixed(4)}</span></div><h2>Push spread · relative change</h2><div id="neuron-spread">${spreadChart(row.spread[0],'Selected neuron spread')}</div><div class="direction"><button data-sign="0" aria-pressed="true">+0.6</button><button data-sign="1" aria-pressed="false">−0.6</button></div><button id="add-steer" class="primary" ${reachable?'':'hidden'}>Add to steer</button>`;
  $('#selection').hidden=false;$('#close-selection').onclick=onClose;
  $('#selection').querySelectorAll('[data-sign]').forEach(button=>button.onclick=()=>{const sign=+button.dataset.sign;$('#selection').querySelectorAll('[data-sign]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));$('#neuron-spread').innerHTML=spreadChart(row.spread[sign],'Selected neuron spread');onSign(sign);});
  $('#add-steer').onclick=onSteer;
}
