import { config } from './config.js';
const details = new Map();
export async function load(name) {
  const response = await fetch(config.data + name);
  if (!response.ok) throw new Error(`Cannot load ${name} (${response.status})`);
  return response;
}
export async function compressed(name) {
  const response = await load(name);
  const bytes=new Uint8Array(await response.arrayBuffer());
  if(bytes[0]===31&&bytes[1]===139){const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));return new Response(stream).json();}
  return JSON.parse(new TextDecoder().decode(bytes));
}
export function detail(layer) {
  if (!details.has(layer)) details.set(layer, compressed(`L${layer}.json.gz`).catch(error => { details.delete(layer); throw error; }));
  return details.get(layer);
}
export const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
