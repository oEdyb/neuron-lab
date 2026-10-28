import { config } from './config.js';
const details = new Map();
export async function load(name) {
  const response = await fetch(config.data + name);
  if (!response.ok) throw new Error(`Cannot load ${name} (${response.status})`);
  return response;
}
export async function compressed(name) {
  const response = await load(name);
  const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).json();
}
export function detail(layer) {
  if (!details.has(layer)) details.set(layer, compressed(`L${layer}.json.gz`).catch(error => { details.delete(layer); throw error; }));
  return details.get(layer);
}
export const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
