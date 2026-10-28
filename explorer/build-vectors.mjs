import {pipeline,env} from '@huggingface/transformers';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {gunzipSync,gzipSync} from 'node:zlib';
env.cacheDir=new URL('./data/model-cache/',import.meta.url).pathname;
const existing=JSON.parse(gunzipSync(await readFile('data/vector-words.json.gz')));const words=[...new Set([...existing,...Object.keys(JSON.parse(gunzipSync(await readFile('data/signed-search.json.gz'))))])];
await writeFile('data/vector-words.json.gz',gzipSync(JSON.stringify(words)));
const extractor=await pipeline('feature-extraction','Xenova/paraphrase-multilingual-MiniLM-L12-v2',{dtype:'q8',device:'cpu',session_options:{intraOpNumThreads:4,interOpNumThreads:1}});
const vectors=new Int8Array(words.length*384);
let start=0;try{const previous=await readFile('data/word-vectors.bin');vectors.set(previous);start=previous.length/384;}catch{}
for(let i=start;i<words.length;i+=128){const batch=words.slice(i,i+128);const out=await extractor(batch.map(w=>w.trim()||w),{pooling:'mean',normalize:true});for(let j=0;j<out.data.length;j++)vectors[i*384+j]=Math.round(out.data[j]*127);if(i%1280===0){await writeFile('data/word-vectors.partial',vectors);await writeFile('data/vector-progress.json',JSON.stringify({done:i+batch.length,total:words.length}));console.log(i+batch.length,words.length);}}
await writeFile('data/word-vectors.bin',vectors);await writeFile('data/vector-progress.json',JSON.stringify({done:words.length,total:words.length}));
const queries=['death','pirate','money','猫'];const q=await extractor(queries,{pooling:'mean',normalize:true});await writeFile('data/query-check.json',JSON.stringify(Object.fromEntries(queries.map((s,i)=>[s,Array.from(q.data.slice(i*384,(i+1)*384))]))));
