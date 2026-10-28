let ready;
async function init(){
 const {pipeline,env}=await import('./vendor/transformers.web.min.js');
 env.allowLocalModels=true;env.allowRemoteModels=false;env.localModelPath=new URL('./data/model-cache/',import.meta.url).href;env.backends.onnx.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;env.backends.onnx.wasm.numThreads=1;
 env.useCustomCache=true;env.customCache={async match(key){if(!String(key).endsWith('/onnx/model_quantized.onnx'))return undefined;const base=new URL('./data/model-cache/Xenova/paraphrase-multilingual-MiniLM-L12-v2/onnx/',import.meta.url),manifest=await(await fetch(new URL('parts.json',base))).json();const parts=await Promise.all(manifest.files.map(async file=>{const r=await fetch(new URL(file,base));if(!r.ok)throw new Error(`Cannot load ${file}`);return r.arrayBuffer();}));return new Response(new Blob(parts),{headers:{'Content-Length':String(manifest.bytes)}});},async put(){}};
 const extractor=await pipeline('feature-extraction','Xenova/paraphrase-multilingual-MiniLM-L12-v2',{dtype:'q8',device:'wasm'});
 const vectors=new Int8Array(await(await fetch('./data/word-vectors.bin')).arrayBuffer());
 return {extractor,vectors};
}
async function run({data:{id,query}}){try{
 ready??=init().catch(e=>{ready=null;throw e;});const {extractor,vectors}=await ready;
 const out=await extractor(query,{pooling:'mean',normalize:true}),q=out.data,top=[];
 for(let i=0;i<vectors.length/384;i++){let dot=0,norm=0;for(let j=0;j<384;j++){const v=vectors[i*384+j];dot+=v*q[j];norm+=v*v;}const similarity=norm?dot/Math.sqrt(norm):0;if(similarity>.48){top.push({i,similarity});}}
 top.sort((a,b)=>b.similarity-a.similarity);postMessage({id,terms:top.slice(0,80)});
}catch(e){postMessage({id,error:`Meaning search unavailable: ${e.message}. Try Exact word.`});}}
let queue=Promise.resolve();onmessage=event=>{queue=queue.then(()=>run(event));};
