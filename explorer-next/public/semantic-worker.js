let ready;
async function init(){
 const {pipeline,env}=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm');
 env.allowLocalModels=false;
 env.backends.onnx.wasm.numThreads=1;
 const extractor=await pipeline('feature-extraction','Xenova/paraphrase-multilingual-MiniLM-L12-v2',{revision:'2c4055b12046f11709e9df2c122e59ffbdc2f900',dtype:'q8',device:'wasm'});
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
