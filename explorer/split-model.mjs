import {readFile,writeFile} from 'node:fs/promises';
const path='data/model-cache/Xenova/paraphrase-multilingual-MiniLM-L12-v2/onnx/';
const data=await readFile(path+'model_quantized.onnx'),size=32*1024*1024,files=[];
for(let i=0;i<data.length;i+=size){const name=`model-part-${files.length}.bin`;await writeFile(path+name,data.subarray(i,i+size));files.push(name);}
await writeFile(path+'parts.json',JSON.stringify({bytes:data.length,files}));
