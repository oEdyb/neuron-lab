import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
const root=fileURLToPath(new URL('.',import.meta.url));
const args=Object.fromEntries(process.argv.slice(2).map(arg=>arg.replace(/^--/,'').split(/=(.*)/s).slice(0,2)));
const port=Number(args.port||8797), upstream=args['model-server'];
const mime={'.mjs':'text/javascript','.wasm':'application/wasm','.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.png':'image/png','.md':'text/markdown'};
createServer(async(req,res)=>{
 try{
  const url=new URL(req.url,'http://localhost');
  if(upstream&&((req.method==='GET'&&url.pathname==='/model/api/state')||(req.method==='POST'&&url.pathname==='/model/api/chat'))){
   const chunks=[];for await(const chunk of req)chunks.push(chunk);
   const response=await fetch(upstream.replace(/\/$/,'')+url.pathname.replace('/model',''),{method:req.method,headers:{'Content-Type':'application/json'},...(req.method==='POST'?{body:Buffer.concat(chunks)}:{})});
   res.writeHead(response.status,{'Content-Type':response.headers.get('content-type')||'application/json'});
   await pipeline(Readable.fromWeb(response.body),res);return;
  }
  if(req.method!=='GET'&&req.method!=='HEAD'){res.writeHead(405);res.end();return;}
  let path=resolve(root,'.'+decodeURIComponent(url.pathname));
  if(path!==root.slice(0,-1)&&!path.startsWith(root)){res.writeHead(403);res.end();return;}
  let info=await stat(path);
  if(info.isDirectory()){path=resolve(path,'index.html');info=await stat(path);}
  res.writeHead(200,{'Content-Type':mime[extname(path)]||'application/octet-stream','Content-Length':info.size,'Cache-Control':'no-cache'});
  if(req.method==='HEAD')res.end();else await pipeline(createReadStream(path),res);
 }catch(error){if(!res.headersSent){res.writeHead(error.code==='ENOENT'?404:502,{'Content-Type':'application/json'});res.end(JSON.stringify({ok:false,error:error.message}));}else res.end();}
}).listen(port,'127.0.0.1',()=>console.log(`http://127.0.0.1:${port}${upstream?' · local model bridge':''}`));
