import { createServer } from 'node:http'
import { createReadStream } from 'node:fs'
import { stat } from 'node:fs/promises'
import { resolve, extname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
const args = Object.fromEntries(process.argv.slice(2).map(a=>{const [k,v]=a.replace(/^--/,'').split(/=(.*)/s);return [k,v??true]}))
const root=fileURLToPath(new URL('.',import.meta.url)), upstream=args['model-server']||'http://127.0.0.1:8792'
const vite=args.static?null:await (await import('vite')).createServer({root,server:{middlewareMode:true}})
const types={'.js':'text/javascript','.css':'text/css','.html':'text/html','.json':'application/json','.wasm':'application/wasm','.svg':'image/svg+xml','.png':'image/png'}
createServer(async(req,res)=>{
 try{
  const url=new URL(req.url,'http://localhost')
  if(/^\/(?:api|data)\//.test(url.pathname)&&['GET','POST','PATCH','DELETE'].includes(req.method)){
   const controller=new AbortController()
   res.on('close',()=>{if(!res.writableEnded)controller.abort()})
   const chunks=[];for await(const c of req)chunks.push(c)
   const response=await fetch(upstream+url.pathname+url.search,{method:req.method,headers:{'Content-Type':'application/json',...(req.headers.prefer?{Prefer:req.headers.prefer}:{})},signal:controller.signal,...(['POST','PATCH'].includes(req.method)?{body:Buffer.concat(chunks)}:{})})
   res.writeHead(response.status,{'Content-Type':response.headers.get('content-type')||'application/json','Cache-Control':'no-store'})
   await pipeline(Readable.fromWeb(response.body),res);return
  }
  if(vite){vite.middlewares(req,res);return}
  const directory=resolve(root,'dist'),path=resolve(directory,'.'+decodeURIComponent(url.pathname.endsWith('/')?url.pathname+'index.html':url.pathname))
  if(!path.startsWith(directory+'/')){res.writeHead(403);res.end();return}
  const info=await stat(path);res.writeHead(200,{'Content-Type':types[extname(path)]||'application/octet-stream','Content-Length':info.size})
  if(req.method==='HEAD')res.end();else await pipeline(createReadStream(path),res)
 }catch(error){if(!res.headersSent){res.writeHead(error.code==='ENOENT'?404:error.status||502,{'Content-Type':'application/json'});res.end(JSON.stringify({ok:false,error:error.message}))}else res.end()}
}).listen(Number(args.port||8800),args.host||'127.0.0.1',()=>console.log('Explorer: http://'+(args.host||'127.0.0.1')+':'+Number(args.port||8800)))
