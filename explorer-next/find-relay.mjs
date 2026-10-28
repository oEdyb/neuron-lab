import { request } from 'node:http'
import { randomUUID } from 'node:crypto'
const jobs=new Map()
class FindError extends Error{constructor(message,status){super(message);this.status=status}}
function forward(upstream,body){return new Promise((resolve,reject)=>{const req=request(new URL('/api/find',upstream),{method:'POST',headers:{'Content-Type':'application/json','Content-Length':body.length},timeout:0},res=>{const chunks=[];res.on('data',c=>chunks.push(c));res.on('error',reject);res.on('end',()=>{try{const result=JSON.parse(Buffer.concat(chunks).toString());if(res.statusCode>=400||!result.ok)reject(new FindError([result.error,result.fix].filter(Boolean).join(' ')||`Find failed (${res.statusCode})`,res.statusCode>=400?res.statusCode:502));else resolve(result)}catch(e){reject(e)}})});req.on('error',reject);req.end(body)})}
export async function findRelay(req,res,url,upstream){
 const match=url.pathname.match(/^\/model\/api\/find\/([a-f0-9-]+)$/)
 if(req.method==='GET'&&match){const job=jobs.get(match[1]);res.writeHead(job?200:404,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(job??{error:'Run unavailable'}));return true}
 if(req.method!=='POST'||url.pathname!=='/model/api/find')return false
 const chunks=[];for await(const c of req)chunks.push(c)
 const body=Buffer.concat(chunks)
 if(req.headers.prefer==='respond-async'){
  const id=randomUUID();jobs.set(id,{status:'running'})
  forward(upstream,body).then(result=>jobs.set(id,{status:'done',result}),error=>jobs.set(id,{status:'error',error:error.message})).finally(()=>{const timer=setTimeout(()=>jobs.delete(id),24*3600*1000);timer.unref()})
  res.writeHead(202,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify({id}));return true
 }
 const result=await forward(upstream,body);res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(result));return true
}
