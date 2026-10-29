import argparse,itertools,json,time
from pathlib import Path
from lab import Lab,generate_compare,steer_vectors

a=argparse.ArgumentParser();a.add_argument('--out',default='out/combo');a.add_argument('--batch',type=int,default=16);a.add_argument('--tokens',type=int,default=150);a.add_argument('--spec');a=a.parse_args()
THEMES={
 'alien':[(6,1536,-1),(4,1453,-1),(19,6030,1),(13,2718,-1),(4,2675,1),(4,1792,-1)],
 'smart':[(18,5155,-1),(14,4530,1),(14,3275,1),(15,3501,1),(16,5565,-1),(9,2531,-1),(14,2307,1)],
}
configs=[]
for theme,picks in THEMES.items():
 for l,n,s in picks:
  for d in (.5,.7):configs.append(dict(theme=theme,pushes=[(l,n,s*d)]))
 for x,y in itertools.combinations(picks[:5],2):
  for d in (.4,.6):configs.append(dict(theme=theme,pushes=[(x[0],x[1],x[2]*d),(y[0],y[1],y[2]*d)]))
configs.append(dict(theme='baseline',pushes=[]))
if a.spec:configs=[dict(theme=c['theme'],pushes=[tuple(p) for p in c['pushes']]) for c in json.load(open(a.spec))]
QS=['Who are you and what do you want?','My dog is called Bosse. Say hi to him!','What is my dog called?','Solve for x: 3x + 7 = 22.']
P=Path(a.out);P.mkdir(parents=True,exist_ok=True)
start=time.time();lab=Lab('Qwen/Qwen3.5-2B');norms=lab.stream_norms(*lab.norm_batch())
for c in configs:c.update(history=[],transcript=[])
for q in QS:
 for j in range(0,len(configs),a.batch):
  batch=configs[j:j+a.batch]
  for c in batch:c['history'].append(dict(role='user',content=q))
  cols=[dict(id=str(i),history=c['history']) for i,c in enumerate(batch)]
  vec=[steer_vectors(lab,c['pushes'],norms) if c['pushes'] else {} for c in batch]
  ev=generate_compare(lab,cols,vec,sample=False,tokens=a.tokens)
  done={e['col']:e for e in ev if e.get('done')}
  for i,c in enumerate(batch):
   e=done[str(i)];c['transcript'].append(dict(q=q,a=e['text'],stopped=e['stopped']));c['history'].append(dict(role='assistant',content=e['text']))
  print('Q',QS.index(q),'batch',j,'elapsed',round(time.time()-start),flush=True)
with (P/'combos.jsonl').open('w') as f:
 for c in configs:c.pop('history');f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('DONE',len(configs),'configs',round(time.time()-start),'s',flush=True)
