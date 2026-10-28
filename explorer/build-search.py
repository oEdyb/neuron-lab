import json,gzip,array
from pathlib import Path
p=Path('data');m=json.loads((p/'manifest.json').read_text());index={};strength=array.array('f');offset=0
for layer in m['layers']:
 d=json.load(gzip.open(p/f"L{layer['layer']}.json.gz"))
 for n,row in enumerate(d['rows']):
  for sign in range(2):
   strength.append(row['kl'][sign]/d['p95'][sign])
   direct={word.strip().lower() for word in row['up' if sign==0 else 'down'] if word.strip()}
   persona={word.strip().lower() for word in row.get('persona',[[],[]])[sign] if word.strip()}
   for word in direct:index.setdefault(word,[]).append((offset+n)*2+sign)
   for word in persona-direct:index.setdefault(word,[]).append((offset+n)*2+sign+sum(l['count'] for l in m['layers'])*2)
 offset+=len(d['rows'])
with gzip.open(p/'signed-search.json.gz','wt') as f:json.dump(index,f,ensure_ascii=False,separators=(',',':'))
(p/'signed-strength.bin').write_bytes(strength.tobytes())
print(len(index),'signed words',len(strength),'signed strengths')
