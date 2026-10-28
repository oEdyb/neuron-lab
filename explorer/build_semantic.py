import urllib.request,zipfile,io,json,gzip
from pathlib import Path
p=Path(__file__).parent/'data'
z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip').read()))
groups=[]
for kind in ['noun','verb','adj','adv']:
 for line in z.read('wordnet/data.'+kind).decode().splitlines():
  if not line or not line[0].isdigit():continue
  fields=line.split('|')[0].split();count=int(fields[3],16);words=[fields[4+i*2].replace('_',' ').lower() for i in range(count)]
  if len(words)>1:groups.append(words)
groups.extend([['death','die','dead','dying','kill','killed','killing','murder','mortality','deceased','funeral','corpse','死','死亡','杀','död'],['happy','happiness','joy','joyful','glad','delighted','幸福','快乐'],['sad','sadness','grief','sorrow','unhappy','悲'],['fear','afraid','scared','terror','horror','恐'],['love','affection','romance','adoration','爱'],['money','cash','wealth','currency','finance','金','钱'],['war','battle','combat','conflict','战争'],['computer','computing','software','code','programming','电脑'],['ocean','sea','water','marine','海']])
with gzip.open(p/'semantic.json.gz','wt',encoding='utf8') as f:json.dump(groups,f,ensure_ascii=False,separators=(',',':'))
(p/'WORDNET-LICENSE.txt').write_bytes(z.read('wordnet/LICENSE'))
print(len(groups),'semantic synonym sets')
