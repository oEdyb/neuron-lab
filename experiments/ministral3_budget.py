"""Conservative GPU cost guard: count entire app lifetimes, including queue time."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

RATES = dict(H100=.001097, B200=.001736)
OUT = Path('out')


def check():
    known={}
    for path in OUT.glob('ministral3-*.log'):
        for app_id in re.findall(r'https://modal.com/apps/[^/]+/[^/]+/(ap-[A-Za-z0-9]+)',path.read_text()):
            known[app_id]='B200' if 'b200' in path.name else 'H100'
    apps=json.loads(subprocess.check_output(['uv','run','modal','app','list','--json'],text=True))
    now=datetime.now(timezone.utc)
    rows=[]
    for app in apps:
        if app['app_id'] not in known:
            continue
        gpu=known[app['app_id']]
        start=datetime.fromisoformat(app['created_at'])
        end=datetime.fromisoformat(app['stopped_at']) if app['stopped_at'] else now
        seconds=max(0,(end-start).total_seconds())
        rows.append(dict(app=app['app_id'],gpu=gpu,created_at=app['created_at'],
                         stopped_at=app['stopped_at'],seconds=seconds,upper_usd=seconds*RATES[gpu]))
    if '--stop-map-after-l0' in sys.argv:
        log=(OUT/'ministral3-map.log').read_text()
        original=re.search(r'/main/(ap-[A-Za-z0-9]+)',log).group(1)
        if re.search(r'^L0: [0-9]+s',log,re.M) and any(r['app']==original and r['stopped_at'] is None for r in rows):
            listing=subprocess.run(['uv','run','modal','volume','ls','neuron-lab-hf',
                'maps-ministral3/Ministral-3-3B-Instruct-2512/L0.npz','--json'],capture_output=True,text=True)
            if listing.returncode==0 and json.loads(listing.stdout):
                subprocess.run(['uv','run','modal','app','stop','--yes',original],check=True)
                print('L0 persisted; stopped original all-layer worker',flush=True)
    total=sum(r['upper_usd'] for r in rows)
    active=[r for r in rows if r['stopped_at'] is None]
    reserve=max(.4,90*sum(RATES[r['gpu']] for r in active)+.1)
    threshold=20-reserve
    report=dict(as_of=now.isoformat(),gpu_upper_usd=total,stop_threshold_usd=threshold,
                rates_per_second=RATES,apps=rows,
                method='Entire app lifetime including queue, startup and shutdown; GPU-only upper bound, not invoice.')
    (OUT/'ministral3-spend.json').write_text(json.dumps(report,indent=2)+'\n')
    if active and total>=threshold:
        for row in rows:
            if row['stopped_at'] is None:
                subprocess.run(['uv','run','modal','app','stop','--yes',row['app']],check=True)
        raise SystemExit('Stopped known task apps before $20 GPU budget')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--log-dir',type=Path,default=OUT)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--stop-map-after-l0',action='store_true')
    args=parser.parse_args()
    OUT=args.log_dir
    OUT.mkdir(parents=True,exist_ok=True)
    while True:
        report=check()
        print(f"GPU upper bound ${report['gpu_upper_usd']:.3f}",flush=True)
        if '--once' in sys.argv:
            break
        time.sleep(30)
