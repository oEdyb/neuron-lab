import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import VIEW_MODEL, VIEW_GRID, VIEW_DOTS
import map_data

router = APIRouter()
ROOT = Path(__file__).parent / 'maps' / VIEW_MODEL.split('/')[-1] / 'view'


@lru_cache(maxsize=24)
def layer_rows(layer, stamp):
    path = ROOT / f'L{layer}.json'
    if not path.exists():
        path = map_data.data_dir() / f'L{layer}.json.gz'
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())


@router.get('/api/view/search')
def search(q: str = ''):
    q = q.strip().casefold()
    if len(q) < 2:
        return {'rows': []}
    matches = []
    manifest = ROOT / 'manifest.json'
    if not manifest.exists():
        manifest = map_data.data_dir() / 'manifest.json'
    if not manifest.exists():
        return {'rows': []}
    for item in json.loads(manifest.read_text())['layers']:
        layer = item['layer']
        path = ROOT / f'L{layer}.json'
        if not path.exists():
            path = map_data.data_dir() / f'L{layer}.json.gz'
        data = layer_rows(layer, path.stat().st_mtime_ns)
        for n, row in enumerate(data['rows']):
            if any(q in word.casefold() for word in row['up'] + row['down']):
                score = max(row['kl'][i] / max(data['p95'][i], 1e-9) for i in range(2))
                matches.append({'layer': layer, 'n': n, 'words': row['up'][:4], 'score': score})
    return {'rows': sorted(matches, key=lambda r: -r['score'])[:30]}


@router.get('/api/view/{name}')
def data(name: str):
    if name != 'manifest.json' and not re.fullmatch(r'L\d+\.(bin|json)', name):
        raise HTTPException(404)
    path = ROOT / name
    if not path.exists():
        path = map_data.data_dir() / name
    if not path.exists() and name.endswith('.json') and name != 'manifest.json':
        path = path.with_suffix('.json.gz')
        if path.exists():
            return json.loads(gzip.decompress(path.read_bytes()))
    if not path.exists():
        raise HTTPException(404, f"Map isn't downloaded. Run: {map_data.download_command()}")
    if name == 'manifest.json':
        return json.loads(path.read_text()) | {'grid': VIEW_GRID, 'dots': VIEW_DOTS}
    compressed = path.with_suffix(path.suffix + '.gz')
    if compressed.exists() and compressed.stat().st_mtime >= path.stat().st_mtime:
        return FileResponse(compressed, media_type='application/json' if name.endswith('.json') else 'application/octet-stream', headers={'Content-Encoding': 'gzip'})
    return FileResponse(path)
