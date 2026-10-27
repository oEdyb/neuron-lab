import gzip
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

from config import VIEW_MODEL, VIEW_WORDS, VIEW_GRID, VIEW_DOTS


def build():
    root = Path(__file__).parent / 'maps' / VIEW_MODEL.split('/')[-1]
    meta = json.loads((root / 'meta.json').read_text())
    out = root / 'view'
    out.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(VIEW_MODEL, local_files_only=True)
    snapshot = Path(snapshot_download(VIEW_MODEL, local_files_only=True))
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    torch.set_num_threads(4)
    words = {}

    def decode(ids):
        result = []
        for i in ids:
            i = int(i)
            if i not in words:
                words[i] = tok.decode([i])
            result.append(words[i])
        return result

    layers = []
    for layer in range(meta['layers']):
        source = root / f'L{layer}.npz'
        if not source.exists():
            continue
        geometry, detail = out / f'L{layer}.bin', out / f'L{layer}.json'
        if not geometry.exists() or not detail.exists() or min(geometry.stat().st_mtime, detail.stat().st_mtime) < source.stat().st_mtime:
            a = np.load(source)
            key = f'model.language_model.layers.{layer}.mlp.down_proj.weight'
            with safe_open(snapshot / index[key], framework='pt') as weights:
                w = weights.get_tensor(key).float().T.contiguous()
            w -= w.mean(0)
            torch.manual_seed(0)
            u, s, _ = torch.pca_lowrank(w, q=6, niter=4, center=False)
            xy = (u[:, :2] * s[:2]).numpy()
            xy /= np.maximum(np.quantile(np.abs(xy), .995, axis=0), 1e-8)
            xy = np.clip(xy, -1.15, 1.15)
            p95 = [float(np.quantile(a[f'null_{sign}_kl'], .95)) for sign in ('up', 'down')]
            ratio = np.maximum(a['up_kl'] / max(p95[0], 1e-9), a['down_kl'] / max(p95[1], 1e-9))
            np.column_stack([xy, ratio]).astype('<f4').tofile(geometry)
            rows = []
            for n in range(len(xy)):
                rows.append({sign: decode(a[f'{sign}_tok'][n, :VIEW_WORDS]) for sign in ('up', 'down') } | {
                    'persona': [decode(a[f'{sign}_persona'][n]) for sign in ('up', 'down')],
                    'kl': [round(float(a[f'{sign}_kl'][n]), 6) for sign in ('up', 'down')],
                    'spread': [[round(float(v), 4) for v in a[f'{sign}_spread'][n]] for sign in ('up', 'down')],
                })
            detail.write_text(json.dumps({'p95': p95, 'rows': rows}, ensure_ascii=False, separators=(',', ':')))
            print(f'L{layer}: {len(xy)} neurons', flush=True)
        for path in (geometry, detail):
            compressed = path.with_suffix(path.suffix + '.gz')
            if not compressed.exists() or compressed.stat().st_mtime < path.stat().st_mtime:
                compressed.write_bytes(gzip.compress(path.read_bytes(), compresslevel=6))
        layers.append({'layer': layer, 'count': geometry.stat().st_size // 12})
    manifest = dict(model=VIEW_MODEL, layers=layers, total_layers=meta['layers'], neurons=meta['neurons'], dose=meta['dose'], grid=VIEW_GRID, dots=VIEW_DOTS, projection='PCA of down_proj columns, independently scaled per layer')
    (out / 'manifest.json').write_text(json.dumps(manifest))
    print(json.dumps(manifest))


if __name__ == '__main__':
    build()
