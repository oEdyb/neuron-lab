import gzip
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

from config import VIEW_MODEL, VIEW_WORDS, VIEW_GRID, VIEW_DOTS, MINISTRAL_BF16


def build():
    root = Path(__file__).parent / 'maps' / VIEW_MODEL.split('/')[-1]
    meta = json.loads((root / 'meta.json').read_text())
    out = root / 'view'
    out.mkdir(exist_ok=True)
    weights_id = meta.get('weights', VIEW_MODEL)
    loading = {'revision': meta['revision']} if meta.get('revision') else {}
    tokenizer_options = {'fix_mistral_regex': True} if weights_id == MINISTRAL_BF16 else {}
    tok = AutoTokenizer.from_pretrained(weights_id, local_files_only=True, **loading, **tokenizer_options)
    snapshot = Path(snapshot_download(weights_id, local_files_only=True, **loading))
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

    layers, summaries, signed_strength, signed_index = [], [], [], {}
    offset = 0
    for layer in range(meta['layers']):
        source = root / f'L{layer}.npz'
        if not source.exists():
            continue
        geometry, detail = out / f'L{layer}.bin', out / f'L{layer}.json'
        if not geometry.exists() or not detail.exists() or min(geometry.stat().st_mtime, detail.stat().st_mtime) < source.stat().st_mtime:
            a = np.load(source)
            keys = [key for key in index if key.endswith(f'.layers.{layer}.mlp.down_proj.weight')]
            if len(keys) != 1:
                raise ValueError(f'L{layer}: expected one text down_proj tensor, found {keys}')
            key = keys[0]
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
        data = json.loads(detail.read_text())
        count = len(data['rows'])
        strengths = np.array([[row['kl'][s] / max(data['p95'][s], 1e-9) for s in range(2)]
                              for row in data['rows']], dtype='<f4')
        signed_strength.append(strengths.reshape(-1))
        for n, row in enumerate(data['rows']):
            for s, name in enumerate(('up', 'down')):
                for word in set(row[name]):
                    signed_index.setdefault(word.lower(), []).append((offset + n) * 2 + s)
        strongest = np.argsort(-strengths.max(1))[:16]
        summaries.append(dict(layer=layer, count=count, above=int((strengths.max(1) > 1).sum()),
                              top=[dict(n=int(n), **data['rows'][n]) for n in strongest]))
        layers.append({'layer': layer, 'count': count})
        offset += count
    manifest = dict(model=VIEW_MODEL, layers=layers, total_layers=meta['layers'], neurons=meta['neurons'], dose=meta['dose'], grid=VIEW_GRID, dots=VIEW_DOTS, projection='PCA of down_proj columns, independently scaled per layer')
    manifest.update(summaries=summaries, capabilities=dict(exact_search=True, semantic_search=False, connections=False))
    (out / 'signed-search.json.gz').write_bytes(gzip.compress(json.dumps(signed_index, ensure_ascii=False,
                                                            separators=(',', ':')).encode()))
    np.concatenate(signed_strength).astype('<f4').tofile(out / 'signed-strength.bin')
    (out / 'manifest.json').write_text(json.dumps(manifest))
    print(json.dumps(manifest))


if __name__ == '__main__':
    build()
