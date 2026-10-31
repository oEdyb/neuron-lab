import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from build_view import build
from connections import build as build_connections


def indexes(root, device):
    manifest = json.loads((root / 'manifest.json').read_text())
    index = defaultdict(set)
    strengths = []
    offset = 0
    total = sum(item['count'] for item in manifest['layers']) * 2
    for item in manifest['layers']:
        data = json.loads(gzip.decompress((root / f"L{item['layer']}.json.gz").read_bytes()))
        for n, row in enumerate(data['rows']):
            for j, sign in enumerate(('up', 'down')):
                measurement = (offset + n) * 2 + j
                strengths.append(row['kl'][j] / max(data['p95'][j], 1e-9))
                for word in row[sign]:
                    index[word].add(measurement)
                for word in row.get('persona', [[], []])[j]:
                    index[word].add(measurement + total)
        offset += item['count']
    vocabulary = sorted(index)
    (root / 'signed-search.json.gz').write_bytes(gzip.compress(json.dumps({w: sorted(index[w]) for w in vocabulary}, ensure_ascii=False, separators=(',', ':')).encode()))
    np.array(strengths, dtype='<f4').tofile(root / 'signed-strength.bin')
    (root / 'vector-words.json.gz').write_bytes(gzip.compress(json.dumps(vocabulary, ensure_ascii=False).encode()))
    name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    with (root / 'word-vectors.bin').open('wb') as f, torch.inference_mode():
        for lo in range(0, len(vocabulary), 256):
            batch = tokenizer([w.strip() or ' ' for w in vocabulary[lo:lo + 256]], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            normalized = torch.nn.functional.normalize(pooled, dim=1)
            (normalized.cpu().numpy() * 127).round().clip(-127, 127).astype('i1').tofile(f)
    print(f'{len(vocabulary)} vector words', flush=True)


def main():
    ap = argparse.ArgumentParser(description='Build complete downloadable explorer data.')
    ap.add_argument('--model', required=True)
    ap.add_argument('--maps', type=Path, default=Path('maps'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    build(a.model, a.maps)
    root = a.maps / a.model.split('/')[-1] / 'view'
    indexes(root, a.device)
    build_connections(argparse.Namespace(model=a.model, out=str(root / 'connections'), device=a.device, k=24, chunk=512))


if __name__ == '__main__':
    main()
