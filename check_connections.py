import gzip
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from connections import RECORD, Links, calculate, weights


def expected(read, side, layer, neuron, layers, n):
    values = []
    if side == 0:
        norm = 1 + read(layer, 'post_attention_layernorm')
        gate = read(layer, 'mlp.gate_proj')[neuron] * norm
        up = read(layer, 'mlp.up_proj')[neuron] * norm
        for i in range(layer):
            down = read(i, 'mlp.down_proj')
            values.append(((gate @ down) * (up @ down)).numpy())
        offset = 0
    else:
        down = read(layer, 'mlp.down_proj')[:, neuron]
        for j in range(layer + 1, layers):
            v = (1 + read(j, 'post_attention_layernorm')) * down
            values.append(((read(j, 'mlp.gate_proj') @ v) * (read(j, 'mlp.up_proj') @ v)).numpy())
        offset = (layer + 1) * n
    raw = np.concatenate(values)
    magnitude = np.abs(raw).astype(np.float64)
    top = np.argsort(-magnitude)[:24]
    return top + offset, (magnitude[top] - magnitude.mean()) / magnitude.std(), np.sign(raw[top])


def check_builder():
    rng = np.random.default_rng(4)
    layers, n, k, dim = 4, 11, 3, 7
    tensors = {(l, key): torch.tensor(rng.normal(size=shape), dtype=torch.float32)
               for l in range(layers) for key, shape in [('mlp.down_proj', (dim, n)),
               ('mlp.gate_proj', (n, dim)), ('mlp.up_proj', (n, dim)), ('post_attention_layernorm', (dim,))]}
    read = lambda layer, key: tensors[layer, key]
    raw = {}
    for j in range(1, layers):
        for i in range(j):
            v = (1 + read(j, 'post_attention_layernorm'))[:, None] * read(i, 'mlp.down_proj')
            raw[i, j] = ((read(j, 'mlp.gate_proj') @ v) * (read(j, 'mlp.up_proj') @ v)).numpy()
    for chunk in [1, 4, 11]:
        links = Links(layers, n, k)
        calculate(read, SimpleNamespace(chunk=chunk), links, 'cpu')
        with tempfile.TemporaryDirectory() as temp:
            for layer in range(layers):
                links.write(Path(temp), layer)
                saved = np.fromfile(Path(temp) / f'L{layer}.bin', dtype=RECORD).reshape(2, n, k)
                for side in range(2):
                    eligible = range(layer) if side == 0 else range(layer + 1, layers)
                    if not eligible:
                        assert (saved[side]['layer'] == -1).all()
                        continue
                    for neuron in range(n):
                        values = np.concatenate([raw[i, layer][neuron] if side == 0 else raw[layer, i][:, neuron] for i in eligible])
                        magnitude = np.abs(values).astype(float)
                        top = np.argsort(-magnitude)[:k]
                        indices = np.array(list(eligible))[top // n] * n + top % n
                        actual = saved[side, neuron]
                        np.testing.assert_array_equal(actual['layer'].astype(int) * n + actual['neuron'], indices)
                        np.testing.assert_array_equal(actual['sign'], np.sign(values[top]))
                        np.testing.assert_allclose(actual['z'], (magnitude[top] - magnitude.mean()) / magnitude.std(), rtol=.002)


def main():
    check_builder()
    folder = Path('explorer-next/public/data/connections')
    meta = json.loads((folder / 'manifest.json').read_text())
    n, layers, k = meta['neurons'], meta['layers'], meta['k']
    for layer in range(layers):
        data = np.fromfile(folder / f'L{layer}.bin', dtype=RECORD).reshape(2, n, k)
        assert np.isfinite(data['z']).all()
        for side in range(2):
            links = data[side]
            if (side == 0 and layer == 0) or (side == 1 and layer == layers - 1):
                assert (links['layer'] == -1).all()
                continue
            assert (links['layer'] >= 0).all() and (links['layer'] < layers).all()
            assert ((links['layer'] < layer) if side == 0 else (links['layer'] > layer)).all()
            assert (links['neuron'] >= 0).all() and (links['neuron'] < n).all()
            assert np.isin(links['sign'], [-1, 1]).all()
            assert (np.diff(links['z'].astype(float), axis=-1) <= 0).all()
            keys = np.sort(links['layer'].astype(int) * n + links['neuron'], axis=-1)
            assert (np.diff(keys, axis=-1) > 0).all()
    read = weights(meta['model'])
    report = dict(model=meta['model'], seconds=meta['seconds'], bytes=sum(p.stat().st_size for p in folder.iterdir()),
                  selected=dict(layer=17, neuron=5244), checks=['chunked builder matches full-matrix reference on all tiny graph neurons with chunk 1, 4, 11', 'all files: bounds, signs, ordering, unique links, empty boundaries'])
    data = np.fromfile(folder / 'L17.bin', dtype=RECORD).reshape(2, n, k)
    torch.set_num_threads(4)
    with torch.inference_mode():
        for side, label in enumerate(['upstream', 'downstream']):
            indices, z, signs = expected(read, side, 17, 5244, layers, n)
            actual = data[side, 5244]
            np.testing.assert_array_equal(actual['layer'].astype(int) * n + actual['neuron'], indices)
            np.testing.assert_array_equal(actual['sign'], signs)
            np.testing.assert_allclose(actual['z'], z, rtol=.001, atol=.02)
            report['checks'].append(f'L17 N5244 {label}: all 24 match independent CPU matrix-vector ranking, sign and z')
            rows = []
            for link in actual[:5]:
                layer, neuron = int(link['layer']), int(link['neuron'])
                with gzip.open(f'explorer-next/public/data/L{layer}.json.gz') as f:
                    words = json.load(f)['rows'][neuron]['up'][:5]
                rows.append(dict(layer=layer, neuron=neuron, sign=int(link['sign']), z=float(link['z']), words=words))
            report[label] = rows
    path = Path('explorer-next/shots/web/data-checks.json')
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
