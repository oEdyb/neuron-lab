import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


RECORD = np.dtype([('layer', 'i1'), ('neuron', '<i2'), ('z', '<f2'), ('sign', 'i1')])


def weights(model):
    folder = Path(snapshot_download(model))
    index_path = folder / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())['weight_map']
    else:
        with safe_open(folder / 'model.safetensors', framework='pt') as f:
            index = dict.fromkeys(f.keys(), 'model.safetensors')

    def read(layer, name):
        key = f'model.language_model.layers.{layer}.{name}.weight'
        with safe_open(folder / index[key], framework='pt') as f:
            return f.get_tensor(key).float()

    return read


def merge(scores, ids, values, indices, k):
    values = np.concatenate((scores, values), axis=-1)
    indices = np.concatenate((ids, indices), axis=-1)
    keep = np.argpartition(values, -k, axis=-1)[..., -k:]
    scores[:] = np.take_along_axis(values, keep, -1)
    ids[:] = np.take_along_axis(indices, keep, -1)


class Links:
    def __init__(self, layers, n, k, counts=None):
        self.layers, self.n, self.k = layers, n, k
        self.counts_per_layer = counts or [n] * layers
        shape = (2, layers, n)
        self.scores = np.zeros((*shape, k), dtype=np.float32)
        self.ids = np.full((*shape, k), -1, dtype=np.int32)
        self.totals = np.zeros((*shape, 2), dtype=np.float64)
        self.counts = np.zeros((2, layers), dtype=np.int64)

    def add(self, raw, i, j, lo):
        magnitude = np.abs(raw)
        self.totals[1, i, lo:lo + raw.shape[1], 0] += magnitude.sum(0, dtype=np.float64)
        self.totals[1, i, lo:lo + raw.shape[1], 1] += np.square(magnitude, dtype=np.float64).sum(0)
        self.totals[0, j, :raw.shape[0], 0] += magnitude.sum(1, dtype=np.float64)
        self.totals[0, j, :raw.shape[0], 1] += np.square(magnitude, dtype=np.float64).sum(1)
        b = np.argpartition(magnitude, -self.k, axis=0)[-self.k:]
        val = np.take_along_axis(magnitude, b, 0).T
        sign = np.take_along_axis(raw, b, 0).T < 0
        encoded = (j * self.n + b.T) * 2 + sign
        merge(self.scores[1, i, lo:lo + raw.shape[1]], self.ids[1, i, lo:lo + raw.shape[1]], val, encoded, self.k)
        k = min(self.k, raw.shape[1])
        b = np.argpartition(magnitude, -k, axis=1)[:, -k:]
        val = np.take_along_axis(magnitude, b, 1)
        sign = np.take_along_axis(raw, b, 1) < 0
        encoded = (i * self.n + lo + b) * 2 + sign
        merge(self.scores[0, j, :raw.shape[0]], self.ids[0, j, :raw.shape[0]], val, encoded, self.k)

    def add_tensor(self, raw, i, j, lo):
        magnitude = raw.abs()
        precise = magnitude.double()
        width, count = raw.shape[1], raw.shape[0]
        for side, layer, selection, dimension in ((1, i, slice(lo, lo + width), 0), (0, j, slice(0, count), 1)):
            self.totals[side, layer, selection, 0] += precise.sum(dimension).cpu().numpy()
            self.totals[side, layer, selection, 1] += precise.square().sum(dimension).cpu().numpy()
        values, indices = magnitude.topk(self.k, dim=0)
        negative = raw.gather(0, indices) < 0
        encoded = ((j * self.n + indices) * 2 + negative).T.cpu().numpy()
        merge(self.scores[1, i, lo:lo + width], self.ids[1, i, lo:lo + width], values.T.cpu().numpy(), encoded, self.k)
        values, indices = magnitude.topk(min(self.k, width), dim=1)
        negative = raw.gather(1, indices) < 0
        encoded = ((i * self.n + lo + indices) * 2 + negative).cpu().numpy()
        merge(self.scores[0, j, :count], self.ids[0, j, :count], values.cpu().numpy(), encoded, self.k)

    def write(self, folder, layer):
        records = np.zeros((2, self.n, self.k), dtype=RECORD)
        for side in range(2):
            order = np.argsort(-self.scores[side, layer], axis=-1, kind='stable')
            strength = np.take_along_axis(self.scores[side, layer], order, -1)
            index = np.take_along_axis(self.ids[side, layer], order, -1)
            count = max(1, self.counts[side, layer])
            mean = self.totals[side, layer, :, 0] / count
            std = np.sqrt(np.maximum(0, self.totals[side, layer, :, 1] / count - mean ** 2))
            z = np.divide(strength - mean[:, None], std[:, None], out=np.zeros_like(strength), where=std[:, None] > 0)
            valid = index >= 0
            records[side]['layer'] = np.where(valid, index // 2 // self.n, -1)
            records[side]['neuron'] = np.where(valid, index // 2 % self.n, -1)
            records[side]['z'] = np.where(valid, z, 0)
            records[side]['sign'] = np.where(valid, np.where(index % 2, -1, 1), 0)
        path = folder / f'L{layer}.bin'
        records[:, :self.counts_per_layer[layer]].tofile(str(path) + '.tmp')
        os.replace(str(path) + '.tmp', path)


def calculate(read, a, links, dev):
    start = time.time()
    with torch.inference_mode():
        for j in range(1, links.layers):
            gate = read(j, 'mlp.gate_proj').to(dev)
            up = read(j, 'mlp.up_proj').to(dev)
            norm_name = 'pre_feedforward_layernorm' if a.model.startswith('google/gemma') else 'post_attention_layernorm'
            norm = (1 + read(j, norm_name)).to(dev)
            for i in range(j):
                down = read(i, 'mlp.down_proj').to(dev)
                for lo in range(0, links.counts_per_layer[i], a.chunk):
                    hi = min(links.counts_per_layer[i], lo + a.chunk)
                    v = norm[:, None] * down[:, lo:hi]
                    p = (gate @ v) * (up @ v)
                    if dev == 'cuda':
                        links.add_tensor(p, i, j, lo)
                    else:
                        links.add(p.cpu().numpy(), i, j, lo)
                links.counts[1, i] += links.counts_per_layer[j]
                links.counts[0, j] += links.counts_per_layer[i]
                del down, v, p
            print(f'L{j}: {time.time() - start:.1f}s', flush=True)
            del gate, up, norm
            if dev == 'mps':
                torch.mps.empty_cache()


def build(a):
    torch.set_num_threads(4)
    read = weights(a.model)
    config = json.loads((Path(snapshot_download(a.model)) / 'config.json').read_text())['text_config']
    layers = config['num_hidden_layers']
    counts = [read(l, 'mlp.down_proj').shape[1] for l in range(layers)]
    n = max(counts)
    if a.k > n:
        raise ValueError('k must not exceed the number of neurons')
    links = Links(layers, n, a.k, counts)
    dev = a.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    start = time.time()
    calculate(read, a, links, dev)
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=True)
    for layer in range(layers):
        links.write(folder, layer)
    meta = dict(version=1, model=a.model, layers=layers, neurons=n, counts=counts, k=a.k, record_bytes=6,
                order=['upstream', 'downstream'], device=dev, seconds=round(time.time() - start, 2),
                method='(gate @ ((1 + pre_mlp_norm) * down)) * (up @ ((1 + pre_mlp_norm) * down))',
                ranking='global |P| z-score per neuron over all eligible layers; population standard deviation',
                estimate='Weights only; ignores attention, activation nonlinearity and input-dependent RMS scale.')
    (folder / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser(description='Build direct-path weight estimates for the Web view.')
    ap.add_argument('--model', default='Qwen/Qwen3.5-2B')
    ap.add_argument('--out', default='explorer-next/public/data/connections')
    ap.add_argument('--k', type=int, default=24)
    ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    a = ap.parse_args()
    if a.k < 1 or a.chunk < 1:
        ap.error('k and chunk must be positive')
    build(a)


if __name__ == '__main__':
    main()
