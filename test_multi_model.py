import gc
import gzip
import json
import weakref

from fastapi.testclient import TestClient

import map_data
import server
from config import VIEW_MODELS


def test_switch_updates_map_routes_and_releases_previous_model(tmp_path, monkeypatch):
    monkeypatch.setenv('NEURON_LAB_MAPS', str(tmp_path))
    monkeypatch.setattr(map_data, 'active_model', VIEW_MODELS[0])
    monkeypatch.setattr(server, 'engine', server.Engine())
    previous = []

    class Lab:
        def __init__(self, model):
            gc.collect()
            assert all(ref() is None for ref in previous)
            previous.append(weakref.ref(self))
            self.model_id, self.dev = model, 'cpu'
            self.n_layers = 24 if model == VIEW_MODELS[0] else 35
            self.n_neurons = 6144 if self.n_layers == 24 else [6144] * 15 + [12288] * 20

        def norm_batch(self):
            return ()

        def stream_norms(self):
            return []

    monkeypatch.setattr(server, 'Lab', Lab)
    for model, layers in zip(VIEW_MODELS, (24, 35)):
        root = tmp_path / model.split('/')[-1]
        (root / 'connections').mkdir(parents=True)
        manifest = dict(model=model, layers=[dict(layer=i, count=6144 if i < 15 or layers == 24 else 12288) for i in range(layers)])
        (root / 'manifest.json').write_text(json.dumps(manifest))
        for filename in ('signed-search.json.gz', 'signed-strength.bin', 'vector-words.json.gz', 'word-vectors.bin', 'connections/manifest.json'):
            (root / filename).touch()
        for i in range(layers):
            (root / f'L{i}.bin').write_bytes(model.encode())
            (root / f'connections/L{i}.bin').touch()
            (root / f'L{i}.json.gz').write_bytes(gzip.compress(json.dumps(dict(p95=[1, 1], rows=[dict(up=[model], down=['other'], kl=[2, 1])])).encode()))
    with TestClient(server.app) as client:
        for model, layers in [(VIEW_MODELS[0], 24), (VIEW_MODELS[1], 35), (VIEW_MODELS[0], 24)]:
            result = client.post('/api/load', json=dict(model=model))
            assert result.status_code == 200, result.text
            assert result.json()['model'] == model
            manifest = client.get('/data/manifest.json').json()
            assert manifest['model'] == model
            assert len(manifest['layers']) == layers
            assert client.get('/api/top?limit=1').json()['model'] == model
            assert client.get('/api/neuron?layer=0&n=0').json()['signs'][0]['words'] == [model]
            assert client.get('/api/view/manifest.json').json()['model'] == model
            assert client.get('/api/view/search?q=other').json()['rows']
