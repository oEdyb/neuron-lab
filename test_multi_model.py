import gc
import gzip
import json
import weakref

from fastapi.testclient import TestClient

import map_data
import library
import view_api
import server
from config import VIEW_MODELS


def test_switch_updates_map_routes_and_releases_previous_model(tmp_path, monkeypatch):
    monkeypatch.setenv('NEURON_LAB_MAPS', str(tmp_path))
    monkeypatch.setattr(library, 'MAPS', tmp_path / 'absent')
    monkeypatch.setattr(view_api, 'view_root', lambda: tmp_path / 'absent-view')
    monkeypatch.setattr(map_data, 'active_model', VIEW_MODELS[0])
    monkeypatch.setattr(server, 'engine', server.Engine())
    previous = []

    class Lab:
        def __init__(self, model, bf16=False):
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


def test_connection_records_use_each_layers_width(tmp_path):
    import numpy as np
    from connections import Links, RECORD

    links = Links(2, 4, 1, counts=[2, 4])
    links.add(np.array([[-1, 2], [3, -4], [5, 6], [7, -8]], dtype=np.float32), 0, 1, 0)
    links.counts[1, 0] = 4
    links.counts[0, 1] = 2
    for layer, count in enumerate((2, 4)):
        links.write(tmp_path, layer)
        assert (tmp_path / f'L{layer}.bin').stat().st_size == 2 * count * 6
    upstream = np.fromfile(tmp_path / 'L1.bin', dtype=RECORD).reshape(2, 4, 1)
    assert upstream[0, 3, 0]['layer'] == 0
    assert upstream[0, 3, 0]['neuron'] == 1
    assert upstream[0, 3, 0]['sign'] == -1


def test_tensor_connection_ranking_matches_numpy():
    import numpy as np
    import torch
    from connections import Links

    raw = np.random.default_rng(4).normal(size=(17, 12)).astype(np.float32)
    reference, tensor = (Links(2, 17, 3, counts=[12, 17]) for _ in range(2))
    reference.add(raw, 0, 1, 0)
    tensor.add_tensor(torch.from_numpy(raw), 0, 1, 0)
    np.testing.assert_array_equal(tensor.ids, reference.ids)
    np.testing.assert_array_equal(tensor.scores, reference.scores)
    np.testing.assert_allclose(tensor.totals, reference.totals, atol=1e-12)
