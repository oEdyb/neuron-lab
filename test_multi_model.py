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


def fake_lab(monkeypatch):
    class Lab:
        def __init__(self, model, bf16=False):
            self.model_id, self.dev, self.n_layers, self.n_neurons = model, 'cpu', 1, 1

        def norm_batch(self):
            return ()

        def stream_norms(self):
            return []

    monkeypatch.setattr(server, 'Lab', Lab)
    monkeypatch.setattr(server, 'engine', server.Engine())


def write_map(root, model):
    (root / 'connections').mkdir(parents=True)
    (root / 'manifest.json').write_text(json.dumps(dict(model=model, layers=[dict(layer=0, count=1)])))
    for name in ('signed-search.json.gz', 'signed-strength.bin', 'vector-words.json.gz', 'word-vectors.bin',
                 'connections/manifest.json', 'connections/L0.bin', 'L0.bin', 'L0.json.gz'):
        (root / name).touch()


def test_bf16_id_uses_the_ministral_map_not_qwens(tmp_path, monkeypatch):
    from config import MINISTRAL, MINISTRAL_BF16
    fake_lab(monkeypatch)
    monkeypatch.setenv('NEURON_LAB_MAPS', str(tmp_path))
    monkeypatch.setattr(map_data, 'active_model', 'Qwen/Qwen3.5-2B')
    write_map(tmp_path / 'Qwen3.5-2B', 'Qwen/Qwen3.5-2B')
    write_map(tmp_path / MINISTRAL.split('/')[-1], MINISTRAL)
    server.engine.load(MINISTRAL_BF16)
    assert map_data.active_model == MINISTRAL
    assert server.engine.lab.model_id == MINISTRAL


def test_unmapped_model_is_refused_instead_of_borrowing_another_map(tmp_path, monkeypatch):
    fake_lab(monkeypatch)
    monkeypatch.setenv('NEURON_LAB_MAPS', str(tmp_path))
    monkeypatch.setattr(map_data, 'active_model', 'Qwen/Qwen3.5-2B')
    write_map(tmp_path / 'Qwen3.5-2B', 'Qwen/Qwen3.5-2B')
    try:
        server.engine.load('Qwen/Qwen3.5-4B')
    except server.Fail as e:
        assert e.code == 'map_missing' and 'Qwen3.5-4B' in e.fix
    else:
        raise AssertionError('Qwen 4B loaded with the 2B map')
    assert map_data.active_model == 'Qwen/Qwen3.5-2B'
    assert server.engine.lab is None


def test_failed_switch_keeps_the_running_model(tmp_path, monkeypatch):
    from config import MINISTRAL
    fake_lab(monkeypatch)
    monkeypatch.setenv('NEURON_LAB_MAPS', str(tmp_path))
    monkeypatch.setattr(map_data, 'active_model', 'Qwen/Qwen3.5-2B')
    write_map(tmp_path / 'Qwen3.5-2B', 'Qwen/Qwen3.5-2B')
    server.engine.load('Qwen/Qwen3.5-2B')
    with TestClient(server.app) as client:
        result = client.post('/api/load', json=dict(model=MINISTRAL))
        assert result.status_code == 503
        assert result.json()['code'] == 'map_missing' and MINISTRAL in result.json()['fix']
        state = client.get('/api/state').json()
    assert state['model'] == 'Qwen/Qwen3.5-2B' and state['configured_model'] == 'Qwen/Qwen3.5-2B'


def test_server_starts_without_a_map_and_says_how_to_fix_it(tmp_path, monkeypatch, capsys):
    from config import MINISTRAL
    fake_lab(monkeypatch)
    monkeypatch.setenv('NEURON_LAB_MAPS', str(tmp_path))
    monkeypatch.setattr(server.sys, 'argv', ['server.py', '--model', MINISTRAL, '--no-open'])
    monkeypatch.setattr(server.uvicorn, 'run', lambda *a, **k: None)
    monkeypatch.setattr(map_data, 'active_model', 'Qwen/Qwen3.5-2B')
    server.main()
    out = capsys.readouterr().out
    assert f'No complete map for {MINISTRAL}' in out and map_data.download_command(MINISTRAL) in out
    assert map_data.active_model == MINISTRAL
    assert server.engine.lab is None


def test_library_entry_without_model_uses_the_active_model(tmp_path, monkeypatch):
    from config import MINISTRAL, MINISTRAL_BF16
    monkeypatch.setattr(map_data, 'active_model', MINISTRAL)
    assert library.validate(dict(layer=1, n=2, dose=.5))['model'] == MINISTRAL
    assert library.validate(dict(model=MINISTRAL_BF16, layer=1, n=2, dose=.5))['model'] == MINISTRAL


def test_seed_entries_stay_on_their_own_model(monkeypatch):
    from config import MINISTRAL
    monkeypatch.setattr(map_data, 'active_model', MINISTRAL)
    assert all(entry['model'] == library.VIEW_MODEL for entry in library.seeds())


def test_raw_map_without_a_cached_tokenizer_still_answers(tmp_path, monkeypatch):
    import numpy as np
    from config import MINISTRAL

    def no_tokenizer(model):
        raise OSError('not cached')

    name = MINISTRAL.split('/')[-1]
    view = tmp_path / name / 'view'
    view.mkdir(parents=True)
    (view / 'manifest.json').write_text(json.dumps(dict(model=MINISTRAL, layers=[dict(layer=0, count=1)])))
    (view / 'L0.json').write_text(json.dumps(dict(p95=[1, 1], rows=[dict(up=['a'], down=['b'], kl=[1, 1])])))
    np.savez(tmp_path / name / 'L0.npz', up_frame_tok=np.zeros((1, 1, 1), int), down_frame_tok=np.zeros((1, 1, 1), int))
    monkeypatch.setattr(library, 'MAPS', tmp_path)
    monkeypatch.setattr(library, 'tokenizer', no_tokenizer)
    monkeypatch.setattr(map_data, 'active_model', MINISTRAL)
    result = library.neuron(0, 0)
    assert result['sentence_words_available'] is False
    assert result['signs'][0]['words'] == ['a']
