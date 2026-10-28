import json
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

import map_data
import server


def fixture_map(root):
    root.mkdir(parents=True)
    (root / 'manifest.json').write_text(json.dumps(dict(model=map_data.VIEW_MODEL, layers=[dict(layer=17)])))
    (root / 'connections').mkdir()
    for name in ('signed-search.json.gz', 'signed-strength.bin', 'vector-words.json.gz', 'word-vectors.bin', 'connections/manifest.json', 'connections/L17.bin'):
        (root / name).write_bytes(b'fixture')
    (root / 'L17.bin').write_bytes(b'plane')
    (root / 'L17.json.gz').write_bytes(b'details')


def test_dataset_layout_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(map_data, 'ROOT', tmp_path)
    monkeypatch.delenv('NEURON_LAB_MAPS', raising=False)
    def download(**kwargs):
        assert kwargs['repo_id'] == map_data.MAP_REPO
        assert kwargs['repo_type'] == 'dataset'
        assert kwargs['token'] is False
        assert kwargs['local_dir'] == tmp_path / 'data'
        fixture_map(tmp_path / 'data' / map_data.MODEL_NAME)
    snapshot = Mock(side_effect=download)
    monkeypatch.setattr(map_data, 'snapshot_download', snapshot)
    assert map_data.ensure_maps()
    assert map_data.ensure_maps()
    assert snapshot.call_count == 1


def test_failed_download_still_serves_ui_and_api(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(map_data, 'ROOT', tmp_path)
    monkeypatch.delenv('NEURON_LAB_MAPS', raising=False)
    monkeypatch.setattr(map_data, 'snapshot_download', Mock(side_effect=RuntimeError('dataset unavailable')))
    assert not map_data.ensure_maps()
    assert 'uv run python map_data.py' in capsys.readouterr().out
    with TestClient(server.app) as client:
        assert 'assets/index-' in client.get('/').text
        assert client.get('/api/maps').json()['available'] is False
        assert client.get('/data/manifest.json').status_code == 404
        assert client.get('/api/library').json()['entries']
        assert client.get('/api/state').json()['model'] is None


def test_local_folder_is_served_without_download(tmp_path, monkeypatch):
    root = tmp_path / 'local'
    fixture_map(root)
    monkeypatch.setenv('NEURON_LAB_MAPS', str(root))
    monkeypatch.setattr(map_data, 'snapshot_download', Mock(side_effect=AssertionError('no network')))
    assert map_data.ensure_maps()
    with TestClient(server.app) as client:
        assert client.get('/data/L17.bin').content == b'plane'
        assert client.get('/data/%2e%2e/server.py').status_code == 404


def test_async_find_contract(monkeypatch):
    async def result(body):
        return dict(ok=True, rows=[dict(layer=17, n=5244)])
    monkeypatch.setattr(server, 'run_find', result)
    with TestClient(server.app) as client:
        response = client.post('/api/find', headers={'Prefer': 'respond-async'}, json={})
        assert response.status_code == 202
        job = client.get('/api/find/' + response.json()['id']).json()
        assert job == dict(status='done', result=dict(ok=True, rows=[dict(layer=17, n=5244)]))
