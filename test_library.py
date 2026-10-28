import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import library
import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(library, 'LIBRARY', tmp_path / 'library.json')
    monkeypatch.setattr(library, 'seeds', lambda: [])
    monkeypatch.setattr(server.engine, 'load', lambda *_: pytest.fail('must not load a model'))
    monkeypatch.setattr(server.engine, 'need', lambda: pytest.fail('must not need a model'))
    root = tmp_path / 'maps/Qwen3.5-2B/view'
    root.mkdir(parents=True)
    monkeypatch.setattr(library, 'MAPS', tmp_path / 'maps')
    monkeypatch.setattr(library, 'PUBLIC_MAP', tmp_path / 'public')
    (root / 'manifest.json').write_text(json.dumps({'layers': [{'layer': 17}]}))
    (root / 'L17.json').write_text(json.dumps({'p95': [2, 4], 'rows': [
        {'up': ['Evil'] + ['word'] * 20, 'down': ['kind'], 'kl': [8, 4], 'persona': [['villain'], ['friend']], 'frame_words': [[['cruel']], [['nice']]]},
        {'up': ['devilish'], 'down': ['EVILISH'], 'kl': [4, 24], 'persona': [[], []]}
    ]}))
    with TestClient(server.app) as c:
        yield c


def add(client, **changes):
    return client.post('/api/library', json=dict(model=library.VIEW_MODEL, layer=17, n=0, dose=.6, name='Evil', source='agent') | changes)


def test_crud_upsert_and_sign(client):
    assert client.get('/api/library').json()['entries'] == []
    assert not library.LIBRARY.exists()
    entry = add(client).json()['entry']
    assert library.LIBRARY.exists()
    assert entry['source'] == 'agent' and entry['added']
    changed = add(client, dose=.4, name='Renamed').json()['entry']
    assert changed['id'] == entry['id'] and changed['dose'] == .4
    minus = add(client, dose=-.6).json()['entry']
    null = add(client, dose=None).json()['entry']
    assert len(client.get('/api/library').json()['entries']) == 3
    assert client.get('/api/library?model=other').json()['entries'] == []
    patched = client.patch('/api/library/'+entry['id'], json={'name': 'New', 'note': 'Test', 'tags': ['a']}).json()['entry']
    assert (patched['name'], patched['note'], patched['tags']) == ('New', 'Test', ['a'])
    assert client.patch('/api/library/'+minus['id'], json={'dose': .3}).status_code == 409
    assert client.delete('/api/library/'+null['id']).status_code == 200
    assert client.delete('/api/library/'+null['id']).status_code == 404
    assert len(json.loads(library.LIBRARY.read_text())) == 2


@pytest.mark.parametrize('change', [{'dose': 3}, {'dose': 'bad'}, {'tags': 'bad'}, {'layer': -1}, {'n': True}, {'source': 'robot'}, {'model': 'other'}, {'name': 4}, {'extra': 1}])
def test_validation(client, change):
    response = add(client, **change)
    assert response.status_code == 400
    assert set(response.json()) == {'ok', 'code', 'error', 'fix'}
    assert not response.json()['ok']


def test_patch_and_origin_validation(client):
    entry = add(client).json()['entry']
    assert client.patch('/api/library/'+entry['id'], json={'n': 3}).status_code == 400
    assert client.patch('/api/library/'+entry['id'], json={'name': ''}).status_code == 400
    assert client.patch('/api/library/'+entry['id'], json={'dose': .4}, headers={'Origin': 'https://example.com'}).status_code == 403
    assert client.delete('/api/library/'+entry['id'], headers={'Origin': 'https://example.com'}).status_code == 403
    assert client.post('/api/library', content='[]', headers={'Content-Type': 'application/json'}).status_code == 400


def test_search_top_and_neuron(client):
    rows = client.get('/api/search', params={'q': 'EVIL'}).json()['rows']
    assert [(r['n'], r['sign'], r['kl_ratio']) for r in rows] == [(1, '-', 6), (0, '+', 4), (1, '+', 2)]
    assert len(rows[1]['words']) == 12
    assert len(client.get('/api/search', params={'q': 'evil', 'sign': '+', 'limit': 1}).json()['rows']) == 1
    assert client.get('/api/search?q=evil&sign=x').status_code == 400
    assert client.get('/api/top?limit=bad').status_code == 400
    assert client.get('/api/neuron?layer=-1&n=0').status_code == 400
    assert client.get('/api/neuron?layer=17&n=9').status_code == 404
    assert client.get('/api/top?layer=16').json()['rows'] == []
    assert client.get('/api/top?layer=17&limit=1').json()['rows'][0]['kl_ratio'] == 6
    neuron = client.get('/api/neuron?layer=17&n=0').json()
    assert [s['sign'] for s in neuron['signs']] == ['+', '-']
    assert neuron['signs'][0]['sentences'] == [['cruel']]
    assert neuron['signs'][0]['persona'] == ['villain']


def test_atomic_concurrent_writes(client):
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda n: library.upsert(dict(layer=17, n=n, name=str(n))), range(25)))
    assert len(json.loads(library.LIBRARY.read_text())) == 25
    assert not list(library.LIBRARY.parent.glob('.library-*'))


def test_seed_ids_stable_and_duplicate_sign_deduped(client, monkeypatch):
    monkeypatch.setattr(library, 'seeds', lambda: [library.validate(dict(layer=17, n=5244, dose=.4, name='Dark Realist')), library.validate(dict(layer=17, n=5244, dose=.6, name='Evil'))])
    first = client.get('/api/library').json()['entries']
    assert first[0]['name'] == 'Evil' and len(first) == 1
    assert client.get('/api/library').json()['entries'][0]['id'] == first[0]['id']
    assert client.patch('/api/library/'+first[0]['id'], json={'note': 'saved'}).status_code == 200
    assert library.LIBRARY.exists()


def test_combo_crud_and_order_independent_upsert(client):
    pushes = [dict(layer=17, n=5244, dose=.6), dict(layer=17, n=604, dose=.5)]
    response = client.post('/api/library', json=dict(model=library.VIEW_MODEL, name='Evil + God', source='agent', pushes=pushes))
    assert response.status_code == 200
    entry = response.json()['entry']
    assert entry['pushes'] == pushes and not {'layer', 'n', 'dose'} & entry.keys()
    same = client.post('/api/library', json=dict(name='God + Evil', pushes=list(reversed(pushes)))).json()['entry']
    assert same['id'] == entry['id'] and same['name'] == 'God + Evil'
    changed = client.patch('/api/library/'+entry['id'], json={'pushes': [dict(pushes[0], dose=.4), pushes[1]], 'note': 'mix'}).json()['entry']
    assert changed['pushes'][0]['dose'] == .4 and changed['note'] == 'mix'
    assert client.patch('/api/library/'+entry['id'], json={'dose': .5}).status_code == 400
    single = add(client).json()['entry']
    assert client.patch('/api/library/'+single['id'], json={'pushes': pushes}).status_code == 400
    assert len(client.get('/api/library').json()['entries']) == 2
    assert client.delete('/api/library/'+entry['id']).status_code == 200
    assert len(client.get('/api/library').json()['entries']) == 1


@pytest.mark.parametrize('pushes', [[], 'bad', [dict(layer=17, n=0, dose=3)], [dict(layer=17, n=0)], [dict(layer=17, n=0, dose=True)], [dict(layer=17, n=0, dose=.4)] * 2])
def test_bad_combos(client, pushes):
    r = client.post('/api/library', json={'name': 'combo', 'pushes': pushes})
    assert r.status_code == 400
    assert r.json()['code'] == 'bad_library'


def test_combo_cannot_mix_single_fields(client):
    r = client.post('/api/library', json=dict(layer=17, n=0, dose=.6, pushes=[dict(layer=17, n=1, dose=.6)]))
    assert r.status_code == 400


def test_help_advertises_all_library_and_map_routes(client):
    help = client.get('/api/help').json()['endpoints']
    assert all(path in help for path in ['GET /api/library', 'POST /api/library', 'PATCH /api/library/{id}', 'DELETE /api/library/{id}', 'GET /api/search', 'GET /api/neuron', 'GET /api/top'])
    assert 'combo' in help['POST /api/library']


def test_static_gzip_map_fallback(client, monkeypatch, tmp_path):
    import gzip
    source = library.MAPS / 'Qwen3.5-2B/view'
    public = tmp_path / 'public'
    public.mkdir()
    (public / 'manifest.json').write_bytes((source / 'manifest.json').read_bytes())
    (public / 'L17.json.gz').write_bytes(gzip.compress((source / 'L17.json').read_bytes()))
    monkeypatch.setattr(library, 'MAPS', tmp_path / 'absent-maps')
    assert client.get('/api/search?q=evil&limit=1').json()['rows'][0]['n'] == 1
    assert client.get('/api/neuron?layer=17&n=0').json()['signs'][0]['words'][0] == 'Evil'
