import fcntl
import gzip
import json
import math
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from config import MODELS, VIEW_MODEL

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / 'library.json'
MAPS = ROOT / 'maps'
from map_data import data_dir, download_command

PUBLIC_MAP = data_dir()


class LibraryError(Exception):
    def __init__(self, code, error, fix='GET /api/help', status=400):
        self.code, self.error, self.fix, self.status = code, error, fix, status


def validate_push(raw):
    if not isinstance(raw, dict) or set(raw) != {'layer', 'n', 'dose'}:
        raise LibraryError('bad_library', 'each push needs layer, n and dose')
    for key in ('layer', 'n'):
        if type(raw[key]) is not int or raw[key] < 0:
            raise LibraryError('bad_library', f'{key} must be a nonnegative integer')
    if type(raw['dose']) not in (float, int) or not math.isfinite(raw['dose']) or abs(raw['dose']) > 2:
        raise LibraryError('bad_library', 'push dose must be finite and between -2 and 2')
    return raw


def validate(raw, patch=False):
    allowed = {'name', 'note', 'tags', 'dose', 'pushes'} if patch else {'id', 'model', 'layer', 'n', 'dose', 'pushes', 'name', 'note', 'tags', 'source', 'added'}
    if set(raw) - allowed:
        raise LibraryError('bad_library', 'unknown Library fields')
    out = dict(raw) if patch else dict(model=VIEW_MODEL, name='', note='', tags=[], source='you') | raw
    for key in ('name', 'note'):
        if key in out and (not isinstance(out[key], str) or len(out[key]) > (200 if key == 'name' else 10000)):
            raise LibraryError('bad_library', f'{key} must be text within its length limit')
    if 'tags' in out and (not isinstance(out['tags'], list) or len(out['tags']) > 50 or any(not isinstance(t, str) or len(t) > 100 for t in out['tags'])):
        raise LibraryError('bad_library', 'tags must be a list of up to 50 short strings')
    if out.get('dose') is not None and (type(out['dose']) not in (float, int) or not math.isfinite(out['dose']) or abs(out['dose']) > 2):
        raise LibraryError('bad_library', 'dose must be null or a finite number between -2 and 2')
    if 'pushes' in out:
        if not isinstance(out['pushes'], list) or not 1 <= len(out['pushes']) <= 32:
            raise LibraryError('bad_library', 'combos need 1–32 pushes')
        for push in out['pushes']:
            validate_push(push)
        keys = [(p['layer'], p['n']) for p in out['pushes']]
        if len(keys) != len(set(keys)):
            raise LibraryError('bad_library', 'combine repeated neurons into one signed push')
        if any(k in out for k in ('layer', 'n', 'dose')):
            raise LibraryError('bad_library', 'use pushes for a combo or layer/n/dose for a neuron')
    if not patch:
        if out['model'] not in MODELS or out['source'] not in ('you', 'agent'):
            raise LibraryError('bad_library', 'unsupported model or source')
        if 'pushes' not in out:
            out.setdefault('dose', None)
            validate_push({k: out.get(k) if k != 'dose' else out.get(k) or 0 for k in ('layer', 'n', 'dose')})
        out['id'] = uuid.uuid4().hex
        out['added'] = datetime.now(timezone.utc).isoformat()
        out['name'] = out['name'].strip() or (' + '.join(f"L{p['layer']} N{p['n']}" for p in out['pushes']) if 'pushes' in out else f"L{out['layer']} N{out['n']}")
    elif 'name' in out and not out['name'].strip():
        raise LibraryError('bad_library', 'name cannot be empty')
    return out


def identity(entry):
    if 'pushes' in entry:
        return entry['model'], tuple(sorted((p['layer'], p['n'], p['dose']) for p in entry['pushes']))
    dose = entry['dose']
    return entry['model'], entry['layer'], entry['n'], None if dose is None else (dose > 0) - (dose < 0)


def seeds():
    source = ROOT / 'out/hunt/shortlist.json'
    if source.exists():
        raw = json.loads(source.read_text()) + [dict(layer=17, n=5244, dose=.6, name='Evil')]
        return [validate(dict(model=VIEW_MODEL, layer=r['layer'], n=r['n'], dose=r['dose'], name=r['name'], note=r.get('why', ''), source='agent')) for r in raw]
    return [validate(r) for r in json.loads((ROOT / 'examples/library.json').read_text())]


def initial():
    unique = {}
    for entry in seeds():
        entry['id'] = uuid.uuid5(uuid.NAMESPACE_URL, repr(identity(entry))).hex
        entry['added'] = datetime.fromtimestamp((ROOT / 'examples/library.json').stat().st_mtime, timezone.utc).isoformat()
        unique[identity(entry)] = entry
    return list(unique.values())


@contextmanager
def locked():
    with LIBRARY.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def entries():
    return json.loads(LIBRARY.read_text()) if LIBRARY.exists() else initial()


def save(rows):
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=LIBRARY.parent, prefix='.library-', delete=False) as f:
            name = f.name
            json.dump(rows, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, LIBRARY)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def listing(model=None):
    with locked():
        rows = entries()
    return [r for r in rows if not model or r['model'] == model]


def upsert(raw):
    entry = validate(raw)
    with locked():
        rows = entries()
        existing = next((r for r in rows if identity(r) == identity(entry)), None)
        if existing:
            entry = existing | {k: v for k, v in entry.items() if k in raw and k not in ('id', 'added')}
            rows[rows.index(existing)] = entry
        else:
            rows.append(entry)
        save(rows)
    return entry


def edit(id, raw=None):
    patch = validate(raw, patch=True) if raw is not None else None
    with locked():
        rows = entries()
        entry = next((r for r in rows if r['id'] == id), None)
        if entry is None:
            raise LibraryError('library_not_found', 'Library entry not found', 'GET /api/library', 404)
        if patch is None:
            rows.remove(entry)
        else:
            updated = entry | patch
            validate(updated)
            if any(r['id'] != id and identity(r) == identity(updated) for r in rows):
                raise LibraryError('library_conflict', 'that neuron and dose sign are already saved', 'edit the existing entry', 409)
            rows[rows.index(entry)] = updated
            entry = updated
        save(rows)
    return entry


@lru_cache(maxsize=24)
def read_layer(path, stamp):
    p = Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_text())


def map_layers(only=None):
    root = MAPS / VIEW_MODEL.split('/')[-1] / 'view'
    manifest = root / 'manifest.json'
    if not manifest.exists() and (PUBLIC_MAP / 'manifest.json').exists():
        root, manifest = PUBLIC_MAP, PUBLIC_MAP / 'manifest.json'
    if not manifest.exists():
        raise LibraryError('map_missing', 'map data is missing', download_command(), 404)
    meta = json.loads(manifest.read_text())
    if meta.get('model', VIEW_MODEL) != VIEW_MODEL:
        raise LibraryError('map_mismatch', 'map model does not match the configured view', 'run uv run python build_view.py', 409)
    for item in meta['layers']:
        if only is not None and item['layer'] != only:
            continue
        path = root / f"L{item['layer']}.json"
        if not path.exists():
            path = path.with_suffix('.json.gz')
        yield item['layer'], read_layer(str(path), path.stat().st_mtime_ns)


def ranked(q=None, sign=None, layer=None, limit=20):
    rows = []
    for l, data in map_layers(layer):
        if layer is not None and l != layer:
            continue
        for i, row in enumerate(data['rows']):
            for j, s in enumerate(('+', '-')):
                words = row['up' if j == 0 else 'down']
                if sign and s != sign or q and not any(q.casefold() in w.casefold() for w in words):
                    continue
                rows.append(dict(layer=l, n=row.get('n', i), sign=s, kl_ratio=row['kl'][j] / max(data['p95'][j], 1e-9), words=words[:12]))
    return sorted(rows, key=lambda r: (-r['kl_ratio'], r['layer'], r['n'], r['sign']))[:limit]


@lru_cache(maxsize=3)
def tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, local_files_only=True)


def neuron(layer, n):
    for l, data in map_layers(layer):
        if l != layer:
            continue
        row = next((r for i, r in enumerate(data['rows']) if r.get('n', i) == n), None)
        if row is None:
            break
        frames = row.get('frame_words', [[], []])
        path = MAPS / VIEW_MODEL.split('/')[-1] / f'L{layer}.npz'
        if path.exists():
            import numpy as np
            with np.load(path) as raw:
                tok = tokenizer(VIEW_MODEL)
                frames = [[[tok.decode([int(t)]) for t in sentence[:12]] for sentence in raw[f'{s}_frame_tok'][n]] for s in ('up', 'down')]
        return dict(model=VIEW_MODEL, layer=layer, n=n, sentence_words_available=bool(frames[0] or frames[1]), signs=[dict(sign=s, words=row['up' if j == 0 else 'down'][:12], persona=(row.get('persona') or [[], []])[j][:12], sentences=[w[:12] for w in frames[j]], kl_ratio=row['kl'][j] / max(data['p95'][j], 1e-9)) for j, s in enumerate(('+', '-'))])
    raise LibraryError('neuron_not_found', 'neuron is outside the map', 'GET /api/top', 404)
