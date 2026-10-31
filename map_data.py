import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from config import MAP_REPO, VIEW_MODEL, MODELS

ROOT = Path(__file__).resolve().parent
MODEL_NAME = VIEW_MODEL.split('/')[-1]


active_model = VIEW_MODEL


def data_dir(model=None):
    name = (model or active_model).split("/")[-1]
    source = os.environ.get('NEURON_LAB_MAPS', MAP_REPO)
    path = Path(source).expanduser()
    if path.is_dir():
        return path / name if (path / name).is_dir() else path
    return ROOT / 'data' / name


def available(model=None):
    model = model or active_model
    root = data_dir(model)
    try:
        meta = json.loads((root / 'manifest.json').read_text())
        required = ['signed-search.json.gz', 'signed-strength.bin', 'vector-words.json.gz', 'word-vectors.bin', 'connections/manifest.json']
        return meta['model'] == model and all((root / name).is_file() for name in required) and bool(meta['layers']) and all(
            (root / f"L{item['layer']}.bin").is_file() and
            (root / f"L{item['layer']}.json.gz").is_file() and
            (root / 'connections' / f"L{item['layer']}.bin").is_file() for item in meta['layers'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def download_command():
    return f'uv run python map_data.py --model {active_model}'


def ensure_maps(model=None):
    model = model or active_model
    name = model.split("/")[-1]
    if available(model):
        return True
    source = os.environ.get('NEURON_LAB_MAPS', MAP_REPO)
    print(f'Downloading maps from {source}…', flush=True)
    try:
        if Path(source).expanduser().is_dir() or source.startswith(('/', '.', '~')):
            raise ValueError(f'No complete {name} map in {source}')
        snapshot_download(repo_id=source, repo_type='dataset', token=False, local_dir=ROOT / 'data',
                          allow_patterns=[f'{name}/**'])
        if not available(model):
            raise ValueError(f'Dataset has no complete {name} map')
        return True
    except Exception as error:
        print(f"Map isn't downloaded: {error}\nRun: {download_command()}", flush=True)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download an explorer map dataset.')
    parser.add_argument('--model', choices=MODELS, default=VIEW_MODEL)
    args = parser.parse_args()
    raise SystemExit(0 if ensure_maps(args.model) else 1)
