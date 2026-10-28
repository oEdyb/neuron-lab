import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from config import MAP_REPO, VIEW_MODEL

ROOT = Path(__file__).resolve().parent
MODEL_NAME = VIEW_MODEL.split('/')[-1]


def data_dir():
    source = os.environ.get('NEURON_LAB_MAPS', MAP_REPO)
    path = Path(source).expanduser()
    if path.is_dir():
        return path / MODEL_NAME if (path / MODEL_NAME).is_dir() else path
    return ROOT / 'data' / MODEL_NAME


def available():
    root = data_dir()
    try:
        meta = json.loads((root / 'manifest.json').read_text())
        required = ['signed-search.json.gz', 'signed-strength.bin', 'vector-words.json.gz', 'word-vectors.bin', 'connections/manifest.json']
        return meta['model'] == VIEW_MODEL and all((root / name).is_file() for name in required) and bool(meta['layers']) and all(
            (root / f"L{item['layer']}.bin").is_file() and
            (root / f"L{item['layer']}.json.gz").is_file() and
            (root / 'connections' / f"L{item['layer']}.bin").is_file() for item in meta['layers'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def download_command():
    return 'uv run python map_data.py'


def ensure_maps():
    if available():
        return True
    source = os.environ.get('NEURON_LAB_MAPS', MAP_REPO)
    print(f'Downloading maps from {source}…', flush=True)
    try:
        if Path(source).expanduser().is_dir() or source.startswith(('/', '.', '~')):
            raise ValueError(f'No complete {MODEL_NAME} map in {source}')
        snapshot_download(repo_id=source, repo_type='dataset', token=False, local_dir=ROOT / 'data',
                          allow_patterns=[f'{MODEL_NAME}/**'])
        if not available():
            raise ValueError(f'Dataset has no complete {MODEL_NAME} map')
        return True
    except Exception as error:
        print(f"Map isn't downloaded: {error}\nRun: {download_command()}", flush=True)
        return False


if __name__ == '__main__':
    argparse.ArgumentParser(description='Download the explorer map dataset.').parse_args()
    raise SystemExit(0 if ensure_maps() else 1)
