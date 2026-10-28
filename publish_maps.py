import argparse
import shutil
import tempfile
from pathlib import Path

from config import MAP_REPO, VIEW_MODEL

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description='Publish explorer data as a Hugging Face dataset.')
    ap.add_argument('--source', type=Path, default=ROOT / 'explorer/data')
    ap.add_argument('--repo', default=MAP_REPO)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    files = sorted(p for p in a.source.rglob('*') if p.is_file() and not p.name.startswith('.'))
    if not files or not (a.source / 'manifest.json').is_file():
        ap.error('source must contain the explorer map and manifest.json')
    name = VIEW_MODEL.split('/')[-1]
    for path in files:
        print(f'{name}/{path.relative_to(a.source)}\t{path.stat().st_size}')
    print(f'{len(files)} files, {sum(p.stat().st_size for p in files):,} bytes total')
    if a.dry_run:
        return
    from huggingface_hub import HfApi
    with tempfile.TemporaryDirectory(prefix='neuron-maps-') as folder:
        target = Path(folder) / name
        target.mkdir()
        for path in files:
            dest = target / path.relative_to(a.source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        api = HfApi()
        api.create_repo(repo_id=a.repo, repo_type='dataset', exist_ok=True)
        api.upload_folder(repo_id=a.repo, repo_type='dataset', folder_path=folder)


if __name__ == '__main__':
    main()
