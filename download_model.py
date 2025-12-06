#!/usr/bin/env python
"""
download_model.py

Download and extract a model archive (tar.gz or zip) into the project's
model folder expected by `Duuba-AI.py`.

Usage:
  python download_model.py --url <archive_url> [--dest ./Cocoa/Tensorflow/workspace/models/my_ssd_mobnetpod]

If `MODEL_URL` environment variable is set and `--url` not provided, it will be used.
"""

import argparse
import os
import shutil
import tarfile
import zipfile
import urllib.request
from pathlib import Path
import re

try:
    import requests
except Exception:
    requests = None


def download_and_extract(url: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    archive_path = dest / 'model_archive.tmp'
    print(f'Downloading model from {url} to {archive_path}...')

    # Handle Google Drive shared file links specially to get a direct download
    drive_match = re.search(r'd/([a-zA-Z0-9_-]+)', url)
    if drive_match:
        file_id = drive_match.group(1)
        direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        # Prefer requests for handling large-file confirm tokens
        if requests:
            session = requests.Session()
            response = session.get(direct_url, stream=True)
            token = None
            for k, v in response.cookies.items():
                if k.startswith('download_warning'):
                    token = v
                    break
            if token:
                params = {'id': file_id, 'export': 'download', 'confirm': token}
                response = session.get('https://drive.google.com/uc', params=params, stream=True)

            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            print('Download finished.')
        else:
            # Fallback to direct uc link via urllib (may fail on large files requiring confirmation)
            urllib.request.urlretrieve(direct_url, archive_path)
            print('Download finished (urllib fallback).')
        # proceed to extraction below
    else:
        urllib.request.urlretrieve(url, archive_path)
        print('Download finished.')

    try:
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(path=dest)
        elif zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(path=dest)
        else:
            # Not an archive: move into place (single file)
            shutil.move(str(archive_path), str(dest / Path(url).name))
            print('Saved single-file model to', dest)
    finally:
        if archive_path.exists():
            try:
                archive_path.unlink()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='URL to model archive (tar.gz or zip).')
    parser.add_argument('--dest', default=os.path.join('Cocoa','Tensorflow','workspace','models','my_ssd_mobnetpod'), help='Destination folder to extract into')
    args = parser.parse_args()

    url = args.url or os.environ.get('MODEL_URL')
    if not url:
        print('ERROR: no URL provided (use --url or set MODEL_URL).')
        raise SystemExit(1)

    dest = Path(args.dest)
    download_and_extract(url, dest)
    print('Model is ready at', dest)


if __name__ == '__main__':
    main()
