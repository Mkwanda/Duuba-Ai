#!/usr/bin/env python
"""
download_model.py

Small utility to download and extract a model archive (tar.gz or zip) into the
workspace models folder expected by `Duuba-AI.py`.

Usage:
  python download_model.py --url <archive_url> [--dest ./Cocoa/Tensorflow/workspace/models/my_ssd_mobnetpod]

Alternatively set `MODEL_URL` environment variable and run without args.

Note: For private storage you'll need to provide signed URLs or other auth.
"""

import argparse
import os
import shutil
import tarfile
import zipfile
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--url', help='URL to model archive (tar.gz or zip).')
parser.add_argument('--dest', default=os.path.join('Cocoa','Tensorflow','workspace','models','my_ssd_mobnetpod'), help='Destination folder to extract into')
args = parser.parse_args()

url = args.url or os.environ.get('MODEL_URL')
if not url:
    print('ERROR: no URL provided (use --url or set MODEL_URL).')
    raise SystemExit(1)

dest = Path(args.dest)
dest.mkdir(parents=True, exist_ok=True)

archive_path = dest / 'model_archive.tmp'

print(f'Downloading model from {url} to {archive_path}...')
urllib.request.urlretrieve(url, archive_path)
print('Download finished.')

# Try to extract
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
except Exception as e:
    print('Extraction failed:', e)
    raise
finally:
    if archive_path.exists():
        try:
            archive_path.unlink()
        except Exception:
            pass

print('Model is ready at', dest)
print('Run `streamlit run Duuba-AI.py` in this workspace to start the app.')
