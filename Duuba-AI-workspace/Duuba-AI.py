#!/usr/bin/env python
# Minimal wrapper copy of Duuba-AI for lightweight workspace
# This file is intended to be the same runtime entrypoint as the project root
# `Duuba-AI.py` but placed inside `Duuba-AI-workspace` so deployment tools can
# point to a small workspace directory.

import os
import shutil
from pathlib import Path

# Ensure we run the app from the workspace root (this file's directory)
HERE = Path(__file__).parent.resolve()
# If the full app exists in repository root, copy it here for convenience
ROOT_APP = Path(__file__).parents[1] / 'Duuba-AI.py'
TARGET_APP = HERE / 'Duuba-AI.py'
if ROOT_APP.exists() and not TARGET_APP.exists():
    shutil.copy2(ROOT_APP, TARGET_APP)

# Copy small supporting files if present at repo root
supporting = ['requirements-deploy.txt', 'Procfile', 'Dockerfile', 'DEPLOY.md']
for fname in supporting:
    src = Path(__file__).parents[1] / fname
    dst = HERE / fname
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)

# Copy label map if available
src_label = Path(__file__).parents[1] / 'annotations' / 'label_map.pbtxt'
if src_label.exists():
    dst_label_dir = HERE / 'annotations'
    dst_label_dir.mkdir(exist_ok=True)
    dst_label = dst_label_dir / 'label_map.pbtxt'
    if not dst_label.exists():
        shutil.copy2(src_label, dst_label)

# Copy local model checkpoint directory if present in the repository root
# This avoids committing large binaries while allowing the workspace to be
# self-contained at runtime when the repo already has the model.
root_model_dir = Path(__file__).parents[1] / 'my_ssd_mobnetpod'
if root_model_dir.exists() and root_model_dir.is_dir():
    dst_model_dir = HERE / 'Cocoa' / 'Tensorflow' / 'workspace' / 'models' / 'my_ssd_mobnetpod'
    dst_model_dir.mkdir(parents=True, exist_ok=True)
    # Copy all files from the root model dir (this is a local copy, kept out of git by default)
    for item in root_model_dir.iterdir():
        src_item = item
        dst_item = dst_model_dir / item.name
        if src_item.is_dir():
            # Recursively copy directories if they don't exist
            if not dst_item.exists():
                shutil.copytree(src_item, dst_item)
        else:
            if not dst_item.exists():
                shutil.copy2(src_item, dst_item)

# If model not present locally, leave a README with instructions; the download script
# `download_model.py` (added next) can be used to fetch a model from a URL.
if not (HERE / 'Cocoa' / 'Tensorflow' / 'workspace' / 'models' / 'my_ssd_mobnetpod').exists():
    readme = HERE / 'PLACE_MODEL_HERE.md'
    if not readme.exists():
        readme.write_text(
            'Place your trained model files (ckpt-*, pipeline.config or exported SavedModel) in:\n'
            '  ./Cocoa/Tensorflow/workspace/models/my_ssd_mobnetpod/\n\n'
            'Or set the environment variable MODEL_URL to a tar.gz containing the model and run:\n'
            '  python download_model.py\n'
        )

# Provide quick user hint and then exec the copied app
print('Duuba-AI workspace prepared in', HERE)
print('To run the app:')
print('  streamlit run Duuba-AI.py')

# Launch the actual app in this workspace if present
if TARGET_APP.exists():
    # Execute Streamlit from here
    # Before exec, if model missing try auto-download if MODEL_URL provided
    model_dir = HERE / 'Cocoa' / 'Tensorflow' / 'workspace' / 'models' / 'my_ssd_mobnetpod'
    if not model_dir.exists():
        model_url = os.environ.get('MODEL_URL')
        if model_url:
            print('MODEL_URL found; attempting to download model...')
            # run the downloader script if present
            downloader = HERE / 'download_model.py'
            if downloader.exists():
                os.execvp('python', ['python', str(downloader), '--url', model_url])
            else:
                print('download_model.py not found; please add it or place the model manually.')
        else:
            print('Model not found in workspace; starting app may fail. See PLACE_MODEL_HERE.md for instructions.')
    os.execvp('streamlit', ['streamlit', 'run', str(TARGET_APP)])
else:
    print('Error: could not find Duuba-AI.py in repo root to copy. Place your main app in the repository root named Duuba-AI.py.')
