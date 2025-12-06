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

# Provide quick user hint and then exec the copied app
print('Duuba-AI workspace prepared in', HERE)
print('To run the app:')
print('  streamlit run Duuba-AI.py')

# Launch the actual app in this workspace if present
if TARGET_APP.exists():
    # Execute Streamlit from here
    os.execvp('streamlit', ['streamlit', 'run', str(TARGET_APP)])
else:
    print('Error: could not find Duuba-AI.py in repo root to copy. Place your main app in the repository root named Duuba-AI.py.')
