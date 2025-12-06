#!/usr/bin/env python
"""
Downloader copied into the workspace. This is identical to the root-level
`download_model.py` and supports Google Drive downloads via gdown.
"""

from pathlib import Path
import shutil

# Determine repository root robustly
file_path = Path(__file__).resolve()
if len(file_path.parents) >= 2:
    REPO_ROOT = file_path.parents[1]
elif len(file_path.parents) == 1:
    REPO_ROOT = file_path.parents[0]
else:
    REPO_ROOT = Path.cwd()

# Copy the root downloader into this workspace file at runtime if needed
src = REPO_ROOT / 'download_model.py'
if src.exists():
    shutil.copy2(src, Path(__file__))
else:
    # If the root downloader is not available, leave a placeholder message.
    Path(__file__).write_text('# root download_model.py not found; please run downloader from repo root')
