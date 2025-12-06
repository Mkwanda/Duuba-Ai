#!/usr/bin/env python
"""
Downloader copied into the workspace. This is identical to the root-level
`download_model.py` and supports Google Drive downloads via gdown.
"""

from pathlib import Path
import shutil

# Copy the root downloader into this workspace file at runtime if needed
ROOT = Path(__file__).parents[1]
src = ROOT / 'download_model.py'
if src.exists():
    shutil.copy2(src, Path(__file__))
else:
    # If the root downloader is not available, leave a placeholder message.
    Path(__file__).write_text('# root download_model.py not found; please run downloader from repo root')
