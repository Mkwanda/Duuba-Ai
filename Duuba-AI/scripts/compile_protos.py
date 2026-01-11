#!/usr/bin/env python
"""Compile TensorFlow Object Detection .proto files into a local vendor package.

This script will:
- Ensure grpc_tools is installed (installs it if missing)
- Clone the TensorFlow models repo (shallow) to a temp dir
- Compile protoc files from models/research/object_detection/protos
  into a local `vendor/object_detection` package
"""
import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = ROOT / 'vendor' / 'object_detection'
MODELS_REPO = 'https://github.com/tensorflow/models.git'

def ensure_grpc_tools():
    try:
        import grpc_tools  # noqa: F401
        return
    except Exception:
        print('grpc_tools not found; installing grpcio-tools...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'grpcio-tools'])

def clone_models(tmpdir):
    dst = Path(tmpdir) / 'models'
    subprocess.check_call(['git', 'clone', '--depth', '1', MODELS_REPO, str(dst)])
    return dst

def compile_protos(models_dir, out_dir):
    proto_dir = models_dir / 'research' / 'object_detection' / 'protos'
    if not proto_dir.exists():
        raise RuntimeError(f'protos dir not found at {proto_dir}')

    out_dir.mkdir(parents=True, exist_ok=True)
    # Ensure package __init__ files
    (out_dir / '__init__.py').write_text('# vendor object_detection package')
    protos_pkg = out_dir / 'protos'
    protos_pkg.mkdir(exist_ok=True)
    (protos_pkg / '__init__.py').write_text('# protos package')

    # Compile each .proto file
    proto_files = sorted(proto_dir.glob('*.proto'))
    print(f'Compiling {len(proto_files)} protos to {out_dir}')

    for proto in proto_files:
        cmd = [
            sys.executable, '-m', 'grpc_tools.protoc',
            f'-I{str(models_dir / "research")}',
            f'--python_out={str(out_dir)}',
            f'--grpc_python_out={str(out_dir)}',
            str(proto)
        ]
        print('Running:', ' '.join(cmd))
        subprocess.check_call(cmd)

    print('Done compiling protos')

def main():
    ensure_grpc_tools()
    tmpdir = tempfile.mkdtemp(prefix='tfmodels_')
    try:
        models_dir = clone_models(tmpdir)
        compile_protos(models_dir, VENDOR_DIR)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == '__main__':
    main()
