#!/usr/bin/env python
"""
setup_for_web.py

Prepares the Duuba-AI project for web deployment by:
1. Checking required files exist
2. Verifying model files or MODEL_URL is configured
3. Testing imports
4. Creating necessary directories
"""

import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def check_directory(dirpath, create=False):
    """Check if a directory exists, optionally create it."""
    if Path(dirpath).exists():
        print(f"✅ Directory exists: {dirpath}")
        return True
    else:
        if create:
            Path(dirpath).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dirpath}")
            return True
        else:
            print(f"⚠️  Directory not found: {dirpath}")
            return False

def check_model_files():
    """Check if model files exist."""
    model_dir = Path('my_ssd_mobnetpod')
    required_files = ['checkpoint', 'pipeline.config']
    checkpoint_files = list(model_dir.glob('ckpt-*.index'))
    
    if not model_dir.exists():
        print(f"⚠️  Model directory not found: {model_dir}")
        print("   You'll need to set MODEL_URL environment variable for auto-download")
        return False
    
    all_exist = True
    for f in required_files:
        if not (model_dir / f).exists():
            print(f"❌ Missing model file: {model_dir / f}")
            all_exist = False
        else:
            print(f"✅ Found: {model_dir / f}")
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files (ckpt-*.index) found in {model_dir}")
        all_exist = False
    else:
        print(f"✅ Found checkpoint files: {checkpoint_files}")
    
    return all_exist

def check_imports():
    """Check if required packages can be imported."""
    required_packages = [
        'streamlit',
        'tensorflow',
        'object_detection',
        'cv2',
        'numpy',
        'PIL'
    ]
    
    all_imported = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ Can import: {package}")
        except ImportError:
            print(f"❌ Cannot import: {package}")
            all_imported = False
    
    return all_imported

def check_env_vars():
    """Check environment variables."""
    model_url = os.environ.get('MODEL_URL')
    if model_url:
        print(f"✅ MODEL_URL is set: {model_url[:50]}...")
        return True
    else:
        print("⚠️  MODEL_URL not set")
        print("   Set it with: export MODEL_URL='your-model-url'")
        return False

def main():
    print("=" * 60)
    print("Duuba-AI Web Deployment Readiness Check")
    print("=" * 60)
    print()
    
    print("1. Checking required files...")
    print("-" * 60)
    check_file('Duuba-AI.py', 'Main application')
    check_file('requirements.txt', 'Requirements file')
    check_file('Procfile', 'Procfile (for Heroku)')
    check_file('Dockerfile', 'Dockerfile')
    check_file('download_model.py', 'Model downloader')
    check_file('.streamlit/config.toml', 'Streamlit config')
    print()
    
    print("2. Checking/creating directories...")
    print("-" * 60)
    check_directory('annotations', create=True)
    check_directory('.streamlit', create=False)
    print()
    
    print("3. Checking model files...")
    print("-" * 60)
    model_exists = check_model_files()
    print()
    
    print("4. Checking environment variables...")
    print("-" * 60)
    env_configured = check_env_vars()
    print()
    
    print("5. Checking Python packages...")
    print("-" * 60)
    imports_ok = check_imports()
    print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if not model_exists and not env_configured:
        print("⚠️  WARNING: No model files found AND MODEL_URL not set")
        print("   Action required:")
        print("   1. Either copy model files to my_ssd_mobnetpod/ directory")
        print("   2. Or set MODEL_URL environment variable")
        print()
    
    if not imports_ok:
        print("❌ FAILED: Some required packages are missing")
        print("   Run: pip install -r requirements.txt")
        print()
        sys.exit(1)
    
    if model_exists or env_configured:
        print("✅ READY FOR DEPLOYMENT!")
        print()
        print("Next steps:")
        print("  • Local test: streamlit run Duuba-AI.py")
        print("  • Deploy to Streamlit Cloud: push to GitHub and connect")
        print("  • Deploy to Heroku: git push heroku main")
        print("  • Docker: docker build -t duuba-ai . && docker run -p 8501:8501 duuba-ai")
        print()
        print("See WEB_DEPLOYMENT.md for detailed instructions")
    else:
        print("⚠️  Configuration needed before deployment")
        print("   See notes above")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
