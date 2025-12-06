Deployment notes for Duuba-AI workspace

This folder contains a lightweight copy of the app and minimal deployment files.

To run locally (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-deploy.txt
streamlit run Duuba-AI.py
```

Docker build / run:

```powershell
docker build -t duuba-ai .
docker run -p 8501:8501 duuba-ai
```

Streamlit Cloud / Heroku:
- Point the service to this folder and use `Duuba-AI.py` as the main file.
- Ensure the model checkpoint (not included) is placed under `models/<model_name>/` matching the paths in `Duuba-AI.py`.
