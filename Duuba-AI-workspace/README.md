Duuba-AI workspace

This folder contains a lightweight workspace with the files required to run the Duuba-AI Streamlit app.

Notes:
- Large model checkpoints and TFRecords are intentionally omitted to keep this workspace small.
- To run locally, install the packages in `requirements-deploy.txt` and place your model checkpoint under `models/<model_name>/` following the paths used by `Duuba-AI.py`.

Commands to run locally (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-deploy.txt
streamlit run Duuba-AI.py
```
