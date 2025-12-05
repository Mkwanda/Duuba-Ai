Deployment options

1) Streamlit Community Cloud (recommended for Streamlit apps)
- Create an account at https://share.streamlit.io/
-- Link your GitHub repo and choose the repo/branch and `Duuba-AI.py` as the main file.
- Streamlit will install requirements from `requirements-deploy.txt` if present.

2) Heroku (using Procfile)
- Install the Heroku CLI, create an app: `heroku create your-app-name`
- Push to Heroku: `git push heroku main`
- Ensure `requirements-deploy.txt` is used; you can rename to `requirements.txt` for Heroku or set buildpacks.

3) Docker
- Build: `docker build -t duuba-ai .`
- Run: `docker run -p 8501:8501 duuba-ai`

Notes
- Large TensorFlow models may increase image size substantially; consider using GPU-enabled hosts or cloud ML services for production.
- The repository includes `requirements-deploy.txt` with a minimal set of packages for hosting. Adjust as needed.
