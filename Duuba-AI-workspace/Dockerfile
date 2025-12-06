FROM python:3.10-slim

WORKDIR /app

COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "Duuba-AI.py", "--server.port=8501", "--server.address=0.0.0.0"]
