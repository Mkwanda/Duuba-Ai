FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy only requirements first for cache
COPY requirements-deploy.txt ./

RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy app
COPY . /app

# Expose port
EXPOSE 8501

# Run streamlit
CMD ["streamlit", "run", "COCOA Train model new.py", "--server.port=8501", "--server.address=0.0.0.0"]
