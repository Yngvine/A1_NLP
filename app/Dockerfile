FROM mcr.microsoft.com/devcontainers/python:1-3.13-bookworm

WORKDIR /app

# Install system dependencies
RUN rm -f /etc/apt/sources.list.d/yarn.list && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "--workers", "1", "flask_app:app"]
