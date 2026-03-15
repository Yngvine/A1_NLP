FROM mcr.microsoft.com/devcontainers/python:1-3.13-bookworm

WORKDIR /app

RUN pip install --no-cache-dir streamlit requests

COPY streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
