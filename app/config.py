import os

# ChromaDB
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = "rsvlm_captions"

# MinIO
MINIO_HOST = os.getenv("MINIO_HOST", "localhost")
MINIO_PORT = int(os.getenv("MINIO_PORT", "9000"))
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "documents"

# LLM (llama-server)
LLM_HOST = os.getenv("LLM_HOST", "localhost")
LLM_PORT = int(os.getenv("LLM_PORT", "8080"))
LLM_URL = f"http://{LLM_HOST}:{LLM_PORT}"

# Data paths
CAPTIONS_PATH = os.getenv("CAPTIONS_PATH", "RSVLM-QA-captions.parquet")
QUESTIONS_PATH = os.getenv("QUESTIONS_PATH", "RSVLM-QA-questions.parquet")

# Ingestion
DEFAULT_INGEST_LIMIT = int(os.getenv("DEFAULT_INGEST_LIMIT", "1000"))

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval
DEFAULT_TOP_K = 5

# Agent
MAX_AGENT_ITERATIONS = 3
