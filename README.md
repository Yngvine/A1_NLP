# Assignment 3: Document Retrieval-Augmented Generation System

A complete RAG (Retrieval-Augmented Generation) system for querying remote sensing image captions from the RSVLM-QA dataset. Built with ChromaDB, LangGraph, Flask, and Streamlit.

## Architecture

```
User <-> Streamlit UI (:8501) <-> Flask REST API (:5000)
                                       |
                       +---------------+---------------+
                       |               |               |
                 ChromaDB (:8000)  MinIO (:9000)  llama-server (:8080)
```

| Service          | Description                                                 | Port      |
| ---------------- | ----------------------------------------------------------- | --------- |
| **llama-server** | Quantized LLM (Mistral 7B) via llama.cpp                    | 8080      |
| **MinIO**        | Object store for raw documents                              | 9000/9001 |
| **ChromaDB**     | Vector database for caption embeddings                      | 8000      |
| **Flask API**    | REST API orchestrating ingestion, retrieval, and generation | 5000      |
| **Streamlit**    | Web UI for interactive querying                             | 8501      |

## RAG Pipeline

The system uses a **LangGraph ReAct-style agent** for the query pipeline:

1. **Plan Action**: Decide whether to search for more context or generate an answer
2. **Search**: Embed the query and retrieve relevant captions from ChromaDB (all-MiniLM-L6-v2 bi-encoder)
3. **Generate**: Build a grounded prompt with retrieved passages and generate an answer via the local LLM

The agent can perform up to 3 search iterations, refining queries between rounds.

## Quick Start

### 1. Download the LLM model

```bash
# Mistral 7B Instruct (recommended, ~4.4 GB)
mkdir -p models
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Alternative: TinyLlama (lightweight for development, ~0.7 GB)
# wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### 2. Start all services

```bash
docker compose up --build
```

Note: The current `llama-server` config uses CUDA (`ghcr.io/ggml-org/llama.cpp:server-cuda`), so Docker Desktop must have NVIDIA GPU support enabled.

Wait ~30-60 seconds for llama-server to load the model.

### 3. Ingest captions

```bash
curl -X POST http://localhost:5000/documents \
  -H "Content-Type: application/json" \
  -d '{"source": "dataset", "limit": 1000}'
```

### 4. Query the system

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What kind of area has a highway interchange?", "top_k": 5}'
```

Or open the Streamlit UI at **http://localhost:8501**.

## API Endpoints

| Method   | Endpoint          | Description                                                |
| -------- | ----------------- | ---------------------------------------------------------- |
| `POST`   | `/documents`      | Upload a PDF/DOCX file or ingest captions from the dataset |
| `GET`    | `/documents`      | List all indexed documents                                 |
| `DELETE` | `/documents/{id}` | Delete a document and its chunks                           |
| `POST`   | `/query`          | Submit a question; returns answer + source passages        |
| `GET`    | `/health`         | Health check for all services                              |

### Examples

```bash
# Health check
curl http://localhost:5000/health

# Ingest 500 captions
curl -X POST http://localhost:5000/documents \
  -H "Content-Type: application/json" \
  -d '{"source": "dataset", "limit": 500}'

# Upload a PDF
curl -X POST http://localhost:5000/documents -F "file=@document.pdf"

# Query
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What recreational facilities can be seen near a highway interchange?", "top_k": 5}'

# List documents
curl http://localhost:5000/documents

# Delete a document
curl -X DELETE http://localhost:5000/documents/caption-42
```

## Project Structure

```
├── docker-compose.yml          # Orchestrates all 5 services
├── data/                       # Parquet datasets (mounted into containers)
├── app/
│   ├── Dockerfile              # Flask API container
│   ├── requirements.txt        # Python dependencies
│   ├── config.py               # Configuration constants
│   ├── flask_app.py            # REST API (5 endpoints)
│   ├── rag_pipeline.py         # LangGraph ReAct agent
│   ├── indexer.py              # ChromaDB indexing + embeddings
│   ├── retriever.py            # ChromaDB search
│   ├── llm_client.py           # llama-server HTTP client
│   ├── minio_client.py         # MinIO object store client
│   └── document_parser.py      # PDF/DOCX parsing + chunking
├── ui/
│   ├── Dockerfile              # Streamlit container
│   └── streamlit_app.py        # Web UI
├── eval/
│   ├── eval_dataset.json       # 18 annotated evaluation questions
│   └── evaluate.py             # Retrieval metrics (Hit Rate, MRR, Precision @k)
├── results/
│   └── eval_results.json       # Evaluation output
└── models/                     # GGUF model files for llama-server
```

## Evaluation

The evaluation dataset contains 18 hand-annotated questions across 6 question types (spatial, count, presence, overall, object, quantity). Run the evaluation:

```bash
cd eval
python evaluate.py
```

Computes Hit Rate@k, MRR, and Precision@k for k=1, 3, 5.

## Configuration

Key constants in `app/config.py` (overridable via environment variables):

| Variable               | Default          | Description                       |
| ---------------------- | ---------------- | --------------------------------- |
| `DEFAULT_INGEST_LIMIT` | 1000             | Number of captions to ingest      |
| `EMBEDDING_MODEL`      | all-MiniLM-L6-v2 | Sentence transformer model        |
| `DEFAULT_TOP_K`        | 5                | Number of results to retrieve     |
| `MAX_AGENT_ITERATIONS` | 3                | Max LangGraph agent search rounds |

## Known Limitations

- The in-memory document registry is lost on container restart (captions remain in ChromaDB and MinIO)
- llama-server startup can take 30-60 seconds; queries return 503 until the LLM is ready
- Evaluation metrics depend on the default 1000-caption ingestion; results change with different limits
- Some count/presence questions may match multiple captions with similar content (e.g., "How many ships?")
