#set page(margin: 2.2cm)
#set text(font: "Libertinus Serif", size: 11pt)

#align(center)[
  #text(size: 22pt, weight: "bold")[Assignment 3 Report]
  #v(6pt)
  #text(size: 14pt)[Document Retrieval-Augmented Generation (RAG) System]
  #v(10pt)
  #text(size: 10pt)[Date: March 15, 2026]
]

#v(18pt)

= Overview

This project implements a local RAG pipeline for remote-sensing captions from the RSVLM-QA dataset.  
The goal is to answer natural-language questions using retrieved evidence instead of generating unsupported answers.

The system combines:
- ChromaDB for vector search
- MinIO for document storage
- Flask API as orchestration layer
- Streamlit UI for user interaction
- llama.cpp server as local LLM backend

= What The System Is Doing

For each user question, the backend executes an iterative retrieval-and-generation process:

1. Plan action: decide whether to retrieve more context or answer.
2. Search: retrieve top relevant caption chunks from ChromaDB.
3. Generate: build a grounded prompt with retrieved passages and ask the LLM for the final answer.

This loop can run for multiple iterations (bounded by configuration), and returns:
- final answer
- source passages
- reasoning trace
- search queries used

= Architecture And Data Flow

#align(center)[
  User -> Streamlit (8501) -> Flask API (5000)  
  Flask API -> ChromaDB (8000), MinIO (9000/9001), llama-server (8080)
]

Data flow summary:
- Ingestion endpoint reads dataset captions (or uploaded files), chunks text, stores objects in MinIO, and indexes embeddings in ChromaDB.
- Query endpoint runs the RAG agent and returns a grounded response with provenance.

= How It Works In Practice

== Ingestion

The /documents endpoint supports:
- dataset ingestion with a configurable limit
- file upload (PDF/DOCX), parsing, chunking, indexing

== Retrieval

Retriever computes embedding similarity and returns top-k candidate passages.
Duplicate chunks are filtered to keep context concise.

More detail on retrieval:
- Query encoding: the user question is converted to a dense embedding vector using the same sentence-transformer family used at index time.
- Vector search: ChromaDB compares the query embedding with stored caption/chunk embeddings and returns the nearest neighbors.
- Relevance ranking: results are ordered by similarity, then exposed as source chunks with relevance scores and metadata (image path, tags, document id).
- Deduplication across iterations: when the agent performs multiple search rounds, already-seen chunk ids are removed so each iteration contributes new evidence.
- Top-k control: `top_k` determines how many passages are finally returned to the client, while the internal loop can temporarily inspect more evidence before final answer synthesis.

In practice, retrieval is the grounding mechanism of the whole system: it narrows the evidence space from the full dataset to a small, relevant context window that the LLM can reason over.

== LLM Agent (Decision + Reasoning)

The query pipeline is implemented as a stateful ReAct-style graph with three nodes: `plan_action`, `search`, and `generate`.

Agent state includes:
- original query
- retrieved chunks accumulated so far
- search queries used in each round
- reasoning trace
- current iteration counter
- selected action (`search` or `generate`)

How the agent decides:
1. First iteration always starts with `search`.
2. After retrieval, `plan_action` asks the LLM a meta-question: do we already have enough context to answer?
3. If the answer is no and max iterations are not reached, the graph loops back to `search`.
4. Otherwise, it transitions to `generate` and produces the final response.

How multi-step search works:
- On later iterations, the agent can ask the LLM to rewrite/refine the search query.
- This often improves recall by exploring alternate phrasing (for example, using related objects, relations, or scene terms).
- Each round appends a reasoning message, so the API can return a transparent reasoning trace.

Final answer generation:
- The generator builds a constrained prompt containing only retrieved passages.
- It explicitly instructs the model to avoid unsupported claims and to answer from provided evidence.
- If the LLM is still warming up or unavailable, the system returns a graceful fallback summary based on top retrieved evidence instead of a raw server exception.

== Generation

The generator uses only retrieved passages to answer.
If context is insufficient, the model states that explicitly.

== Resilience

The LLM client includes retries for transient 503 responses during model warmup.
If the LLM is temporarily unavailable, the API returns a graceful fallback answer instead of exposing raw technical errors.

= Runtime Notes

- GPU acceleration is enabled for llama.cpp using the CUDA image and NVIDIA runtime in Docker Compose.
- Health endpoint reports per-service readiness and global status.
- First startup can be slow while model tensors are loaded into memory.

= How To Run

1. Start services:

#block[
  docker compose up --build
]

2. Check health:

#block[
  curl http://localhost:5000/health
]

3. Ingest captions:

#block[
  curl -X POST http://localhost:5000/documents \
    -H "Content-Type: application/json" \
    -d '{"source": "dataset", "limit": 1000}'
]

4. Query:

#block[
  curl -X POST http://localhost:5000/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What kind of area has a highway interchange?", "top_k": 5}'
]

= Conclusion

This RAG system provides grounded question answering over remote-sensing text data with local infrastructure.  
Its main strengths are traceability (source passages), modular architecture, and robust runtime behavior (health checks, retries, graceful fallback, and GPU-backed LLM inference).
