import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request

from config import DEFAULT_INGEST_LIMIT, DEFAULT_TOP_K
from document_parser import chunk_text, parse_docx, parse_pdf
from indexer import CaptionIndexer
from llm_client import LLMClient
from minio_client import MinIOClient
from rag_pipeline import run_rag_agent
from retriever import CaptionRetriever

app = Flask(__name__)

# Lazy singletons
_indexer = None
_retriever = None
_minio = None
_llm = None

# In-memory document registry
document_registry: dict[str, dict] = {}


def get_indexer():
    global _indexer
    if _indexer is None:
        _indexer = CaptionIndexer()
    return _indexer


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = CaptionRetriever()
    return _retriever


def get_minio():
    global _minio
    if _minio is None:
        _minio = MinIOClient()
    return _minio


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


@app.route("/health", methods=["GET"])
def health():
    services = {}

    try:
        services["chromadb"] = (
            "ok" if get_retriever().health_check() else "unavailable"
        )
    except Exception:
        services["chromadb"] = "unavailable"

    try:
        services["minio"] = "ok" if get_minio().health_check() else "unavailable"
    except Exception:
        services["minio"] = "unavailable"

    try:
        services["llm"] = "ok" if get_llm().health_check() else "unavailable"
    except Exception:
        services["llm"] = "unavailable"

    all_ok = all(s == "ok" for s in services.values())
    return jsonify({"status": "healthy" if all_ok else "degraded", "services": services})


@app.route("/documents", methods=["POST"])
def upload_document():
    # Case 1: JSON body - ingest captions from dataset
    if request.is_json:
        data = request.get_json() or {}
        limit = data.get("limit", DEFAULT_INGEST_LIMIT)
        source = data.get("source", "dataset")

        # Allow null/0 to mean "ingest all captions".
        if limit is None:
            limit = 0

        if source == "dataset":
            result = get_indexer().ingest(limit=limit)
            return jsonify(
                {
                    "message": f"Ingested {result['ingested']} captions",
                    "total_indexed": result["collection_count"],
                }
            ), 201

    # Case 2: File upload - PDF or DOCX
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send JSON or multipart file."}), 400

    file = request.files["file"]
    filename = file.filename or "unknown"
    file_bytes = file.read()

    # Parse document
    try:
        if filename.lower().endswith(".pdf"):
            text = parse_pdf(file_bytes)
        elif filename.lower().endswith(".docx"):
            text = parse_docx(file_bytes)
        else:
            return jsonify({"error": "Unsupported file type. Use PDF or DOCX."}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    # Generate ID and store in MinIO
    doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    get_minio().store_document(
        doc_id,
        file_bytes,
        filename,
        content_type="application/pdf"
        if filename.lower().endswith(".pdf")
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    # Chunk and index
    chunks = chunk_text(text)
    index_result = get_indexer().index_text_chunks(doc_id, chunks, filename)

    # Register
    document_registry[doc_id] = {
        "id": doc_id,
        "filename": filename,
        "upload_date": datetime.now(timezone.utc).isoformat(),
        "chunks": index_result["indexed_chunks"],
    }

    return jsonify(
        {
            "document_id": doc_id,
            "filename": filename,
            "chunks_indexed": index_result["indexed_chunks"],
        }
    ), 201


@app.route("/documents", methods=["GET"])
def list_documents():
    docs = get_indexer().get_all_documents()
    return jsonify({"total": len(docs), "documents": docs})


@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id: str):
    try:
        get_indexer().delete_document(doc_id)
        document_registry.pop(doc_id, None)
        return jsonify({"message": f"Document {doc_id} deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Provide a 'question' field in JSON body."}), 400

    question = data["question"]
    top_k = data.get("top_k", DEFAULT_TOP_K)

    try:
        result = run_rag_agent(question, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Query failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
