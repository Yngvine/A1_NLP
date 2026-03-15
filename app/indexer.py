import json

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    CAPTIONS_PATH,
    CHROMA_COLLECTION,
    CHROMA_HOST,
    CHROMA_PORT,
    DEFAULT_INGEST_LIMIT,
    EMBEDDING_MODEL,
)
from minio_client import MinIOClient


class CaptionIndexer:
    def __init__(self):
        self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.minio = MinIOClient()

    def load_captions(self, limit: int = DEFAULT_INGEST_LIMIT) -> pd.DataFrame:
        df = pd.read_parquet(CAPTIONS_PATH)
        df = df[df["caption"].str.len() > 0]
        if limit is None or int(limit) <= 0:
            return df
        return df.head(int(limit))

    def ingest(self, limit: int = DEFAULT_INGEST_LIMIT) -> dict:
        df = self.load_captions(limit)

        doc_ids = []
        texts = []
        metadatas = []

        for _, row in df.iterrows():
            doc_id = f"caption-{row['id']}"
            doc_ids.append(doc_id)
            texts.append(row["caption"])

            tags = row["tags"]
            if hasattr(tags, "tolist"):
                tags = tags.tolist()
            tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)

            metadatas.append(
                {
                    "source_id": str(row["id"]),
                    "image": row["image"],
                    "tags": tag_str,
                    "source": "RSVLM-QA",
                }
            )

            # Store in MinIO
            caption_json = json.dumps(
                {
                    "id": str(row["id"]),
                    "image": row["image"],
                    "caption": row["caption"],
                    "tags": tags if isinstance(tags, list) else [str(tags)],
                }
            ).encode()
            self.minio.store_document(doc_id, caption_json, "caption.json")

        # Batch embed
        embeddings = self.embedder.encode(
            texts, show_progress_bar=True, batch_size=64
        ).tolist()

        # Upsert into ChromaDB in batches
        batch_size = 500
        for i in range(0, len(doc_ids), batch_size):
            end = min(i + batch_size, len(doc_ids))
            self.collection.upsert(
                ids=doc_ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )

        return {"ingested": len(doc_ids), "collection_count": self.collection.count()}

    def index_text_chunks(
        self, doc_id: str, chunks: list[str], filename: str
    ) -> dict:
        """Index text chunks from an uploaded PDF/DOCX document."""
        chunk_ids = [f"{doc_id}-chunk-{i}" for i in range(len(chunks))]
        metadatas = [
            {"source_id": doc_id, "image": "", "tags": "", "source": filename}
            for _ in chunks
        ]

        embeddings = self.embedder.encode(
            chunks, show_progress_bar=False, batch_size=64
        ).tolist()

        self.collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return {"indexed_chunks": len(chunks)}

    def delete_document(self, doc_id: str):
        # Delete from ChromaDB (get all IDs matching doc_id prefix)
        all_docs = self.collection.get(where={"source_id": doc_id})
        if all_docs["ids"]:
            self.collection.delete(ids=all_docs["ids"])
        # Also try the exact ID
        try:
            self.collection.delete(ids=[doc_id])
        except Exception:
            pass
        # Delete from MinIO
        self.minio.delete_document(doc_id)

    def get_all_documents(self) -> list:
        result = self.collection.get(include=["metadatas"])
        docs = {}
        for id_, meta in zip(result["ids"], result["metadatas"]):
            source_id = meta.get("source_id", id_)
            if source_id not in docs:
                docs[source_id] = {
                    "id": source_id,
                    "source": meta.get("source", ""),
                    "image": meta.get("image", ""),
                    "tags": meta.get("tags", ""),
                }
        return list(docs.values())

    def get_collection_count(self) -> int:
        return self.collection.count()
