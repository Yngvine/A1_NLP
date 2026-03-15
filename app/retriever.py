import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION,
    CHROMA_HOST,
    CHROMA_PORT,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
)


class CaptionRetriever:
    def __init__(self):
        self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance_score": 1.0 - results["distances"][0][i],
                }
            )
        return chunks

    def health_check(self) -> bool:
        try:
            self.chroma_client.heartbeat()
            return True
        except Exception:
            return False
