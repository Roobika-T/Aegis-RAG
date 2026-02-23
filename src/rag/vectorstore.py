from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer


class InMemoryVectorStore:
    """
    Simple FAISS-backed vector store to simulate the external knowledge base in RAG.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._encoder = SentenceTransformer(model_name)
        self._dim = self._encoder.get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatIP(self._dim)
        self._docs: List[str] = []

    def add_texts(self, texts: List[str]) -> None:
        embeddings = self._encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self._index.add(embeddings)
        self._docs.extend(texts)

    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._docs:
            return []
        q_emb = self._encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self._index.search(q_emb, top_k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._docs):
                continue
            results.append((self._docs[idx], float(score)))
        return results

