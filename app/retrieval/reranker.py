from typing import List, Tuple

from sentence_transformers import CrossEncoder

from app.config import settings


class Reranker:
    """
    Cross-encoder re-ranking of candidate chunks.

    Why cross-encoder after FAISS + BM25?
    ───────────────────────────────────────
    FAISS uses a bi-encoder: query and chunk are embedded separately,
    then compared by cosine similarity. Fast but less precise.

    Cross-encoder reads (query + chunk) TOGETHER in one pass — much more
    accurate because it sees the full context of both at once.

    Workflow:
      hybrid_search() → top 100 candidates  (fast, approximate)
      reranker.rerank() → top 20 final       (slow, precise)

    We only run the cross-encoder on 100 candidates, not all chunks —
    so it stays fast even with thousands of documents.
    """

    def __init__(self):
        self._model = CrossEncoder(settings.RERANKER_MODEL)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],   # [(chunk_id, chunk_text), ...]
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Score each (query, chunk_text) pair and return top_k sorted best-first.
        Returns list of (chunk_id, score).
        """
        if not candidates:
            return []

        pairs  = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip([cid for cid, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(cid, float(score)) for cid, score in ranked[:top_k]]


# Singleton — loaded once, reused across all queries
reranker = Reranker()
