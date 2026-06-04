from typing import List, Tuple

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import settings


# Keep ONNX-runtime threads low for a small VPS (2 GB / 2 vCPU).
_THREADS = 2


class Reranker:
    """
    Cross-encoder re-ranking of candidate chunks.

    Uses fastembed (ONNX runtime) instead of sentence-transformers (PyTorch)
    so it runs without the heavy torch dependency — the underlying model is the
    same ms-marco-MiniLM cross-encoder.

    Why cross-encoder after FAISS + BM25?
    ───────────────────────────────────────
    FAISS/BM25 score query and chunk separately (fast but approximate).
    The cross-encoder reads (query + chunk) TOGETHER in one pass — much more
    accurate. We only run it on ~25 candidates, so it stays fast.
    """

    def __init__(self):
        self._model = TextCrossEncoder(
            model_name=settings.RERANKER_MODEL,
            threads=_THREADS,
        )
        # Warm up the ONNX session at startup so the FIRST real query is fast.
        try:
            list(self._model.rerank("warmup", ["warmup text"]))
        except Exception:
            pass

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

        docs   = [text for _, text in candidates]
        scores = list(self._model.rerank(query, docs))

        ranked = sorted(
            zip([cid for cid, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(cid, float(score)) for cid, score in ranked[:top_k]]


# Singleton — loaded once, reused across all queries
reranker = Reranker()
