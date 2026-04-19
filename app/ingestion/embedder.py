from typing import List
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings


# Number of parallel threads for embedding.
# sentence-transformers releases the GIL during inference, so threads help.
# Keep at 2 to avoid OOM; increase to 4 if you have 16GB+ RAM.
_EMBED_WORKERS = 2

# Optimal single-batch size for all-MiniLM-L6-v2 on CPU.
# Larger = more RAM; smaller = more overhead. 256 is a sweet spot.
_SINGLE_BATCH = 256


class Embedder:
    """
    Converts text into 384-dim float32 vectors using sentence-transformers.
    Loaded once and reused — model stays in memory across calls.

    When embed_texts receives a large list it splits it into sub-batches and
    runs them in parallel threads, cutting CPU embedding time roughly in half.
    """

    DIMENSION = 384  # output size of all-MiniLM-L6-v2

    def __init__(self):
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def embed_texts(
        self, texts: List[str], batch_size: int = _SINGLE_BATCH
    ) -> np.ndarray:
        """
        Embed a list of texts in parallel batches.
        Returns a float32 numpy array of shape (len(texts), 384).
        normalize_embeddings=True means dot product == cosine similarity.
        """
        if len(texts) <= batch_size:
            # Small list — single call, no overhead
            return self._encode_batch(texts, batch_size)

        # Split into sub-batches and run in parallel
        sub_batches = [
            texts[i : i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]

        results: List[np.ndarray] = [None] * len(sub_batches)  # type: ignore

        def _run(idx: int, batch: List[str]) -> tuple[int, np.ndarray]:
            return idx, self._encode_batch(batch, batch_size)

        with ThreadPoolExecutor(max_workers=_EMBED_WORKERS) as pool:
            futures = {pool.submit(_run, i, b): i for i, b in enumerate(sub_batches)}
            for future in futures:
                idx, arr = future.result()
                results[idx] = arr

        return np.vstack(results).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns shape (1, 384) — ready to pass directly to FAISS search.
        """
        embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    # ── Private ──────────────────────────────────────────────────────────

    def _encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Run a single encoding pass — called from main thread or worker threads."""
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)


# Singleton — loaded once when the module is first imported
embedder = Embedder()
