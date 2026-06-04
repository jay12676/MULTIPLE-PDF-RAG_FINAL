from typing import List

import numpy as np
from fastembed import TextEmbedding

from app.config import settings


# Keep ONNX-runtime threads low so the model fits comfortably on a small VPS
# (2 GB / 2 vCPU). Raise to 4 if you have more cores/RAM.
_THREADS = 2

# Optimal batch size for all-MiniLM-L6-v2 on CPU. Larger = more RAM.
_SINGLE_BATCH = 256


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Make every row a unit vector so dot product == cosine similarity."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class Embedder:
    """
    Converts text into 384-dim float32 vectors.

    Uses fastembed (ONNX runtime) instead of sentence-transformers (PyTorch).
    The model is identical (all-MiniLM-L6-v2) but the runtime is ~10x lighter
    on RAM, which is what lets the whole app fit inside a 2 GB droplet.

    Loaded once and reused — the model stays in memory across calls.
    """

    DIMENSION = 384  # output size of all-MiniLM-L6-v2

    def __init__(self):
        # fastembed downloads a small quantized ONNX model on first use and
        # caches it; subsequent loads are instant.
        self._model = TextEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            threads=_THREADS,
        )
        # Warm up the ONNX session at startup so the FIRST real embed call is
        # fast (otherwise the first PDF pays the one-time session-init cost).
        try:
            list(self._model.embed(["warmup"]))
        except Exception:
            pass

    def embed_texts(
        self, texts: List[str], batch_size: int = _SINGLE_BATCH
    ) -> np.ndarray:
        """
        Embed a list of texts.
        Returns a float32 numpy array of shape (len(texts), 384), L2-normalised.
        """
        if not texts:
            return np.zeros((0, self.DIMENSION), dtype=np.float32)

        vecs = np.asarray(
            list(self._model.embed(texts, batch_size=batch_size)),
            dtype=np.float32,
        )
        return _l2_normalize(vecs)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns shape (1, 384) — ready to pass directly to FAISS search.
        """
        vec = np.asarray(list(self._model.embed([query])), dtype=np.float32)
        return _l2_normalize(vec)


# Singleton — loaded once when the module is first imported
embedder = Embedder()
