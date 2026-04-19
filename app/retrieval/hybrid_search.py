from typing import Dict, List, Tuple

from app.ingestion.embedder import embedder
from app.storage.bm25_store import bm25_store
from app.storage.vector_store import vector_store


def hybrid_search(
    project_id: str,
    query: str,
    top_k: int = 100,
) -> List[Tuple[str, float]]:
    """
    Runs FAISS (dense) + BM25 (sparse) search in parallel and merges
    results using Reciprocal Rank Fusion (RRF).

    Why both?
      FAISS  → "medication side effects" matches "drug adverse reactions"  (meaning)
      BM25   → "FDA"  only matches chunks that literally say "FDA"          (exact words)

    Returns list of (chunk_id, rrf_score) sorted best-first, length = top_k.
    """
    # Dense search — embed query then search FAISS
    query_vec    = embedder.embed_query(query)
    dense_hits   = vector_store.search(project_id, query_vec, top_k=30)

    # Sparse search — BM25 keyword matching
    sparse_hits  = bm25_store.search(project_id, query, top_k=30)

    # Merge with RRF
    return _rrf(dense_hits, sparse_hits, top_k=top_k)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf(
    list_a: List[Tuple[str, float]],
    list_b: List[Tuple[str, float]],
    k: int = 60,
    top_k: int = 100,
) -> List[Tuple[str, float]]:
    """
    RRF formula:  score(d) = Σ  1 / (k + rank_i)

    k=60 is the standard constant — dampens the effect of very high ranks
    so that a chunk ranked #1 in one list doesn't completely dominate.

    A chunk appearing in BOTH lists gets scored twice → rises to the top.
    This is exactly what we want: chunks relevant by both meaning AND keywords.
    """
    scores: Dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(list_a):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, (chunk_id, _) in enumerate(list_b):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:top_k]
