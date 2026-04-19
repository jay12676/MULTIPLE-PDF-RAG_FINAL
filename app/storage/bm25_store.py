import os
import pickle
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from app.config import settings


class BM25Store:
    """
    Keyword-based search index (one per project_id).
    Stored as a pickle file on disk alongside the FAISS index.

    Why BM25 alongside FAISS?
    ─────────────────────────
    FAISS works on meaning  → "medication side effects" matches "drug adverse reactions"
    BM25  works on words    → "FDA" only matches chunks that literally contain "FDA"

    BM25 is critical for:
      • Exact names     — person names, company names, drug names
      • Acronyms        — FDA, RCT, GDP, API, NLP
      • Specific codes  — ICD-10 codes, legal article numbers
      • Unique IDs      — contract numbers, case IDs

    Both indexes are searched on every query.
    Results are merged using Reciprocal Rank Fusion (done in hybrid_search.py).
    """

    def __init__(self):
        # project_id → BM25Okapi index
        self._indexes: dict[str, BM25Okapi] = {}
        # project_id → list of chunk_ids (position matches BM25 corpus position)
        self._chunk_ids: dict[str, List[str]] = {}
        # project_id → tokenized corpus (needed to rebuild index on new additions)
        self._corpus: dict[str, List[List[str]]] = {}
        # tracks disk file mtime at last load — detects Celery worker updates
        self._mtimes: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        project_id: str,
        texts: List[str],
        chunk_ids: List[str],
    ):
        """
        Add a batch of chunks to the BM25 index for a project.
        Rebuilds the full index after adding (BM25Okapi is not incremental).

        Empty-tokenized chunks are excluded: they can never match a keyword
        query, and BM25Okapi raises ZeroDivisionError on an empty corpus.
        """
        self._load(project_id)

        # Keep only chunks that produce at least one token after tokenization.
        # Chunks with no tokens (empty text, image-only pages) are useless for
        # keyword search and would cause BM25Okapi to crash with ZeroDivisionError.
        for text, cid in zip(texts, chunk_ids):
            tokens = self._tokenize(text)
            if tokens:
                self._corpus[project_id].append(tokens)
                self._chunk_ids[project_id].append(cid)

        # Only build the index when we have at least one non-empty document.
        if self._corpus[project_id]:
            self._indexes[project_id] = BM25Okapi(self._corpus[project_id])
        # else: index stays None — search() already handles None by returning []

        self._save(project_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        project_id: str,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Returns list of (chunk_id, score) sorted best-first.
        Only returns chunks with score > 0 (i.e. at least one keyword matched).
        """
        self._load(project_id)

        index = self._indexes.get(project_id)
        if index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = index.get_scores(tokenized_query)

        # sort indices by score descending, keep top_k
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        chunk_ids = self._chunk_ids[project_id]
        return [
            (chunk_ids[i], float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_project(self, project_id: str):
        """Remove BM25 index for a project from memory and disk."""
        self._indexes.pop(project_id, None)
        self._chunk_ids.pop(project_id, None)
        self._corpus.pop(project_id, None)

        path = self._store_path(project_id)
        if os.path.exists(path):
            os.remove(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer + lowercase."""
        return text.lower().split()

    def _store_path(self, project_id: str) -> str:
        dir_path = os.path.join(settings.INDEX_DIR, project_id)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, "bm25.pkl")

    def _load(self, project_id: str):
        """
        Load from disk if not already in memory or if the on-disk file has
        been updated since last load.

        The Celery worker runs in a separate process and writes new data to
        disk after each ingest. Checking the file's mtime lets the FastAPI
        process detect those writes and refresh its in-memory state.
        """
        path = self._store_path(project_id)
        if os.path.exists(path):
            current_mtime = os.path.getmtime(path)
            if (project_id not in self._indexes
                    or self._mtimes.get(project_id, -1) != current_mtime):
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self._indexes[project_id]   = data["index"]
                self._chunk_ids[project_id] = data["chunk_ids"]
                self._corpus[project_id]    = data["corpus"]
                self._mtimes[project_id]    = current_mtime
        elif project_id not in self._indexes:
            # First time — start empty
            self._indexes[project_id]   = None
            self._chunk_ids[project_id] = []
            self._corpus[project_id]    = []
            self._mtimes[project_id]    = -1

    def _save(self, project_id: str):
        """Persist index, chunk IDs, and corpus to disk."""
        with open(self._store_path(project_id), "wb") as f:
            pickle.dump(
                {
                    "index":     self._indexes[project_id],
                    "chunk_ids": self._chunk_ids[project_id],
                    "corpus":    self._corpus[project_id],
                },
                f,
            )


# Singleton — shared across the whole app
bm25_store = BM25Store()
