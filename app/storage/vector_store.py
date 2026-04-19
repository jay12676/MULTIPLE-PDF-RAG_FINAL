import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

from app.config import settings


class VectorStore:
    """
    FAISS HNSW index, one shard per project_id.

    - Loaded lazily from disk on first access.
    - Written to disk after every batch of additions.
    - Only active project shards stay in RAM.

    Why HNSW?
      Approximate nearest-neighbour search in O(log n) time.
      Fast enough for millions of chunks without needing a GPU.
    """

    DIMENSION = 384          # must match embedder output
    HNSW_M = 32              # number of connections per node (higher = more accurate, more RAM)
    EF_CONSTRUCTION = 200    # build-time accuracy
    EF_SEARCH = 50           # query-time accuracy

    def __init__(self):
        self._indexes: dict[str, faiss.Index] = {}
        # maps position-in-FAISS-index → chunk_id  (per project)
        self._id_maps: dict[str, List[str]] = {}
        # tracks disk file mtime at last load — detects Celery worker updates
        self._mtimes: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        project_id: str,
        embeddings: np.ndarray,      # shape (N, 384)
        chunk_ids: List[str],
    ) -> List[int]:
        """
        Add embeddings to the project's index AND immediately save to disk.
        Use add_batch() + flush() instead when adding multiple batches to
        avoid redundant disk writes.
        Returns the FAISS positions assigned to each chunk.
        """
        self._load(project_id)

        start_pos = self._indexes[project_id].ntotal
        self._indexes[project_id].add(embeddings)
        self._id_maps[project_id].extend(chunk_ids)

        self._save(project_id)

        return list(range(start_pos, start_pos + len(chunk_ids)))

    def add_batch(
        self,
        project_id: str,
        embeddings: np.ndarray,
        chunk_ids: List[str],
    ) -> List[int]:
        """
        Add a batch of embeddings to the in-memory index WITHOUT saving to disk.
        Call flush(project_id) once after all batches are added.

        This avoids rewriting the full .index file after every 512-chunk batch
        (which adds ~0.5–1s per batch = 3–6s wasted for a 3000-chunk document).
        """
        self._load(project_id)

        start_pos = self._indexes[project_id].ntotal
        self._indexes[project_id].add(embeddings)
        self._id_maps[project_id].extend(chunk_ids)
        # No _save() here — caller must call flush() when done

        return list(range(start_pos, start_pos + len(chunk_ids)))

    def flush(self, project_id: str):
        """
        Persist the current in-memory index to disk.
        Call this once after all add_batch() calls are complete.
        """
        if project_id in self._indexes:
            self._save(project_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        project_id: str,
        query_embedding: np.ndarray,   # shape (1, 384)
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Returns list of (chunk_id, score) sorted best-first.
        Score is cosine similarity (0–1, higher = more relevant).
        """
        self._load(project_id)
        index = self._indexes[project_id]

        if index.ntotal == 0:
            return []

        k = min(top_k, index.ntotal)
        scores, positions = index.search(query_embedding, k)

        results = []
        id_map = self._id_maps[project_id]
        for score, pos in zip(scores[0], positions[0]):
            if 0 <= pos < len(id_map):
                results.append((id_map[pos], float(score)))

        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_project(self, project_id: str):
        """Remove all vectors for a project from memory and disk."""
        self._indexes.pop(project_id, None)
        self._id_maps.pop(project_id, None)

        for path in [self._index_path(project_id), self._map_path(project_id)]:
            if os.path.exists(path):
                os.remove(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _project_dir(self, project_id: str) -> str:
        path = os.path.join(settings.INDEX_DIR, project_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _index_path(self, project_id: str) -> str:
        return os.path.join(self._project_dir(project_id), "faiss.index")

    def _map_path(self, project_id: str) -> str:
        return os.path.join(self._project_dir(project_id), "chunk_ids.pkl")

    def _load(self, project_id: str):
        """
        Load index from disk into memory if not already loaded or if the
        on-disk file has been updated since last load.

        The Celery worker runs in a separate process and writes new indexes
        to disk after each ingest. Using the file's mtime lets the FastAPI
        process detect those writes and refresh its in-memory state without
        any cross-process signalling.
        """
        index_path = self._index_path(project_id)
        map_path   = self._map_path(project_id)

        if os.path.exists(index_path) and os.path.exists(map_path):
            current_mtime = os.path.getmtime(index_path)
            # Re-load if: first time, or disk file is newer than what we have
            if (project_id not in self._indexes
                    or self._mtimes.get(project_id, -1) != current_mtime):
                self._indexes[project_id] = faiss.read_index(index_path)
                with open(map_path, "rb") as f:
                    self._id_maps[project_id] = pickle.load(f)
                self._mtimes[project_id] = current_mtime
        elif project_id not in self._indexes:
            # First time for this project — create a fresh HNSW index
            index = faiss.IndexHNSWFlat(self.DIMENSION, self.HNSW_M)
            index.hnsw.efConstruction = self.EF_CONSTRUCTION
            index.hnsw.efSearch = self.EF_SEARCH
            self._indexes[project_id] = index
            self._id_maps[project_id] = []
            self._mtimes[project_id] = -1

    def _save(self, project_id: str):
        """Persist index and ID map to disk."""
        faiss.write_index(
            self._indexes[project_id],
            self._index_path(project_id),
        )
        with open(self._map_path(project_id), "wb") as f:
            pickle.dump(self._id_maps[project_id], f)


# Singleton — one VectorStore shared across the whole app
vector_store = VectorStore()
