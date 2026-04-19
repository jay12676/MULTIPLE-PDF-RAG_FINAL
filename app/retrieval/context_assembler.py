from dataclasses import dataclass
from typing import List, Tuple

from app.storage.db import SessionLocal
from app.storage import models


@dataclass
class RetrievedChunk:
    """One chunk with all metadata needed to build LLM context + citations."""
    chunk_id:    str
    doc_id:      str
    filename:    str
    title:       str
    page_number: int
    section:     str
    text:        str
    score:       float


def assemble_context(
    ranked_chunks: List[Tuple[str, float]],   # [(chunk_id, score), ...]
) -> List[RetrievedChunk]:
    """
    FAISS and BM25 only return chunk_ids + scores — no text.
    This function fetches the actual text and metadata from SQLite
    so the LLM has something to read and we can generate citations.

    Returns chunks in rank order (highest score first).
    """
    if not ranked_chunks:
        return []

    chunk_ids = [cid for cid, _ in ranked_chunks]
    score_map = {cid: score for cid, score in ranked_chunks}

    db = SessionLocal()
    try:
        # Single JOIN query — one DB round-trip for all chunks
        rows = (
            db.query(models.Chunk, models.Document)
            .join(
                models.Document,
                models.Chunk.doc_id == models.Document.doc_id,
            )
            .filter(models.Chunk.chunk_id.in_(chunk_ids))
            .all()
        )

        results = [
            RetrievedChunk(
                chunk_id    = chunk.chunk_id,
                doc_id      = chunk.doc_id,
                filename    = doc.filename,
                title       = doc.title or doc.filename,
                page_number = chunk.page_number,
                section     = chunk.section,
                text        = chunk.text,
                score       = score_map.get(chunk.chunk_id, 0.0),
            )
            for chunk, doc in rows
        ]

        # Restore rank order (DB query doesn't preserve it)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    finally:
        db.close()
