import json
import os

import numpy as np
from fastapi import APIRouter, HTTPException

from app.storage.db import SessionLocal
from app.storage import models
from app.storage.vector_store import vector_store
from app.storage.bm25_store import bm25_store
from app.ingestion.embedder import embedder

router = APIRouter()


@router.get("/documents/guide")
def get_source_guide(doc_id: str):
    """Return the auto-generated summary and key topics for a document."""
    db = SessionLocal()
    try:
        guide = db.query(models.SourceGuide).filter_by(doc_id=doc_id).first()
        if not guide:
            raise HTTPException(status_code=404, detail="Source guide not found")
        return {
            "doc_id":     guide.doc_id,
            "summary":    guide.summary,
            "key_topics": json.loads(guide.key_topics or "[]"),
            "created_at": guide.created_at.isoformat(),
        }
    finally:
        db.close()


@router.get("/documents")
def list_documents(project_id: str):
    """
    Return all documents in a project.
    Auto-removes any whose PDF file was manually deleted from disk.
    """
    db = SessionLocal()
    try:
        docs = (
            db.query(models.Document)
            .filter_by(project_id=project_id)
            .order_by(models.Document.upload_date.desc())
            .all()
        )

        # ── Live sync: remove records for files deleted from disk ────────
        removed_any = False
        for doc in list(docs):
            if doc.file_path and not os.path.exists(doc.file_path):
                db.query(models.SourceGuide).filter_by(doc_id=doc.doc_id).delete()
                db.delete(doc)
                docs.remove(doc)
                removed_any = True

        if removed_any:
            db.commit()
            # Rebuild indexes so deleted content is no longer searchable
            vector_store.delete_project(project_id)
            bm25_store.delete_project(project_id)
            remaining = db.query(models.Chunk).filter_by(project_id=project_id).all()
            if remaining:
                texts     = [c.text     for c in remaining]
                chunk_ids = [c.chunk_id for c in remaining]
                BATCH = 512
                all_positions = []
                for i in range(0, len(texts), BATCH):
                    embs = embedder.embed_texts(texts[i:i+BATCH])
                    pos  = vector_store.add(project_id, embs, chunk_ids[i:i+BATCH])
                    all_positions.extend(pos)
                for chunk, new_idx in zip(remaining, all_positions):
                    chunk.faiss_idx = new_idx
                db.commit()
                bm25_store.add(project_id, texts, chunk_ids)
        # ────────────────────────────────────────────────────────────────

        return [
            {
                "doc_id":      d.doc_id,
                "filename":    d.filename,
                "title":       d.title,
                "page_count":  d.page_count,
                "status":      d.status,
                "upload_date": d.upload_date.isoformat() if d.upload_date else None,
            }
            for d in docs
        ]
    finally:
        db.close()


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """
    Delete a document and all its chunks from DB and disk.
    Rebuilds the FAISS and BM25 indexes from the remaining chunks
    so that deleted document's content is no longer searchable.
    """
    db = SessionLocal()
    try:
        doc = db.query(models.Document).filter_by(doc_id=doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        project_id = doc.project_id

        # Remove raw PDF from disk
        if doc.file_path and os.path.exists(doc.file_path):
            os.remove(doc.file_path)

        # Delete doc + chunks from DB (chunks cascade)
        db.delete(doc)
        db.commit()

        # ── Rebuild indexes from remaining chunks ─────────────────────
        # Wipe existing indexes for the project
        vector_store.delete_project(project_id)
        bm25_store.delete_project(project_id)

        # Load all chunks that still belong to this project
        remaining = (
            db.query(models.Chunk)
            .filter_by(project_id=project_id)
            .all()
        )

        if remaining:
            texts     = [c.text      for c in remaining]
            chunk_ids = [c.chunk_id  for c in remaining]

            # Re-embed and rebuild FAISS
            BATCH = 512
            all_positions = []
            for i in range(0, len(texts), BATCH):
                batch_texts = texts[i : i + BATCH]
                batch_ids   = chunk_ids[i : i + BATCH]
                embeddings  = embedder.embed_texts(batch_texts)
                positions   = vector_store.add(project_id, embeddings, batch_ids)
                all_positions.extend(positions)

            # Update faiss_idx in DB to new positions
            for chunk, new_idx in zip(remaining, all_positions):
                chunk.faiss_idx = new_idx
            db.commit()

            # Rebuild BM25
            bm25_store.add(project_id, texts, chunk_ids)

        return {"status": "deleted", "doc_id": doc_id}
    finally:
        db.close()
