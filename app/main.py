import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.storage.db import create_tables, SessionLocal
from app.storage import models
from app.storage.vector_store import vector_store
from app.storage.bm25_store import bm25_store
from app.api.routes import upload, documents, jobs, query
from app.api.routes import contradictions, gaps, questions, insights, timeline, brief

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuMind API",
    description="Multi-PDF RAG backend — upload, index, query",
    version="1.0.0",
)

# Allow Streamlit (port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Create DB tables and sync DB with files on disk."""
    create_tables()
    _sync_deleted_files()


def _sync_deleted_files():
    """
    Remove DB records for any document whose PDF file no longer exists on disk.
    Also wipes and rebuilds the FAISS + BM25 indexes for affected projects.
    """
    db = SessionLocal()
    try:
        all_docs = db.query(models.Document).all()
        affected_projects = set()

        for doc in all_docs:
            if doc.file_path and not os.path.exists(doc.file_path):
                logger.info(f"[sync] File missing on disk, removing from DB: {doc.filename}")
                affected_projects.add(doc.project_id)
                db.delete(doc)

        if affected_projects:
            db.commit()
            # Wipe indexes for affected projects — they will be rebuilt from
            # remaining chunks below
            for pid in affected_projects:
                vector_store.delete_project(pid)
                bm25_store.delete_project(pid)

            # Rebuild indexes from whatever chunks remain
            from app.ingestion.embedder import embedder
            for pid in affected_projects:
                remaining = db.query(models.Chunk).filter_by(project_id=pid).all()
                if not remaining:
                    continue
                texts     = [c.text     for c in remaining]
                chunk_ids = [c.chunk_id for c in remaining]
                BATCH = 512
                all_positions = []
                for i in range(0, len(texts), BATCH):
                    embs = embedder.embed_texts(texts[i:i+BATCH])
                    pos  = vector_store.add(pid, embs, chunk_ids[i:i+BATCH])
                    all_positions.extend(pos)
                for chunk, new_idx in zip(remaining, all_positions):
                    chunk.faiss_idx = new_idx
                db.commit()
                bm25_store.add(pid, texts, chunk_ids)
                logger.info(f"[sync] Rebuilt indexes for project={pid} chunks={len(remaining)}")
    except Exception as e:
        logger.error(f"[sync] Startup sync failed: {e}")
        db.rollback()
    finally:
        db.close()


# Register routes
app.include_router(upload.router,         tags=["Upload"])
app.include_router(documents.router,      tags=["Documents"])
app.include_router(jobs.router,           tags=["Jobs"])
app.include_router(query.router,          tags=["Query"])
app.include_router(contradictions.router, tags=["Contradictions"])
app.include_router(gaps.router,           tags=["Knowledge Gaps"])
app.include_router(questions.router,      tags=["Smart Questions"])
app.include_router(insights.router,       tags=["Insights"])
app.include_router(timeline.router,       tags=["Timeline"])
app.include_router(brief.router,          tags=["Brief"])


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
