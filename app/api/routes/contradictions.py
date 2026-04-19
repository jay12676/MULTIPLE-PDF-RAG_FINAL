"""
POST /contradictions/run?project_id=...  — run pipeline and return results immediately (synchronous)
GET  /contradictions?project_id=...      — return cached contradictions
"""
import logging

from fastapi import APIRouter
from app.storage.db import SessionLocal
from app.storage import models
from app.reasoning.contradiction_graph import contradiction_chain

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/contradictions/run", tags=["Contradictions"])
def run_contradictions(project_id: str):
    """Run contradiction detection synchronously and return results immediately."""
    try:
        contradiction_chain.invoke({
            "project_id":     project_id,
            "doc_chunks":     {},
            "contradictions": [],
        })
    except Exception as e:
        logger.error(f"Contradiction pipeline error: {e}")
    return get_contradictions(project_id)


@router.get("/contradictions", tags=["Contradictions"])
def get_contradictions(project_id: str):
    """Return all stored contradictions for a project."""
    db = SessionLocal()
    try:
        rows = (
            db.query(models.Contradiction)
            .filter_by(project_id=project_id)
            .order_by(models.Contradiction.confidence.desc())
            .all()
        )
        result = []
        for r in rows:
            doc1 = db.query(models.Document).filter_by(doc_id=r.doc_id_1).first()
            doc2 = db.query(models.Document).filter_by(doc_id=r.doc_id_2).first()
            result.append({
                "id":          r.id,
                "topic":       r.topic,
                "doc_a":       doc1.filename if doc1 else r.doc_id_1,
                "text_a":      r.text_1,
                "doc_b":       doc2.filename if doc2 else r.doc_id_2,
                "text_b":      r.text_2,
                "explanation": r.explanation,
                "confidence":  r.confidence,
                "created_at":  r.created_at.isoformat(),
            })
        return result
    finally:
        db.close()
