"""
POST /gaps/run?project_id=...   — run pipeline and return results immediately (synchronous)
GET  /gaps?project_id=...       — return cached knowledge gaps
"""
import logging

from fastapi import APIRouter
from app.storage.db import SessionLocal
from app.storage import models
from app.reasoning.gap_graph import gap_chain

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/gaps/run", tags=["Knowledge Gaps"])
def run_gaps(project_id: str):
    """Run the gap pipeline synchronously and return results immediately."""
    try:
        gap_chain.invoke({
            "project_id":    project_id,
            "doc_summaries": [],
            "gaps":          [],
        })
    except Exception as e:
        logger.error(f"Gap pipeline error: {e}")
    # Return fresh results after pipeline completes
    return get_gaps(project_id)


@router.get("/gaps", tags=["Knowledge Gaps"])
def get_gaps(project_id: str):
    """Return all stored knowledge gaps for a project."""
    db = SessionLocal()
    try:
        rows = (
            db.query(models.KnowledgeGap)
            .filter_by(project_id=project_id)
            .order_by(models.KnowledgeGap.created_at.desc())
            .all()
        )
        return [
            {
                "id":          r.id,
                "topic":       r.topic,
                "description": r.description,
                "created_at":  r.created_at.isoformat(),
            }
            for r in rows
        ]
    finally:
        db.close()
