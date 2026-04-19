"""
POST /insights/run?project_id=...  — run pipeline and return results immediately (synchronous)
GET  /insights?project_id=...      — return cached proactive insights
"""
import logging

from fastapi import APIRouter
from app.storage.db import SessionLocal
from app.storage import models
from app.reasoning.insights_graph import insights_chain

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/insights/run", tags=["Insights"])
def run_insights(project_id: str, topic: str = ""):
    """Run insight mining synchronously and return results immediately."""
    try:
        insights_chain.invoke({
            "project_id": project_id,
            "topic":      topic,
            "evidence":   "",
            "doc_map":    {},
            "insights":   [],
        })
    except Exception as e:
        logger.error(f"Insights pipeline error: {e}")
    return get_insights(project_id)


@router.get("/insights", tags=["Insights"])
def get_insights(project_id: str):
    """Return all stored insights for a project."""
    db = SessionLocal()
    try:
        rows = (
            db.query(models.Insight)
            .filter_by(project_id=project_id)
            .order_by(models.Insight.insight_type, models.Insight.created_at.desc())
            .all()
        )
        result = []
        for r in rows:
            supporting_filenames = []
            if r.supporting_docs:
                for doc_id in r.supporting_docs.split(","):
                    doc = db.query(models.Document).filter_by(doc_id=doc_id.strip()).first()
                    if doc:
                        supporting_filenames.append(doc.filename)
            result.append({
                "id":              r.id,
                "insight_type":    r.insight_type,
                "title":           r.title,
                "description":     r.description,
                "supporting_docs": supporting_filenames,
                "created_at":      r.created_at.isoformat(),
            })
        return result
    finally:
        db.close()
