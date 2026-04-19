"""
POST /questions/run?project_id=...  — run pipeline and return results immediately (synchronous)
GET  /questions?project_id=...      — return cached smart questions
"""
import logging

from fastapi import APIRouter
from app.storage.db import SessionLocal
from app.storage import models
from app.reasoning.questions_graph import questions_chain

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/questions/run", tags=["Smart Questions"])
def run_questions(project_id: str, topic: str = ""):
    """Run question generation synchronously and return results immediately."""
    try:
        questions_chain.invoke({
            "project_id": project_id,
            "topic":      topic,
            "context":    "",
            "questions":  [],
        })
    except Exception as e:
        logger.error(f"Questions pipeline error: {e}")
    return get_questions(project_id)


@router.get("/questions", tags=["Smart Questions"])
def get_questions(project_id: str):
    """Return all stored smart questions for a project."""
    db = SessionLocal()
    try:
        rows = (
            db.query(models.SmartQuestion)
            .filter_by(project_id=project_id)
            .order_by(models.SmartQuestion.category, models.SmartQuestion.created_at)
            .all()
        )
        return [
            {
                "id":       r.id,
                "question": r.question,
                "category": r.category,
            }
            for r in rows
        ]
    finally:
        db.close()
