"""
POST /timeline?project_id=...   — generate and return document evolution timeline
(on-demand, not cached — runs the full LangGraph pipeline synchronously)
"""
import logging

from fastapi import APIRouter
from app.reasoning.timeline_graph import timeline_chain

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/timeline", tags=["Timeline"])
def get_timeline(project_id: str):
    """
    Run the timeline pipeline and return events immediately.
    May take 10-30s depending on document count.
    """
    try:
        result = timeline_chain.invoke({
            "project_id": project_id,
            "doc_texts":  [],
            "raw_events": [],
            "timeline":   [],
        })
        return {"timeline": result.get("timeline", [])}
    except Exception as e:
        logger.error(f"Timeline pipeline error: {e}")
        return {"timeline": [], "error": str(e)}
