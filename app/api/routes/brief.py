"""
POST /brief?project_id=...  — generate a 3-sentence collection briefing
"""
import json
import logging

from fastapi import APIRouter
from langchain_groq import ChatGroq

from app.config import settings
from app.storage.db import SessionLocal
from app.storage import models

router = APIRouter()
logger = logging.getLogger(__name__)

_llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.3,
    max_tokens=300,
)


@router.post("/brief", tags=["Brief"])
def generate_brief(project_id: str):
    """
    Generates a 3-sentence briefing about the whole document collection
    using the per-document source guides as input.
    """
    db = SessionLocal()
    try:
        docs = (
            db.query(models.Document)
            .filter_by(project_id=project_id, status="indexed")
            .all()
        )
        if not docs:
            return {"brief": "No indexed documents found in this project."}

        # Build summary input from source guides
        parts = []
        for doc in docs:
            guide = db.query(models.SourceGuide).filter_by(doc_id=doc.doc_id).first()
            if guide and guide.summary:
                topics = json.loads(guide.key_topics or "[]")
                topic_str = ", ".join(topics) if topics else "general topics"
                parts.append(f"- {doc.filename}: {guide.summary} (Topics: {topic_str})")
            else:
                parts.append(f"- {doc.filename}: (no guide available)")

        if not parts:
            return {"brief": "Source guides not yet generated. Upload and index documents first."}

        summaries = "\n".join(parts)
        prompt = (
            f"Based on these document summaries from a research collection, "
            f"write a 3-sentence briefing that describes:\n"
            f"1. What this collection is about overall\n"
            f"2. The key themes or findings\n"
            f"3. What questions this collection can help answer\n\n"
            f"Documents:\n{summaries}\n\n"
            f"Return ONLY the 3-sentence briefing, no labels or headings."
        )

        resp  = _llm.invoke(prompt)
        brief = resp.content.strip()
        return {"brief": brief}

    except Exception as e:
        logger.error(f"Brief generation failed: {e}")
        return {"brief": "Could not generate collection brief."}
    finally:
        db.close()
