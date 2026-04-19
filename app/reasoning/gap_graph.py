"""
Task 14 — Knowledge Gap Pipeline
=================================
Analyses what topics the uploaded documents DON'T cover well,
given what they claim to be about.

Flow:
  summarise_docs → identify_gaps → save_gaps → END
"""
import json
import logging
from typing import TypedDict, List

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from app.config import settings
from app.storage.db import SessionLocal
from app.storage import models

logger = logging.getLogger(__name__)

_llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.3,
    max_tokens=1024,
)


# ── State ────────────────────────────────────────────────────────────────────

class GapState(TypedDict):
    project_id: str
    doc_summaries: List[dict]   # [{"filename": str, "summary": str}]
    gaps:          List[dict]


# ── Nodes ────────────────────────────────────────────────────────────────────

def summarise_docs(state: GapState) -> GapState:
    """Collect level-1 document summary chunks (or first few level-3 chunks)."""
    db = SessionLocal()
    try:
        docs = (
            db.query(models.Document)
            .filter_by(project_id=state["project_id"], status="indexed")
            .all()
        )
        summaries: List[dict] = []
        for doc in docs:
            # Level-1 is the document-level summary
            summary_chunk = (
                db.query(models.Chunk)
                .filter_by(doc_id=doc.doc_id, level=1)
                .first()
            )
            if summary_chunk:
                text = summary_chunk.text
            else:
                # Fall back to first 3 level-3 chunks
                chunks = (
                    db.query(models.Chunk)
                    .filter_by(doc_id=doc.doc_id, level=3)
                    .limit(3)
                    .all()
                )
                text = " ".join(c.text for c in chunks)
            if text:
                summaries.append({"filename": doc.filename, "summary": text[:1500]})
        return {**state, "doc_summaries": summaries}
    finally:
        db.close()


def identify_gaps(state: GapState) -> GapState:
    """Ask the LLM to identify what topics are missing or underexplored."""
    if not state["doc_summaries"]:
        return {**state, "gaps": []}

    combined = "\n\n".join(
        f"[{s['filename']}]\n{s['summary']}" for s in state["doc_summaries"]
    )

    prompt = f"""You are a research librarian reviewing a collection of documents.

Document collection overview:
{combined}

Identify 3-6 KNOWLEDGE GAPS — important topics, methods, or evidence that are
ABSENT or insufficiently covered given what these documents address.
Focus on gaps that would be meaningful to a researcher using this collection.

Return a JSON array. Each item:
{{
  "topic": "<gap topic in 5 words or less>",
  "description": "<2-3 sentences explaining what is missing and why it matters>"
}}

Respond with only the JSON array, no markdown."""

    try:
        resp = _llm.invoke([
            SystemMessage(content="You are a research librarian. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        gaps = json.loads(raw)
    except Exception as e:
        logger.warning(f"Gap identification failed: {e}")
        gaps = []

    return {**state, "gaps": gaps}


def save_gaps(state: GapState) -> GapState:
    """Persist gaps to DB."""
    db = SessionLocal()
    try:
        db.query(models.KnowledgeGap).filter_by(
            project_id=state["project_id"]
        ).delete()

        for item in state["gaps"]:
            db.add(models.KnowledgeGap(
                project_id  = state["project_id"],
                topic       = item.get("topic", "Unknown"),
                description = item.get("description", ""),
            ))
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save knowledge gaps: {e}")
        db.rollback()
    finally:
        db.close()
    return state


# ── Graph ────────────────────────────────────────────────────────────────────

def build_gap_graph():
    g = StateGraph(GapState)
    g.add_node("summarise_docs", summarise_docs)
    g.add_node("identify_gaps",  identify_gaps)
    g.add_node("save_gaps",      save_gaps)

    g.set_entry_point("summarise_docs")
    g.add_edge("summarise_docs", "identify_gaps")
    g.add_edge("identify_gaps",  "save_gaps")
    g.add_edge("save_gaps",      END)
    return g.compile()


gap_chain = build_gap_graph()
