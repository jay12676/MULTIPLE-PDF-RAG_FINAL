"""
Task 16 — Proactive Insight Engine
=====================================
Proactively surfaces patterns, trends, and anomalies the user may not think to ask about.
Types: pattern | trend | anomaly

Flow:
  collect_evidence → mine_insights → save_insights → END
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
    temperature=0.4,
    max_tokens=1200,
)


# ── State ────────────────────────────────────────────────────────────────────

class InsightsState(TypedDict):
    project_id:     str
    topic:          str          # optional focus topic from the user (empty = all topics)
    evidence:       str          # assembled text from docs
    doc_map:        dict         # doc_id → filename
    insights:       List[dict]


# ── Nodes ────────────────────────────────────────────────────────────────────

def collect_evidence(state: InsightsState) -> InsightsState:
    """Collect level-2 section chunks from all indexed documents."""
    db = SessionLocal()
    try:
        docs = (
            db.query(models.Document)
            .filter_by(project_id=state["project_id"], status="indexed")
            .all()
        )
        doc_map: dict = {}
        parts: List[str] = []
        for doc in docs:
            doc_map[doc.doc_id] = doc.filename
            chunks = (
                db.query(models.Chunk)
                .filter_by(doc_id=doc.doc_id, level=2)
                .limit(5)
                .all()
            )
            if not chunks:
                chunks = (
                    db.query(models.Chunk)
                    .filter_by(doc_id=doc.doc_id, level=3)
                    .limit(3)
                    .all()
                )
            if chunks:
                excerpt = " ".join(c.text[:400] for c in chunks)
                parts.append(f"[{doc.filename}]\n{excerpt}")

        evidence = "\n\n---\n\n".join(parts[:12])
        return {**state, "evidence": evidence, "doc_map": doc_map}
    finally:
        db.close()


def mine_insights(state: InsightsState) -> InsightsState:
    """Ask the LLM to surface non-obvious insights."""
    if not state["evidence"]:
        return {**state, "insights": []}

    filenames = ", ".join(state["doc_map"].values())
    topic = state.get("topic", "").strip()
    topic_line = (
        f"\nFocus specifically on the topic: **{topic}**\n"
        if topic else
        "\nSurface insights across all themes in the documents.\n"
    )

    prompt = f"""You are an expert analyst. You have access to excerpts from these documents:
{filenames}

Excerpts:
{state['evidence']}
{topic_line}
Proactively surface 4-7 non-obvious insights the researcher may not think to ask about.
Each insight should be one of:
- "pattern"  — a recurring theme or consistent finding across documents
- "trend"    — a change or progression over time or across studies
- "anomaly"  — something surprising, inconsistent, or unexpected

Return a JSON array. Each item:
{{
  "insight_type": "pattern|trend|anomaly",
  "title": "<insight title in 8 words or less>",
  "description": "<2-3 sentences. Be specific — name documents, figures, or concepts.>",
  "supporting_docs": ["<filename1>", "<filename2>"]
}}

Respond with only the JSON array, no markdown."""

    try:
        resp = _llm.invoke([
            SystemMessage(content="You are an expert analyst. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        insights = json.loads(raw)
    except Exception as e:
        logger.warning(f"Insight mining failed: {e}")
        insights = []

    return {**state, "insights": insights}


def save_insights(state: InsightsState) -> InsightsState:
    """Persist insights to DB."""
    db = SessionLocal()
    try:
        # Reverse map: filename → doc_id
        name_to_id = {v: k for k, v in state["doc_map"].items()}

        db.query(models.Insight).filter_by(
            project_id=state["project_id"]
        ).delete()

        for item in state["insights"]:
            # Resolve supporting doc filenames to IDs
            supporting_ids = [
                name_to_id[fn]
                for fn in item.get("supporting_docs", [])
                if fn in name_to_id
            ]
            db.add(models.Insight(
                project_id      = state["project_id"],
                insight_type    = item.get("insight_type", "pattern"),
                title           = item.get("title", ""),
                description     = item.get("description", ""),
                supporting_docs = ",".join(supporting_ids),
            ))
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save insights: {e}")
        db.rollback()
    finally:
        db.close()
    return state


# ── Graph ────────────────────────────────────────────────────────────────────

def build_insights_graph():
    g = StateGraph(InsightsState)
    g.add_node("collect_evidence", collect_evidence)
    g.add_node("mine_insights",    mine_insights)
    g.add_node("save_insights",    save_insights)

    g.set_entry_point("collect_evidence")
    g.add_edge("collect_evidence", "mine_insights")
    g.add_edge("mine_insights",    "save_insights")
    g.add_edge("save_insights",    END)
    return g.compile()


insights_chain = build_insights_graph()
