"""
Task 17 — Document Evolution Timeline
========================================
Builds a chronological timeline of how findings / methodologies evolved across
the uploaded documents.  Run on-demand when the user opens the Timeline tab.

Flow:
  extract_dates_and_claims → order_timeline → format_timeline → END

Output: list of timeline events, each with a date, document, and key claim.
Results are returned (not persisted) since they are expensive to regenerate
and displayed immediately.
"""
import json
import logging
import re
from typing import TypedDict, List, Optional

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
    temperature=0.2,
    max_tokens=1500,
)


# ── State ────────────────────────────────────────────────────────────────────

class TimelineState(TypedDict):
    project_id:      str
    doc_texts:       List[dict]   # [{"filename": str, "upload_date": str, "text": str}]
    raw_events:      List[dict]   # extracted but unsorted
    timeline:        List[dict]   # final sorted timeline


# ── Nodes ────────────────────────────────────────────────────────────────────

def extract_dates_and_claims(state: TimelineState) -> TimelineState:
    """
    For each document ask the LLM to extract:
    - Publication year / date (from abstract, headers, or content)
    - 2-3 key claims or findings
    """
    db = SessionLocal()
    try:
        docs = (
            db.query(models.Document)
            .filter_by(project_id=state["project_id"], status="indexed")
            .order_by(models.Document.upload_date)
            .all()
        )
        doc_texts: List[dict] = []
        for doc in docs:
            # Use level-1 summary + first 2 level-3 paragraphs
            chunks = []
            l1 = db.query(models.Chunk).filter_by(doc_id=doc.doc_id, level=1).first()
            if l1:
                chunks.append(l1.text[:600])
            l3s = (
                db.query(models.Chunk)
                .filter_by(doc_id=doc.doc_id, level=3)
                .limit(2)
                .all()
            )
            chunks.extend(c.text[:400] for c in l3s)
            doc_texts.append({
                "filename":    doc.filename,
                "upload_date": doc.upload_date.strftime("%Y-%m-%d"),
                "text":        "\n".join(chunks),
            })
        return {**state, "doc_texts": doc_texts}
    finally:
        db.close()


def order_timeline(state: TimelineState) -> TimelineState:
    """Ask the LLM to extract timeline events from each document."""
    raw_events: List[dict] = []

    for doc in state["doc_texts"]:
        prompt = f"""Extract timeline information from this document.

Filename: {doc['filename']}
Upload date: {doc['upload_date']}

Content:
{doc['text']}

Extract:
1. The publication year or date range mentioned in the document (e.g. "2021", "2019-2022").
   If not found, use the upload date year.
2. 2-3 key findings, claims, or methodological advances from this document.

Return a JSON object:
{{
  "year": "<4-digit year or year range like 2019-2021>",
  "filename": "{doc['filename']}",
  "events": [
    {{"claim": "<one concise key finding or claim>", "importance": "high|medium|low"}}
  ]
}}

Respond with only the JSON object, no markdown."""

        try:
            resp = _llm.invoke([
                SystemMessage(content="You are a timeline extractor. Return only valid JSON."),
                HumanMessage(content=prompt),
            ])
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            event = json.loads(raw)
            raw_events.append(event)
        except Exception as e:
            logger.warning(f"Timeline extraction failed for {doc['filename']}: {e}")
            raw_events.append({
                "year":     doc["upload_date"][:4],
                "filename": doc["filename"],
                "events":   [],
            })

    return {**state, "raw_events": raw_events}


def format_timeline(state: TimelineState) -> TimelineState:
    """Sort events chronologically, deduplicate, and flatten into a list of timeline items."""
    def _sort_key(e):
        year_str = str(e.get("year", "9999"))
        m = re.match(r"(\d{4})", year_str)
        return int(m.group(1)) if m else 9999

    sorted_events = sorted(state["raw_events"], key=_sort_key)

    timeline: List[dict] = []
    seen_claims: set = set()          # deduplicate identical claims

    for doc_event in sorted_events:
        year     = doc_event.get("year", "Unknown")
        filename = doc_event.get("filename", "")
        for ev in doc_event.get("events", []):
            claim = ev.get("claim", "").strip()
            if not claim:
                continue
            # Skip exact duplicates (same claim text already seen)
            claim_key = claim.lower()[:80]
            if claim_key in seen_claims:
                continue
            seen_claims.add(claim_key)
            timeline.append({
                "year":       year,
                "filename":   filename,
                "claim":      claim,
                "importance": ev.get("importance", "medium"),
            })

    return {**state, "timeline": timeline}


# ── Graph ────────────────────────────────────────────────────────────────────

def build_timeline_graph():
    g = StateGraph(TimelineState)
    g.add_node("extract_dates_and_claims", extract_dates_and_claims)
    g.add_node("order_timeline",           order_timeline)
    g.add_node("format_timeline",          format_timeline)

    g.set_entry_point("extract_dates_and_claims")
    g.add_edge("extract_dates_and_claims", "order_timeline")
    g.add_edge("order_timeline",           "format_timeline")
    g.add_edge("format_timeline",          END)
    return g.compile()


timeline_chain = build_timeline_graph()
