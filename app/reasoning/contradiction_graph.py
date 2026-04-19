"""
Task 13 — Contradiction Detection Pipeline
==========================================
Scans all indexed documents in a project for contradictory claims.

Flow:
  load_chunks → compare_pairs → extract_contradictions → save → END

Strategy:
- Pull level-3 paragraph chunks for each document (representative content).
- Group by document; for every pair of documents ask the LLM to find contradictions.
- Persist findings to the contradictions table so the API can serve them instantly.
"""
import json
import logging
from itertools import combinations
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
    temperature=0,
    max_tokens=1024,
)

MAX_CHUNKS_PER_DOC = 6    # keep prompt sizes manageable


# ── State ────────────────────────────────────────────────────────────────────

class ContradictionState(TypedDict):
    project_id:    str
    doc_chunks:    dict        # {doc_id: {"filename": str, "chunks": [str]}}
    contradictions: List[dict]


# ── Nodes ────────────────────────────────────────────────────────────────────

def load_chunks(state: ContradictionState) -> ContradictionState:
    """Load level-3 paragraph chunks for each indexed document."""
    db = SessionLocal()
    try:
        docs = (
            db.query(models.Document)
            .filter_by(project_id=state["project_id"], status="indexed")
            .all()
        )
        doc_chunks: dict = {}
        for doc in docs:
            chunks = (
                db.query(models.Chunk)
                .filter_by(doc_id=doc.doc_id, level=3)
                .limit(MAX_CHUNKS_PER_DOC)
                .all()
            )
            if chunks:
                doc_chunks[doc.doc_id] = {
                    "filename": doc.filename,
                    "chunks": [c.text for c in chunks],
                }
        return {**state, "doc_chunks": doc_chunks}
    finally:
        db.close()


def compare_pairs(state: ContradictionState) -> ContradictionState:
    """For every pair of documents, ask the LLM to find contradictions."""
    doc_chunks = state["doc_chunks"]
    all_contradictions: List[dict] = []

    doc_ids = list(doc_chunks.keys())
    for id_a, id_b in combinations(doc_ids, 2):
        info_a = doc_chunks[id_a]
        info_b = doc_chunks[id_b]

        text_a = "\n---\n".join(c[:300] for c in info_a["chunks"][:5])
        text_b = "\n---\n".join(c[:300] for c in info_b["chunks"][:5])

        prompt = f"""You are a scientific fact-checker comparing two documents.

Document A: {info_a['filename']}
{text_a}

Document B: {info_b['filename']}
{text_b}

Find DIRECT CONTRADICTIONS — claims in Document A that explicitly contradict claims in Document B.
Ignore minor differences in emphasis, scope, or context.

Return a JSON array (may be empty). Each item:
{{
  "topic": "<short topic label>",
  "text_a": "<exact or paraphrased contradicting sentence from A>",
  "text_b": "<exact or paraphrased contradicting sentence from B>",
  "explanation": "<one-sentence explanation of why these contradict>",
  "confidence": <float 0-1>
}}

Respond with only the JSON array, no markdown."""

        try:
            resp = _llm.invoke([
                SystemMessage(content="You are an expert fact-checker. Return only valid JSON."),
                HumanMessage(content=prompt),
            ])
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            items = json.loads(raw)
            for item in items:
                item["doc_id_1"] = id_a
                item["doc_id_2"] = id_b
                all_contradictions.append(item)
        except Exception as e:
            logger.warning(f"Contradiction compare failed for {id_a}/{id_b}: {e}")

    return {**state, "contradictions": all_contradictions}


def save_contradictions(state: ContradictionState) -> ContradictionState:
    """Persist new contradictions to DB (clear old ones first)."""
    db = SessionLocal()
    try:
        # Remove stale results for this project
        db.query(models.Contradiction).filter_by(
            project_id=state["project_id"]
        ).delete()

        for item in state["contradictions"]:
            if item.get("confidence", 0) < 0.4:
                continue  # skip low-confidence noise
            db.add(models.Contradiction(
                project_id  = state["project_id"],
                topic       = item.get("topic", "General"),
                doc_id_1    = item.get("doc_id_1"),
                text_1      = item.get("text_a", ""),
                doc_id_2    = item.get("doc_id_2"),
                text_2      = item.get("text_b", ""),
                explanation = item.get("explanation", ""),
                confidence  = item.get("confidence", 0.5),
            ))
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save contradictions: {e}")
        db.rollback()
    finally:
        db.close()
    return state


# ── Graph ────────────────────────────────────────────────────────────────────

def build_contradiction_graph():
    g = StateGraph(ContradictionState)
    g.add_node("load_chunks",        load_chunks)
    g.add_node("compare_pairs",      compare_pairs)
    g.add_node("save_contradictions", save_contradictions)

    g.set_entry_point("load_chunks")
    g.add_edge("load_chunks",        "compare_pairs")
    g.add_edge("compare_pairs",      "save_contradictions")
    g.add_edge("save_contradictions", END)
    return g.compile()


contradiction_chain = build_contradiction_graph()
