"""
Task 15 — Smart Question Generator
====================================
Generates high-value research questions the user SHOULD ask about their documents.
Categories: comparison, factual, gap, trend.

Flow:
  load_context → generate_questions → save_questions → END
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
    temperature=0.5,
    max_tokens=1024,
)


# ── State ────────────────────────────────────────────────────────────────────

class QuestionsState(TypedDict):
    project_id: str
    topic:      str          # optional focus topic from the user (empty = all topics)
    context:    str          # assembled snippet of document content
    questions:  List[dict]


# ── Nodes ────────────────────────────────────────────────────────────────────

def load_context(state: QuestionsState) -> QuestionsState:
    """Collect a representative cross-section of content from all indexed docs."""
    db = SessionLocal()
    try:
        # One level-2 section chunk per document (gives topics without full detail)
        docs = (
            db.query(models.Document)
            .filter_by(project_id=state["project_id"], status="indexed")
            .all()
        )
        snippets: List[str] = []
        for doc in docs:
            chunks = (
                db.query(models.Chunk)
                .filter_by(doc_id=doc.doc_id, level=2)
                .limit(4)
                .all()
            )
            if not chunks:
                chunks = (
                    db.query(models.Chunk)
                    .filter_by(doc_id=doc.doc_id, level=3)
                    .limit(2)
                    .all()
                )
            if chunks:
                text = " | ".join(c.text[:300] for c in chunks)
                snippets.append(f"[{doc.filename}] {text}")

        context = "\n\n".join(snippets[:10])  # cap at 10 docs
        return {**state, "context": context}
    finally:
        db.close()


def generate_questions(state: QuestionsState) -> QuestionsState:
    """Generate 5 smart research questions about the given topic from the documents."""
    if not state["context"]:
        return {**state, "questions": []}

    topic = state.get("topic", "").strip()
    topic_line = (
        f"The user wants questions specifically about: **{topic}**"
        if topic else
        "Cover the most important and interesting topics from the documents."
    )

    prompt = f"""You are a research advisor helping a user explore their documents.

Document collection snippets:
{state['context']}

{topic_line}

Generate exactly 5 smart, specific, insightful questions a student or researcher should ask.
- Questions must be directly answerable from these documents
- Make them specific and thought-provoking, not generic
- Do NOT add category labels, prefixes, or numbering

Return ONLY a JSON array of 5 plain question strings. Example format:
["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]

Respond with only the JSON array, no markdown, no explanation."""

    try:
        resp = _llm.invoke([
            SystemMessage(content="You are a research advisor. Return only a valid JSON array of strings."),
            HumanMessage(content=prompt),
        ])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        # Support both ["q1","q2",...] and [{"question":"q1"},...] formats
        questions = []
        for item in parsed[:5]:
            if isinstance(item, str):
                questions.append({"question": item, "category": "smart"})
            elif isinstance(item, dict):
                questions.append({"question": item.get("question", ""), "category": "smart"})
    except Exception as e:
        logger.warning(f"Question generation failed: {e}")
        questions = []

    return {**state, "questions": questions}



def save_questions(state: QuestionsState) -> QuestionsState:
    """Persist generated questions to DB."""
    db = SessionLocal()
    try:
        db.query(models.SmartQuestion).filter_by(
            project_id=state["project_id"]
        ).delete()

        for item in state["questions"]:
            db.add(models.SmartQuestion(
                project_id = state["project_id"],
                question   = item.get("question", ""),
                category   = item.get("category", "factual"),
            ))
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save smart questions: {e}")
        db.rollback()
    finally:
        db.close()
    return state


# ── Graph ────────────────────────────────────────────────────────────────────

def build_questions_graph():
    g = StateGraph(QuestionsState)
    g.add_node("load_context",      load_context)
    g.add_node("generate_questions", generate_questions)
    g.add_node("save_questions",    save_questions)

    g.set_entry_point("load_context")
    g.add_edge("load_context",       "generate_questions")
    g.add_edge("generate_questions", "save_questions")
    g.add_edge("save_questions",     END)
    return g.compile()


questions_chain = build_questions_graph()
