import json
import logging
import re

import groq
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.memory.session_manager import session_manager
from app.reasoning.qa_graph import qa_chain
from app.storage.db import SessionLocal
from app.storage import models

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Request schema ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    project_id: str
    session_id: str  = "default"
    top_k:      int  = 20
    stream:     bool = True


# ── Endpoint ────────────────────────────────────────────────────────────────

@router.post("/query")
async def query(req: QueryRequest):
    """
    Run the full RAG pipeline for a user question and stream the answer.

    Streaming format (Server-Sent Events):
      data: {"type": "token",    "content": "word "}   ← one per word
      data: {"type": "metadata", "citations": [...], "confidence": 0.87, ...}
      data: [DONE]

    Streamlit reads this with:
      response = requests.post(..., stream=True)
      for line in response.iter_lines(): ...
    """
    # 1. Load conversation history from Redis
    history = session_manager.get_history(req.session_id)

    # 2. Build initial LangGraph state
    initial_state = {
        "user_query":           req.question,
        "project_id":           req.project_id,
        "session_id":           req.session_id,
        "conversation_history": history,
        "sub_questions":        [],
        "retrieved_chunks":     [],
        "aggregated_evidence":  "",
        "iterations":           0,
        "final_answer":         "",
        "citations":            [],
        "confidence":           0.0,
        "follow_up_questions":  [],
    }

    # 3. Run LangGraph (decompose → retrieve → aggregate → reason → format)
    try:
        result = qa_chain.invoke(initial_state)
    except groq.RateLimitError as e:
        error_msg = _rate_limit_message(str(e))
        logger.warning(f"Groq rate limit hit: {e}")
        if req.stream:
            return StreamingResponse(
                _stream_error(error_msg),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return {"answer": error_msg, "citations": [], "confidence": 0.0, "follow_up_questions": []}
    except Exception as e:
        logger.error(f"Query pipeline failed: {e}", exc_info=True)
        # Surface Groq rate limit that may be wrapped inside LangChain exceptions
        err_str = str(e)
        if "rate_limit" in err_str.lower() or "429" in err_str:
            error_msg = _rate_limit_message(err_str)
        else:
            error_msg = "⚠️ **Server error.** Something went wrong while answering your question. Please try again shortly."
        if req.stream:
            return StreamingResponse(
                _stream_error(error_msg),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return {"answer": error_msg, "citations": [], "confidence": 0.0, "follow_up_questions": []}

    # 4. Save turn to session memory
    session_manager.add_turn(req.session_id, req.question, result["final_answer"])

    # 5. Save to audit log
    _save_audit(req, result)

    # 6. Return streaming or JSON response
    if req.stream:
        return StreamingResponse(
            _stream(result),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return {
        "answer":              result["final_answer"],
        "citations":           result["citations"],
        "confidence":          result["confidence"],
        "follow_up_questions": result["follow_up_questions"],
    }


# ── Streaming generator ─────────────────────────────────────────────────────

async def _stream(result: dict):
    """
    Yields the answer word-by-word first (so Streamlit renders it live),
    then yields one final chunk with citations + metadata.
    """
    answer = result.get("final_answer", "")

    # Stream answer tokens
    words = answer.split(" ")
    for i, word in enumerate(words):
        token = word if i == 0 else f" {word}"
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    # Send structured metadata as the last chunk
    metadata = {
        "type":                "metadata",
        "citations":           result.get("citations", []),
        "confidence":          result.get("confidence", 0.0),
        "follow_up_questions": result.get("follow_up_questions", []),
    }
    yield f"data: {json.dumps(metadata)}\n\n"
    yield "data: [DONE]\n\n"


# ── Error helpers ───────────────────────────────────────────────────────────

def _rate_limit_message(error_str: str) -> str:
    """Build a human-readable rate-limit message, including retry time if available."""
    retry_match = re.search(r"[Pp]lease try again in ([\w\d.]+)", error_str)
    retry_info = f" Please try again in **{retry_match.group(1)}**." if retry_match else " Please try again in a few minutes."
    return (
        f"⚠️ **Rate limit reached.** The AI model has used its daily token quota.{retry_info}"
    )


async def _stream_error(message: str):
    """Stream a single error token so the frontend renders it as a chat bubble."""
    yield f"data: {json.dumps({'type': 'token', 'content': message})}\n\n"
    yield f"data: {json.dumps({'type': 'metadata', 'citations': [], 'confidence': 0.0, 'follow_up_questions': []})}\n\n"
    yield "data: [DONE]\n\n"


# ── Audit log ───────────────────────────────────────────────────────────────

def _save_audit(req: QueryRequest, result: dict):
    """Write every question + answer to the audit_logs table."""
    db = SessionLocal()
    try:
        chunk_ids = ",".join(
            str(c.get("number", "")) for c in result.get("citations", [])
        )
        db.add(
            models.AuditLog(
                project_id    = req.project_id,
                session_id    = req.session_id,
                question      = req.question,
                answer        = result.get("final_answer", ""),
                chunks_used   = chunk_ids,
                confidence    = result.get("confidence", 0.0),
                model_version = settings.GROQ_MODEL,
            )
        )
        db.commit()
    except Exception as e:
        logger.warning(f"Audit log write failed: {e}")
    finally:
        db.close()
