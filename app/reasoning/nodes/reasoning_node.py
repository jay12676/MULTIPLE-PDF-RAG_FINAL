import logging

from langchain_groq import ChatGroq

from app.config import settings
from app.reasoning.state import QAState

logger = logging.getLogger(__name__)

_llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

_SYSTEM = """You are a precise document analysis assistant.
- Answer ONLY from the provided evidence — do not use outside knowledge.
- Cite every factual claim using [number] notation matching the evidence list.
- If the evidence does not fully answer the question, clearly state what is missing.
- Be concise, accurate, and factual."""


def reasoning_node(state: QAState) -> QAState:
    """
    Node 4 — GROQ LLM reads the aggregated evidence and writes the answer.

    Includes the last 3 conversation turns so follow-up questions
    like "What about the second one?" have context.
    """
    # Last 3 turns of conversation (6 messages: 3 user + 3 assistant)
    history = state.get("conversation_history", [])[-6:]

    messages = [{"role": "system", "content": _SYSTEM}]
    messages += history
    messages.append({
        "role": "user",
        "content": (
            f"Evidence from documents:\n\n"
            f"{state['aggregated_evidence']}\n\n"
            f"---\n\n"
            f"Question: {state['user_query']}\n\n"
            f"Answer using ONLY the evidence above. Cite sources with [number]."
        ),
    })

    response = _llm.invoke(messages)

    # Confidence from top-3 reranker scores.
    # ms-marco cross-encoder outputs raw logits in range roughly [-10, +10].
    # We linearly normalise to [0.05, 0.99] so low-relevance docs show low
    # confidence and high-relevance docs show high confidence.
    chunks = state["retrieved_chunks"]
    if chunks:
        top_scores = sorted([c.score for c in chunks], reverse=True)[:3]
        avg_logit  = sum(top_scores) / len(top_scores)
        # Clamp to [-10, +10] then map to [0.05, 0.99]
        clamped    = max(-10.0, min(10.0, avg_logit))
        confidence = round(0.05 + (clamped + 10.0) / 20.0 * 0.94, 2)
    else:
        confidence = 0.05

    return {
        **state,
        "final_answer": response.content,
        "confidence":   confidence,
        "iterations":   state.get("iterations", 0) + 1,
    }
