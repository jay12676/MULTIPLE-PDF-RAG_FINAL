import json
import logging
import re

from langchain_groq import ChatGroq

from app.config import settings
from app.reasoning.state import QAState

logger = logging.getLogger(__name__)

_llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

_PROMPT = """Break this question into 2-4 focused sub-questions suitable for searching a document database.
If the question is simple and self-contained, return just 1 sub-question (the original question).

Question: {query}

Rules:
- Each sub-question should be independently searchable
- Cover different aspects of the original question
- Keep sub-questions concise

Return ONLY a valid JSON array of strings. No explanation.
Example: ["sub-question 1", "sub-question 2"]"""

# Patterns that indicate a multi-part query worth decomposing via LLM
_COMPOUND_RE = re.compile(
    r"\b(compare|versus|vs\.?|difference|differences|both|between|contrast|"
    r"relationship|advantages|disadvantages|pros|cons|similarities|"
    r"tradeoff|trade-off|how does .+? affect|what are .+? and)\b",
    re.IGNORECASE,
)


def _needs_decomposition(query: str) -> bool:
    """
    Return True only when the query is genuinely multi-faceted.

    Short queries (<= 10 words) almost never benefit from decomposition —
    the LLM call just adds latency with no retrieval gain.
    Longer queries are only decomposed when they contain explicit comparison
    or multi-topic keywords.
    """
    if len(query.split()) <= 10:
        return False
    return bool(_COMPOUND_RE.search(query))


def query_decomposer(state: QAState) -> QAState:
    """
    Node 1 — Breaks the user's question into sub-questions.

    Fast path: simple / short queries skip the LLM call entirely and
    go straight to retrieval as a single sub-question.

    LLM path: only fires for complex multi-topic questions (e.g.
    "Compare the revenue and risk strategy across all reports").
    """
    query = state["user_query"]

    if not _needs_decomposition(query):
        logger.info(f"Simple query — skipping decomposition: {query!r}")
        return {
            **state,
            "sub_questions": [query],
            "iterations": state.get("iterations", 0),
        }

    try:
        response = _llm.invoke(_PROMPT.format(query=query))
        sub_questions = json.loads(response.content)
        if not isinstance(sub_questions, list) or not sub_questions:
            sub_questions = [query]
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e} — using original query")
        sub_questions = [query]

    logger.info(f"Sub-questions: {sub_questions}")

    return {
        **state,
        "sub_questions": sub_questions,
        "iterations": state.get("iterations", 0),
    }
