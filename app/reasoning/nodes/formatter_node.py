import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_groq import ChatGroq

from app.config import settings
from app.reasoning.state import QAState

logger = logging.getLogger(__name__)

_llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

_STOPWORDS = {
    "the","a","an","is","are","was","were","of","in","to","and","or","it",
    "this","that","with","for","on","at","by","from","be","has","have","had",
    "not","but","as","its","also","which","their","they","we","he","she","i",
    "do","did","so","if","about","can","more","than","into","will","been",
}


def formatter_node(state: QAState) -> QAState:
    """
    Node 5 — Builds citations list and generates follow-up questions.

    All LLM work (follow-ups + per-citation quote fallbacks) runs
    concurrently via a thread pool, so this node is bounded by the
    slowest single call rather than the sum of all calls.
    """
    chunks = state["retrieved_chunks"]
    answer = state["final_answer"]

    # Collect only chunks that are actually cited in the answer
    to_cite = []
    seen_snippets: set = set()
    for i, chunk in enumerate(chunks, start=1):
        if f"[{i}]" not in answer:
            continue
        snippet = chunk.text[:250]
        if snippet in seen_snippets:
            continue
        seen_snippets.add(snippet)
        to_cite.append((i, chunk))

    # Dispatch follow-up generation and all quote extractions concurrently
    with ThreadPoolExecutor(max_workers=len(to_cite) + 1) as executor:
        follow_up_future = executor.submit(
            _generate_follow_ups, state["user_query"], answer
        )

        quote_futures = {}
        for i, chunk in to_cite:
            answer_context = _extract_answer_context(answer, i)
            quote_futures[i] = executor.submit(
                _extract_exact_quote, chunk.text, answer_context
            )

        # Collect results
        quotes: dict = {}
        for i, future in quote_futures.items():
            try:
                quotes[i] = future.result()
            except Exception as e:
                logger.warning(f"Quote extraction failed for citation {i}: {e}")
                quotes[i] = ""

        follow_ups = follow_up_future.result()

    # Build citations in original order
    citations = []
    for i, chunk in to_cite:
        exact_quote = quotes.get(i, "")
        snippet = chunk.text[:250]
        logger.info(
            f"[citation {i}] file={chunk.filename} page={chunk.page_number} "
            f"chunk_words={len(chunk.text.split())} "
            f"quote_words={len(exact_quote.split()) if exact_quote else 0} "
            f"exact_quote_preview={exact_quote[:80]!r}"
        )
        citations.append({
            "number":       i,
            "doc_id":       chunk.doc_id,
            "title":        chunk.title,
            "filename":     chunk.filename,
            "page_number":  chunk.page_number,
            "section":      chunk.section,
            "text_snippet": snippet + "…" if len(chunk.text) > 250 else chunk.text,
            "exact_quote":  exact_quote,
        })

    return {
        **state,
        "citations":           citations,
        "follow_up_questions": follow_ups,
    }


# ── Exact Quote Helpers ───────────────────────────────────────────────────────

def _extract_answer_context(answer: str, citation_number: int) -> str:
    """Extract the 60-word window around the [n] citation marker in the answer."""
    marker = f"[{citation_number}]"
    pos = answer.find(marker)
    if pos == -1:
        return answer[:300]

    words = answer.split()
    char_count = 0
    marker_word_idx = 0
    for idx, w in enumerate(words):
        char_count += len(w) + 1
        if char_count >= pos:
            marker_word_idx = idx
            break

    start = max(0, marker_word_idx - 30)
    end   = min(len(words), marker_word_idx + 30)
    return " ".join(words[start:end])


def _score_sentence(sentence: str, context: str) -> int:
    """Count non-stopword token overlaps between sentence and answer context."""
    sent_tokens    = {w.lower().strip(".,;:!?()[]\"'") for w in sentence.split()} - _STOPWORDS
    context_tokens = {w.lower().strip(".,;:!?()[]\"'") for w in context.split()}  - _STOPWORDS
    return len(sent_tokens & context_tokens)


def _extract_exact_quote(chunk_text: str, answer_context: str) -> str:
    """
    Find the 1-3 sentences in chunk_text that best support the answer_context.

    Pass 1: keyword overlap scoring (instant, no LLM)
    Pass 2: LLM fallback if Pass 1 yields < 12 words
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if s.strip()]
    if not sentences:
        return ""

    scored = [(s, _score_sentence(s, answer_context)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_sent = scored[0][0]
    best_idx  = sentences.index(best_sent)

    result_sentences = [best_sent]
    if scored[0][1] > 0:
        if best_idx + 1 < len(sentences) and _score_sentence(sentences[best_idx + 1], answer_context) > 0:
            result_sentences.append(sentences[best_idx + 1])
        elif best_idx - 1 >= 0 and _score_sentence(sentences[best_idx - 1], answer_context) > 0:
            result_sentences = [sentences[best_idx - 1]] + result_sentences

    exact_quote = " ".join(result_sentences)

    words = exact_quote.split()
    if len(words) > 80:
        exact_quote = " ".join(words[:80]) + "…"

    if len(exact_quote.split()) < 12:
        logger.info(f"[exact_quote] Pass 1 too short ({len(exact_quote.split())} words), using LLM fallback")
        exact_quote = _llm_extract_quote(chunk_text, answer_context)
    else:
        logger.info(f"[exact_quote] Pass 1 succeeded ({len(exact_quote.split())} words)")

    return exact_quote


def _llm_extract_quote(chunk_text: str, answer_context: str) -> str:
    """Fallback: ask LLM to extract the exact supporting sentence(s)."""
    prompt = (
        f"From the following passage, extract the 1-2 sentences that most directly "
        f"support this claim: \"{answer_context[:200]}\"\n\n"
        f"Passage:\n{chunk_text[:600]}\n\n"
        f"Return ONLY the exact sentence(s) from the passage, nothing else."
    )
    try:
        resp = _llm.invoke(prompt)
        quote = resp.content.strip().strip('"').strip("'")
        words = quote.split()
        return " ".join(words[:80]) + ("…" if len(words) > 80 else "")
    except Exception as e:
        logger.warning(f"LLM quote extraction failed: {e}")
        return ""


# ── Follow-up Questions ───────────────────────────────────────────────────────

def _generate_follow_ups(question: str, answer: str) -> list:
    prompt = (
        f"The user asked: {question}\n"
        f"The answer summary: {answer[:400]}\n\n"
        f"Suggest exactly 3 short natural follow-up questions the user might ask next.\n"
        f"Return ONLY a valid JSON array of 3 strings. No explanation."
    )
    try:
        response   = _llm.invoke(prompt)
        follow_ups = json.loads(response.content)
        if isinstance(follow_ups, list):
            return follow_ups[:3]
    except Exception as e:
        logger.warning(f"Follow-up generation failed: {e}")
    return []
