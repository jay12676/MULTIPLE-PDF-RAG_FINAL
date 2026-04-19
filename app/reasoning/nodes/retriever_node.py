import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from app.reasoning.state import QAState
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import reranker
from app.retrieval.context_assembler import assemble_context, RetrievedChunk

logger = logging.getLogger(__name__)


def _retrieve_for_subq(project_id: str, sub_q: str) -> List[RetrievedChunk]:
    """
    Full retrieval pipeline for a single sub-question.

    hybrid_search()    → top-100 by FAISS + BM25 + RRF
    assemble_context() → fetch chunk text from SQLite (top-25)
    reranker.rerank()  → cross-encoder scores → keeps top-10
    """
    hits = hybrid_search(project_id, sub_q, top_k=100)
    candidates = assemble_context(hits[:25])
    pairs = [(c.chunk_id, c.text) for c in candidates]
    reranked = reranker.rerank(sub_q, pairs, top_k=10)
    score_map = {cid: score for cid, score in reranked}
    result = []
    for chunk in candidates:
        if chunk.chunk_id in score_map:
            chunk.score = score_map[chunk.chunk_id]
            result.append(chunk)
    return result


def retriever_node(state: QAState) -> QAState:
    """
    Node 2 — Runs hybrid search for every sub-question in parallel,
    deduplicates results, reranks, and returns top-6 chunks.

    Sub-questions are dispatched concurrently so a 3-sub-question
    query takes ~1× retrieval time instead of 3×.
    """
    project_id    = state["project_id"]
    sub_questions = state["sub_questions"]

    seen: Dict[str, RetrievedChunk] = {}

    with ThreadPoolExecutor(max_workers=min(4, len(sub_questions))) as executor:
        futures = {
            executor.submit(_retrieve_for_subq, project_id, sub_q): sub_q
            for sub_q in sub_questions
        }
        for future in as_completed(futures):
            try:
                for chunk in future.result():
                    if chunk.chunk_id not in seen or seen[chunk.chunk_id].score < chunk.score:
                        seen[chunk.chunk_id] = chunk
            except Exception as e:
                logger.warning(f"Sub-question retrieval failed: {e}")

    final_chunks = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:6]

    logger.info(
        f"Retrieved {len(final_chunks)} unique chunks "
        f"from {len(sub_questions)} sub-questions"
    )

    return {**state, "retrieved_chunks": final_chunks}
