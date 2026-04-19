from typing import TypedDict, List


class QAState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────
    user_query:           str
    project_id:           str
    session_id:           str
    conversation_history: List[dict]   # [{"role": "user/assistant", "content": "..."}]

    # ── Intermediate (filled by nodes) ────────────────────────────────
    sub_questions:       List[str]
    retrieved_chunks:    list          # List[RetrievedChunk] from context_assembler
    aggregated_evidence: str           # formatted evidence string sent to LLM
    iterations:          int           # how many reasoning loops have run (max 3)

    # ── Output ────────────────────────────────────────────────────────
    final_answer:        str
    citations:           List[dict]
    confidence:          float
    follow_up_questions: List[str]
