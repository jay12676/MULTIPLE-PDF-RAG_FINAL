from app.reasoning.state import QAState


def aggregator_node(state: QAState) -> QAState:
    """
    Node 3 — Formats retrieved chunks into a single evidence string
    that the LLM can read.

    Each chunk is labelled [1], [2], ... so the LLM can cite them
    in its answer and we can map citations back to real doc/page info.
    """
    chunks = state["retrieved_chunks"]

    evidence_parts = []
    for i, chunk in enumerate(chunks, start=1):
        evidence_parts.append(
            f"[{i}] {chunk.title}  |  Page {chunk.page_number}  |  {chunk.section}\n"
            f"{chunk.text}"
        )

    aggregated = "\n\n---\n\n".join(evidence_parts)

    return {**state, "aggregated_evidence": aggregated}
