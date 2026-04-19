import json
import logging

from langchain_groq import ChatGroq

from app.workers.celery_app import celery_app
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import Chunker
from app.ingestion.embedder import embedder
from app.storage.vector_store import vector_store
from app.storage.bm25_store import bm25_store
from app.storage.db import SessionLocal
from app.storage import models
from app.reasoning.gap_graph import gap_chain
from app.reasoning.contradiction_graph import contradiction_chain
from app.config import settings

logger = logging.getLogger(__name__)

_llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
    max_tokens=512,
)

BATCH_SIZE = 512   # number of chunks embedded per forward pass


# ════════════════════════════════════════════════════════════════════════════
# MAIN TASK — ingest_pdf
# Completes as fast as possible (parse → chunk → embed → store).
# Post-processing (source guide, gaps, contradictions) is dispatched as
# separate async tasks so it never adds to the upload's perceived latency.
# ════════════════════════════════════════════════════════════════════════════

@celery_app.task(bind=True, name="ingest_pdf")
def ingest_pdf(self, doc_id: str, pdf_path: str, project_id: str):
    """
    Full ingestion pipeline for one PDF.
    Runs entirely in the background — FastAPI returns immediately after queuing.

    Progress updates are stored in Redis and polled by Streamlit:
        0%  → job received
        10% → parsing started
        35% → chunking done
        55% → embedding started
        85% → FAISS + BM25 stored
        95% → saving to SQLite
       100% → indexed (post-processing dispatched separately)

    Returns:
        { status, doc_id, pages, total_chunks }
    """
    db = SessionLocal()
    parser = PDFParser()
    chunker = Chunker()

    try:
        # ── 1. Mark document as processing ───────────────────────────────
        doc = db.query(models.Document).filter_by(doc_id=doc_id).first()
        if not doc:
            return {"status": "failed", "error": "Document not found in DB"}

        doc.status = "processing"
        db.commit()
        _progress(self, 10, "parsing")

        # ── 2. Parse PDF (parallel page processing) ───────────────────────
        pages = list(parser.parse(pdf_path))
        doc.page_count = len(pages)
        db.commit()
        _progress(self, 35, "chunking")

        # ── 3. Chunk into levels ──────────────────────────────────────────
        chunks = chunker.chunk_document(doc_id, pages)
        total_chunks = len(chunks)
        _progress(self, 50, "embedding")

        # ── 4. Embed + store in FAISS (in batches, single disk write at end) ──
        all_texts    = [c.text      for c in chunks]
        all_ids      = [c.chunk_id  for c in chunks]
        faiss_positions: list[int] = []

        for i in range(0, total_chunks, BATCH_SIZE):
            batch_texts = all_texts[i : i + BATCH_SIZE]
            batch_ids   = all_ids[i : i + BATCH_SIZE]

            embeddings = embedder.embed_texts(batch_texts)
            # add_batch() accumulates in memory — NO disk write per batch
            positions  = vector_store.add_batch(project_id, embeddings, batch_ids)
            faiss_positions.extend(positions)

            # progress moves from 50% to 80% while embedding
            pct = 50 + int((i / total_chunks) * 30)
            _progress(self, pct, "embedding")

        # Single disk write for the entire FAISS index (was 1 write per batch)
        vector_store.flush(project_id)

        # ── 5. Store all chunks in BM25 index ────────────────────────────
        bm25_store.add(project_id, all_texts, all_ids)
        _progress(self, 85, "indexing keywords")

        # ── 6. Save chunk text + metadata to SQLite ───────────────────────
        for chunk, faiss_idx in zip(chunks, faiss_positions):
            db.add(
                models.Chunk(
                    chunk_id    = chunk.chunk_id,
                    doc_id      = chunk.doc_id,
                    project_id  = project_id,
                    page_number = chunk.page_number,
                    level       = chunk.level,
                    section     = chunk.section,
                    text        = chunk.text,
                    word_count  = chunk.word_count,
                    faiss_idx   = faiss_idx,
                )
            )
        db.commit()
        _progress(self, 95, "saving")

        # ── 7. Mark as indexed ────────────────────────────────────────────
        doc.status = "indexed"
        db.commit()
        _progress(self, 100, "indexed")

        logger.info(
            f"[ingest_pdf] doc_id={doc_id} pages={len(pages)} chunks={total_chunks}"
        )

        # ── 8. Dispatch post-processing as SEPARATE async tasks ───────────
        # These run after the main task completes — they do NOT block the
        # upload progress bar or add to the user-visible ingestion time.
        # Each task has its own retry logic and won't fail the main ingest.

        # 8a. Source guide (fast — 1 LLM call, ~2–5s)
        chunk_data = [{"chunk_id": c.chunk_id, "text": c.text, "level": c.level}
                      for c in chunks]
        run_source_guide.apply_async(
            args=[doc_id, chunk_data],
            countdown=1,          # start 1s after this task finishes
        )

        # 8b. Knowledge gaps (moderate — ~15–30s)
        run_gaps_task.apply_async(
            args=[project_id],
            countdown=3,          # slight delay so source guide starts first
        )

        # 8c. Contradictions (slow — only when 2+ docs, ~20–40s)
        db2 = SessionLocal()
        indexed_count = (
            db2.query(models.Document)
            .filter_by(project_id=project_id, status="indexed")
            .count()
        )
        db2.close()
        if indexed_count >= 2:
            run_contradictions_task.apply_async(
                args=[project_id],
                countdown=5,      # let gaps start first to avoid rate-limit spikes
            )

        return {
            "status":       "indexed",
            "doc_id":       doc_id,
            "pages":        len(pages),
            "total_chunks": total_chunks,
        }

    except Exception as exc:
        logger.error(f"[ingest_pdf] FAILED doc_id={doc_id} error={exc}")
        try:
            doc = db.query(models.Document).filter_by(doc_id=doc_id).first()
            if doc:
                doc.status = "failed"
                db.commit()
        except Exception:
            pass
        raise   # re-raise so Celery marks the task as FAILURE

    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING TASKS  (run asynchronously after main ingest completes)
# ════════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="run_source_guide",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    ignore_result=True,
)
def run_source_guide(self, doc_id: str, chunk_data: list):
    """
    Generate a 2-3 sentence summary + key topics for a document.
    Runs asynchronously — result is saved to DB when done.
    """
    try:
        db = SessionLocal()
        # Find the L1 chunk (document summary)
        l1_text = next(
            (c["text"] for c in chunk_data if c["level"] == 1),
            None,
        )
        if not l1_text or len(l1_text.split()) < 20:
            db.close()
            return

        prompt = (
            f"Read this document excerpt and return a JSON object with exactly two keys:\n"
            f"- \"summary\": a 2-3 sentence summary of what this document covers\n"
            f"- \"key_topics\": an array of 3-5 key topics or concepts discussed\n\n"
            f"Excerpt:\n{l1_text[:800]}\n\n"
            f"Return ONLY valid JSON, no markdown, no explanation."
        )
        resp = _llm.invoke(prompt)
        raw  = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        summary    = data.get("summary", "")
        key_topics = json.dumps(data.get("key_topics", []))

        db.query(models.SourceGuide).filter_by(doc_id=doc_id).delete()
        db.add(models.SourceGuide(
            doc_id     = doc_id,
            summary    = summary,
            key_topics = key_topics,
        ))
        db.commit()
        db.close()
        logger.info(f"[source_guide] done for doc_id={doc_id}")

    except Exception as exc:
        logger.warning(f"[source_guide] failed for doc_id={doc_id}: {exc}")
        try:
            self.retry(exc=exc)
        except Exception:
            pass


@celery_app.task(
    name="run_gaps_task",
    bind=True,
    max_retries=1,
    default_retry_delay=30,
    ignore_result=True,
)
def run_gaps_task(self, project_id: str):
    """
    Auto-run the Knowledge Gap analysis. Runs async after ingest completes.
    """
    try:
        gap_chain.invoke({
            "project_id": project_id,
            "context":    "",
            "gaps":       [],
        })
        logger.info(f"[gaps] auto-run done for project={project_id}")
    except Exception as exc:
        logger.warning(f"[gaps] auto-run failed for project={project_id}: {exc}")
        try:
            self.retry(exc=exc)
        except Exception:
            pass


@celery_app.task(
    name="run_contradictions_task",
    bind=True,
    max_retries=1,
    default_retry_delay=30,
    ignore_result=True,
)
def run_contradictions_task(self, project_id: str):
    """
    Auto-run Contradiction Detection. Runs async after ingest completes.
    Only dispatched when 2+ docs are indexed.
    """
    try:
        contradiction_chain.invoke({
            "project_id":     project_id,
            "doc_chunks":     {},
            "contradictions": [],
        })
        logger.info(f"[contradictions] auto-run done for project={project_id}")
    except Exception as exc:
        logger.warning(f"[contradictions] auto-run failed for project={project_id}: {exc}")
        try:
            self.retry(exc=exc)
        except Exception:
            pass


# ── Helper ─────────────────────────────────────────────────────────────────

def _progress(task, percent: int, stage: str):
    """Push a progress update into Redis so GET /jobs/{id} can return it."""
    task.update_state(
        state="PROGRESS",
        meta={"progress": percent, "stage": stage},
    )
