import os
import uuid
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.storage.db import SessionLocal
from app.storage import models
from app.workers.tasks import ingest_pdf

router = APIRouter()

# Only accept PDF files
ALLOWED_TYPES = {"application/pdf"}


@router.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    project_id: str = Form(...),
):
    """
    Accepts one or more PDF files and queues one Celery ingestion job per file.
    Returns immediately — no waiting for processing to finish.

    Streamlit then polls GET /jobs/{job_id} to track each file's progress.
    """
    accepted = []
    rejected = []
    db = SessionLocal()

    try:
        for file in files:
            # ── Validate: must be a PDF ───────────────────────────────
            is_pdf = (
                file.content_type in ALLOWED_TYPES
                or (file.filename or "").lower().endswith(".pdf")
            )
            if not is_pdf:
                rejected.append(file.filename)
                continue

            # ── Save file to disk ─────────────────────────────────────
            doc_id = str(uuid.uuid4())
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            file_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.pdf")

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            # ── Create document record in SQLite ──────────────────────
            doc = models.Document(
                doc_id     = doc_id,
                project_id = project_id,
                filename   = file.filename,
                title      = (file.filename or "").replace(".pdf", ""),
                file_path  = file_path,
                status     = "queued",
            )
            db.add(doc)
            db.commit()

            # ── Queue Celery background job ───────────────────────────
            task = ingest_pdf.delay(doc_id, file_path, project_id)

            accepted.append({
                "doc_id":   doc_id,
                "job_id":   task.id,
                "filename": file.filename,
            })

    finally:
        db.close()

    if not accepted and rejected:
        raise HTTPException(
            status_code=400,
            detail=f"No valid PDFs found. Rejected: {rejected}",
        )

    return {
        "accepted":       accepted,
        "rejected":       rejected,
        "document_count": len(accepted),
    }
