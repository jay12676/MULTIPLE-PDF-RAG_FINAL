from fastapi import APIRouter
from app.workers.celery_app import celery_app

router = APIRouter()


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """
    Streamlit polls this endpoint every second to update the progress bar.

    Possible responses:
      queued      → job is waiting for a worker
      processing  → worker is running, progress 0–99
      completed   → done, progress 100
      failed      → something went wrong
    """
    task = celery_app.AsyncResult(job_id)

    if task.state == "PENDING":
        return {"status": "queued", "progress": 0, "stage": "waiting"}

    if task.state == "STARTED":
        return {"status": "processing", "progress": 5, "stage": "starting"}

    if task.state == "PROGRESS":
        meta = task.info or {}
        return {
            "status":   "processing",
            "progress": meta.get("progress", 0),
            "stage":    meta.get("stage", ""),
        }

    if task.state == "SUCCESS":
        return {
            "status":   "completed",
            "progress": 100,
            "stage":    "done",
            "result":   task.result,
        }

    if task.state == "FAILURE":
        return {
            "status":   "failed",
            "progress": 0,
            "error":    str(task.info),
        }

    return {"status": task.state.lower(), "progress": 0}
