from celery import Celery
from app.config import settings

celery_app = Celery(
    "documind",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,

    # lets us poll task progress via GET /jobs/{id}
    task_track_started=True,

    # ── Latency optimisations ─────────────────────────────────────────────
    # Run 2 workers in parallel so post-processing tasks (gaps, contradictions)
    # can execute concurrently with the next upload's main ingest task.
    worker_concurrency=2,

    # Prefetch only 1 task per worker — prevents a slow gaps/contradiction
    # task from blocking a fast ingest task in the same worker slot.
    worker_prefetch_multiplier=1,

    # Keep task results for 1 hour (enough for the Streamlit progress bar).
    result_expires=3600,

    # Use a fair round-robin schedule so quick tasks aren't starved by long ones.
    task_acks_late=True,
)
