import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

# Make sure the DB directory exists before creating the engine
os.makedirs(settings.DB_DIR, exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    # Required for SQLite when used across threads (Celery workers)
    connect_args={"check_same_thread": False},
    # Keep connections alive — avoids reconnect overhead
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session, closes it after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Call once at app startup to create all tables if they don't exist."""
    from app.storage import models  # noqa: F401 — import triggers model registration
    Base.metadata.create_all(bind=engine)
