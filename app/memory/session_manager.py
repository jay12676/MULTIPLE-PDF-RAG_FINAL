import json
import logging

import redis

from app.config import settings

logger = logging.getLogger(__name__)

_redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

SESSION_TTL  = 3600   # conversation expires after 1 hour of inactivity
MAX_MESSAGES = 20     # keep last 20 messages (10 turns)


class SessionManager:
    """
    Stores conversation history per session_id in Redis.

    Why Redis?
    - Fast reads/writes (in-memory)
    - Automatic expiry via TTL — no manual cleanup needed
    - Each Celery worker and FastAPI instance shares the same Redis
      so sessions work correctly even with multiple processes
    """

    def get_history(self, session_id: str) -> list:
        """Return the full message history for a session."""
        try:
            raw = _redis.get(f"session:{session_id}")
            return json.loads(raw) if raw else []
        except Exception as e:
            logger.warning(f"Failed to read session {session_id}: {e}")
            return []

    def add_turn(self, session_id: str, question: str, answer: str):
        """Append one user+assistant turn and reset the TTL."""
        try:
            history = self.get_history(session_id)
            history.append({"role": "user",      "content": question})
            history.append({"role": "assistant",  "content": answer})

            # Rolling window — keep only the most recent messages
            history = history[-MAX_MESSAGES:]

            _redis.setex(
                f"session:{session_id}",
                SESSION_TTL,
                json.dumps(history),
            )
        except Exception as e:
            logger.warning(f"Failed to save session {session_id}: {e}")

    def clear(self, session_id: str):
        """Delete a session (user clicks 'New Chat')."""
        _redis.delete(f"session:{session_id}")


session_manager = SessionManager()
