import json
import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


class APIClient:
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url

    # ── Upload ──────────────────────────────────────────────────────────

    def upload_pdfs(self, files: list, project_id: str) -> dict:
        file_tuples = [
            ("files", (f.name, f.getvalue(), "application/pdf"))
            for f in files
        ]
        resp = requests.post(
            f"{self.base_url}/upload",
            files=file_tuples,
            data={"project_id": project_id},
        )
        return resp.json()

    def get_job_status(self, job_id: str) -> dict:
        resp = requests.get(f"{self.base_url}/jobs/{job_id}")
        return resp.json()

    # ── Documents ───────────────────────────────────────────────────────

    def get_documents(self, project_id: str) -> list:
        resp = requests.get(
            f"{self.base_url}/documents",
            params={"project_id": project_id},
        )
        return resp.json() if resp.ok else []

    def delete_document(self, doc_id: str) -> dict:
        resp = requests.delete(f"{self.base_url}/documents/{doc_id}")
        return resp.json()

    def get_source_guide(self, doc_id: str) -> dict:
        resp = requests.get(
            f"{self.base_url}/documents/guide",
            params={"doc_id": doc_id},
        )
        return resp.json() if resp.ok else {}

    def generate_brief(self, project_id: str) -> dict:
        resp = requests.post(
            f"{self.base_url}/brief",
            params={"project_id": project_id},
            timeout=60,
        )
        return resp.json() if resp.ok else {"brief": ""}

    # ── Query ───────────────────────────────────────────────────────────

    def query_stream(self, question: str, project_id: str, session_id: str):
        """Returns a streaming response object."""
        return requests.post(
            f"{self.base_url}/query",
            json={
                "question":   question,
                "project_id": project_id,
                "session_id": session_id,
                "stream":     True,
            },
            stream=True,
            timeout=120,
        )

    def parse_stream(self, response) -> tuple[str, dict]:
        """
        Reads a streaming response and returns:
          (full_answer_text, metadata_dict)
        metadata contains: citations, confidence, follow_up_questions
        """
        # Handle non-200 responses before attempting SSE parsing
        if not response.ok:
            status = response.status_code
            try:
                body = response.text or ""
            except Exception:
                body = ""

            # Groq rate limit surfaced as 500 from the backend
            if status == 429 or "rate_limit" in body.lower() or "ratelimit" in body.lower():
                return (
                    "⚠️ **Rate limit reached.** The AI model has hit its daily token limit. "
                    "Please wait a few minutes and try again.",
                    {},
                )
            if status == 500 and ("rate" in body.lower() or "429" in body):
                return (
                    "⚠️ **Rate limit reached.** The AI model has hit its daily token limit. "
                    "Please wait a few minutes and try again.",
                    {},
                )
            return (
                f"⚠️ **Server error ({status}).** Something went wrong on the backend. "
                "Please try again shortly.",
                {},
            )

        full_answer = ""
        metadata    = {}

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                if chunk.get("type") == "token":
                    full_answer += chunk.get("content", "")
                elif chunk.get("type") == "metadata":
                    metadata = chunk
            except json.JSONDecodeError:
                pass

        return full_answer, metadata

    # ── Unique Features ─────────────────────────────────────────────────

    def run_contradictions(self, project_id: str) -> list:
        """Runs pipeline synchronously — returns results directly when done."""
        resp = requests.post(
            f"{self.base_url}/contradictions/run",
            params={"project_id": project_id},
            timeout=300,
        )
        return resp.json() if resp.ok else []

    def get_contradictions(self, project_id: str) -> list:
        resp = requests.get(
            f"{self.base_url}/contradictions",
            params={"project_id": project_id},
        )
        return resp.json() if resp.ok else []

    def run_gaps(self, project_id: str) -> list:
        """Runs pipeline synchronously — returns results directly when done."""
        resp = requests.post(
            f"{self.base_url}/gaps/run",
            params={"project_id": project_id},
            timeout=300,
        )
        return resp.json() if resp.ok else []

    def get_gaps(self, project_id: str) -> list:
        resp = requests.get(
            f"{self.base_url}/gaps",
            params={"project_id": project_id},
        )
        return resp.json() if resp.ok else []

    def run_questions(self, project_id: str, topic: str = "") -> list:
        """Runs pipeline synchronously — returns results directly when done."""
        resp = requests.post(
            f"{self.base_url}/questions/run",
            params={"project_id": project_id, "topic": topic},
            timeout=300,
        )
        return resp.json() if resp.ok else []

    def get_questions(self, project_id: str) -> list:
        resp = requests.get(
            f"{self.base_url}/questions",
            params={"project_id": project_id},
        )
        return resp.json() if resp.ok else []

    def run_insights(self, project_id: str, topic: str = "") -> list:
        """Runs pipeline synchronously — returns results directly when done."""
        resp = requests.post(
            f"{self.base_url}/insights/run",
            params={"project_id": project_id, "topic": topic},
            timeout=300,
        )
        return resp.json() if resp.ok else []

    def get_insights(self, project_id: str) -> list:
        resp = requests.get(
            f"{self.base_url}/insights",
            params={"project_id": project_id},
        )
        return resp.json() if resp.ok else []

    def get_timeline(self, project_id: str) -> list:
        resp = requests.post(
            f"{self.base_url}/timeline",
            params={"project_id": project_id},
            timeout=60,
        )
        return resp.json().get("timeline", []) if resp.ok else []

    # ── Health ──────────────────────────────────────────────────────────

    def is_online(self) -> bool:
        """Try up to 3 times (6s total) — backend may still be loading models."""
        import time
        for attempt in range(3):
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=3)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            if attempt < 2:
                time.sleep(2)
        return False


api = APIClient()
