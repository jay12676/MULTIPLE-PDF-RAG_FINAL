import re
import uuid
from dataclasses import dataclass
from typing import List, Tuple

from app.ingestion.pdf_parser import ParsedPage
from app.config import settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_number: int
    level: int       # 1=doc summary  2=section  3=paragraph  4=sentence
    section: str
    text: str
    word_count: int


# Common academic / professional section headings
_SECTION_PATTERN = re.compile(
    r"^(abstract|introduction|background|related work|literature review|"
    r"methods?|methodology|materials?|experimental|experiments?|"
    r"results?|findings?|evaluation|discussion|conclusion|"
    r"summary|overview|references?|bibliography|appendix|"
    r"acknowledgements?|contributions?)[\s:\.\-]*$",
    re.IGNORECASE,
)


class Chunker:
    """
    Breaks a list of ParsedPage objects into 4 levels of chunks.

    Level 1 — 1 chunk per document (first 500 words, used for doc-level Q&A)
    Level 2 — 1 chunk per detected section (Introduction, Methods, etc.)
    Level 3 — sliding-window paragraph chunks  ← main retrieval unit
    Level 4 — individual sentence chunks (used for exact citation pinpointing)
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size   # words per paragraph chunk
        self.overlap = overlap         # overlapping words between consecutive chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(
        self, doc_id: str, pages: List[ParsedPage]
    ) -> List[Chunk]:
        """
        Main entry point.  Returns all chunks for a document across all 4 levels.
        """
        page_texts: List[Tuple[int, str]] = [
            (p.page_number, p.text)
            for p in pages
            if p.text.strip()
        ]

        if not page_texts:
            return []

        chunks: List[Chunk] = []

        # detect sections once, reused across levels 2–4
        sections = self._detect_sections(page_texts)

        chunks += self._level1_summary(doc_id, page_texts)
        chunks += self._level2_sections(doc_id, sections)
        chunks += self._level3_paragraphs(doc_id, page_texts, sections)
        # Level 4 (sentence chunks) skipped — too many chunks for large books,
        # level 3 paragraph chunks are sufficient for retrieval quality.

        return chunks

    # ------------------------------------------------------------------
    # Level builders
    # ------------------------------------------------------------------

    def _level1_summary(
        self, doc_id: str, page_texts: List[Tuple[int, str]]
    ) -> List[Chunk]:
        full_text = " ".join(text for _, text in page_texts)
        # keep first 500 words as the document summary chunk
        summary = " ".join(full_text.split()[: 500])
        return [self._make_chunk(doc_id, 1, 1, "full_document", summary)]

    def _level2_sections(
        self,
        doc_id: str,
        sections: List[Tuple[str, str, int]],   # (name, text, page_num)
    ) -> List[Chunk]:
        chunks = []
        for name, text, page_num in sections:
            if len(text.split()) > 10:
                chunks.append(
                    self._make_chunk(doc_id, page_num, 2, name, text)
                )
        return chunks

    def _level3_paragraphs(
        self,
        doc_id: str,
        page_texts: List[Tuple[int, str]],
        sections: List[Tuple[str, str, int]],
    ) -> List[Chunk]:
        chunks = []
        for page_num, page_text in page_texts:
            section = self._section_for_page(page_num, sections)
            for window in self._sliding_window(page_text):
                if len(window.split()) > 20:
                    chunks.append(
                        self._make_chunk(doc_id, page_num, 3, section, window)
                    )
        return chunks

    def _level4_sentences(
        self,
        doc_id: str,
        page_texts: List[Tuple[int, str]],
        sections: List[Tuple[str, str, int]],
    ) -> List[Chunk]:
        chunks = []
        for page_num, page_text in page_texts:
            section = self._section_for_page(page_num, sections)
            for sent in self._split_sentences(page_text):
                # skip very short / noisy sentences
                if len(sent.split()) >= 8:
                    chunks.append(
                        self._make_chunk(doc_id, page_num, 4, section, sent)
                    )
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        doc_id: str,
        page_number: int,
        level: int,
        section: str,
        text: str,
    ) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            page_number=page_number,
            level=level,
            section=section,
            text=text.strip(),
            word_count=len(text.split()),
        )

    def _sliding_window(self, text: str) -> List[str]:
        """Split text into overlapping word windows of size chunk_size."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        windows = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            windows.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += self.chunk_size - self.overlap
        return windows

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences on . ! ? boundaries."""
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    def _detect_sections(
        self, page_texts: List[Tuple[int, str]]
    ) -> List[Tuple[str, str, int]]:
        """
        Scan every line across all pages.
        When a line matches a known section heading, start a new section.
        Returns list of (section_name, section_text, start_page).
        """
        sections: List[Tuple[str, str, int]] = []
        current_name = "main"
        current_lines: List[str] = []
        current_page = page_texts[0][0]

        for page_num, page_text in page_texts:
            for line in page_text.split("\n"):
                stripped = line.strip()
                if _SECTION_PATTERN.match(stripped):
                    # save the section we just finished
                    if current_lines:
                        sections.append(
                            (current_name, " ".join(current_lines), current_page)
                        )
                    current_name = stripped.lower().rstrip(":.-")
                    current_lines = []
                    current_page = page_num
                else:
                    if stripped:
                        current_lines.append(stripped)

        # save the final section
        if current_lines:
            sections.append(
                (current_name, " ".join(current_lines), current_page)
            )

        return sections

    def _section_for_page(
        self, page_num: int, sections: List[Tuple[str, str, int]]
    ) -> str:
        """Return the section that was active on the given page."""
        result = "main"
        for name, _, section_page in sections:
            if section_page <= page_num:
                result = name
        return result
