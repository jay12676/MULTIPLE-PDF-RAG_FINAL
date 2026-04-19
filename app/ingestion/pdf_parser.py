import fitz  # PyMuPDF
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Generator, List
import logging
import os
import shutil

logger = logging.getLogger(__name__)

# ── OCR engine selection ────────────────────────────────────────────────────
# Priority: Tesseract (fast C++) > EasyOCR (slow but no-install fallback)

def _find_tesseract() -> str | None:
    """Return tesseract executable path if found, else None."""
    # Check PATH first
    if shutil.which("tesseract"):
        return shutil.which("tesseract")
    # Common Windows install paths
    for path in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\LOQ\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]:
        if os.path.exists(path):
            return path
    return None

_TESSERACT_PATH = _find_tesseract()
_easyocr_reader = None   # lazy-loaded only if Tesseract not found

if _TESSERACT_PATH:
    logger.info(f"[OCR] Using Tesseract: {_TESSERACT_PATH}")
else:
    logger.info("[OCR] Tesseract not found — will use EasyOCR (slower)")


def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        logger.info("[OCR] Loading EasyOCR model...")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("[OCR] EasyOCR ready.")
    return _easyocr_reader


# ── Data class ──────────────────────────────────────────────────────────────

@dataclass
class ParsedPage:
    page_number: int
    text: str
    has_tables: bool
    is_ocr: bool
    word_count: int


# ── Parser ──────────────────────────────────────────────────────────────────

# Number of threads used for parallel page parsing.
# Each thread processes one page independently (render + OCR if needed).
# Keep at 4 to avoid overwhelming CPU on EasyOCR; Tesseract is thread-safe.
_PARSE_WORKERS = 4


class PDFParser:
    """
    Streams a PDF page-by-page.
    Handles digital PDFs (direct text) and scanned/image PDFs (OCR fallback).
    Uses Tesseract if installed (fast), otherwise EasyOCR (no install needed).

    Pages are processed in parallel using a thread pool — this gives a
    significant speedup for mixed PDFs where some pages need OCR and
    others can be extracted directly.
    """

    MIN_CHARS_FOR_DIGITAL = 50

    def parse(self, pdf_path: str) -> List[ParsedPage]:
        """
        Parse all pages in parallel and return them in correct page order.
        Uses ThreadPoolExecutor so CPU-bound OCR pages don't block each other.
        """
        doc = fitz.open(pdf_path)
        total = len(doc)

        # Pre-extract raw fitz page data (must be done in main thread)
        # We pass page_index only — each worker re-opens the doc to avoid
        # sharing a non-thread-safe fitz.Document object.
        results: List[ParsedPage] = [None] * total  # type: ignore

        def _parse_one(page_index: int) -> tuple[int, ParsedPage]:
            # Each thread opens its own fitz document handle
            _doc = fitz.open(pdf_path)
            page = _doc[page_index]
            result = self._process_page(page, page_index + 1)
            _doc.close()
            return page_index, result

        with ThreadPoolExecutor(max_workers=_PARSE_WORKERS) as pool:
            futures = {pool.submit(_parse_one, i): i for i in range(total)}
            for future in as_completed(futures):
                try:
                    idx, parsed = future.result()
                    results[idx] = parsed
                except Exception as e:
                    idx = futures[future]
                    logger.warning(f"[pdf_parser] page {idx+1} failed: {e}")
                    # Insert a blank page so indexing doesn't skip the slot
                    results[idx] = ParsedPage(
                        page_number=idx + 1,
                        text="",
                        has_tables=False,
                        is_ocr=False,
                        word_count=0,
                    )

        doc.close()
        return results

    def validate(self, pdf_path: str) -> tuple[bool, str]:
        if not os.path.exists(pdf_path):
            return False, "File not found"
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
        except Exception as e:
            return False, f"Cannot open PDF: {e}"
        if count == 0:
            return False, "PDF has 0 pages"
        return True, f"Valid PDF — {count} pages"

    def get_page_count(self, pdf_path: str) -> int:
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        return n

    # ── Private helpers ──────────────────────────────────────────────────

    def _process_page(self, page: fitz.Page, page_number: int) -> ParsedPage:
        text = page.get_text("text").strip()
        is_ocr = False

        if len(text) < self.MIN_CHARS_FOR_DIGITAL:
            logger.info(f"[pdf_parser] Page {page_number}: running OCR")
            text = self._ocr_page(page, page_number)
            is_ocr = True

        has_tables, table_text = self._extract_tables(page)
        if has_tables:
            text = text + "\n\n" + table_text

        return ParsedPage(
            page_number=page_number,
            text=text.strip(),
            has_tables=has_tables,
            is_ocr=is_ocr,
            word_count=len(text.split()),
        )

    def _ocr_page(self, page: fitz.Page, page_number: int) -> str:
        """Render page to image and OCR it — Tesseract if available, else EasyOCR."""
        try:
            if _TESSERACT_PATH:
                return self._ocr_tesseract(page)
            else:
                return self._ocr_easyocr(page)
        except Exception as e:
            logger.warning(f"[pdf_parser] OCR page {page_number} failed: {e}")
            return ""

    def _ocr_tesseract(self, page: fitz.Page) -> str:
        """Fast path: use Tesseract (C++ engine, ~1s/page)."""
        import pytesseract
        from PIL import Image
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH

        # 2x zoom = ~150 dpi, good quality for printed text
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(img, config="--psm 6")

    def _ocr_easyocr(self, page: fitz.Page) -> str:
        """Fallback: use EasyOCR (no install needed, ~5-15s/page on CPU)."""
        # 1.5x zoom instead of 2x — 44% less pixels, noticeably faster
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        reader = _get_easyocr()
        results = reader.readtext(img, detail=0, paragraph=False)
        return "\n".join(results)

    def _extract_tables(self, page: fitz.Page) -> tuple[bool, str]:
        try:
            tables = page.find_tables()
            if not tables.tables:
                return False, ""
            rows = []
            for table in tables.tables:
                for row in table.extract():
                    rows.append(" | ".join(str(c).strip() if c else "" for c in row))
            return True, "\n".join(rows)
        except Exception as e:
            logger.warning(f"[pdf_parser] Table extraction failed: {e}")
            return False, ""
