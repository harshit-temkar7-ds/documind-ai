"""
DocuMind AI — PDF Processing Service
─────────────────────────────────────
Business Logic:
  1. Extract raw text from PDF using PyMuPDF (fastest Python PDF lib)
  2. Clean and normalize the text
  3. Split into overlapping chunks using LangChain's RecursiveCharacterTextSplitter
  4. Return structured Chunk objects ready for embedding

Why RecursiveCharacterTextSplitter?
  - Tries to split on paragraphs → sentences → words (in that order)
  - Preserves semantic meaning better than fixed-size splits
  - Overlap ensures context isn't lost at chunk boundaries
"""

import fitz  # PyMuPDF
import re
import hashlib
import tiktoken
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core import settings, get_logger

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """Represents a single text chunk from a PDF document."""
    chunk_id:    str          # unique hash ID
    doc_id:      str          # parent document ID
    doc_name:    str          # original filename
    text:        str          # chunk text content
    page_number: int          # source page number
    chunk_index: int          # position within document
    char_count:  int          # character length
    metadata:    dict = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Represents a fully processed PDF document."""
    doc_id:       str
    filename:     str
    total_pages:  int
    total_chunks: int
    chunks:       List[DocumentChunk]
    file_size_kb: float


class PDFProcessor:
    """
    Handles all PDF ingestion:
      - Text extraction page by page
      - Text cleaning (remove noise, fix spacing)
      - Recursive chunking with overlap
    """

    def __init__(self):
        self.encoding = self._load_encoding()
        if settings.chunk_unit == "chars":
            length_fn = len
        elif self.encoding is not None:
            length_fn = lambda value: len(self.encoding.encode(value))  # noqa: E731
        else:
            # Offline fallback: ~4 characters per token.
            length_fn = lambda value: max(1, len(value) // 4)  # noqa: E731
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=length_fn,
            # Try splitting on paragraphs → newlines → sentences → words
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
        logger.info(
            f"PDFProcessor initialized | chunk_size={settings.chunk_size} "
            f"overlap={settings.chunk_overlap}"
        )

    @staticmethod
    def _load_encoding():
        """tiktoken downloads its vocabulary on first use; fall back if that is impossible."""
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # network blocked, cache not writable, ...
            logger.warning(
                f"tiktoken encoding unavailable ({exc}); using ~4 chars/token estimate"
            )
            return None

    @property
    def _overlap_chars(self) -> int:
        """Cross-page overlap expressed in characters (tokens are ~4 chars)."""
        factor = 1 if settings.chunk_unit == "chars" else 4
        return settings.chunk_overlap * factor

    def _generate_doc_id(self, path: Path) -> str:
        """Hash incrementally to avoid reading the complete PDF into memory."""
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()[:16]

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text:
          - Remove excessive whitespace and blank lines
          - Fix broken hyphenated words (e.g. 'impor-\ntant' → 'important')
          - Normalize unicode characters
        """
        # Fix hyphenated line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)
        # Collapse more than 2 newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove page header/footer noise (short lines with only numbers)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        return text.strip()

    def _extract_text_by_page(self, pdf_path: str):
        """
        Extract text from each page individually.
        Returns (pages, total_pages) where pages is a list of {page_number, text}
        dicts (empty pages skipped) and total_pages is the real PDF page count.
        """
        pages = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            raise ValueError("Invalid or unreadable PDF file.") from exc

        ocr_enabled = True  # switched off after the first OCR failure (e.g. no Tesseract)
        try:
            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc[page_num]
                cleaned = self._clean_text(page.get_text("text", sort=True))
                if not cleaned and ocr_enabled:
                    try:
                        cleaned = self._ocr_page(page)
                    except Exception as exc:  # ImportError, TesseractNotFoundError, ...
                        ocr_enabled = False
                        logger.warning(f"OCR unavailable, skipping scanned pages: {exc}")
                if cleaned:  # skip empty pages
                    pages.append({
                        "page_number": page_num + 1,  # 1-indexed
                        "text": cleaned
                    })
        finally:
            doc.close()

        logger.info(f"Extracted {len(pages)} text pages from {Path(pdf_path).name}")
        return pages, total_pages

    def _ocr_page(self, page) -> str:
        """OCR one page (needs the pytesseract package and the Tesseract binary)."""
        import io
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        text = pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))))
        return self._clean_text(text)

    def process(self, pdf_path: str) -> ProcessedDocument:
        """
        Main entry point: process a PDF file end-to-end.

        Args:
            pdf_path: Absolute path to the PDF file

        Returns:
            ProcessedDocument with all chunks ready for embedding
        """
        path = Path(pdf_path)
        filename = path.name
        file_size_kb = path.stat().st_size / 1024
        doc_id = self._generate_doc_id(path)

        logger.info(f"Processing PDF: {filename} ({file_size_kb:.1f} KB)")

        # Step 1: Extract text page by page
        pages, total_pages = self._extract_text_by_page(pdf_path)
        if not pages:
            raise ValueError(f"No extractable text found in {filename}. "
                             "File may be scanned/image-only (OCR needs Tesseract installed).")

        # Step 2: Chunk each page separately (preserves page metadata)
        all_chunks: List[DocumentChunk] = []
        chunk_index = 0

        previous_tail = ""
        for page_data in pages:
            overlap = min(self._overlap_chars, len(previous_tail))
            page_text = (previous_tail[-overlap:] + "\n" if overlap else "") + page_data["text"]
            page_chunks = self.splitter.split_text(page_text)
            previous_tail = page_data["text"][-self._overlap_chars:]

            for chunk_text in page_chunks:
                if len(chunk_text.strip()) < 20:  # skip trivially short chunks
                    continue

                chunk_id = hashlib.md5(
                    f"{doc_id}_{chunk_index}".encode()
                ).hexdigest()[:16]

                all_chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    doc_name=filename,
                    text=chunk_text.strip(),
                    page_number=page_data["page_number"],
                    chunk_index=chunk_index,
                    char_count=len(chunk_text),
                    metadata={
                        "source": filename,
                        "page": page_data["page_number"],
                        "doc_id": doc_id,
                    }
                ))
                chunk_index += 1

        logger.info(
            f"Chunking complete | {filename} → {len(all_chunks)} chunks "
            f"across {total_pages} pages"
        )

        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            total_pages=total_pages,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
            file_size_kb=round(file_size_kb, 2),
        )
