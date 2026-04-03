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
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
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
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            # Try splitting on paragraphs → newlines → sentences → words
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
        logger.info(
            f"PDFProcessor initialized | chunk_size={settings.chunk_size} "
            f"overlap={settings.chunk_overlap}"
        )

    def _generate_doc_id(self, filename: str, file_size: int) -> str:
        """Generate a deterministic unique ID for a document."""
        raw = f"{filename}_{file_size}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

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

    def _extract_text_by_page(self, pdf_path: str) -> List[dict]:
        """
        Extract text from each page individually.
        Returns list of {page_number, text} dicts.
        """
        pages = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  # plain text extraction
            cleaned = self._clean_text(text)
            if cleaned:  # skip empty pages
                pages.append({
                    "page_number": page_num + 1,  # 1-indexed
                    "text": cleaned
                })

        doc.close()
        logger.info(f"Extracted {len(pages)} pages from {Path(pdf_path).name}")
        return pages

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
        doc_id = self._generate_doc_id(filename, int(file_size_kb))

        logger.info(f"Processing PDF: {filename} ({file_size_kb:.1f} KB)")

        # Step 1: Extract text page by page
        pages = self._extract_text_by_page(pdf_path)
        if not pages:
            raise ValueError(f"No extractable text found in {filename}. "
                             "File may be scanned/image-only.")

        total_pages = pages[-1]["page_number"] if pages else 0

        # Step 2: Chunk each page separately (preserves page metadata)
        all_chunks: List[DocumentChunk] = []
        chunk_index = 0

        for page_data in pages:
            page_chunks = self.splitter.split_text(page_data["text"])

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
