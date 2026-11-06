import fitz
from backend.services.pdf_processor import PDFProcessor

def test_text_cleaning_and_hyphenation():
    assert PDFProcessor()._clean_text("impor-\ntant   words") == "important words"

def test_content_doc_id_is_deterministic_and_content_based(tmp_path):
    processor = PDFProcessor()
    a = tmp_path / "a.pdf"; b = tmp_path / "b.pdf"
    a.write_bytes(b"same bytes"); b.write_bytes(b"same bytes")
    assert processor._generate_doc_id(a) == processor._generate_doc_id(b)
    b.write_bytes(b"different bytes")
    assert processor._generate_doc_id(a) != processor._generate_doc_id(b)

def test_real_pdf_page_metadata_and_short_chunk_skipping(tmp_path):
    path = tmp_path / "sample.pdf"
    pdf = fitz.open(); page = pdf.new_page()
    page.insert_text((72, 72), "This is enough meaningful text to create a chunk with page metadata.")
    pdf.save(path); pdf.close()
    result = PDFProcessor().process(str(path))
    assert result.total_pages == 1
    assert result.chunks and all(chunk.page_number == 1 for chunk in result.chunks)
    assert all(len(chunk.text) >= 20 for chunk in result.chunks)

def test_scanned_or_empty_pdf_raises_value_error(tmp_path):
    path = tmp_path / "blank.pdf"
    pdf = fitz.open(); pdf.new_page(); pdf.save(path); pdf.close()
    try:
        PDFProcessor().process(str(path))
    except ValueError as exc:
        assert "No extractable text" in str(exc)
    else:
        raise AssertionError("Textless PDF must be rejected")
