import re

import pytest

from document_processor import DocumentError, DocumentProcessor

SAMPLE = "\n\n".join(
    f"Paragraph {p}. " + " ".join(f"Sentence {p}-{s} talks about topic{p}x{s} in some detail." for s in range(12))
    for p in range(15)
)


def test_short_text_is_a_single_chunk():
    chunks = DocumentProcessor(chunk_size=100, chunk_overlap=10).chunk_text("Just one line.")
    assert [c["text"] for c in chunks] == ["Just one line."]


def test_chunks_respect_size_bound_and_indexing():
    processor = DocumentProcessor(chunk_size=64, chunk_overlap=8)
    chunks = processor.chunk_text(SAMPLE)

    assert len(chunks) > 1
    assert all(c["text"].strip() for c in chunks)
    # chunk_size is a hard cap, overlap included (4 chars per estimated token)
    assert all(len(c["text"]) <= 64 * 4 for c in chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert {c["total_chunks"] for c in chunks} == {len(chunks)}


def test_chunks_cover_every_word():
    chunks = DocumentProcessor(chunk_size=64, chunk_overlap=8).chunk_text(SAMPLE)
    covered = set(re.findall(r"\w+", " ".join(c["text"] for c in chunks)))
    assert set(re.findall(r"\w+", SAMPLE)) <= covered


def test_overlap_never_pushes_a_chunk_past_chunk_size():
    # Paragraphs just under the 40-char cap leave no room for a 20-char overlap
    text = "\n\n".join(f"paragraph {i:02d} " + "x" * 22 for i in range(6))
    chunks = DocumentProcessor(chunk_size=10, chunk_overlap=5).chunk_text(text)

    assert len(chunks) == 6
    assert all(len(c["text"]) <= 10 * 4 for c in chunks)


def test_text_without_separators_is_force_split():
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=5)
    chunks = processor.chunk_text("x" * 1000)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 50 * 4 for c in chunks)


def test_blank_lines_never_make_an_empty_chunk():
    text = "aaaa bbbb" + "\n" * 30 + "cccc dddd eeee ffff"
    chunks = DocumentProcessor(chunk_size=4, chunk_overlap=0).chunk_text(text)
    assert [c["text"] for c in chunks] == ["aaaa bbbb", "cccc dddd eeee", "ffff"]


def test_zero_overlap_does_not_repeat_text():
    text = " ".join(f"word{i}" for i in range(40))
    chunks = DocumentProcessor(chunk_size=10, chunk_overlap=0).chunk_text(text)

    joined = " ".join(c["text"] for c in chunks)
    assert len(chunks) > 1
    assert joined.split() == text.split()


def test_overlap_carries_text_between_chunks():
    text = " ".join(f"word{i}" for i in range(40))
    chunks = DocumentProcessor(chunk_size=10, chunk_overlap=3).chunk_text(text)

    for previous, current in zip(chunks, chunks[1:]):
        first_word = current["text"].split()[0]
        assert first_word in previous["text"]


@pytest.mark.parametrize("size, overlap", [(100, 100), (100, 150), (100, -1), (0, 0)])
def test_invalid_chunk_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        DocumentProcessor(chunk_size=size, chunk_overlap=overlap)


def test_process_file_attaches_filename_and_content_hash(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nSome content worth indexing.", encoding="utf-8")
    processor = DocumentProcessor()

    result = processor.process_file(str(path))

    assert result["filename"] == "notes.md"
    assert all(c["chunk_config"] == processor.chunk_config for c in result["chunks"])
    file_hash = processor.compute_file_hash(str(path))
    assert all(c["filename"] == "notes.md" and c["content_hash"] == file_hash for c in result["chunks"])


def test_content_hash_changes_with_content(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("first version", encoding="utf-8")
    before = DocumentProcessor.compute_file_hash(str(path))
    path.write_text("second version", encoding="utf-8")
    assert DocumentProcessor.compute_file_hash(str(path)) != before


def test_non_utf8_text_falls_back_to_latin1(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_bytes("café crème".encode("latin-1"))
    assert DocumentProcessor().load_file(str(path)) == "café crème"


def test_unsupported_and_empty_files_are_rejected(tmp_path):
    unsupported = tmp_path / "image.png"
    unsupported.write_bytes(b"data")
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    processor = DocumentProcessor()

    assert processor.validate_file(str(unsupported))[0] is False
    assert processor.validate_file(str(empty)) == (False, "File is empty")


def make_pdf(path, pages):
    """Write a PDF with one page per list of lines."""
    import pymupdf

    pdf = pymupdf.open()
    for lines in pages:
        page = pdf.new_page()
        for i, line in enumerate(lines):
            page.insert_text((72, 72 + 14 * i), line)
    pdf.save(path)
    pdf.close()
    return str(path)


def test_pdf_text_is_extracted(tmp_path):
    path = make_pdf(tmp_path / "report.pdf", [["Cats purr when content."], ["Dogs bark at strangers."]])

    result = DocumentProcessor().process_file(path)

    assert "Cats purr when content." in result["text"] and "Dogs bark at strangers." in result["text"]
    assert result["file_type"] == "PDF Document" and result["total_chunks"] == 1


def test_pdf_chunks_record_the_pages_they_come_from(tmp_path):
    topics = ["cats", "dogs", "owls"]
    pages = [[f"Line {i} about {topic}, with a few more words." for i in range(8)] for topic in topics]
    path = make_pdf(tmp_path / "animals.pdf", pages)

    chunks = DocumentProcessor(chunk_size=40, chunk_overlap=0).process_file(path)["chunks"]

    assert len(chunks) > 3
    for chunk in chunks:
        mentioned = [t for t in topics if f"about {t}" in chunk["text"]]
        assert mentioned == topics[chunk["page_start"] - 1:chunk["page_end"]], chunk
    assert {c["page_start"] for c in chunks} == {1, 2, 3}


def test_a_chunk_spanning_pages_records_the_range(tmp_path):
    path = make_pdf(tmp_path / "short.pdf", [["One."], ["Two."], ["Three."]])

    [chunk] = DocumentProcessor().process_file(path)["chunks"]

    assert (chunk["page_start"], chunk["page_end"]) == (1, 3)


def test_blank_pages_keep_the_numbering(tmp_path):
    path = make_pdf(tmp_path / "gaps.pdf", [["Intro text."], [], ["Page three."]])

    chunks = DocumentProcessor(chunk_size=4, chunk_overlap=0).process_file(path)["chunks"]

    assert [(c["text"], c["page_start"], c["page_end"]) for c in chunks] == [
        ("Intro text.", 1, 1), ("Page three.", 3, 3)]


def test_text_files_have_no_page_numbers(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Just some notes.", encoding="utf-8")

    [chunk] = DocumentProcessor().process_file(str(path))["chunks"]

    assert "page_start" not in chunk and "page_end" not in chunk


def test_unreadable_files_raise_document_error(tmp_path):
    blank = tmp_path / "blank.txt"
    blank.write_text("   \n\n  ", encoding="utf-8")
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not really a pdf")
    processor = DocumentProcessor()

    with pytest.raises(DocumentError, match="No text content"):
        processor.load_file(str(blank))
    with pytest.raises(DocumentError, match="Error extracting text from PDF"):
        processor.load_file(str(broken))
    with pytest.raises(DocumentError, match="File does not exist"):
        processor.process_file(str(tmp_path / "missing.txt"))
