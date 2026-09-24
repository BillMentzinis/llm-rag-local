import re

import pytest

from document_processor import DocumentProcessor

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
    size, overlap = 64 * 4, 8 * 4

    assert len(chunks) > 1
    assert all(c["text"].strip() for c in chunks)
    # A chunk may carry the previous chunk's overlap on top of a full-size piece
    assert all(len(c["text"]) <= size + overlap for c in chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert {c["total_chunks"] for c in chunks} == {len(chunks)}


def test_chunks_cover_every_word():
    chunks = DocumentProcessor(chunk_size=64, chunk_overlap=8).chunk_text(SAMPLE)
    covered = set(re.findall(r"\w+", " ".join(c["text"] for c in chunks)))
    assert set(re.findall(r"\w+", SAMPLE)) <= covered


def test_text_without_separators_is_force_split():
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=5)
    chunks = processor.chunk_text("x" * 1000)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 50 * 4 for c in chunks)


def test_zero_overlap_is_respected():
    assert DocumentProcessor(chunk_size=100, chunk_overlap=0).chunk_overlap == 0


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
