"""
Document Processing Module for RAG Pipeline.
Handles file parsing, text extraction, and chunking for various file types.
"""

import os
import re
import bisect
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pymupdf
from config import SUPPORTED_EXTENSIONS, FILE_CONFIG, RAG_CONFIG

# Put between a PDF's pages; a paragraph break, so chunks tend to end at page boundaries
PAGE_BREAK = "\n\n"


class DocumentError(Exception):
    """A file that can't be read or indexed; the message says why."""


class DocumentProcessor:
    """
    Processes documents for RAG pipeline: extracts text and chunks it.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize DocumentProcessor.

        Args:
            chunk_size: Number of tokens per chunk (default from config)
            chunk_overlap: Number of overlapping tokens (default from config)
        """
        self.chunk_size = chunk_size if chunk_size is not None else RAG_CONFIG["chunk_size"]
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else RAG_CONFIG["chunk_overlap"]
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_size")
        self.max_file_size = FILE_CONFIG["max_file_size_mb"] * 1024 * 1024  # Convert to bytes

    @property
    def chunk_config(self) -> str:
        """Chunking settings as stored with each chunk, e.g. "200/30"."""
        return f"{self.chunk_size}/{self.chunk_overlap}"

    def is_supported(self, filename: str) -> bool:
        """
        Check if file type is supported.

        Args:
            filename: Name of the file

        Returns:
            True if file type is supported
        """
        ext = os.path.splitext(filename)[1].lower()
        return ext in SUPPORTED_EXTENSIONS

    def get_file_type(self, filename: str) -> Optional[str]:
        """
        Get human-readable file type.

        Args:
            filename: Name of the file

        Returns:
            File type description or None if unsupported
        """
        ext = os.path.splitext(filename)[1].lower()
        return SUPPORTED_EXTENSIONS.get(ext)

    def validate_file(self, file_path: str) -> tuple[bool, Optional[str]]:
        """
        Validate file before processing.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, "File does not exist"

        if not self.is_supported(os.path.basename(file_path)):
            return False, f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"

        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            return False, f"File too large. Max size: {FILE_CONFIG['max_file_size_mb']}MB"

        if file_size == 0:
            return False, "File is empty"

        return True, None

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """
        Compute a SHA-256 hash of a file's bytes, used to detect re-uploads.

        Args:
            file_path: Path to the file

        Returns:
            Hex digest
        """
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def extract_pages_from_pdf(self, file_path: str) -> List[str]:
        """
        Extract the text of each page of a PDF file using PyMuPDF.

        Args:
            file_path: Path to PDF file

        Returns:
            Text of each page, in order
        """
        try:
            with pymupdf.open(file_path) as doc:
                return [page.get_text() for page in doc]
        except Exception as e:
            raise DocumentError(f"Error extracting text from PDF: {str(e)}") from e

    def extract_text_from_text_file(self, file_path: str) -> str:
        """
        Extract text from text-based files (txt, md, code files).

        Args:
            file_path: Path to text file

        Returns:
            Extracted text
        """
        try:
            # Try primary encoding
            with open(file_path, 'r', encoding=FILE_CONFIG["encoding"]) as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to alternative encoding
            try:
                with open(file_path, 'r', encoding=FILE_CONFIG["encoding_fallback"]) as f:
                    return f.read()
            except Exception as e:
                raise DocumentError(f"Error reading text file: {str(e)}") from e

    def read_document(self, file_path: str) -> Tuple[str, Optional[List[int]]]:
        """
        Validate a file and extract its text.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (text, page_starts): page_starts lists where each page
            starts in the text for a PDF, and is None for other files

        Raises:
            DocumentError: If the file isn't valid or has no text
        """
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            raise DocumentError(error_msg)

        page_starts = None
        if os.path.splitext(file_path)[1].lower() == ".pdf":
            pages = self.extract_pages_from_pdf(file_path)
            page_starts, offset = [], 0
            for page in pages:
                page_starts.append(offset)
                offset += len(page) + len(PAGE_BREAK)
            text = PAGE_BREAK.join(pages)
        else:
            # All other supported types are text-based
            text = self.extract_text_from_text_file(file_path)

        if not text.strip():
            raise DocumentError("No text content found in file")

        return text, page_starts

    def load_file(self, file_path: str) -> str:
        """
        Validate a file and extract its text.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content

        Raises:
            DocumentError: If the file isn't valid or has no text
        """
        return self.read_document(file_path)[0]

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: 1 token ~= 4 chars).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks using recursive character splitting.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Convert token sizes to character approximations
        chunk_size_chars = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4

        # Splitting hierarchy: paragraphs -> sentences -> words
        separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]

        chunks = []
        chunks_text = self._recursive_split(text, chunk_size_chars, overlap_chars, separators)

        # Create chunk dictionaries with metadata
        for i, chunk_text in enumerate(chunks_text):
            chunk_dict = {
                "text": chunk_text,
                "chunk_index": i,
                "total_chunks": len(chunks_text),
                "char_count": len(chunk_text),
                "estimated_tokens": self.estimate_tokens(chunk_text),
            }

            # Add custom metadata if provided
            if metadata:
                chunk_dict.update(metadata)

            chunks.append(chunk_dict)

        return chunks

    @staticmethod
    def _add_page_numbers(chunks: List[Dict], text: str, page_starts: List[int]) -> None:
        """
        Record the pages each chunk comes from, as 1-based page_start and page_end.

        Chunks are pieces of the text in order, so each is looked up from where
        the previous one starts.

        Args:
            chunks: Chunk dictionaries from chunk_text(text)
            text: The text that was chunked
            page_starts: Where each page starts in the text
        """
        position = 0
        for chunk in chunks:
            found = text.find(chunk["text"], position)
            if found < 0:
                continue
            position = found
            chunk["page_start"] = bisect.bisect_right(page_starts, found)
            chunk["page_end"] = bisect.bisect_right(page_starts, found + len(chunk["text"]) - 1)

    def _recursive_split(self, text: str, chunk_size: int, overlap: int,
                        separators: List[str], current_sep_idx: int = 0) -> List[str]:
        """
        Recursively split text using hierarchy of separators.

        Args:
            text: Text to split
            chunk_size: Target chunk size in characters
            overlap: Overlap size in characters
            separators: List of separators in priority order
            current_sep_idx: Current separator index

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # If we've exhausted all separators, force split by characters
        if current_sep_idx >= len(separators):
            return self._force_split(text, chunk_size, overlap)

        separator = separators[current_sep_idx]
        splits = text.split(separator)

        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            # Add separator back (except for last split)
            piece = split + (separator if i < len(splits) - 1 else "")

            # If single piece is larger than chunk_size, recursively split with next separator
            if len(piece) > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Try next separator level
                sub_chunks = self._recursive_split(piece, chunk_size, overlap,
                                                  separators, current_sep_idx + 1)
                chunks.extend(sub_chunks)
                continue

            # If adding piece would exceed chunk_size, save current chunk
            if len(current_chunk) + len(piece) > chunk_size and current_chunk:
                if current_chunk.strip():  # a run of blank lines isn't a chunk
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap, trimmed so the chunk stays within chunk_size
                current_chunk = self._get_overlap(current_chunk, min(overlap, chunk_size - len(piece))) + piece
            else:
                current_chunk += piece

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _force_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Force split text by character count when no good separators found.

        Args:
            text: Text to split
            chunk_size: Target chunk size
            overlap: Overlap size

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

        return [c for c in chunks if c.strip()]

    def _get_overlap(self, text: str, overlap_size: int) -> str:
        """
        Get overlap portion from end of text.

        Args:
            text: Source text
            overlap_size: Number of characters to overlap

        Returns:
            Overlap text
        """
        if overlap_size <= 0:
            return ""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]

    def process_file(self, file_path: str) -> Dict:
        """
        Complete file processing: extract text and chunk.

        Args:
            file_path: Path to file

        Returns:
            Dictionary containing:
                - filename: Original filename
                - file_type: File type description
                - file_path: Path to file
                - text: Extracted text
                - chunks: List of chunk dictionaries (with page_start and page_end for PDFs)
                - metadata: File metadata
        """
        filename = os.path.basename(file_path)

        # Extract text
        text, page_starts = self.read_document(file_path)

        # Prepare metadata
        metadata = {
            "filename": filename,
            "file_type": self.get_file_type(filename),
            "upload_timestamp": datetime.now().isoformat(),
            "content_hash": self.compute_file_hash(file_path),
            "chunk_config": self.chunk_config,
        }

        # Chunk text
        chunks = self.chunk_text(text, metadata)
        if page_starts:
            self._add_page_numbers(chunks, text, page_starts)

        return {
            "filename": filename,
            "file_type": self.get_file_type(filename),
            "file_path": file_path,
            "text": text,
            "chunks": chunks,
            "metadata": metadata,
            "total_chunks": len(chunks),
            "total_chars": len(text),
            "estimated_tokens": self.estimate_tokens(text),
        }


if __name__ == "__main__":
    # Simple test
    processor = DocumentProcessor()
    print("DocumentProcessor initialized successfully!")
    print(f"Chunk size: {processor.chunk_size} tokens")
    print(f"Chunk overlap: {processor.chunk_overlap} tokens")
    print(f"Supported extensions: {list(SUPPORTED_EXTENSIONS.keys())}")
