"""
RAG Pipeline Module.
Orchestrates document processing, retrieval, and generation for RAG functionality.
"""

import os
from typing import Callable, List, Dict, Optional, Tuple
from document_processor import DocumentProcessor
from llm_backends import LLMBackend
from vector_store_manager import VectorStoreManager
from config import (CHAT_CONFIG, GENERATION_CONFIG, RAG_CONFIG, RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT,
                    SUMMARY_CONFIG, SUMMARY_COMBINE_PROMPT, SUMMARY_CONDENSE_PROMPT, SUMMARY_PART_PROMPT,
                    SUMMARY_WHOLE_PROMPT)

# Rough per-message cost of a chat template's role markers, in tokens
MESSAGE_OVERHEAD_TOKENS = 8

# Overlap between consecutive chunks shorter than this isn't looked for when
# joining them back up (a short match is likely a coincidence)
MIN_OVERLAP_CHARS = 20
MAX_OVERLAP_CHARS = 1000


def select_history(history: Optional[List[Dict]], max_turns: int) -> List[List[Dict]]:
    """
    Pick the most recent complete question/answer pairs from a conversation.

    Pairs keep the roles alternating from a user message, which some chat
    templates require. A question whose answer has no text (it failed, or was
    stopped before any output) is left out along with that answer, as are
    unpaired messages.

    Args:
        history: Earlier messages [{"role": "user/assistant", "content": "..."}]
        max_turns: Most pairs to keep

    Returns:
        List of [user message, assistant message] pairs, oldest first
    """
    turns = []
    question = None
    for msg in history or []:
        content = msg.get("content") or ""
        if msg.get("role") == "user":
            question = {"role": "user", "content": content}
        elif msg.get("role") == "assistant" and question is not None:
            if content.strip():
                turns.append([question, {"role": "assistant", "content": content}])
            question = None
    return turns[-max_turns:] if max_turns > 0 else []


def source_label(metadata: Dict) -> str:
    """
    Name a chunk's source: its file and pages (PDFs), or its chunk number.

    Args:
        metadata: The chunk's metadata

    Returns:
        e.g. "report.pdf, p. 3", "report.pdf, pp. 3-4" or "notes.txt, Chunk 2"
    """
    filename = metadata.get("filename", "unknown")
    first, last = metadata.get("page_start"), metadata.get("page_end", metadata.get("page_start"))
    if first is None:
        return f"{filename}, Chunk {metadata.get('chunk_index', 0) + 1}"
    return f"{filename}, p. {first}" if last == first else f"{filename}, pp. {first}-{last}"



def join_chunks(texts: List[str]) -> str:
    """
    Join a document's consecutive chunks back into running text, leaving out
    the text each chunk repeats from the end of the one before.

    Args:
        texts: Chunk texts, in order

    Returns:
        The joined text
    """
    joined = ""
    for text in texts:
        if not joined:
            joined = text
            continue
        longest = min(len(joined), len(text), MAX_OVERLAP_CHARS)
        overlap = next((k for k in range(longest, MIN_OVERLAP_CHARS - 1, -1) if joined.endswith(text[:k])), 0)
        joined += text[overlap:] if overlap else "\n\n" + text
    return joined


def page_range(chunks: List[Dict]) -> Optional[Tuple[int, int]]:
    """First and last page of some chunks of a PDF, or None for other files."""
    starts = [c["metadata"]["page_start"] for c in chunks if "page_start" in c["metadata"]]
    ends = [c["metadata"].get("page_end", c["metadata"]["page_start"]) for c in chunks if "page_start" in c["metadata"]]
    return (min(starts), max(ends)) if starts else None


def describe_pages(pages: Optional[Tuple[int, int]]) -> str:
    """e.g. " (pages 3-7)", " (page 3)", or "" when pages aren't known."""
    if not pages:
        return ""
    return f" (page {pages[0]})" if pages[0] == pages[1] else f" (pages {pages[0]}-{pages[1]})"


def split_into_parts(chunks: List[Dict], max_tokens: int, count_tokens: Callable[[str], int]) -> List[Dict]:
    """
    Group a document's chunks, in order, into parts of about max_tokens.

    A chunk is never split. Chunk sizes are added up, overlaps included, so a
    part is never larger than estimated.

    Args:
        chunks: The document's chunks ({"text", "metadata"}), in order
        max_tokens: Most tokens in one part
        count_tokens: Counts the tokens in a text

    Returns:
        List of {"text", "pages"} dictionaries
    """
    return [{"text": join_chunks([c["text"] for c in group]), "pages": page_range(group)}
            for group in _group_by_tokens(chunks, max_tokens, count_tokens)]


def _group_by_tokens(items: List[Dict], max_tokens: int, count_tokens: Callable[[str], int]) -> List[List[Dict]]:
    """Split items ({"text", ...}), in order, into runs whose texts add up to at most max_tokens."""
    groups, current, size = [], [], 0
    for item in items:
        tokens = count_tokens(item["text"])
        if current and size + tokens > max_tokens:
            groups.append(current)
            current, size = [], 0
        current.append(item)
        size += tokens
    if current:
        groups.append(current)
    return groups


class RAGPipeline:
    """
    Coordinates RAG workflow: document ingestion, retrieval, and context-enhanced generation.
    """

    def __init__(self, llm: Optional[LLMBackend] = None, vector_store: VectorStoreManager = None,
                 doc_processor: DocumentProcessor = None, config: Dict = None):
        """
        Initialize RAG Pipeline.

        Args:
            llm: Backend that generates answers; ingestion and retrieval work without one
            vector_store: VectorStoreManager instance (creates new if None)
            doc_processor: DocumentProcessor instance (creates new if None)
            config: Optional RAG configuration override
        """
        self.llm = llm
        self.vector_store = vector_store or VectorStoreManager()
        self.doc_processor = doc_processor or DocumentProcessor()
        self.config = config or RAG_CONFIG

        print("RAG Pipeline initialized successfully!")

    def ingest_document(self, file_path: str) -> Dict:
        """
        Process and index a document for RAG.

        Args:
            file_path: Path to document file

        Re-ingesting a file with the same name replaces the indexed version,
        unless its content and the chunking settings are unchanged, in which
        case it's skipped.

        Returns:
            Dictionary with ingestion results and metadata
        """
        try:
            filename = os.path.basename(file_path)
            previous = self.vector_store.get_document_metadata(filename)
            # PDFs indexed before page numbers were recorded are indexed again
            has_pages = "page_start" in (previous or {}) or not filename.lower().endswith(".pdf")
            if (previous and has_pages
                    and previous.get("chunk_config") == self.doc_processor.chunk_config
                    and previous.get("content_hash") == self.doc_processor.compute_file_hash(file_path)):
                return {
                    "success": True,
                    "skipped": True,
                    "filename": filename,
                    "chunks_added": 0,
                    "message": f"{filename} is already indexed and unchanged"
                }
            replacing = previous is not None

            # Process document
            print(f"Processing file: {file_path}")
            result = self.doc_processor.process_file(file_path)

            # Add to vector store (replaces any earlier version of this file)
            chunks_added = self.vector_store.add_documents(result["chunks"])

            action = "Updated" if replacing else "Processed"
            return {
                "success": True,
                "skipped": False,
                "replaced": replacing,
                "filename": result["filename"],
                "file_type": result["file_type"],
                "chunks_added": chunks_added,
                "total_chars": result["total_chars"],
                "estimated_tokens": result["estimated_tokens"],
                "message": f"{action} {result['filename']}: {chunks_added} chunks indexed"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process document: {str(e)}"
            }

    def retrieve_context(self, query: str, top_k: int = None, min_similarity: float = None,
                         filter_metadata: Dict = None) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User query
            top_k: Number of chunks to retrieve (default from config)
            min_similarity: Minimum similarity threshold (default from config)
            filter_metadata: Optional metadata filter, e.g. {"filename": "doc.pdf"}

        Returns:
            List of retrieved chunk dictionaries
        """
        if self.vector_store.is_empty():
            return []

        top_k = top_k or self.config["top_k"]
        results = self.vector_store.search(query, top_k=top_k, min_similarity=min_similarity,
                                           filter_metadata=filter_metadata)

        return results

    def format_context_for_prompt(self, context_chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string for prompt.

        Args:
            context_chunks: List of retrieved chunk dictionaries

        Returns:
            Formatted context string
        """
        if not context_chunks:
            return ""

        context_parts = []
        for chunk in context_chunks:
            context_parts.append(f"[Source: {source_label(chunk['metadata'])}]\n{chunk['text']}")

        return "\n\n".join(context_parts)

    def build_rag_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Build complete prompt with RAG context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks

        Returns:
            Complete prompt string
        """
        context_text = self.format_context_for_prompt(context_chunks)

        if not context_text:
            # No context available, return query only
            return query

        # Use template from config
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context_text,
            query=query
        )

        return prompt

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if self.llm is not None:
            return self.llm.count_tokens(text)
        # Character-based estimation
        return len(text) // 4

    def truncate_context_to_budget(self, context_chunks: List[Dict], token_budget: int) -> List[Dict]:
        """
        Truncate context chunks to fit within token budget.

        Args:
            context_chunks: List of retrieved chunks
            token_budget: Maximum tokens allowed for context

        Returns:
            Truncated list of chunks
        """
        truncated = []
        total_tokens = 0

        for chunk in context_chunks:
            chunk_tokens = self.estimate_token_count(chunk["text"])

            if total_tokens + chunk_tokens <= token_budget:
                truncated.append(chunk)
                total_tokens += chunk_tokens
            else:
                break

        return truncated

    def stream_response(self, query: str, context_chunks: List[Dict] = None,
                        history: List[Dict] = None, generation_config: Dict = None) -> Dict:
        """
        Start generating a response with optional RAG context, streaming its text.

        The model gets the system prompt, the last few complete question/answer
        pairs, and the question (with any document excerpts), trimmed to fit its
        context window. Generation only starts once the stream is iterated. Closing the stream
        early (or abandoning it, as happens when Streamlit interrupts a script
        run) stops generation.

        Args:
            query: User query
            context_chunks: Optional retrieved context chunks
            history: Optional conversation history [{"role": "user/assistant", "content": "..."}]
            generation_config: Optional generation parameters override

        Returns:
            Dictionary with "stream" (an iterator of text pieces) and metadata
        """
        if self.llm is None:
            raise RuntimeError("No language model is loaded")
        generation_config = generation_config or {}

        chunks = self.truncate_context_to_budget(context_chunks or [], CHAT_CONFIG["rag_context_tokens"])
        turns = select_history(history, CHAT_CONFIG["history_turns"])
        system = {"role": "system", "content": SYSTEM_PROMPT}

        # Fit the prompt in the context window, leaving room for the answer. Drop
        # the oldest turns first, then the least relevant excerpts (chunks arrive
        # ranked); the question itself is always sent.
        max_new_tokens = generation_config.get("max_new_tokens", GENERATION_CONFIG["max_new_tokens"])
        prompt_budget = self.llm.context_window() - max_new_tokens - CHAT_CONFIG["prompt_margin_tokens"]
        trimmed = False
        while True:
            question = {"role": "user", "content": self.build_rag_prompt(query, chunks)}
            messages = [system] + [m for turn in turns for m in turn] + [question]
            if self.count_prompt_tokens(messages) <= prompt_budget:
                break
            if turns:
                turns = turns[1:]
            elif chunks:
                chunks = chunks[:-1]
            else:
                break
            trimmed = True

        return {
            "stream": self.llm.stream_chat(messages, generation_config),
            "sources": chunks,
            "num_sources": len(chunks),
            "rag_enabled": len(chunks) > 0,
            "history_turns": len(turns),
            "trimmed_to_fit": trimmed
        }

    def count_prompt_tokens(self, messages: List[Dict]) -> int:
        """
        Estimate the tokens a list of chat messages takes up in the prompt.

        Args:
            messages: Chat messages

        Returns:
            Estimated token count, including the chat template's role markers
        """
        return sum(self.estimate_token_count(m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in messages)

    def stream_summary(self, filename: str, generation_config: Dict = None,
                       on_progress: Callable[[float, str], None] = None) -> Dict:
        """
        Start summarizing a whole document, streaming the summary's text.

        A document that fits in one prompt is summarized in one go. A longer one
        is split into parts that are summarized one by one; the part summaries
        are then combined (condensed in groups first if they're still too long
        together). The parts are read when the stream is first iterated, before
        the summary's first piece arrives; on_progress reports how far that is.

        Args:
            filename: Name of the indexed document
            generation_config: Generation parameters for the final summary
            on_progress: Called with (fraction done, description) while the parts are read

        Returns:
            Dictionary with "stream" (an iterator of text pieces), "parts"
            (how many parts the document was read in) and "pages" (first and
            last page, or None)
        """
        if self.llm is None:
            raise RuntimeError("No language model is loaded")
        chunks = self.vector_store.get_document_chunks(filename)
        if not chunks:
            raise ValueError(f"{filename} isn't in the document index")

        final_config = {**(generation_config or {}), "temperature": SUMMARY_CONFIG["temperature"]}
        final_tokens = final_config.get("max_new_tokens", GENERATION_CONFIG["max_new_tokens"])
        reserved = (CHAT_CONFIG["prompt_margin_tokens"] + SUMMARY_CONFIG["prompt_tokens"]
                    + self.count_prompt_tokens([{"role": "system", "content": SYSTEM_PROMPT}]) + MESSAGE_OVERHEAD_TOKENS)
        window = self.llm.context_window()
        part_tokens = min(SUMMARY_CONFIG["part_tokens"], window - SUMMARY_CONFIG["part_summary_tokens"] - reserved)
        final_budget = window - final_tokens - reserved  # Room for the text in the final prompt
        if min(part_tokens, final_budget) < 2 * SUMMARY_CONFIG["part_summary_tokens"]:
            raise RuntimeError("The model's context window is too small to summarize documents "
                               "(try fewer Max tokens in the generation settings)")

        pages = page_range(chunks)
        whole = join_chunks([c["text"] for c in chunks])
        if self.estimate_token_count(whole) <= min(part_tokens, final_budget):
            prompt = SUMMARY_WHOLE_PROMPT.format(document=filename, text=whole)
            return {"stream": self.llm.stream_chat(self._summary_messages(prompt), final_config),
                    "parts": 1, "pages": pages}

        parts = split_into_parts(chunks, part_tokens, self.estimate_token_count)
        stream = self._summarize_parts(filename, parts, final_config, part_tokens, final_budget, on_progress)
        return {"stream": stream, "parts": len(parts), "pages": pages}

    def _summarize_parts(self, filename: str, parts: List[Dict], final_config: Dict, part_tokens: int,
                         final_budget: int, on_progress: Optional[Callable[[float, str], None]]):
        """Summarize each part, then stream the combined summary (see stream_summary)."""
        part_config = {**final_config, "max_new_tokens": SUMMARY_CONFIG["part_summary_tokens"]}
        summaries = []
        for i, part in enumerate(parts, 1):
            pages = describe_pages(part["pages"])
            prompt = SUMMARY_PART_PROMPT.format(part=i, parts=len(parts), document=filename, pages=pages,
                                                text=part["text"])
            summary = self._generate(prompt, part_config, on_progress, (i - 1) / len(parts), 1 / len(parts),
                                     f"Reading part {i} of {len(parts)}{pages}")
            summaries.append(f"Part {i}{pages}:\n{summary}")

        # Condense the part summaries in groups until they fit in the final prompt together
        while len(summaries) > 1 and self.estimate_token_count("\n\n".join(summaries)) > final_budget:
            groups = [[s["text"] for s in group] for group in
                      _group_by_tokens([{"text": s} for s in summaries], part_tokens, self.estimate_token_count)]
            if len(groups) >= len(summaries):
                break  # Can't condense any further; send what there is
            summaries = [self._generate(SUMMARY_CONDENSE_PROMPT.format(document=filename,
                                                                       summaries="\n\n".join(group)),
                                        part_config, on_progress, 1.0, 0.0, "Combining the part summaries")
                         for group in groups]

        if on_progress:
            on_progress(1.0, "Writing the summary")
        prompt = SUMMARY_COMBINE_PROMPT.format(document=filename, summaries="\n\n".join(summaries))
        yield from self.llm.stream_chat(self._summary_messages(prompt), final_config)

    def _generate(self, prompt: str, generation_config: Dict, on_progress, start: float, span: float,
                  label: str) -> str:
        """Generate a whole response, reporting progress from start to start + span as it arrives."""
        expected_chars = 4 * generation_config["max_new_tokens"]
        text = ""
        if on_progress:
            on_progress(start, label)
        stream = self.llm.stream_chat(self._summary_messages(prompt), generation_config)
        try:
            for piece in stream:
                text += piece
                if on_progress:
                    on_progress(start + span * min(len(text) / expected_chars, 0.95), label)
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()  # Stops generation if this run was interrupted
        return text.strip()

    @staticmethod
    def _summary_messages(prompt: str) -> List[Dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

    def generate_response(self, query: str, context_chunks: List[Dict] = None,
                         history: List[Dict] = None, generation_config: Dict = None) -> Dict:
        """
        Generate a complete response with optional RAG context.

        Args:
            query: User query
            context_chunks: Optional retrieved context chunks
            history: Optional conversation history [{"role": "user/assistant", "content": "..."}]
            generation_config: Optional generation parameters override

        Returns:
            Dictionary with response and metadata
        """
        result = self.stream_response(query, context_chunks, history, generation_config)
        result["response"] = "".join(result.pop("stream"))
        return result

    def generate_with_rag(self, query: str, history: List[Dict] = None,
                         top_k: int = None, generation_config: Dict = None,
                         filter_metadata: Dict = None) -> Dict:
        """
        Convenience method for RAG-enhanced generation.

        Args:
            query: User query
            history: Optional conversation history
            top_k: Number of context chunks to retrieve
            generation_config: Optional generation parameters
            filter_metadata: Optional metadata filter, e.g. {"filename": "doc.pdf"}

        Returns:
            Dictionary with response and metadata
        """
        # Retrieve context
        context_chunks = self.retrieve_context(query, top_k=top_k, filter_metadata=filter_metadata)

        # Generate response
        result = self.generate_response(
            query=query,
            context_chunks=context_chunks,
            history=history,
            generation_config=generation_config
        )

        return result

    def get_documents(self) -> List[Dict]:
        """
        Get list of all indexed documents.

        Returns:
            List of document info dictionaries
        """
        return self.vector_store.list_documents()

    def delete_document(self, filename: str) -> int:
        """
        Delete a document from the index.

        Args:
            filename: Name of file to delete

        Returns:
            Number of chunks deleted
        """
        return self.vector_store.delete_document(filename)

    def clear_all_documents(self) -> int:
        """
        Clear all documents from the index.

        Returns:
            Number of chunks deleted
        """
        return self.vector_store.clear_all()

    def get_stats(self) -> Dict:
        """
        Get RAG pipeline statistics.

        Returns:
            Dictionary with statistics
        """
        vs_stats = self.vector_store.get_stats()

        return {
            "vector_store": vs_stats,
            "config": {
                "chunk_size": self.doc_processor.chunk_size,
                "chunk_overlap": self.doc_processor.chunk_overlap,
                "top_k": self.config["top_k"],
                "min_similarity": self.config["min_similarity"],
            }
        }


if __name__ == "__main__":
    # Simple initialization test
    print("Testing RAG Pipeline initialization...")
    print("Note: This test only initializes vector store and document processor.")
    print("Full pipeline requires an LLM backend (see llm_backends.py).")

    vector_store = VectorStoreManager()
    doc_processor = DocumentProcessor()

    print("\nComponent initialization successful!")
    print(f"Vector store stats: {vector_store.get_stats()}")
