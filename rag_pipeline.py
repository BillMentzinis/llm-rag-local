"""
RAG Pipeline Module.
Orchestrates document processing, retrieval, and generation for RAG functionality.
"""

import os
import torch
from threading import Event, Thread
from typing import List, Dict, Optional, Any, Iterator
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from config import RAG_CONFIG, RAG_PROMPT_TEMPLATE, TOKEN_BUDGET


class _StopOnEvent(StoppingCriteria):
    """Stops generation once the event is set, e.g. when the reader abandons the stream."""

    def __init__(self, event: Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(),
                          dtype=torch.bool, device=input_ids.device)


class RAGPipeline:
    """
    Coordinates RAG workflow: document ingestion, retrieval, and context-enhanced generation.
    """

    def __init__(self, model: Any, tokenizer: Any, vector_store: VectorStoreManager = None,
                 doc_processor: DocumentProcessor = None, config: Dict = None):
        """
        Initialize RAG Pipeline.

        Args:
            model: Hugging Face model for generation
            tokenizer: Hugging Face tokenizer
            vector_store: VectorStoreManager instance (creates new if None)
            doc_processor: DocumentProcessor instance (creates new if None)
            config: Optional RAG configuration override
        """
        self.model = model
        self.tokenizer = tokenizer
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
            if (previous
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
            filename = chunk["metadata"].get("filename", "unknown")
            chunk_idx = chunk["metadata"].get("chunk_index", 0)
            text = chunk["text"]

            context_parts.append(f"[Source: {filename}, Chunk {chunk_idx + 1}]\n{text}")

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
        # Use tokenizer for accurate count
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except:
            # Fallback to character-based estimation
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

        Generation runs in a background thread and only starts once the stream
        is iterated. Closing the stream early (or abandoning it, as happens when
        Streamlit interrupts a script run) stops generation at the next token.

        Args:
            query: User query
            context_chunks: Optional retrieved context chunks
            history: Optional conversation history [{"role": "user/assistant", "content": "..."}]
            generation_config: Optional generation parameters override

        Returns:
            Dictionary with "stream" (an iterator of text pieces) and metadata
        """
        # Truncate context to fit budget
        context_budget = TOKEN_BUDGET["rag_context"]
        if context_chunks:
            context_chunks = self.truncate_context_to_budget(context_chunks, context_budget)

        # Build prompt with or without RAG context
        if context_chunks:
            prompt_text = self.build_rag_prompt(query, context_chunks)
            sources_used = context_chunks
        else:
            prompt_text = query
            sources_used = []

        # Format conversation history
        messages = []

        # Add history if provided
        if history:
            for msg in history[-3:]:  # Keep last 3 turns to manage context
                messages.append(msg)

        # Add current query
        messages.append({"role": "user", "content": prompt_text})

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        return {
            "stream": self._stream_tokens(inputs, generation_config or {}),
            "sources": sources_used,
            "num_sources": len(sources_used),
            "prompt_tokens": inputs.input_ids.shape[1],
            "rag_enabled": len(sources_used) > 0
        }

    def _stream_tokens(self, inputs: Any, generation_config: Dict) -> Iterator[str]:
        """
        Run model.generate in a background thread and yield decoded text as it arrives.

        Args:
            inputs: Tokenized prompt on the model's device
            generation_config: Generation parameters

        Yields:
            Pieces of response text
        """
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        stop = Event()
        errors = []

        def generate():
            try:
                with torch.no_grad():  # grad mode is per thread
                    self.model.generate(
                        **inputs,
                        **generation_config,
                        pad_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([_StopOnEvent(stop)])
                    )
            except Exception as e:
                errors.append(e)
                streamer.end()  # unblock the reader

        thread = Thread(target=generate, daemon=True)
        thread.start()
        try:
            for text in streamer:
                if text:
                    yield text
            if errors:
                raise errors[0]
        finally:
            # Stop early if the reader went away, and wait so the GPU is free
            # before anything else is generated
            stop.set()
            thread.join()

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
    print("Full pipeline requires model and tokenizer.")

    vector_store = VectorStoreManager()
    doc_processor = DocumentProcessor()

    print("\nComponent initialization successful!")
    print(f"Vector store stats: {vector_store.get_stats()}")
