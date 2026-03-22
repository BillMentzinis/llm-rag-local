"""
Streamlit UI for RAG-Enabled Llama 3.2 3B Chat Application.
"""

import os
import streamlit as st
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import (
    MODEL_CONFIG, GENERATION_CONFIG, RAG_GENERATION_CONFIG,
    UI_CONFIG, RAG_CONFIG, SUPPORTED_EXTENSIONS
)
from rag_pipeline import RAGPipeline
from vector_store_manager import VectorStoreManager
from document_processor import DocumentProcessor


# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"]
)


@st.cache_resource
def load_model_and_tokenizer():
    """
    Load model and tokenizer (cached to avoid reloading).
    """
    with st.spinner("Loading Llama 3.1 8B model... This may take a minute."):
        # Configure quantization
        bnb_config = BitsAndBytesConfig(**MODEL_CONFIG["quantization"])

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG["name"],
            trust_remote_code=MODEL_CONFIG["trust_remote_code"]
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["name"],
            quantization_config=bnb_config,
            device_map=MODEL_CONFIG["device_map"],
            trust_remote_code=MODEL_CONFIG["trust_remote_code"]
        )

        return model, tokenizer


@st.cache_resource
def initialize_rag_pipeline(_model, _tokenizer):
    """
    Initialize RAG pipeline components (cached).

    Args:
        _model: Model instance (underscore prefix prevents hashing)
        _tokenizer: Tokenizer instance

    Returns:
        RAGPipeline instance
    """
    with st.spinner("Initializing RAG pipeline..."):
        vector_store = VectorStoreManager()
        doc_processor = DocumentProcessor()
        pipeline = RAGPipeline(_model, _tokenizer, vector_store, doc_processor)

        return pipeline


def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = True

    if "top_k" not in st.session_state:
        st.session_state.top_k = RAG_CONFIG["top_k"]

    if "temperature" not in st.session_state:
        st.session_state.temperature = GENERATION_CONFIG["temperature"]

    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = GENERATION_CONFIG["max_new_tokens"]

    if "uploaded_files_processed" not in st.session_state:
        st.session_state.uploaded_files_processed = set()

    if "selected_document" not in st.session_state:
        st.session_state.selected_document = "All Documents"

    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = {}


def render_sidebar(pipeline: RAGPipeline):
    """
    Render sidebar with document upload and settings.

    Args:
        pipeline: RAG pipeline instance
    """
    with st.sidebar:
        st.title("Document Management")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=list(ext.strip(".") for ext in SUPPORTED_EXTENSIONS.keys()),
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS.values())}"
        )

        # Process uploaded files
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                process_uploaded_files(uploaded_files, pipeline)

        st.divider()

        # Display indexed documents
        st.subheader("Indexed Documents")
        documents = pipeline.get_documents()

        if documents:
            # Create DataFrame for display
            doc_data = []
            for doc in documents:
                doc_data.append({
                    "Filename": doc["filename"],
                    "Type": doc["file_type"],
                    "Chunks": doc["chunk_count"]
                })

            st.dataframe(doc_data, use_container_width=True, hide_index=True)

            # Document actions
            with st.expander("Document Actions"):
                selected_doc = st.selectbox(
                    "Select document",
                    options=[doc["filename"] for doc in documents],
                    key="doc_action_selector"
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Summarize", key="summarize_btn"):
                        summarize_document(selected_doc, pipeline)
                with col2:
                    if st.button("Delete", key="delete_btn"):
                        count = pipeline.delete_document(selected_doc)
                        if selected_doc in st.session_state.uploaded_files_processed:
                            st.session_state.uploaded_files_processed.remove(selected_doc)
                        st.success(f"Deleted {count} chunks from {selected_doc}")
                        st.rerun()

            # Document management buttons
            if st.button("Clear All Documents"):
                count = pipeline.clear_all_documents()
                st.session_state.uploaded_files_processed.clear()
                st.session_state.document_summaries.clear()
                st.success(f"Cleared {count} chunks")
                st.rerun()

            # Show stats
            with st.expander("Vector Store Stats"):
                stats = pipeline.get_stats()
                st.write(f"Total chunks: {stats['vector_store']['total_chunks']}")
                st.write(f"Total documents: {stats['vector_store']['total_documents']}")
                st.write(f"Embedding model: {stats['vector_store']['embedding_model']}")
                st.write(f"Embedding dimension: {stats['vector_store']['embedding_dimension']}")
        else:
            st.info("No documents indexed yet. Upload files above to get started.")

        st.divider()

        # RAG Settings
        st.subheader("Settings")

        st.session_state.rag_enabled = st.checkbox(
            "Enable RAG",
            value=st.session_state.rag_enabled,
            help="Use document context to enhance responses"
        )

        if st.session_state.rag_enabled and not pipeline.vector_store.is_empty():
            # Document selector
            doc_options = ["All Documents"] + [doc["filename"] for doc in documents]
            st.session_state.selected_document = st.selectbox(
                "Focus on Document",
                options=doc_options,
                index=doc_options.index(st.session_state.selected_document) if st.session_state.selected_document in doc_options else 0,
                help="Retrieve chunks only from selected document, or all documents"
            )

            st.session_state.top_k = st.slider(
                "Context Chunks",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                help="Number of document chunks to retrieve"
            )

        # Generation settings
        with st.expander("Generation Settings"):
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )

            st.session_state.max_tokens = st.slider(
                "Max Tokens",
                min_value=50,
                max_value=2048,
                value=st.session_state.max_tokens,
                step=50,
                help="Maximum length of response"
            )


def summarize_document(filename: str, pipeline: RAGPipeline):
    """
    Generate a summary for a specific document.

    Args:
        filename: Name of the document to summarize
        pipeline: RAG pipeline instance
    """
    with st.spinner(f"Generating summary for {filename}..."):
        # Get all chunks from this document
        doc_info = pipeline.vector_store.get_document_info(filename)

        if not doc_info:
            st.error(f"Document {filename} not found")
            return

        # Retrieve multiple chunks to get good coverage
        summary_prompt = f"Provide a comprehensive summary of the key points and main topics covered in {filename}. Include the most important information and insights."

        # Retrieve more chunks for summarization
        context_chunks = pipeline.retrieve_context(
            query=summary_prompt,
            top_k=min(10, doc_info["chunk_count"]),  # Get up to 10 chunks
            min_similarity=0.3  # Lower threshold for summarization
        )

        # Filter to only this document
        context_chunks = [c for c in context_chunks if c["metadata"]["filename"] == filename]

        if not context_chunks:
            st.warning(f"Could not retrieve content from {filename}")
            return

        # Generate summary
        gen_config = {
            "max_new_tokens": 512,
            "temperature": 0.6,  # Lower for more factual summary
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }

        result = pipeline.generate_response(
            query=summary_prompt,
            context_chunks=context_chunks,
            generation_config=gen_config
        )

        # Store summary
        st.session_state.document_summaries[filename] = result["response"]

        # Add to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": f"Summarize {filename}"
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "sources": context_chunks
        })

        # Show success message
        st.success(f"Summary generated for {filename} - check the chat!")
        st.rerun()


def process_uploaded_files(uploaded_files, pipeline: RAGPipeline):
    """
    Process and index uploaded files.

    Args:
        uploaded_files: List of uploaded file objects
        pipeline: RAG pipeline instance
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        # Skip if already processed
        if uploaded_file.name in st.session_state.uploaded_files_processed:
            continue

        status_text.text(f"Processing {uploaded_file.name}...")

        # Save to temporary file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Process with pipeline
            result = pipeline.ingest_document(temp_path)

            if result["success"]:
                st.success(result["message"])
                st.session_state.uploaded_files_processed.add(uploaded_file.name)
            else:
                st.error(result["message"])

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("Processing complete!")
    st.rerun()


def render_chat_interface(pipeline: RAGPipeline):
    """
    Render main chat interface.

    Args:
        pipeline: RAG pipeline instance
    """
    st.title("Llama 3.1 8B Chat with RAG")

    # Display RAG status
    if st.session_state.rag_enabled and not pipeline.vector_store.is_empty():
        doc_count = len(pipeline.get_documents())
        if st.session_state.selected_document != "All Documents":
            st.info(f"RAG Mode: Active | Focused on: {st.session_state.selected_document}")
        else:
            st.info(f"RAG Mode: Active | {doc_count} documents indexed")
    elif st.session_state.rag_enabled and pipeline.vector_store.is_empty():
        st.warning("RAG enabled but no documents indexed. Upload documents in the sidebar.")
    else:
        st.info("RAG Mode: Disabled")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander(f"Sources ({len(message['sources'])} chunks used)"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**[{i+1}] {source['metadata']['filename']}** (Chunk {source['metadata']['chunk_index'] + 1}, Similarity: {source['similarity']:.2f})")
                        st.text(source["text"][:200] + "..." if len(source["text"]) > 200 else source["text"])
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = generate_response(prompt, pipeline)

                # Display response
                st.markdown(response_data["response"])

                # Display sources if RAG was used
                if response_data["sources"]:
                    with st.expander(f"Sources ({len(response_data['sources'])} chunks used)"):
                        for i, source in enumerate(response_data["sources"]):
                            st.markdown(f"**[{i+1}] {source['metadata']['filename']}** (Chunk {source['metadata']['chunk_index'] + 1}, Similarity: {source['similarity']:.2f})")
                            st.text(source["text"][:200] + "..." if len(source["text"]) > 200 else source["text"])
                            st.divider()

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["response"],
            "sources": response_data["sources"]
        })


def generate_response(prompt: str, pipeline: RAGPipeline) -> dict:
    """
    Generate response using RAG pipeline.

    Args:
        prompt: User prompt
        pipeline: RAG pipeline instance

    Returns:
        Response data dictionary
    """
    # Prepare generation config
    gen_config = {
        "max_new_tokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }

    # Format conversation history
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]  # Exclude current message
    ]

    # Generate with or without RAG
    if st.session_state.rag_enabled and not pipeline.vector_store.is_empty():
        # Retrieve context with optional document filter
        filter_metadata = None
        if st.session_state.selected_document != "All Documents":
            filter_metadata = {"filename": st.session_state.selected_document}

        # Get context chunks
        context_chunks = pipeline.retrieve_context(
            query=prompt,
            top_k=st.session_state.top_k,
            min_similarity=None  # Use default from config
        )

        # Filter by document if specified
        if filter_metadata:
            context_chunks = [c for c in context_chunks if c["metadata"]["filename"] == filter_metadata["filename"]]

        # Generate with context
        result = pipeline.generate_response(
            query=prompt,
            context_chunks=context_chunks,
            history=history,
            generation_config=gen_config
        )
    else:
        result = pipeline.generate_response(
            query=prompt,
            history=history,
            generation_config=gen_config
        )

    return result


def main():
    """
    Main application entry point.
    """
    # Initialize session state
    initialize_session_state()

    # Load model and pipeline
    try:
        model, tokenizer = load_model_and_tokenizer()
        pipeline = initialize_rag_pipeline(model, tokenizer)
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()

    # Render UI
    render_sidebar(pipeline)
    render_chat_interface(pipeline)

    # Footer
    st.sidebar.divider()
    st.sidebar.caption("Llama 3.2 3B with RAG | Powered by Streamlit")


if __name__ == "__main__":
    main()
