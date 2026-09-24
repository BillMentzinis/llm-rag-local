"""
Streamlit UI for the local RAG chat application.
"""

import os
import html
import itertools
import tempfile
import streamlit as st
from datetime import datetime
from typing import Optional

from config import (
    GENERATION_CONFIG, UI_CONFIG, RAG_CONFIG, SUPPORTED_EXTENSIONS, AVAILABLE_MODELS,
    DEFAULT_MODEL, OLLAMA_CONFIG, CHAT_CONFIG
)
from llm_backends import (
    LLMBackend, OLLAMA_PREFIX, TRANSFORMERS_PREFIX, load_backend, release_all,
    list_ollama_models, model_label, short_label, default_model_key, cuda_available
)
from rag_pipeline import RAGPipeline
from vector_store_manager import VectorStoreManager
from document_processor import DocumentProcessor
from chat_manager import save_chat, load_chat, list_chats, delete_chat, auto_name_from_message


NO_CONTEXT_NOTE = "No relevant document context found, so this answer uses the model's general knowledge."
STOPPED_NOTE = "Stopped before the answer was complete."

# Neutral icon avatars that suit the theme (Streamlit's defaults are red and orange)
AVATARS = {"user": ":material/person:", "assistant": ":material/neurology:"}

# A copy button in an st.iframe (the only place a click can reach the clipboard:
# the iframe allows scripts, same-origin access and clipboard-write). The HTML
# starts with markup, so st.iframe always treats it as HTML (srcdoc), never as a
# URL or file path. The text travels in an HTML-escaped data attribute, never as code.
# It borrows the app's text colour, accent and font so it matches the theme,
# and falls back to execCommand where navigator.clipboard is unavailable (plain
# http on a LAN address, which isn't a secure context).
COPY_BUTTON_HTML = """
<style>
  body {{ margin: 0; overflow: hidden; }}
  button {{
    display: inline-flex; align-items: center; gap: 0.5rem; height: 2.5rem;
    padding: 0; border: none; background: none; cursor: pointer;
    font: 14px/1.6 "Source Sans", sans-serif; color: var(--text, rgb(49, 51, 63));
  }}
  button:hover {{ color: var(--accent, var(--text)); }}
  svg {{ width: 16px; height: 16px; fill: currentColor; }}
</style>
<button id="copy" data-text="{text}" title="Copy to clipboard">
  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 18q-.825 0-1.412-.587T7 16V4q0-.825.588-1.412T9 2h9q.825 0 1.413.588T20 4v12q0 .825-.587 1.413T18 18zm0-2h9V4H9zm-4 6q-.825 0-1.412-.587T3 20V7q0-.425.288-.712T4 6t.713.288T5 7v13h10q.425 0 .713.288T16 21t-.288.713T15 22zm4-6V4z"/></svg>
  <span>Copy</span>
</button>
<script>
  const button = document.getElementById("copy");
  const label = button.querySelector("span");
  try {{
    const doc = window.parent.document;
    const parentStyle = window.parent.getComputedStyle(doc.body);
    const root = document.documentElement.style;
    root.setProperty("--text", parentStyle.color);
    button.style.fontFamily = parentStyle.fontFamily;
    // The theme's accent colour, as used by the sidebar's primary button
    const primary = doc.querySelector('[data-testid="stBaseButton-primary"]');
    if (primary) root.setProperty("--accent", window.parent.getComputedStyle(primary).backgroundColor);
  }} catch (e) {{}}
  button.addEventListener("click", async () => {{
    const text = button.dataset.text;
    let copied = false;
    try {{
      await navigator.clipboard.writeText(text);
      copied = true;
    }} catch (e) {{
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      copied = document.execCommand("copy");
      area.remove();
    }}
    label.textContent = copied ? "Copied" : "Copy failed";
    setTimeout(() => {{ label.textContent = "Copy"; }}, 1500);
  }});
</script>
"""


# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"]
)


@st.cache_resource(show_spinner=False)
def load_llm(model_key: str) -> LLMBackend:
    """
    Load a model (cached per model key).

    Args:
        model_key: "transformers:<Hugging Face id>" or "ollama:<model name>"
    """
    with st.spinner(f"Loading {model_label(model_key)}... This may take a minute."):
        return load_backend(model_key)


@st.cache_resource(show_spinner="Loading the document index...")
def load_retrieval():
    """
    Load the vector store and document processor (cached; independent of the model).

    Returns:
        Tuple of (VectorStoreManager, DocumentProcessor)
    """
    return VectorStoreManager(), DocumentProcessor()


@st.cache_data(ttl=10, show_spinner=False)
def available_ollama_models() -> list:
    """Models the local Ollama server has (checked at most every 10 seconds)."""
    return list_ollama_models()


def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False

    if "top_k" not in st.session_state:
        st.session_state.top_k = RAG_CONFIG["top_k"]

    if "temperature" not in st.session_state:
        st.session_state.temperature = GENERATION_CONFIG["temperature"]

    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = GENERATION_CONFIG["max_new_tokens"]

    if "selected_document" not in st.session_state:
        st.session_state.selected_document = "All Documents"

    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = {}

    if "current_chat_name" not in st.session_state:
        st.session_state.current_chat_name = None

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = default_model_key(DEFAULT_MODEL, available_ollama_models())

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def notify(message: str, icon: str = None, kind: str = "success"):
    """
    Queue a message for the next run (anything shown before st.rerun() would be wiped).

    Messages are shown inline rather than as toasts: a toast sent while the
    previous one is still on screen is silently dropped by Streamlit.

    Args:
        message: Text to show
        icon: Optional Material icon, e.g. ":material/check:"
        kind: "success", "info" or "error"
    """
    st.session_state.setdefault("notifications", []).append((kind, message, icon))


def show_notifications():
    """Show the messages queued with notify(); they clear on the next interaction."""
    for kind, message, icon in st.session_state.pop("notifications", []):
        getattr(st, kind)(message, icon=icon)


def current_chat_title() -> str:
    """Name of the current chat: its saved name, or one derived from the first question."""
    if st.session_state.current_chat_name:
        return st.session_state.current_chat_name
    first_user = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), None)
    return auto_name_from_message(first_user) if first_user else datetime.now().strftime("Chat_%Y%m%d_%H%M%S")


def save_current_chat(announce: bool = True):
    """Save the current chat, if it has any messages (button callback)."""
    # Callbacks run before the script, so an answer interrupted by this very
    # click hasn't been recovered yet; do it now so it's saved with its chat
    recover_interrupted_response()
    if not st.session_state.messages:
        return
    name = current_chat_title()
    st.session_state.current_chat_name = name
    save_chat(name, st.session_state.messages)
    if announce:
        notify(f"Saved \u201c{name}\u201d", ":material/check:")


def start_new_chat():
    """Save the current chat and start an empty one (button callback)."""
    save_current_chat(announce=False)
    st.session_state.messages = []
    st.session_state.current_chat_name = None


def open_chat(chat: dict):
    """Switch to a saved chat, saving the current one first (button callback)."""
    if chat["name"] == st.session_state.current_chat_name:
        return
    loaded = load_chat(chat["filepath"])
    if loaded is None:
        notify("Could not load that chat.", ":material/error:", kind="error")
        return
    save_current_chat(announce=False)
    st.session_state.messages = loaded
    st.session_state.current_chat_name = chat["name"]


def remove_chat(chat: dict):
    """Delete a saved chat (button callback)."""
    delete_chat(chat["filepath"])
    if st.session_state.current_chat_name == chat["name"]:
        st.session_state.current_chat_name = None
    notify(f"Deleted \u201c{chat['name']}\u201d", ":material/delete:")


def render_chat_sidebar(pipeline: RAGPipeline):
    """
    Render the chat page's sidebar: chat actions, settings and saved chats.

    Args:
        pipeline: RAG pipeline instance
    """
    with st.sidebar:
        with st.container(horizontal=True):
            st.button("New chat", icon=":material/add:", type="primary", width="stretch",
                      on_click=start_new_chat, help="Save this chat and start a new one")
            st.button("Save", icon=":material/save:", on_click=save_current_chat,
                      disabled=not st.session_state.messages, help="Save this chat")
        show_notifications()

        # Model selector: Ollama's models (while it's running), then the Hugging Face ones
        model_keys = [OLLAMA_PREFIX + m for m in available_ollama_models()]
        model_keys += [TRANSFORMERS_PREFIX + m for m in AVAILABLE_MODELS]
        if st.session_state.selected_model not in model_keys:
            model_keys.insert(0, st.session_state.selected_model)  # e.g. Ollama was stopped
        chosen_model = st.selectbox(
            "Model",
            options=model_keys,
            index=model_keys.index(st.session_state.selected_model),
            format_func=model_label,
            help=f"Ollama models are listed while Ollama is running ({OLLAMA_CONFIG['host']}). "
                 "HF (Hugging Face) models run inside the app and need an NVIDIA GPU."
        )
        if chosen_model != st.session_state.selected_model:
            st.session_state.selected_model = chosen_model
            st.session_state.pop("llm_error", None)
            load_llm.clear()
            release_all()
            st.rerun()

        # RAG settings
        st.session_state.rag_enabled = st.toggle(
            "Answer from documents",
            value=st.session_state.rag_enabled,
            help="Use your indexed documents as context for answers (RAG)"
        )

        if st.session_state.rag_enabled:
            documents = pipeline.get_documents()
            if documents:
                doc_options = ["All Documents"] + [doc["filename"] for doc in documents]
                st.session_state.selected_document = st.selectbox(
                    "Focus on document",
                    options=doc_options,
                    index=doc_options.index(st.session_state.selected_document) if st.session_state.selected_document in doc_options else 0,
                    help="Retrieve chunks only from selected document, or all documents"
                )

                st.session_state.top_k = st.slider(
                    "Context chunks",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.top_k,
                    help="Number of document chunks to retrieve"
                )
            else:
                st.caption("No documents indexed yet.")
                st.page_link(PAGES["documents"], label="Add documents", icon=":material/upload_file:")

        # Generation settings
        with st.expander("Generation settings"):
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )

            st.session_state.max_tokens = st.slider(
                "Max tokens",
                min_value=50,
                max_value=2048,
                value=st.session_state.max_tokens,
                step=50,
                help="Maximum length of response"
            )
            st.caption(f"The model sees up to your last {CHAT_CONFIG['history_turns']} questions and "
                       "answers, fewer if they don't fit its context window.")

        st.divider()

        # Saved chats
        st.subheader("Chats")
        saved = list_chats()
        if not saved:
            st.caption("Saved chats will appear here.")
        for chat in saved:
            is_current = chat["name"] == st.session_state.current_chat_name
            label = chat["name"] if len(chat["name"]) <= 30 else chat["name"][:28].rstrip() + "\u2026"
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.button(label, key=f"load_{chat['filepath']}", type="secondary" if is_current else "tertiary",
                          on_click=open_chat, args=(chat,),
                          help=f"{chat['name']} \u00b7 {chat['message_count']} messages")
                st.button(":material/delete:", key=f"del_{chat['filepath']}", type="tertiary",
                          on_click=remove_chat, args=(chat,), help="Delete this chat")

        st.divider()
        st.caption("Local LLM with RAG | Powered by Streamlit")


def format_timestamp(timestamp: str) -> str:
    """Format an ISO timestamp for display, or return it unchanged if it can't be parsed."""
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return timestamp or ""


def render_documents_page(pipeline: RAGPipeline):
    """
    Render the documents page: upload, the index's contents, and document actions.

    Args:
        pipeline: RAG pipeline instance
    """
    st.title("Documents")
    st.caption("Indexed documents are used as context when **Answer from documents** is on in the chat.")
    show_notifications()

    # Upload
    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Add documents",
            type=list(ext.strip(".") for ext in SUPPORTED_EXTENSIONS.keys()),
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS.values())}"
        )
        if uploaded_files and st.button("Process documents", type="primary", icon=":material/upload:"):
            process_uploaded_files(uploaded_files, pipeline)

        # Results of the last upload, shown once after the rerun
        for result in st.session_state.pop("ingest_results", []):
            if not result["success"]:
                st.error(result["message"], icon=":material/error:")
            elif result.get("skipped"):
                st.info(result["message"], icon=":material/check:")
            else:
                st.success(result["message"], icon=":material/check_circle:")

    documents = pipeline.get_documents()
    if not documents:
        st.info("No documents indexed yet. Add some above to get started.", icon=":material/info:")
        return

    stats = pipeline.get_stats()["vector_store"]
    with st.container(horizontal=True):
        st.metric("Documents", stats["total_documents"])
        st.metric("Chunks", stats["total_chunks"])

    st.dataframe(
        [
            {
                "Document": doc["filename"],
                "Type": doc["file_type"],
                "Chunks": doc["chunk_count"],
                "Added": format_timestamp(doc["upload_timestamp"]),
            }
            for doc in documents
        ],
        width="stretch",
        hide_index=True
    )
    st.caption(f"Embedded with {stats['embedding_model']} ({stats['embedding_dimension']} dimensions)")

    # Document actions
    with st.container(horizontal=True, vertical_alignment="bottom"):
        selected_doc = st.selectbox(
            "Document",
            options=[doc["filename"] for doc in documents],
            key="doc_action_selector"
        )
        if st.button("Summarize", icon=":material/summarize:", key="summarize_btn"):
            summarize_document(selected_doc, pipeline)
        if st.button("Delete", icon=":material/delete:", key="delete_btn"):
            count = pipeline.delete_document(selected_doc)
            notify(f"Deleted {selected_doc} ({count} chunks)", ":material/delete:")
            st.rerun()
        with st.popover("Clear all", icon=":material/delete_sweep:"):
            st.markdown(f"Remove all **{len(documents)}** documents from the index? This can't be undone.")
            if st.button("Clear all documents", type="primary", key="confirm_clear"):
                count = pipeline.clear_all_documents()
                st.session_state.document_summaries.clear()
                notify(f"Cleared all documents ({count} chunks)", ":material/delete_sweep:")
                st.rerun()


def summarize_document(filename: str, pipeline: RAGPipeline):
    """
    Generate a summary for a specific document.

    Args:
        filename: Name of the document to summarize
        pipeline: RAG pipeline instance
    """
    pipeline.llm = get_llm()
    if pipeline.llm is None:
        key, error = st.session_state.llm_error
        st.error(f"Can't summarize: {model_label(key)} didn't load ({error}). "
                 "Pick another model in the Chat page's sidebar.", icon=":material/error:")
        return

    with st.spinner(f"Generating summary for {filename}..."):
        # Get all chunks from this document
        doc_info = pipeline.vector_store.get_document_info(filename)

        if not doc_info:
            st.error(f"Document {filename} not found")
            return

        # Retrieve multiple chunks to get good coverage
        summary_prompt = f"Provide a comprehensive summary of the key points and main topics covered in {filename}. Include the most important information and insights."

        # Retrieve more chunks for summarization, from this document only
        context_chunks = pipeline.retrieve_context(
            query=summary_prompt,
            top_k=min(10, doc_info["chunk_count"]),  # Get up to 10 chunks
            min_similarity=0.0,  # Any chunk of the document is useful for a summary
            filter_metadata={"filename": filename}
        )

        if not context_chunks:
            st.warning(f"Could not retrieve content from {filename}")
            return

        # Generate summary
        gen_config = {**GENERATION_CONFIG, "temperature": 0.6}  # a little more focused for summaries

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

        st.switch_page(PAGES["chat"])


def process_uploaded_files(uploaded_files, pipeline: RAGPipeline):
    """
    Process and index uploaded files.

    Args:
        uploaded_files: List of uploaded file objects
        pipeline: RAG pipeline instance
    """
    progress_bar = st.progress(0.0)
    results = []

    for i, uploaded_file in enumerate(uploaded_files):
        # Keep only the base name so the upload can't escape the temp directory;
        # the file keeps its original name, which becomes the document's name
        filename = os.path.basename(uploaded_file.name)
        progress_bar.progress(i / len(uploaded_files), text=f"Processing {filename}...")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, filename)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process with pipeline (skips files that are indexed and unchanged)
                results.append(pipeline.ingest_document(temp_path))

        except Exception as e:
            results.append({"success": False, "message": f"Error processing {filename}: {str(e)}"})

    progress_bar.progress(1.0, text="Processing complete!")

    # Show the results after the rerun, and give the uploader a new key to clear it
    st.session_state.ingest_results = results
    st.session_state.uploader_key += 1
    st.rerun()


def render_status_bar(pipeline: RAGPipeline):
    """
    Render a compact row of badges: model, document (RAG) state and hardware.

    Args:
        pipeline: RAG pipeline instance
    """
    model_name = short_label(st.session_state.selected_model)
    with st.container(horizontal=True, gap="small"):
        st.badge(model_name, icon=":material/neurology:", color="primary")

        if not st.session_state.rag_enabled:
            st.badge("Documents off", icon=":material/description:", color="gray")
        else:
            doc_count = len(pipeline.get_documents())
            if doc_count == 0:
                st.badge("No documents indexed", icon=":material/warning:", color="orange")
            elif st.session_state.selected_document != "All Documents":
                st.badge(f"Focused on {st.session_state.selected_document}",
                         icon=":material/center_focus_strong:", color="green")
            else:
                st.badge(f"Documents on \u00b7 {doc_count} indexed", icon=":material/description:", color="green")

        if pipeline.llm is not None:
            status = pipeline.llm.status()
            if status:
                st.badge(status, icon=":material/memory:", color="gray")
        else:
            st.badge("Model not loaded", icon=":material/error:", color="red")


def render_welcome():
    """Render the greeting shown in place of an empty chat."""
    key = st.session_state.selected_model
    st.header("What can I help with?", anchor=False)
    where = "via Ollama" if key.startswith(OLLAMA_PREFIX) else "running inside this app"
    hint = f"Answers come from {short_label(key)}, {where}."
    if not st.session_state.rag_enabled:
        hint += " Turn on **Answer from documents** in the sidebar to ask about your files."
    st.caption(hint)


def render_chat_interface(pipeline: RAGPipeline):
    """
    Render main chat interface.

    Args:
        pipeline: RAG pipeline instance
    """
    render_status_bar(pipeline)

    # Read the chat input first (it's pinned to the bottom wherever it's called),
    # so the history knows whether a new answer is about to be generated
    prompt = st.chat_input("Ask me anything...", disabled=pipeline.llm is None)
    regenerate = st.session_state.pop("regenerate_requested", False) and pipeline.llm is not None

    if not st.session_state.messages and not prompt:
        render_welcome()

    # Display chat history
    messages = st.session_state.messages
    for i, message in enumerate(messages):
        with st.chat_message(message["role"], avatar=AVATARS.get(message["role"])):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                # Regenerate removes the answer first, so only offer it when a model can replace it
                can_regenerate = i == len(messages) - 1 and not prompt and pipeline.llm is not None
                render_assistant_details(message, can_regenerate=can_regenerate)

    # Below the history, just above the (disabled) chat input, where it'll be seen
    if pipeline.llm is None:
        render_llm_error()

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(prompt)

        respond(prompt, pipeline)

    # Regenerate: the last answer was already removed, so answer the last question again
    elif regenerate and messages and messages[-1]["role"] == "user":
        respond(messages[-1]["content"], pipeline)


def respond(prompt: str, pipeline: RAGPipeline):
    """
    Generate and stream the assistant's answer to the last user message.

    Args:
        prompt: User prompt being answered (already the last message)
        pipeline: RAG pipeline instance
    """
    # Generate response, streaming it as it's produced
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        message = {"role": "assistant", "content": "", "sources": []}
        # Kept in session state so that if this run is interrupted (the Stop
        # button or any other click), the partial answer survives the rerun
        st.session_state.pending_response = message

        body = st.container()
        stop_slot = st.empty()
        stop_slot.button("Stop generating", key="stop_generation")

        stream = None
        try:
            with body:
                with st.spinner("Thinking..."):
                    response_data = start_response(prompt, pipeline)
                    message["sources"] = response_data["sources"]
                    message["rag_no_context"] = response_data["rag_attempted"] and not response_data["sources"]
                    stream = response_data["stream"]
                    first_piece = next(stream, "")
                st.write_stream(record_stream(itertools.chain([first_piece], stream), message))
        except Exception as e:
            message["error"] = str(e)
        finally:
            # Also runs when Streamlit interrupts the run, which stops generation
            if stream is not None:
                stream.close()

        stop_slot.empty()
        del st.session_state.pending_response
        render_assistant_details(message, can_regenerate=True)

    st.session_state.messages.append(message)


def record_stream(pieces, message: dict):
    """
    Pass streamed text through while accumulating it into the message.

    Args:
        pieces: Iterator of text pieces
        message: Message dictionary whose content is extended in place

    Yields:
        The same text pieces
    """
    for piece in pieces:
        message["content"] += piece
        yield piece


def recover_interrupted_response():
    """
    Keep a response whose generation was interrupted by a rerun.

    Runs before anything else renders, so the sidebar's chat actions see it.
    """
    message = st.session_state.pop("pending_response", None)
    if message is not None:
        message["stopped"] = True
        st.session_state.messages.append(message)


def request_regenerate():
    """Drop the last answer so the next run generates a new one (button callback)."""
    messages = st.session_state.messages
    if messages and messages[-1]["role"] == "assistant":
        messages.pop()
        st.session_state.regenerate_requested = True


def render_message_actions(message: dict, can_regenerate: bool):
    """
    Render the Copy and Regenerate buttons under an assistant message.

    Args:
        message: Assistant message dictionary
        can_regenerate: Whether to offer Regenerate (only for the latest answer)
    """
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        if message["content"]:
            st.iframe(COPY_BUTTON_HTML.format(text=html.escape(message["content"], quote=True)),
                      width=70, height=40)
        if can_regenerate:
            st.button("Regenerate", key="regenerate", icon=":material/refresh:", type="tertiary",
                      on_click=request_regenerate, help="Answer the last question again")


def render_assistant_details(message: dict, can_regenerate: bool = False):
    """
    Render the notes, sources and actions shown under an assistant message.

    Args:
        message: Assistant message dictionary
        can_regenerate: Whether to offer Regenerate (only for the latest answer)
    """
    if message.get("error"):
        st.error(f"Generation failed: {message['error']}")
    elif message.get("stopped"):
        st.caption(STOPPED_NOTE)

    if message.get("rag_no_context"):
        st.caption(NO_CONTEXT_NOTE)

    # Show sources if available
    if message.get("sources"):
        with st.expander(f"Sources ({len(message['sources'])} chunks used)"):
            for i, source in enumerate(message["sources"]):
                st.markdown(f"**[{i+1}] {source['metadata']['filename']}** (Chunk {source['metadata']['chunk_index'] + 1}, Similarity: {source['similarity']:.2f})")
                st.text(source["text"][:200] + "..." if len(source["text"]) > 200 else source["text"])
                st.divider()

    render_message_actions(message, can_regenerate)


def start_response(prompt: str, pipeline: RAGPipeline) -> dict:
    """
    Retrieve context if RAG is on, and start streaming the response.

    Args:
        prompt: User prompt
        pipeline: RAG pipeline instance

    Returns:
        Response data dictionary with a "stream" of text pieces
    """
    # Prepare generation config
    gen_config = {
        **GENERATION_CONFIG,
        "max_new_tokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
    }

    # Earlier messages; the pipeline picks the recent complete question/answer pairs
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
            min_similarity=None,  # Use default from config
            filter_metadata=filter_metadata
        )

        # Fall back to all documents if the focused one has nothing relevant
        if filter_metadata and not context_chunks:
            st.warning(f"No relevant chunks found in '{st.session_state.selected_document}'. Searching all documents.")
            context_chunks = pipeline.retrieve_context(
                query=prompt,
                top_k=st.session_state.top_k,
                min_similarity=None
            )

        # Generate with context
        result = pipeline.stream_response(
            query=prompt,
            context_chunks=context_chunks,
            history=history,
            generation_config=gen_config
        )
        result["rag_attempted"] = True
    else:
        result = pipeline.stream_response(
            query=prompt,
            history=history,
            generation_config=gen_config
        )
        result["rag_attempted"] = False

    return result


def get_llm() -> Optional[LLMBackend]:
    """
    Load the selected model.

    Returns:
        The backend, or None if it failed to load. The error is kept in session
        state so a failing model isn't retried on every rerun.
    """
    key = st.session_state.selected_model
    failed = st.session_state.get("llm_error")
    if failed and failed[0] == key:
        return None
    try:
        return load_llm(key)
    except Exception as e:
        st.session_state.llm_error = (key, str(e))
        return None


def render_llm_error():
    """Explain why the selected model didn't load, with a way to retry."""
    key, error = st.session_state.llm_error
    with st.container(border=True):
        st.error(f"Couldn't load {model_label(key)}: {error}", icon=":material/error:")
        if key.startswith(TRANSFORMERS_PREFIX) and not cuda_available():
            st.markdown(
                "Ollama can run models on this machine instead: install it from "
                "[ollama.com](https://ollama.com), run `ollama pull llama3.1:8b`, and pick the model "
                "from the **Model** list in the sidebar."
            )
        elif key.startswith(OLLAMA_PREFIX):
            st.markdown("Check that Ollama is running, then try again.")
        if st.button("Try again", icon=":material/refresh:"):
            del st.session_state.llm_error
            st.rerun()


def get_pipeline() -> RAGPipeline:
    """
    Build a RAG pipeline on the cached document index, without a model yet.

    Returns:
        RAGPipeline instance; set its llm with get_llm() before generating
    """
    vector_store, doc_processor = load_retrieval()
    return RAGPipeline(None, vector_store, doc_processor)


def chat_page():
    """The chat page."""
    pipeline = get_pipeline()
    render_chat_sidebar(pipeline)  # before loading the model, so it stays usable if loading fails
    pipeline.llm = get_llm()
    render_chat_interface(pipeline)


def documents_page():
    """The documents page (it only needs a model to summarize)."""
    render_documents_page(get_pipeline())


# Filled in by main() so pages can link to each other
PAGES = {}


def main():
    """
    Main application entry point.
    """
    # Initialize session state
    initialize_session_state()
    recover_interrupted_response()

    PAGES["chat"] = st.Page(chat_page, title="Chat", icon=":material/chat:", default=True)
    PAGES["documents"] = st.Page(documents_page, title="Documents", icon=":material/description:",
                                 url_path="documents")
    st.navigation(list(PAGES.values()), position="top").run()


if __name__ == "__main__":
    main()
