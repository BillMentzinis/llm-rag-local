# LLM RAG Local

![Python](https://img.shields.io/badge/python-3.11-blue)

A local Retrieval-Augmented Generation (RAG) chat application with a Streamlit UI. Runs entirely on your machine — no cloud API keys required.

## Quick Start

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate llm-local

# 2. Launch the app
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. On first run it downloads the embedding model (~90 MB). LLM weights are downloaded from Hugging Face on first use and cached at `~/.cache/huggingface/hub/`.

> **Note:** Models gated on Hugging Face (e.g. Llama) require an account with access. Log in with `huggingface-cli login` before starting.

---

## Features

- Multi-model support — switch LLMs from the sidebar without restarting
- Chat interface with 4-bit quantized models (~2–8 GB VRAM depending on model)
- RAG support: upload documents and get cited, grounded answers
- Chat history: save, load, and delete named chat sessions (stored as JSON)
- Supported file types: PDF, TXT, MD, and common code files
- Document management: upload, index, summarize, delete, view stats
- Persistent vector storage with ChromaDB
- Configurable generation and retrieval parameters

## Available Models

| Model | Display Name | VRAM (4-bit) | Notes |
|-------|-------------|--------------|-------|
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 8B | ~4 GB | Default |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama 3.2 3B | ~2 GB | Faster, newer |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen 2.5 7B | ~4 GB | Strong quality |
| `microsoft/Phi-3.5-mini-instruct` | Phi-3.5 Mini 3.8B | ~2 GB | Fast, efficient |
| `microsoft/phi-4` | Phi-4 14B | ~8 GB | Needs larger GPU |

Switch models using the **Model** selector at the top of the sidebar. Switching clears the model cache and reloads — documents and chat history are unaffected.

## Architecture

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| LLM | Configurable (see table above), 4-bit via bitsandbytes |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| PDF parsing | PyMuPDF |

## File Structure

```
LLM/
├── streamlit_app.py        # Main Streamlit UI
├── rag_pipeline.py         # RAG orchestration
├── document_processor.py   # File parsing and chunking
├── vector_store_manager.py # ChromaDB operations
├── chat_manager.py         # Chat save/load/delete helpers
├── config.py               # All configuration parameters
├── environment.yml         # Conda environment
├── tests/                  # pytest suite (runs offline, no GPU needed)
├── chats/                  # Saved chat sessions (auto-created, gitignored)
└── chroma_db/              # Vector database (auto-created, gitignored)
```

## Usage

### Basic Chat (no documents)

1. Start the app
2. Disable "Enable RAG" in the sidebar
3. Chat normally using the model's base knowledge

### RAG-Enhanced Chat

1. **Upload documents** — click "Browse files" in the sidebar, select files, then "Process Documents". Uploading a file with the same name as an indexed one replaces it; unchanged files are skipped
2. **Enable RAG** — ensure "Enable RAG" is checked
3. **Ask questions** — the model retrieves relevant chunks and cites sources

### Chat History

- **Save Chat** — saves the current conversation to `chats/` named after your first message
- **New Chat** — auto-saves the current conversation and starts a fresh session
- **Saved Chats** expander — lists all saved sessions; click a name to load, click **X** to delete

### Adjusting Settings

| Setting | Description |
|---------|-------------|
| Model | Select which LLM to use |
| Context Chunks | Number of document chunks to retrieve (1-10) |
| Temperature | Response randomness (0.1 = focused, 2.0 = creative) |
| Max Tokens | Maximum response length |

## Configuration

All parameters are in `config.py`:

- **RAG**: `chunk_size` (512 tokens), `chunk_overlap` (50), `top_k` (3), `min_similarity` (0.3, cosine similarity)
- **Generation**: `temperature` (0.7), `top_p` (0.9), `max_new_tokens` (512)
- **Models**: `AVAILABLE_MODELS` dict — add or remove models here

## Supported File Types

**Documents:** PDF, TXT, MD
**Code:** Python, JavaScript, Java, C/C++, C#, Go, Rust, Ruby

## Running Tests

```bash
pip install pytest   # already included in environment.yml
pytest tests
```

The tests use a small stand-in embedding model and a temporary ChromaDB, so they run offline without a GPU or model downloads.

## Performance

| Metric | Value |
|--------|-------|
| VRAM (default model) | ~4 GB |
| First run | Slow (downloads model weights) |
| Retrieval | <100 ms |
| Generation | ~20-30 tokens/second (GPU-dependent) |

## Troubleshooting

**Model loads slowly** — expected on first run; subsequent loads use the local cache.

**Switching models uses a lot of VRAM** — the old model is evicted from cache when you switch. If you run out of memory, restart the app.

**Poor retrieval quality** — increase `top_k` in the sidebar, or lower `min_similarity` in `config.py`. When no chunk clears the threshold, the answer is marked as coming from the model's general knowledge.

**Upgrading from an older version** — on first launch, an existing `chroma_db/` is migrated automatically to cosine similarity (a one-time re-embedding of the stored chunks). Older versions stored uploads under a `temp_` prefix (e.g. `temp_report.pdf`); delete those from the sidebar and re-upload to get clean names.

**Out of memory** — reduce `max_new_tokens`, close other GPU apps, or switch to a smaller model.

**Document processing fails** — check file format is supported and size is <10 MB.
