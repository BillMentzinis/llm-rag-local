# LLM RAG Local

![Python](https://img.shields.io/badge/python-3.11-blue)

A local Retrieval-Augmented Generation (RAG) chat application with a Streamlit UI. Runs entirely on your machine — no cloud API keys required.

## Quick Start

There are two ways to run the model. Pick one; you can switch between them from the sidebar at any time.

### Option A: Ollama (any computer)

Works on macOS, Windows and Linux, with or without a GPU (NVIDIA, AMD and Apple GPUs are used when present).

```bash
# 1. Install Ollama from https://ollama.com, then download a model
ollama pull llama3.1:8b

# 2. Install the app's dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Launch
streamlit run streamlit_app.py
```

The models you've pulled appear in the sidebar's **Model** list while Ollama is running. On a machine without an NVIDIA GPU the app starts on your first Ollama model automatically; to choose it explicitly, set `LLM_MODEL=ollama:llama3.1:8b`. If Ollama runs somewhere other than `http://localhost:11434`, set `OLLAMA_HOST`.

### Option B: Hugging Face models inside the app (NVIDIA GPU)

The app loads the model itself, 4-bit quantized with bitsandbytes (~2–8 GB of VRAM depending on the model).

```bash
# 1. Create and activate the conda environment (includes CUDA)
conda env create -f environment.yml
conda activate llm-local

# 2. Launch
streamlit run streamlit_app.py
```

LLM weights are downloaded from Hugging Face on first use and cached at `~/.cache/huggingface/hub/`.

> **Note:** Models gated on Hugging Face (e.g. Llama) require an account with access. Log in with `huggingface-cli login` before starting.

Either way, the app opens at `http://localhost:8501`. Run it from the repository folder so Streamlit picks up the theme in `.streamlit/config.toml`. On first run it downloads the embedding model (~90 MB).

---

## Features

- Two model backends: any model you've pulled into [Ollama](https://ollama.com) (runs on any computer), or Hugging Face models run inside the app, 4-bit quantized (NVIDIA GPU)
- Switch models from the sidebar without restarting
- Streaming responses: text appears as it's generated, with a **Stop generating** button that halts the model and keeps the partial answer
- **Copy** any answer (as raw Markdown) and **Regenerate** the latest one
- Light and dark themes that follow your system setting, and a status bar showing the model, whether answers use your documents, and GPU memory in use
- RAG support: upload documents and get cited, grounded answers
- Chat history: save, load, and delete named chat sessions (stored as JSON)
- Supported file types: PDF, TXT, MD, and common code files
- Document management: upload, index, summarize, delete, view stats
- Persistent vector storage with ChromaDB
- Configurable generation and retrieval parameters

## Available Models

**Ollama:** every model you've pulled (`ollama list`) is offered, e.g. `llama3.1:8b`, `qwen2.5:7b`, `phi4`, `mistral`. Browse more at [ollama.com/library](https://ollama.com/library).

**Hugging Face (NVIDIA GPU):**

| Model | Display Name | VRAM (4-bit) | Notes |
|-------|-------------|--------------|-------|
| `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 8B | ~4 GB | Default |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama 3.2 3B | ~2 GB | Faster, newer |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen 2.5 7B | ~4 GB | Strong quality |
| `microsoft/Phi-3.5-mini-instruct` | Phi-3.5 Mini 3.8B | ~2 GB | Fast, efficient |
| `microsoft/phi-4` | Phi-4 14B | ~8 GB | Needs larger GPU |

Switch models using the **Model** selector in the chat sidebar: Ollama models are listed as "· Ollama", Hugging Face ones as "· HF". Switching releases the previous model's memory; documents and chat history are unaffected.

## Architecture

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| LLM | Ollama, or Hugging Face transformers 4-bit via bitsandbytes (`llm_backends.py`) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| PDF parsing | PyMuPDF |

## File Structure

```
LLM/
├── streamlit_app.py        # Main Streamlit UI
├── rag_pipeline.py         # RAG orchestration
├── llm_backends.py         # Ollama and Hugging Face model backends
├── document_processor.py   # File parsing and chunking
├── vector_store_manager.py # ChromaDB operations
├── chat_manager.py         # Chat save/load/delete helpers
├── config.py               # All configuration parameters
├── .streamlit/config.toml  # Light and dark theme
├── requirements.txt        # pip install (Ollama backend, any OS)
├── environment.yml         # Conda environment (includes CUDA, for Hugging Face models)
├── tests/                  # pytest suite (runs offline, no GPU needed)
├── chats/                  # Saved chat sessions (auto-created, gitignored)
└── chroma_db/              # Vector database (auto-created, gitignored)
```

## Usage

The app has two pages, switched from the bar at the top: **Chat** and **Documents**.

### Basic Chat (no documents)

1. Start the app
2. Leave **Answer from documents** off in the sidebar
3. Chat normally using the model's base knowledge

### RAG-Enhanced Chat

1. **Add documents** — on the **Documents** page, upload files and click **Process documents**. Uploading a file with the same name as an indexed one replaces it; unchanged files are skipped
2. **Turn on RAG** — switch on **Answer from documents** in the chat sidebar, optionally focusing on a single document
3. **Ask questions** — the model retrieves relevant chunks and cites sources

The Documents page also lists what's indexed, and can **Summarize** a document into the chat, **Delete** one, or **Clear all** (after a confirmation).

### Chat History

- **Save** — saves the current conversation to `chats/`, named after your first message
- **New chat** — saves the current conversation and starts a fresh one
- **Chats** list — click a chat to open it (the current one is saved first); the bin icon deletes it

### Adjusting Settings

| Setting | Where | Description |
|---------|-------|-------------|
| Model | Chat sidebar | Select which LLM to use |
| Focus on document | Chat sidebar, with RAG on | Retrieve only from one document |
| Context chunks | Chat sidebar, with RAG on | Number of document chunks to retrieve (1-10) |
| Temperature | Chat sidebar → Generation settings | Response randomness (0.1 = focused, 2.0 = creative) |
| Max tokens | Chat sidebar → Generation settings | Maximum response length |

## Configuration

All parameters are in `config.py`:

- **RAG**: `chunk_size` (200 tokens, overlap included), `chunk_overlap` (30), `top_k` (5), `min_similarity` (0.3, cosine similarity). The embedding model only reads the first 256 tokens of a chunk, so keep `chunk_size` below that if you raise it.
- **Generation**: `temperature` (0.7), `top_p` (0.9), `max_new_tokens` (512)
- **Models**: `DEFAULT_MODEL` (the model used on first launch; override with the `LLM_MODEL` environment variable, e.g. `ollama:llama3.1:8b` or `transformers:meta-llama/Llama-3.1-8B-Instruct`), `OLLAMA_CONFIG` (server address; override with `OLLAMA_HOST`), and `AVAILABLE_MODELS` (the Hugging Face models offered — add or remove models here)

## Supported File Types

**Documents:** PDF, TXT, MD
**Code:** Python, JavaScript, Java, C/C++, C#, Go, Rust, Ruby

## Running Tests

```bash
pip install pytest   # already included in requirements.txt and environment.yml
pytest tests
```

The tests use a small stand-in embedding model, a temporary ChromaDB and a fake Ollama server (`tests/fake_ollama.py`), so they run offline without a GPU, Ollama or model downloads.

## Performance

| Metric | Value |
|--------|-------|
| VRAM (default model) | ~4 GB |
| First run | Slow (downloads model weights) |
| Retrieval | <100 ms |
| Generation | ~20-30 tokens/second on a mid-range GPU; slower on CPU |

## Troubleshooting

**Model loads slowly** — expected on first run; subsequent loads use the local cache. With Ollama, the first answer after starting (or after a few idle minutes) waits while Ollama loads the model.

**No Ollama models in the Model list** — Ollama isn't running, or is running somewhere other than `http://localhost:11434` (set `OLLAMA_HOST`). Check with `ollama list`. The list refreshes every few seconds.

**"Ollama doesn't have the model"** — download it with `ollama pull <model>`.

**"Hugging Face models need an NVIDIA GPU"** — pick an Ollama model instead (see Option A above).

**Switching models uses a lot of VRAM** — the previous Hugging Face model is released when you switch. Ollama keeps a model loaded for a few minutes after its last use, and unloads it sooner if memory is needed. If you run out of memory, restart the app.

**Poor retrieval quality** — increase **Context chunks** in the sidebar, or lower `min_similarity` in `config.py`. When no chunk clears the threshold, the answer is marked as coming from the model's general knowledge.

**Upgrading from an older version** — on first launch, an existing `chroma_db/` is migrated automatically to cosine similarity (a one-time re-embedding of the stored chunks). Older versions stored uploads under a `temp_` prefix (e.g. `temp_report.pdf`); delete those on the Documents page and re-upload to get clean names. Documents indexed before the switch to 200-token chunks keep their old, larger chunks (whose second half isn't searchable) until you re-upload them. Re-uploading an unchanged file is normally skipped, but not when the chunk settings it was indexed with differ from the current ones, so a re-upload always picks up new `chunk_size`/`chunk_overlap` values.

**Out of memory** — reduce `max_new_tokens`, close other GPU apps, or switch to a smaller model.

**Document processing fails** — check file format is supported and size is <10 MB.
